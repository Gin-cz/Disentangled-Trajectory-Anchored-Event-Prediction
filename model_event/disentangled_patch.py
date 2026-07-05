import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 1. 辅助组件：GRL 与 HSIC 核心函数
# ============================================================================

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def hsic_loss(x, y):
    """计算 Hilbert-Schmidt Independence Criterion"""
    B = x.size(0)
    if B < 2: return torch.tensor(0.0, device=x.device)
    
    def rbf_kernel(mat):
        # 使用 p=2 的 cdist 计算欧氏距离平方
        dists = torch.cdist(mat, mat).pow(2)
        # 中位数启发式选择 sigma
        sigma = torch.median(dists[dists > 0]) if dists.max() > 0 else 1.0
        return torch.exp(-dists / (2 * sigma + 1e-7))
    
    K = rbf_kernel(x)
    L = rbf_kernel(y)
    H = torch.eye(B, device=x.device) - (1.0 / B)
    # HSIC 统计量公式
    return torch.trace(K @ H @ L @ H) / ((B - 1) ** 2)

# ============================================================================
# 2. Patch 提取模块 (5-Patch 重叠切分)
# ============================================================================

class TrajPatchEmbed(nn.Module):
    def __init__(self, in_dim=2, embed_dim=128, patch_size=32, stride=8):
        super().__init__()
        # 移除 traj_len 参数
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Conv1d(in_dim, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [N, T, F] -> [N, F, T]
        N, T, F = x.shape
        
        # ✅ 动态检查：如果序列太短，进行填充
        if T < self.patch_size:
            pad_len = self.patch_size - T
            x = F.pad(x, (0, 0, 0, pad_len))  # 在时间维度填充
            T = self.patch_size
        
        x = x.permute(0, 2, 1)
        x = self.proj(x)       # [N, embed_dim, num_patches]
        x = x.permute(0, 2, 1) # [N, num_patches, embed_dim]
        return self.norm(x)
# ============================================================================
# 3. 轨迹投影模块 (解耦分流)
# ============================================================================

class DisentangledTrajProjection(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, out_dim=128, dropout=0.1):
        super().__init__()
        self.patch_embed = TrajPatchEmbed(in_dim=in_dim, embed_dim=hidden_dim)
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.intent_linear = nn.Linear(hidden_dim, out_dim)
        self.intent_nonlinear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, fut_traj):
        T, B, A, F = fut_traj.shape
        # 合并 batch 和 agent 维度处理
        x = fut_traj.permute(1, 2, 0, 3).reshape(B * A, T, F)
        x_patches = self.patch_embed(x)  # [B*A, 5, hidden]
        
        shared = self.shared_encoder(x_patches) 
        
        # 还原维度为 [B, A, 5, D]
        z_lin = self.intent_linear(shared).view(B, A, 5, -1)
        z_nonlin = self.intent_nonlinear(shared).view(B, A, 5, -1)
        traj_residual = self.residual_head(shared).view(B, A, 5, -1)
        
        return z_lin, z_nonlin, traj_residual

# ============================================================================
# 4. 解耦损失函数 (带消融实验开关)
# ============================================================================

class ContrastiveDiscriminator(nn.Module):
    def __init__(self, res_dim=128, event_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(res_dim + event_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x_res, e, alpha=1.0):
        x_res = GradientReversalLayer.apply(x_res, alpha)
        return self.net(torch.cat([x_res, e], dim=-1))

class DisentanglementLoss(nn.Module):
    def __init__(self, align_weight=0.5, adv_weight=0.1, decay_rate=0.5):
        super().__init__()
        self.align_weight = align_weight
        self.adv_weight = adv_weight
        self.decay_rate = decay_rate
        self.discriminator = ContrastiveDiscriminator(128, 128)
        
    def forward(self, z_lin, z_nonlin, traj_residual, event_emb, use_hsic=True):
        """
        Args:
            use_hsic: 消融实验开关。False 时不计算 HSIC 梯度。
        """
        B, A, P, D = z_nonlin.shape 
        device = z_lin.device
        
        # 提取 Past Event 目标
        event_target = event_emb[:, -1, :] if event_emb.dim() == 3 else event_emb
        # 构造 B*A 维度的匹配矩阵
        target_exp = event_target.unsqueeze(1).expand(-1, A, -1).reshape(B * A, D)

        # 1. 线性对齐 (Baseline)
        z_lin_mean = z_lin.mean(dim=2) 
        loss_lin = 1.0 - F.cosine_similarity(z_lin_mean.mean(1), event_target, dim=-1).mean()
        
        # 2. 非线性 HSIC (带时间衰减与开关)
        loss_hsic = torch.tensor(0.0, device=device)
        hsic_val_to_log = 0.0

        if use_hsic:
            for i in range(P):
                z_patch = z_nonlin[:, :, i, :].reshape(B * A, D)
                h_val = hsic_loss(z_patch, target_exp)
                # 指数衰减权重
                weight = self.decay_rate ** i
                loss_hsic = loss_hsic - (weight * h_val)
            loss_hsic = loss_hsic / P
            hsic_val_to_log = -loss_hsic.item()
        else:
            # 即使不训练，也可以在监控中查看当前的 HSIC 分数（取第一个 Patch 观察）
            with torch.no_grad():
                z_patch_0 = z_nonlin[:, :, 0, :].reshape(B * A, D)
                hsic_val_to_log = hsic_loss(z_patch_0, target_exp).item()
        
        # 3. 残差对抗 (针对 5 个 Patch 的所有残差)
        res_flat = traj_residual.reshape(-1, D)
        target_exp_adv = event_target.unsqueeze(1).expand(-1, A * P, -1).reshape(-1, D)
        
        pos_logits = self.discriminator(res_flat, target_exp_adv, alpha=1.0)
        idx = torch.randperm(B, device=device)
        neg_target_exp = event_target[idx].unsqueeze(1).expand(-1, A * P, -1).reshape(-1, D)
        neg_logits = self.discriminator(res_flat, neg_target_exp, alpha=1.0)
        
        loss_adv = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits)) + \
                   F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        
        with torch.no_grad():
            acc = ((torch.sigmoid(pos_logits) > 0.5).float().mean() + \
                   (torch.sigmoid(neg_logits) < 0.5).float().mean()) / 2
            
        return {
            'disentangle_loss': loss_lin * self.align_weight + 0.1 * loss_hsic + self.adv_weight * loss_adv,
            'ortho_loss': loss_lin,
            'align_loss': loss_hsic,
            'adv_loss': loss_adv,
            'disc_acc': acc,
            'hsic_score': hsic_val_to_log, # 用于 Tensorboard 的原始分数
        }

# ============================================================================
# 2. Macro Intent Encoder
# ============================================================================

class MacroIntentEncoder(nn.Module):
    """
    宏观意图编码器
    
    输入: traj_intent + event_emb + prior_emb +context ball!
    输出: intent_features (长期策略特征)
    
    设计理念: 捕获 "做什么" 的高层意图
    """
    def __init__(self,
                 hidden_dim: int = 128,
                 n_players: int = 22,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_players = n_players
        
        # Input projections
        self.intent_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.prior_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.event_seq_proj = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim)
        )
        
        # Cross-attention: intent attends to event sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        
        # Self-attention among temoral events seq
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Fusion with prior
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, 
                traj_intent: torch.Tensor,   # [B, A, D]
                event_emb: torch.Tensor,     # [B, C, D]
                prior_emb: torch.Tensor,     #  [B, A, D]
                ball_all: torch.Tensor):   # [B, C, D]
        """
        Args:
            traj_intent: 从轨迹解耦出的意图特征
            event_emb: 事件序列嵌入
            prior_emb: 先验嵌入（Voronoi等空间先验）
            context_ball: 上下文球特征

        Returns:
            intent_features: [B, A, D] 宏观意图特征
        """
        B, A, D = traj_intent.shape
        
        raw_context = torch.cat([event_emb, ball_all], dim=-1)

        # Project inputs
        intent = self.intent_proj(traj_intent)  # [B, A, D]
        prior = self.prior_proj(prior_emb)      # [B, A, D]
        event_seq = self.event_seq_proj(raw_context)  # [B, C, D]
        
        # Cross-attention:  player's intent attends to event sequence
        # Query: mean intent [B, A, D], Key/Value: event [B, C, D]
        intent_attn, _ = self.cross_attn(
            query=intent,
            key=event_seq,
            value=event_seq,
        )
        intent = self.cross_norm(intent + intent_attn)
        
        # Self-attention among temoral events seq
        intent = self.self_attn(intent)
        
        # Fuse with prior
        fused = self.fusion(torch.cat([intent, prior], dim=-1))
        
        # Residual connection
        intent_features = self.output_norm(fused + traj_intent)
        
        return intent_features


# ============================================================================
# 3. Micro Dynamic Encoder
# ============================================================================

class MicroDynamicEncoder(nn.Module):
    """
    微观动态编码器
    
    输入: traj_dynamic + h_players
    输出: dynamic_features (短期运动细节)
    
    设计理念: 捕获 "怎么做" 的运动细节
    """
    def __init__(self,
                 hidden_dim: int = 128,
                 n_players: int = 22,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_players = n_players
        
        # Input projections
        self.dynamic_proj = nn.Linear(hidden_dim, hidden_dim)
        self.player_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Cross-attention: dynamic attends to player states
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        
        # Self-attention for spatial reasoning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self,
                traj_dynamic: torch.Tensor,  # [B, A, D]
                h_players: torch.Tensor):    # [B, A, D]
        """
        Args:
            traj_dynamic: 从轨迹解耦出的动态特征
            h_players: 球员状态嵌入
            
        Returns:
            dynamic_features: [B, A, D] 微观动态特征
        """
        B, A, D = traj_dynamic.shape
        
        # Project inputs
        dynamic = self.dynamic_proj(traj_dynamic)
        players = self.player_proj(h_players)
        
        # Cross-attention: dynamic attends to player states
        dynamic_attn, _ = self.cross_attn(
            query=dynamic,
            key=players,
            value=players,
        )
        dynamic = self.cross_norm(dynamic + dynamic_attn)
        
        # Self-attention for inter-player dynamics
        dynamic = self.self_attn(dynamic)
        
        # Fuse with original player features
        fused = self.fusion(torch.cat([dynamic, h_players], dim=-1))
        
        # Residual
        dynamic_features = self.output_norm(fused + traj_dynamic)
        
        return dynamic_features


class HierarchicalFusionEventHead_Nonlinear(nn.Module):
    """
    Simplified hierarchical fusion using token concat + Transformer (same style as TransformerEventHead)
    
    Tokens: [ball_token, player_1, ..., player_22]
    Per-player token = concat(intent, dynamic) -> 2H -> d_model
    
    Note: h_players, prior_emb, event_emb already fused in MacroIntentEncoder/MicroDynamicEncoder
    """
    def __init__(self,
                 hidden_dim: int = 128,
                 n_players: int = 22,
                 n_event_types: int = 5,
                 fut_T: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 d_model: int = None,
                 dropout: float = 0.1,
                 dynamic_weight: float = 0.5, # fusion control else: 0.0 not use dynamic, 1.0 only dynamic
                 **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_players = n_players
        self.n_event_types = n_event_types
        self.fut_T = fut_T
        self.d_model = d_model or hidden_dim * 2

        self.dynamic_weight = dynamic_weight

        # Player token: [intent, dynamic] -> 2*H -> d_model
        self.player_in = nn.Linear(2 * hidden_dim, self.d_model)
        
        # Ball token: [context ball, h_ball] -> 2*H -> d_model
        self.ball_in = nn.Linear(2 * hidden_dim, self.d_model)
        
        # Player index embedding
        self.player_index_emb = nn.Embedding(n_players, self.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Prediction heads
        self.head_tgt = nn.Linear(self.d_model, 1)
        self.head_event = nn.Linear(self.d_model, n_event_types)
        self.head_time = nn.Linear(self.d_model, fut_T)
        self.head_pos = nn.Linear(self.d_model, 2)
        
        # Prior bias scale
        self.prior_bias_scale = nn.Parameter(torch.tensor(1.0))
        
    def forward(self,
                h_players: torch.Tensor,   # [B, A, H]
                h_ball: torch.Tensor,      # [B, H]
                context_ball: torch.Tensor,  # [B, 1, H]
                fut_emb,                   # dict {'intent': [B,A,H], 'dynamic': [B,A,H]} or Tensor
                event_emb: torch.Tensor = None,
                prior_emb: torch.Tensor = None):
        """
        Args:
            h_players: [B, A, H] - context player features (for ball token only)
            h_ball: [B, H] - ball features
            prior_emb: [B, A, H] - unused (already in intent/dynamic)
            fut_emb: dict with 'intent' and 'dynamic', or Tensor [B, A, H]
        """
        B, A, H = h_players.shape
        
        # Parse fut_emb
        if isinstance(fut_emb, dict):
            intent_features = fut_emb['intent']      # [B, A, H]
            dynamic_features = fut_emb['dynamic']    # [B, A, H]
        else:
            # Fallback: split fut_emb or duplicate
            print("Warning: fut_emb is not a dict, using fallback splitting/duplication.")
            if fut_emb.dim() == 2:
                intent_features = fut_emb.unsqueeze(1).expand(B, A, H)
                dynamic_features = intent_features
            else:
                intent_features = fut_emb
                dynamic_features = fut_emb

        alpha = self.dynamic_weight
        intent_scale = math.sqrt(1.0 - alpha)   # alpha=0 -> 1.0, alpha=1 -> 0.0
        dynamic_scale = math.sqrt(alpha)        # alpha=0 -> 0.0, alpha=1 -> 1.0

        scaled_intent = intent_scale * intent_features      # [B, A, H]
        scaled_dynamic = dynamic_scale * dynamic_features   # [B, A, H]

        # Per-player token: just intent + dynamic (already contains h_players, prior info)
        player_feat = torch.cat([scaled_intent, scaled_dynamic], dim=-1)  # [B, A, 2*H]
        x_player = self.player_in(player_feat)  # [B, A, d_model]
        
        # Ball token: h_ball + context ball
        ball_token = self.ball_in(torch.cat([h_ball, context_ball.squeeze(1)], dim=-1)).unsqueeze(1)  # [B, 1, d_model]
        
        # Add player index embeddings
        idx = torch.arange(A, device=h_players.device).unsqueeze(0).expand(B, A)
        x_player = x_player + self.player_index_emb(idx)
        
        # Transformer: [player_1, ..., player_A, ball_token]
        tokens = torch.cat([x_player, ball_token], dim=1)  # [B, 1+A, d_model]
        z = self.encoder(tokens)
        
        z_ball = z[:, -1:, :]    # [B, 1, d_model]
        z_player = z[:, :-1, :]   # [B, A, d_model]
        
        # Predictions
        tgt_logits_main = self.head_tgt(z_player).squeeze(-1)  # [B, A]
        prior_bias = (z_player * z_ball.expand_as(z_player)).sum(dim=-1)
        tgt_logits = tgt_logits_main + self.prior_bias_scale * prior_bias
        tgt_probs = F.softmax(tgt_logits, dim=-1)
        time_logits = self.head_time(z_player)    # [B, A, T]

        # 暂时没用到下面这两个，只是保留个接口
        event_logits = self.head_event(z_player)  # [B, A, E]
        pos_pred = self.head_pos(z_player)        # [B, A, 2]
        
        return tgt_probs, event_logits, time_logits, pos_pred, tgt_logits

# ============================================================================
# 4. 完整的解耦事件预测器 (Final Predictor)
# ============================================================================

class DisentangledEventPredictor_Nonlinear(nn.Module):
    def __init__(self, context_encoder, prior_scorer, hidden_dim=128, n_players=22, 
                 fut_T=64, n_event_types=5, dropout=0.1, **kwargs):
        super().__init__()
        
        self.context_encoder = context_encoder
        self.prior_scorer = prior_scorer
        self.hidden_dim = hidden_dim
        self.n_players = n_players
        
        # 1. 轨迹投影 (三头分流)
        self.traj_projection = DisentangledTrajProjection(hidden_dim=hidden_dim)
        
        # 2. 解耦损失函数 (含判别器)
        self.disentangle_loss_fn = DisentanglementLoss(adv_weight=kwargs.get('adv_weight', 0.1))
        
        # 3. 意图特征融合层 (Linear + Nonlinear)
        self.intent_projector = nn.Linear(2 * hidden_dim, hidden_dim)

        self.macro_encoder = MacroIntentEncoder(hidden_dim=hidden_dim, n_players=n_players, dropout=dropout)
        self.micro_encoder = MicroDynamicEncoder(hidden_dim=hidden_dim, n_players=n_players, dropout=dropout)
        
        # 5. 融合预测头 (支持权重消融实验)
        self.event_head = HierarchicalFusionEventHead_Nonlinear(
            hidden_dim=hidden_dim, n_players=n_players, n_event_types=n_event_types, 
            fut_T=fut_T, dropout=dropout, **kwargs
        )
        
        self._disentangle_loss_cache = None

    def forward(self, ctx_dyns, ctx_edges, ctx_seqX, ctx_labels_22, cols, 
                edge_cols=None, fut_pred=None, h_players=None, h_ball=None, 
                team_labels=None, event_emb=None, context_ball=None, ball_all=None,use_hsic=True):
        
        # --- A. 基础上下文特征准备 (与原有接口逻辑同步) ---
        import numpy as np
        if isinstance(ctx_labels_22, (list, tuple, np.ndarray)):
            arr = np.array(ctx_labels_22, dtype=object)
            enc_labels_22 = [ctx_labels_22] if arr.ndim == 2 else ctx_labels_22
        else:
            enc_labels_22 = ctx_labels_22

        if h_players is None or h_ball is None or event_emb is None:
            h_players, h_ball, team_labels, players_all, ball_all, event_emb = self.context_encoder(
                ctx_dyns, ctx_edges, enc_labels_22, cols=cols, edge_cols=edge_cols,
                return_team_labels=True, return_decay=True,
            )
        
        if context_ball is None:
            attn_score = torch.sum(h_players * h_ball.unsqueeze(1), dim=-1, keepdim=True)
            attn_weight = torch.softmax(attn_score, dim=1)
            context_ball = torch.sum(h_players * attn_weight, dim=1, keepdim=True)

        if ball_all is None:
            ball_all = torch.zeros(h_players.size(0), event_emb.size(1), self.hidden_dim, device=h_players.device)

        # 获取空间先验 (Voronoi 等)
        if self.prior_scorer is not None:
            labels_last = [[str(x) for x in b[-1]] for b in enc_labels_22]
            prior_emb = self.prior_scorer(ctx_dyns[:, -1, :], cols, labels_last)
        else:
            prior_emb = torch.zeros_like(h_players)

        # --- B. 核心解耦分流逻辑 ---
        if fut_pred is not None:
            # 1. 轨迹源头分流：得到 [B, A, 5, D]
            z_lin, z_nonlin, traj_residual = self.traj_projection(fut_pred[:, :, :self.n_players, :])
            
            # 2. 意图合并：先对 Patch 维度做处理
            # 方案：将 5 个 Patch 的线性与非线性特征合并，然后求平均还原回 [B, A, D]
            # 这样后面的 MacroIntentEncoder 就不需要改动了
            z_combined = torch.cat([z_lin, z_nonlin], dim=-1) # [B, A, 5, 2*D]
            traj_intent_all_patches = self.intent_projector(z_combined) # [B, A, 5, D]
            
            # 聚合 5 个 Patch 成为 1 个特征向量
            # 你也可以选择只取第一个 Patch (最受上个事件影响) 或者最后一个
            traj_intent = traj_intent_all_patches.mean(dim=2) # [B, A, D]
            
            # 3. 对抗质检：计算解耦损失 (传入原始的 5-Patch 张量)
            if self.training:
                self._disentangle_loss_cache = self.disentangle_loss_fn(z_lin, z_nonlin, traj_residual, event_emb, use_hsic=use_hsic)
        else:
            traj_intent = torch.zeros(h_players.size(0), self.n_players, self.hidden_dim, device=h_players.device)

        # --- C. 物理隔离的编码流 ---
        # 此时 traj_intent 已经是 [B, A, D]，MacroEncoder 运行正常
        intent_features = self.macro_encoder(traj_intent, event_emb, prior_emb, ball_all)
        
        # 动态分支 (物理隔离测试)：彻底抛弃残差轨迹，改为用 h_players 驱动
        # 这样 Micro-Encoder 专注于球员局部状态的自增强
        dynamic_features = self.micro_encoder(h_players, h_players)

        # --- D. 最终分流融合 ---
        fut_emb = {'intent': intent_features, 'dynamic': dynamic_features}
        
        return self.event_head(
            h_players=h_players, h_ball=h_ball, context_ball=context_ball,
            fut_emb=fut_emb, event_emb=event_emb, prior_emb=prior_emb
        )

    def get_disentangle_loss(self):
        """返回最近一次前向传播的解耦损失统计字典"""
        return self._disentangle_loss_cache
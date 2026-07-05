# model_event/disentangled_event_head.py
# -*- coding: utf-8 -*-
"""
Disentangled Event Prediction Head

Architecture:
1. DisentangledTrajProjection: fut_traj -> (traj_intent, traj_dynamic)
   - Orthogonality constraint between intent and dynamic
   - Alignment constraint between intent and event_emb
   
2. MacroIntentEncoder: traj_intent + event_emb + prior_emb -> intent_features
   - Long-term strategy and high-level intent
   
3. MicroDynamicEncoder: traj_dynamic + h_players -> dynamic_features  
   - Short-term motion details
   
4. HierarchicalFusion: intent guides dynamic -> event prediction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Disentangled Trajectory Projection
# ============================================================================

class DisentangledTrajProjection(nn.Module):
    """
    将 fut_traj 投影到解耦的 (traj_intent, traj_dynamic) 空间
    
    约束:
    1. traj_intent ⊥ traj_dynamic (正交性)
    2. traj_intent ≈ event_emb (意图对齐)
    """
    def __init__(self, 
                 in_dim: int = 2,           # fut_traj 的特征维度
                 hidden_dim: int = 128,
                 out_dim: int = 128,         # intent 和 dynamic 的输出维度
                 n_players: int = 22,
                 dropout: float = 0.1,
                 ):
        super().__init__()
        self.out_dim = out_dim
        self.n_players = n_players
        
        # Shared encoder for trajectory
        self.shared_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Intent head: 捕获长期策略意图
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        
        # Dynamic head: 捕获短期运动细节
        self.dynamic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        
        # Temporal pooling weights (learnable exponential decay)
        self.temporal_weight = nn.Parameter(torch.ones(1))
        
        # Layer norms for output
        self.intent_norm = nn.LayerNorm(out_dim)
        self.dynamic_norm = nn.LayerNorm(out_dim)
        
    def forward(self, fut_traj: torch.Tensor, return_sequence: bool = False):
        """
        Args:
            fut_traj: [T, B, A, 2] 预测的未来轨迹
            return_sequence: 是否返回时序特征
            
        Returns:
            traj_intent: [B, A, D] 意图特征
            traj_dynamic: [B, A, D] 动态特征
            (optional) intent_seq, dynamic_seq: [B, T, A, D]
        """
        T, B, A, F = fut_traj.shape
        device = fut_traj.device
        
        # Reshape for processing: [T, B, A, F] -> [B*A, T, F]
        x = fut_traj.permute(1, 2, 0, 3).reshape(B * A, T, F)
        
        # Shared encoding
        shared = self.shared_encoder(x)  # [B*A, T, hidden]
        
        # Disentangled projections
        intent_seq = self.intent_head(shared)    # [B*A, T, out_dim]
        dynamic_seq = self.dynamic_head(shared)  # [B*A, T, out_dim]
        
        # Temporal pooling with exponential decay
        # Intent: 更关注整体趋势，用均匀权重
        # Dynamic: 更关注近期，用指数衰减
        gamma = torch.sigmoid(self.temporal_weight) * 0.5 + 0.5  # [0.5, 1.0]
        
        # Intent pooling (uniform -> captures overall strategy)
        intent_pooled = intent_seq.mean(dim=1)  # [B*A, out_dim]
        
        # Dynamic pooling (exponential decay -> captures recent motion)
        decay_weights = gamma ** torch.arange(T-1, -1, -1, device=device, dtype=torch.float)
        decay_weights = decay_weights / decay_weights.sum()
        decay_weights = decay_weights.view(1, T, 1)
        dynamic_pooled = (dynamic_seq * decay_weights).sum(dim=1)  # [B*A, out_dim]
        
        # Reshape back
        traj_intent = self.intent_norm(intent_pooled.view(B, A, -1))    # [B, A, D]
        traj_dynamic = self.dynamic_norm(dynamic_pooled.view(B, A, -1))  # [B, A, D]
        
        if return_sequence:
            intent_seq = intent_seq.view(B, A, T, -1).permute(0, 2, 1, 3)  # [B, T, A, D]
            dynamic_seq = dynamic_seq.view(B, A, T, -1).permute(0, 2, 1, 3)
            return traj_intent, traj_dynamic, intent_seq, dynamic_seq
        
        return traj_intent, traj_dynamic


class DisentanglementLoss(nn.Module):
    """
    解耦损失函数
    
    1. Orthogonal Loss: 让 intent 和 dynamic 尽量正交
    2. Alignment Loss: 让 intent 和 event_emb 相似
    """
    def __init__(self, 
                 ortho_weight: float = 1.0,
                 align_weight: float = 0.5,
                 align_type: str = 'cosine'):  # 'cosine' or 'infonce'
        
        super().__init__()
        self.ortho_weight = ortho_weight
        self.align_weight = align_weight
        self.align_type = align_type
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def forward(self, 
                traj_intent: torch.Tensor,   # [B, A, D]
                traj_dynamic: torch.Tensor,  # [B, A, D]
                event_emb: torch.Tensor):    # [B, C, D] or [B, D]
        """
        计算解耦损失
        """
        B, A, D = traj_intent.shape
        
        # 1. Orthogonal Loss: intent ⊥ dynamic
        # 计算每个 agent 的 intent 和 dynamic 的余弦相似度
        intent_norm = F.normalize(traj_intent, dim=-1)
        dynamic_norm = F.normalize(traj_dynamic, dim=-1)
        
        # 每个 agent 的相似度
        cos_sim = (intent_norm * dynamic_norm).sum(dim=-1)  # [B, A]
        ortho_loss = cos_sim.pow(2).mean()  # 希望接近 0
        
        # 2. Alignment Loss: intent ≈ event_emb 主要是要在同一个样本空间，但其实不需要值太近。
        # #这只是一个妥协，因为batch size太小不好用 infonce
        if event_emb.dim() == 3:
            event_target = event_emb[:, -1, :]  # [B, D]
        else:
            event_target = event_emb  # [B, D]
        
        # 对 intent 做 agent-wise pooling（用 attention 或 mean）
        intent_pooled = traj_intent.mean(dim=1)  # [B, D]
        
        if self.align_type == 'cosine':
            # Simple cosine similarity
            intent_pooled_norm = F.normalize(intent_pooled, dim=-1)
            event_target_norm = F.normalize(event_target, dim=-1)
            align_loss = 1.0 - (intent_pooled_norm * event_target_norm).sum(dim=-1).mean()
            
        else:  # infonce
            # InfoNCE style contrastive loss
            intent_pooled_norm = F.normalize(intent_pooled, dim=-1)
            event_target_norm = F.normalize(event_target, dim=-1)
            
            # Positive: same sample
            pos_sim = (intent_pooled_norm * event_target_norm).sum(dim=-1) / self.temperature
            
            # Negative: other samples in batch
            neg_sim = torch.mm(intent_pooled_norm, event_target_norm.t()) / self.temperature
            
            # InfoNCE loss
            labels = torch.arange(B, device=traj_intent.device)
            align_loss = F.cross_entropy(neg_sim, labels)
        
        total_loss = self.ortho_weight * ortho_loss + self.align_weight * align_loss
        
        return {
            'disentangle_loss': total_loss,
            'ortho_loss': ortho_loss,
            'align_loss': align_loss,
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

# ============================================================================
# 4. Hierarchical Fusion Event Head (Simplified: Token Concat + Transformer)
# ============================================================================

class HierarchicalFusionEventHead(nn.Module):
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
# 5. Unified Disentangled Event Predictor
# ============================================================================

class DisentangledEventPredictor(nn.Module):
    """
    完整的解耦事件预测器
    
    接口与 ParallelEventPredictor_Unified 保持一致
    """
    def __init__(self,
                 context_encoder,
                 prior_scorer,
                 hidden_dim: int = 128,
                 n_players: int = 22,
                 fut_T: int = 64,
                 n_event_types: int = 5,
                 dropout: float = 0.1,
                 # 解耦相关参数
                 ortho_weight: float = 1.0,
                 align_weight: float = 1.0,
                 align_type: str = 'cosine',
                 # Encoder 参数
                 n_heads: int = 4,
                 n_layers: int = 2,
                 **kwargs):
        super().__init__()
        
        self.context_encoder = context_encoder
        self.prior_scorer = prior_scorer
        self.hidden_dim = hidden_dim
        self.n_players = n_players
        
        # 1. Disentangled trajectory projection
        self.traj_projection = DisentangledTrajProjection(
            in_dim=2,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_players=n_players,
            dropout=dropout,
        )
        
        # 2. Disentanglement loss
        self.disentangle_loss_fn = DisentanglementLoss(
            ortho_weight=ortho_weight,
            align_weight=align_weight,
            align_type=align_type,
        )
        
        # 3. Macro intent encoder
        self.macro_encoder = MacroIntentEncoder(
            hidden_dim=hidden_dim,
            n_players=n_players,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        
        # 4. Micro dynamic encoder
        self.micro_encoder = MicroDynamicEncoder(
            hidden_dim=hidden_dim,
            n_players=n_players,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        
        # 5. Hierarchical fusion event head
        self.event_head = HierarchicalFusionEventHead(
            hidden_dim=hidden_dim,
            n_players=n_players,
            n_event_types=n_event_types,
            fut_T=fut_T,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        
        # 缓存解耦损失（训练时使用）
        self._disentangle_loss_cache = None
        
    def forward(self,
                ctx_dyns, ctx_edges, ctx_seqX, ctx_labels_22,
                cols, edge_cols=None, fut_pred=None, h_players=None, 
                h_ball=None, team_labels=None, event_emb=None,context_ball=None, ball_all=None):
        """
        Forward pass - 接口与 ParallelEventPredictor_Unified 一致
        
        Args:
            ctx_dyns: [B, C, F_dyn]
            ctx_edges: [B, C, F_edge]
            ctx_seqX: [B, T, 23, 2]
            ctx_labels_22: player labels
            cols: column names
            edge_cols: edge column names
            fut_pred: [T, B, A, 2] predicted future trajectories
            context_ball: [B, 1, H] context ball embeddings
            h_players: [B, A, H] precomputed player embeddings
            h_ball: [B, H] precomputed ball embeddings
            team_labels: team labels
            event_emb: [B, C, H] precomputed event embeddings
            ball_all: [B, C, H] precomputed ball embeddings

        Returns:
            tgt_probs, event_logits, time_logits, pos_pred, tgt_logits
        """
        import numpy as np
        
        # Prepare labels
        if isinstance(ctx_labels_22, (list, tuple, np.ndarray)):
            arr = np.array(ctx_labels_22, dtype=object)
            enc_labels_22 = [ctx_labels_22] if arr.ndim == 2 else ctx_labels_22
        else:
            enc_labels_22 = ctx_labels_22

        if h_players is None or h_ball is None or event_emb is None:
            # Prepare labels
            if isinstance(ctx_labels_22, (list, tuple, np.ndarray)):
                arr = np.array(ctx_labels_22, dtype=object)
                enc_labels_22 = [ctx_labels_22] if arr.ndim == 2 else ctx_labels_22
            else:
                enc_labels_22 = ctx_labels_22
            
            # Call context encoder
            h_players, h_ball, team_labels, players_all, ball_all, event_emb = self.context_encoder(
                ctx_dyns, ctx_edges, enc_labels_22,
                cols=cols, edge_cols=edge_cols,
                return_team_labels=True,
                return_decay=True,
            )
        if context_ball is None:
            attn_score = torch.sum(h_players * h_ball.unsqueeze(1), dim=-1, keepdim=True)
            attn_weight = torch.softmax(attn_score, dim=1)
            context_ball = torch.sum(h_players * attn_weight, dim=1, keepdim=True)
        
        # Get prior embeddings
        labels_last = []
        for b in range(len(enc_labels_22)):
            row = enc_labels_22[b][-1]
            if hasattr(row, "tolist"):
                labels_last.append([str(x) for x in row.tolist()])
            elif isinstance(row, (list, tuple)):
                labels_last.append([str(x) for x in row])
            else:
                labels_last.append([str(row)])
                
        if self.prior_scorer is not None:
            prior_emb = self.prior_scorer(ctx_dyns[:, -1, :], cols, labels_last)
        else:
            prior_emb = torch.zeros_like(h_players)
            
        # Disentangle trajectory features
        if fut_pred is not None:
            # fut_pred: [T, B, A, 2] -> 只取前22个球员
            fut_pred_players = fut_pred[:, :, :self.n_players, :]
            traj_intent, traj_dynamic = self.traj_projection(fut_pred_players)
            
            # 计算解耦损失（训练时使用）
            if self.training:
                self._disentangle_loss_cache = self.disentangle_loss_fn(
                    traj_intent, traj_dynamic, event_emb
                )
        else:
            # 如果没有 fut_pred，使用零向量
            B = h_players.shape[0]
            traj_intent = torch.zeros(B, self.n_players, self.hidden_dim, 
                                     device=h_players.device)
            traj_dynamic = torch.zeros_like(traj_intent)
            self._disentangle_loss_cache = None

        if ball_all is None:
            # 兜底：如果 ball_all 不存在，创建零向量
            C = event_emb.size(1) if event_emb.dim() == 3 else 10
            ball_all = torch.zeros(h_players.size(0), C, self.hidden_dim, device=h_players.device)
        
        # Macro intent encoding
        intent_features = self.macro_encoder(traj_intent, event_emb, prior_emb, ball_all)
        
        # Micro dynamic encoding
        dynamic_features = self.micro_encoder(traj_dynamic, h_players)
        fut_emb = {
            'intent': intent_features,    # [B, A, D]
            'dynamic': dynamic_features   # [B, A, D]
        }
        return self.event_head(
            h_players=h_players,
            h_ball=h_ball,
            fut_emb=fut_emb,
            event_emb=event_emb,
            context_ball=context_ball,
        )

    def get_disentangle_loss(self):
        """获取解耦损失（训练时调用）"""
        return self._disentangle_loss_cache
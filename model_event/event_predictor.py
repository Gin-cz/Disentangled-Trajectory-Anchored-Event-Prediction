# -*- coding: utf-8 -*-
"""
Unified Event Heads Module

Supports three event prediction architectures:
1. Transformer - Set-Transformer style cross-player attention
2. Linear - Simple linear fusion baseline
3. Disentangled - Hierarchical fusion with disentangled trajectory features
4. Disentangled_Gated - Hierarchical fusion with automatic gated fusion
5. Volterra - Volterra fusion
6. TGN - Temporal Graph Network
7. Seq2Event - event-sequence baseline adapted to target/time prediction
8. NMSTPP - marked spatio-temporal point-process style baseline without zones
Interface:
    forward(h_players, h_ball, prior_emb, fut_emb) ->
        (tgt_probs, event_logits, time_logits, pos_pred, tgt_logits)

All heads share the same interface:
    forward(h_players, h_ball, prior_emb, fut_emb) -> (tgt_probs, event_logits, time_logits, pos_pred, tgt_logits)
"""

import torch
import torch.nn as nn
import numpy as np

from .linear import LinearEventHead
from .transformer import TransformerEventHead
from .sequence_event import NMSTPPEventHead, Seq2EventHead


# ============================================================================
# Unified Parallel Event Predictor
# ============================================================================
class ParallelEventPredictor_Unified(nn.Module):
    def __init__(
        self,
        context_encoder,
        prior_scorer,
        hidden_dim: int,
        n_players: int = 22,
        fut_T: int = 64,
        n_event_types: int = 5,
        event_head_type: str = 'disentangled',
        d_model: int = 256,
        dropout: float = 0.1,
        dynamic_weight: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.prior_scorer = prior_scorer
        self.n_players = n_players
        self.hidden_dim = hidden_dim
        self.event_head_type = event_head_type
        self.use_hsic = kwargs.get('use_hsic', True)
        self.proj_fut = nn.Linear(2, hidden_dim)

        # Select head type
        if event_head_type == 'transformer':
            self.head = TransformerEventHead(
                hidden_dim=hidden_dim,
                n_event_types=n_event_types,
                fut_T=fut_T,
                n_players=n_players,
                d_model=d_model,
                dropout=dropout,
                **{k: v for k, v in kwargs.items() 
                   if k in ['nhead', 'num_layers', 'dim_feedforward']}
            )
            self.return_decay = True
        elif event_head_type == 'linear':
            self.head = LinearEventHead(
                hidden_dim=hidden_dim,
                n_event_types=n_event_types,
                fut_T=fut_T,
                n_players=n_players,
            )
            self.return_decay = True

        elif event_head_type == 'volterra':
            from .volterra import VolterraFusionEventHead
            self.head = VolterraFusionEventHead(
                hidden_dim=hidden_dim,
                n_event_types=n_event_types,
                fut_T=fut_T,
                n_players=n_players,
                d_model=d_model,
                dropout=dropout,
                volterra_Q=kwargs.get('volterra_Q', 2),
                volterra_scales=kwargs.get('volterra_scales', 2),
                **{k: v for k, v in kwargs.items() 
                   if k in ['nhead', 'num_layers', 'dim_feedforward']}
            )
            self.return_decay = True

        elif event_head_type == 'seq2event':
            self.head = Seq2EventHead(
                hidden_dim=hidden_dim,
                n_event_types=n_event_types,
                fut_T=fut_T,
                n_players=n_players,
                d_model=d_model,
                dropout=dropout,
                **{k: v for k, v in kwargs.items()
                   if k in ['nhead', 'num_layers', 'dim_feedforward']}
            )
            self.return_decay = True

        elif event_head_type == 'nmstpp':
            self.head = NMSTPPEventHead(
                hidden_dim=hidden_dim,
                n_event_types=n_event_types,
                fut_T=fut_T,
                n_players=n_players,
                d_model=d_model,
                dropout=dropout,
                **{k: v for k, v in kwargs.items()
                   if k in ['nhead', 'num_layers', 'dim_feedforward']}
            )
            self.return_decay = True

        elif event_head_type == 'disentangled':
            from .disentangled import (
            DisentangledTrajProjection,
            MacroIntentEncoder, 
            MicroDynamicEncoder,
            HierarchicalFusionEventHead,
            DisentanglementLoss,
            )

            self.return_decay = True                
            # 解耦投影
            self.traj_projection = DisentangledTrajProjection(
                in_dim=2,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                dropout=dropout,
            )
            
            # 解耦损失
            self.disentangle_loss_fn = DisentanglementLoss(
                align_weight=kwargs.get('align_weight', 0.5),
                ortho_weight=kwargs.get('ortho_weight', 1.0) # 普通版不需要 adv_weight
            )
            
            # 双流编码器
            self.macro_encoder = MacroIntentEncoder(
                hidden_dim=hidden_dim,
                n_players=n_players,
                n_heads=kwargs.get('nhead', 4),
                n_layers=kwargs.get('num_layers', 2),
                dropout=dropout,
            )
            
            self.micro_encoder = MicroDynamicEncoder(
                hidden_dim=hidden_dim,
                n_players=n_players,
                n_heads=kwargs.get('nhead', 4),
                n_layers=kwargs.get('num_layers', 2),
                dropout=dropout,
            )
            
            # 融合头
            self.head = HierarchicalFusionEventHead(
                hidden_dim=hidden_dim,
                n_players=n_players,
                n_event_types=n_event_types,
                fut_T=fut_T,
                n_heads=kwargs.get('nhead', 4),
                dropout=dropout,
                dynamic_weight=dynamic_weight,
            )
            
            self._disentangle_loss_cache = None

        elif event_head_type == 'disentangled_nonlinear':
            from .disentangled_nonlinear import (
                DisentangledTrajProjection,
                MacroIntentEncoder, 
                MicroDynamicEncoder,
                HierarchicalFusionEventHead_Nonlinear, 
                DisentanglementLoss,
            )

            self.return_decay = True                
            # 解耦投影
            self.traj_projection = DisentangledTrajProjection(
                in_dim=2,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                dropout=dropout,
            )
            
            # 解耦损失
            self.disentangle_loss_fn = DisentanglementLoss(
                align_weight=kwargs.get('align_weight', 0.5),
                adv_weight=0.1  # 或者从 kwargs 获取：kwargs.get('adv_weight', 0.1)
            )

            self.intent_projector = nn.Linear(2 * hidden_dim, hidden_dim)
            
            # 双流编码器
            self.macro_encoder = MacroIntentEncoder(
                hidden_dim=hidden_dim,
                n_players=n_players,
                n_heads=kwargs.get('nhead', 4),
                n_layers=kwargs.get('num_layers', 2),
                dropout=dropout,
            )
            
            self.micro_encoder = MicroDynamicEncoder(
                hidden_dim=hidden_dim,
                n_players=n_players,
                n_heads=kwargs.get('nhead', 4),
                n_layers=kwargs.get('num_layers', 2),
                dropout=dropout,
            )
            
            # 融合头 - 非线性版本
            self.head = HierarchicalFusionEventHead_Nonlinear(
                hidden_dim=hidden_dim,
                n_players=n_players,
                n_event_types=n_event_types,
                fut_T=fut_T,
                n_heads=kwargs.get('nhead', 4),
                dropout=dropout,
                dynamic_weight=dynamic_weight,
            )
            
            self._disentangle_loss_cache = None

        elif event_head_type == 'disentangled_patch':
            # 引入你写的 patch 版本相关组件
            from .disentangled_patch import (
                DisentangledTrajProjection,
                MacroIntentEncoder,
                MicroDynamicEncoder,
                HierarchicalFusionEventHead_Nonlinear, # Patch版通常配合非线性头
                DisentanglementLoss,
            )

            self.return_decay = True
            
            # 1. 轨迹投影 (使用 5-Patch Conv1d)
            self.traj_projection = DisentangledTrajProjection(
                in_dim=2,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                dropout=dropout,
            )
            
            # 2. 解耦损失 (带 HSIC 衰减)
            self.disentangle_loss_fn = DisentanglementLoss(
                align_weight=kwargs.get('align_weight', 0.5),
                adv_weight=0.1,
                decay_rate=kwargs.get('decay_rate', 0.5)
            )

            # 投影聚合层：将 [B, A, 5, 2*H] 映射回 [B, A, H] 以适配后续 Encoder
            self.intent_projector = nn.Linear(2 * hidden_dim, hidden_dim)
            
            # 3. 双流编码器
            self.macro_encoder = MacroIntentEncoder(
                hidden_dim=hidden_dim, n_players=n_players,
                n_heads=kwargs.get('nhead', 4), n_layers=kwargs.get('num_layers', 2),
                dropout=dropout,
            )
            self.micro_encoder = MicroDynamicEncoder(
                hidden_dim=hidden_dim, n_players=n_players,
                n_heads=kwargs.get('nhead', 4), n_layers=kwargs.get('num_layers', 2),
                dropout=dropout,
            )
            
            # 4. 融合预测头
            self.head = HierarchicalFusionEventHead_Nonlinear(
                hidden_dim=hidden_dim, n_players=n_players, n_event_types=n_event_types,
                fut_T=fut_T, n_heads=kwargs.get('nhead', 4), dropout=dropout,
                dynamic_weight=dynamic_weight,
            )
            self._disentangle_loss_cache = None
            print(f"🔥 [DEBUG] Model Initialized! use_hsic is set to: {self.use_hsic}")
            
        elif event_head_type == 'disentangled_gated':
            from .disentangled_gated import (
                DisentangledTrajProjection,
                MacroIntentEncoder, 
                MicroDynamicEncoder,
                HierarchicalFusionEventHead,  # 门控版本
                DisentanglementLoss,
            )

            self.return_decay = True                
            
            # 解耦投影（与 disentangled 相同）
            self.traj_projection = DisentangledTrajProjection(
                in_dim=2,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                dropout=dropout,
            )
            
            # 解耦损失（与 disentangled 相同）
            self.disentangle_loss_fn = DisentanglementLoss(
                align_weight=kwargs.get('align_weight', 0.5),
                ortho_weight=kwargs.get('ortho_weight', 1.0),
            )
            
            # 双流编码器（与 disentangled 相同）
            self.macro_encoder = MacroIntentEncoder(
                hidden_dim=hidden_dim,
                n_players=n_players,
                n_heads=kwargs.get('nhead', 4),
                n_layers=kwargs.get('num_layers', 2),
                dropout=dropout,
            )
            
            self.micro_encoder = MicroDynamicEncoder(
                hidden_dim=hidden_dim,
                n_players=n_players,
                n_heads=kwargs.get('nhead', 4),
                n_layers=kwargs.get('num_layers', 2),
                dropout=dropout,
            )
            
            # 融合头 - 使用门控融合（NEW）
            self.head = HierarchicalFusionEventHead(
                hidden_dim=hidden_dim,
                n_players=n_players,
                n_event_types=n_event_types,
                fut_T=fut_T,
                n_heads=kwargs.get('nhead', 4),
                dropout=dropout,
                use_gated_fusion=True,  # 关键：启用门控融合
            )
            
            self._disentangle_loss_cache = None
        
        elif event_head_type == 'tgn':
            from .tgn import TGNEventHead
            self.head = TGNEventHead(
                hidden_dim=hidden_dim,
                n_event_types=n_event_types,
                fut_T=fut_T,
                n_players=n_players,
                n_heads=kwargs.get('nhead', 4),
                n_layers=kwargs.get('num_layers', 2),
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown event_head_type: {event_head_type}")
        
        self.volterra_module = None

    def forward(self, ctx_dyns, ctx_edges, ctx_seqX, ctx_labels_22, cols,
                edge_cols=None, fut_pred=None,
                h_players=None, h_ball=None, team_labels=None, event_emb=None, 
                context_ball=None, prior_emb=None, ball_all=None, players_all=None, **kwargs): 
        
        # Context Encoder
        # 如果外部已经传入，直接使用；否则才计算
        if h_players is None or h_ball is None or event_emb is None:
            if isinstance(ctx_labels_22, (list, tuple, np.ndarray)):
                arr = np.array(ctx_labels_22, dtype=object)
                enc_labels_22 = [ctx_labels_22] if arr.ndim == 2 else ctx_labels_22
            else:
                enc_labels_22 = ctx_labels_22
            

            h_players, h_ball, team_labels, players_all, ball_all, event_emb = self.context_encoder(
                ctx_dyns, ctx_edges, enc_labels_22,
                cols=cols, edge_cols=edge_cols,
                return_team_labels=True,
            )
            # Compute context_ball if needed
            if context_ball is None:
                attn_score = torch.sum(h_players * h_ball.unsqueeze(1), dim=-1, keepdim=True)
                attn_weight = torch.softmax(attn_score, dim=1)
                context_ball = (attn_weight * h_players).sum(dim=1, keepdim=True)

        else:
            # 外部传入时，需要准备 enc_labels_22（用于 prior_scorer）
            if isinstance(ctx_labels_22, (list, tuple, np.ndarray)):
                arr = np.array(ctx_labels_22, dtype=object)
                enc_labels_22 = [ctx_labels_22] if arr.ndim == 2 else ctx_labels_22
            else:
                enc_labels_22 = ctx_labels_22
        
        # Prior Scorer 
        if prior_emb is None:
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
                B, A, _ = h_players.shape  # ✅ 统一用池化版本的形状
                prior_emb = torch.zeros(B, A, self.hidden_dim, device=h_players.device)

        # =====================================================================
        # disentangled head
        # =====================================================================
        if self.event_head_type == 'disentangled':
            B = h_players.size(0)

            if fut_pred is not None:
                fut_pred_players = fut_pred[:, :, :self.n_players, :]  # [T,B,A,2]
                traj_intent, traj_dynamic = self.traj_projection(fut_pred_players)

                if self.training:
                    self._disentangle_loss_cache = self.disentangle_loss_fn(
                        traj_intent, traj_dynamic, event_emb
                    )
                else:
                    self._disentangle_loss_cache = None

                if ball_all is None:
                    # 如果外部没传且没调用 context_encoder，创建零向量
                    C = event_emb.size(1) if event_emb.dim() == 3 else 1
                    ball_all = torch.zeros(B, C, self.hidden_dim, device=h_players.device)
                    print("Warning: ball_all is None in disentangled head, using zeros.")

                intent_features  = self.macro_encoder(traj_intent, event_emb, prior_emb, ball_all)
                dynamic_features = self.micro_encoder(traj_dynamic, h_players)
          
            else:
                # fut_pred None
                print("Warning: fut_pred is None in disentangled head.")
                self._disentangle_loss_cache = None      # <- 补这里（无 fut 清空）
                intent_features  = torch.zeros(B, self.n_players, self.hidden_dim, device=h_players.device)
                dynamic_features = torch.zeros(B, self.n_players, self.hidden_dim, device=h_players.device)

            fut_emb_disentangled = {"intent": intent_features, "dynamic": dynamic_features}

            return self.head(
                h_players=h_players,
                h_ball=h_ball,
                context_ball=context_ball,
                fut_emb=fut_emb_disentangled,
                event_emb=event_emb,
                prior_emb=prior_emb,
            )
        elif self.event_head_type == 'disentangled_patch':
            B = h_players.size(0)

            if fut_pred is not None:
                # 1. 5-Patch 投影: z_lin/nonlin 形状均为 [B, A, 5, H]
                fut_pred_players = fut_pred[:, :, :self.n_players, :]
                z_lin, z_nonlin, traj_residual = self.traj_projection(fut_pred_players)

                # 2. 计算 Patch-wise Loss (传递 use_hsic 参数)
                if self.training:
                    self._disentangle_loss_cache = self.disentangle_loss_fn(
                        z_lin, z_nonlin, traj_residual, event_emb, use_hsic=self.use_hsic
                    )
                else:
                    self._disentangle_loss_cache = None

                # 3. 特征聚合: [B, A, 5, 2*H] -> [B, A, 5, H] -> [B, A, H]
                z_combined = torch.cat([z_lin, z_nonlin], dim=-1)
                traj_intent = self.intent_projector(z_combined).mean(dim=2) # 聚合时间维度
                
                intent_features = self.macro_encoder(traj_intent, event_emb, prior_emb, ball_all)
                # 物理隔离：动态流用球员自身状态驱动
                dynamic_features = self.micro_encoder(h_players, h_players)
            else:
                self._disentangle_loss_cache = None
                intent_features = torch.zeros(B, self.n_players, self.hidden_dim, device=h_players.device)
                dynamic_features = torch.zeros_like(intent_features)

            fut_emb_disentangled = {"intent": intent_features, "dynamic": dynamic_features}

            return self.head(
                h_players=h_players, h_ball=h_ball, context_ball=context_ball,
                fut_emb=fut_emb_disentangled, event_emb=event_emb, prior_emb=prior_emb,
            )
        
        elif self.event_head_type == 'disentangled_gated':
            # 与 disentangled 完全相同的处理逻辑
            B = h_players.size(0)

            if fut_pred is not None:
                fut_pred_players = fut_pred[:, :, :self.n_players, :]
                traj_intent, traj_dynamic = self.traj_projection(fut_pred_players)

                if self.training:
                    self._disentangle_loss_cache = self.disentangle_loss_fn(
                        traj_intent, traj_dynamic, event_emb
                    )
                else:
                    self._disentangle_loss_cache = None

                if ball_all is None:
                    C = event_emb.size(1) if event_emb.dim() == 3 else 1
                    ball_all = torch.zeros(B, C, self.hidden_dim, device=h_players.device)

                intent_features  = self.macro_encoder(traj_intent, event_emb, prior_emb, ball_all)
                dynamic_features = self.micro_encoder(traj_dynamic, h_players)
            else:
                self._disentangle_loss_cache = None
                intent_features  = torch.zeros(B, self.n_players, self.hidden_dim, device=h_players.device)
                dynamic_features = torch.zeros(B, self.n_players, self.hidden_dim, device=h_players.device)

            fut_emb_disentangled = {"intent": intent_features, "dynamic": dynamic_features}

            return self.head(
                h_players=h_players,
                h_ball=h_ball,
                context_ball=context_ball,
                fut_emb=fut_emb_disentangled,
                event_emb=event_emb,
                prior_emb=prior_emb,
            ) 
        # =====================================================================
        # 新增：disentangled_nonlinear 处理逻辑
        # =====================================================================
        elif self.event_head_type == 'disentangled_nonlinear':
            B = h_players.size(0)

            if fut_pred is not None:
                # 1. 执行三头投影：[T, B, A, 2] -> z_lin, z_nonlin, traj_residual
                fut_pred_players = fut_pred[:, :, :self.n_players, :]
                z_lin, z_nonlin, traj_residual = self.traj_projection(fut_pred_players)

                # 2. 训练时计算三路解耦损失 (HSIC + 对抗 + 线性)
                if self.training:
                    self._disentangle_loss_cache = self.disentangle_loss_fn(
                        z_lin, z_nonlin, traj_residual, event_emb
                    )
                else:
                    self._disentangle_loss_cache = None

                # 3. 意图合并与编码：Linear + Nonlinear -> Macro
                # 确保你已经在 __init__ 里定义了 self.intent_projector = nn.Linear(2*H, H)
                traj_intent = self.intent_projector(torch.cat([z_lin, z_nonlin], dim=-1))
                intent_features = self.macro_encoder(traj_intent, event_emb, prior_emb, ball_all)
                
                # 4. 物理隔离：动态分支不再接收轨迹残差，改用 h_players 状态自增强
                dynamic_features = self.micro_encoder(h_players, h_players)
            else:
                # 兜底处理 (无预测轨迹时)
                self._disentangle_loss_cache = None
                intent_features = torch.zeros(B, self.n_players, self.hidden_dim, device=h_players.device)
                dynamic_features = torch.zeros_like(intent_features)

            # 5. 封装并进入非线性版本的 Fusion Head
            fut_emb_disentangled = {"intent": intent_features, "dynamic": dynamic_features}

            return self.head(
                h_players=h_players,
                h_ball=h_ball,
                context_ball=context_ball,
                fut_emb=fut_emb_disentangled,
                event_emb=event_emb,
                prior_emb=prior_emb,
            )
        
        elif self.event_head_type == 'tgn':
            # ✅ TGN 使用完整序列
            if fut_pred is not None:
                fut_pred_players = fut_pred[:, :, :self.n_players, :]
                fut_traj_emb = self.proj_fut(fut_pred_players[-8:].mean(dim=0))
            else:
                B = h_players.size(0)
                fut_traj_emb = torch.zeros(B, self.n_players, self.hidden_dim, device=h_players.device)
            
            return self.head(
                h_players=players_all,    # ✅ [B, C, A, H]
                h_ball=ball_all,          # ✅ [B, C, H]
                prior_emb=prior_emb,      # [B, A, H]
                fut_emb=fut_traj_emb,     # [B, A, H]
                event_emb=event_emb,      # [B, C, H]
                context_ball=context_ball,# [B, 1, H] or [B, A, H]
            )
        
        else: # Other heads
            # 将 fut_pred 投影为 fut_emb [B, A, H]
            if fut_pred is not None:
                # fut_pred: [T, B, A, 2] -> take only players
                fut_pred_players = fut_pred[:, :, :self.n_players, :]  # [T, B, 22, 2]
                
                # 简单时序池化：取最后几帧的平均
                fut_traj_emb = self.proj_fut(fut_pred_players[-8:].mean(dim=0))  # [B, 22, 2] -> [B, 22, H]
            else:
                # 无 fut_pred 时使用零向量
                B = h_players.size(0)
                fut_traj_emb = torch.zeros(B, self.n_players, self.hidden_dim, device=h_players.device)
            
            # 调用统一的 head 接口
            return self.head(
                h_players=h_players,
                h_ball=h_ball,
                prior_emb=prior_emb,
                fut_emb=fut_traj_emb,
                event_emb=event_emb,
                context_ball=context_ball,
            )
        
    def get_disentangle_loss(self):
            # 增加对 nonlinear 类型的判断
            if self.event_head_type in ['disentangled', 'disentangled_gated', 'disentangled_nonlinear','disentangled_patch']:
                cache = getattr(self, '_disentangle_loss_cache', None)
                if cache is not None:
                    return cache
            
            # 统一样本返回结构，防止外层训练脚本 key_error
            device = next(self.parameters()).device
            return {
                'disentangle_loss': torch.tensor(0.0, device=device),
                'ortho_loss': torch.tensor(0.0, device=device),
                'align_loss': torch.tensor(0.0, device=device),
                'disc_acc': torch.tensor(0.5, device=device),  # 默认随机水平
                'hsic_score': torch.tensor(0.0, device=device),
            }


# Factory function
def create_event_predictor(
    context_encoder,
    prior_scorer,
    hidden_dim: int,
    event_head_type: str = 'volterra',
    **kwargs
):
    """
    Factory function to create event predictor.
    
    Args:
        context_encoder: ContextEncoder instance
        prior_scorer: TgtPlayerScorer instance (or None)
        hidden_dim: hidden dimension
        event_head_type: 'transformer' | 'linear'
        **kwargs: additional arguments for the head
        
    Returns:
        ParallelEventPredictor_Unified instance
    """
    return ParallelEventPredictor_Unified(
        context_encoder=context_encoder,
        prior_scorer=prior_scorer,
        hidden_dim=hidden_dim,
        event_head_type=event_head_type,
        **kwargs
    )

def gather_neighbors(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    x:   [B, A, D]
    idx: [B, A, K]  neighbor indices for each i
    out: [B, A, K, D]
    """
    B, A, D = x.shape
    K = idx.shape[-1]
    x_exp = x.unsqueeze(1).expand(B, A, A, D)  # [B, i, j, D]
    idx_exp = idx.unsqueeze(-1).expand(B, A, K, D)
    return torch.gather(x_exp, dim=2, index=idx_exp)

ParallelEventPredictor_Transformer = lambda **kw: ParallelEventPredictor_Unified(event_head_type='transformer', **kw)
ParallelEventPredictor_Linear = lambda **kw: ParallelEventPredictor_Unified(event_head_type='linear', **kw)
ParallelEventPredictor_Volterra = lambda **kw: ParallelEventPredictor_Unified(event_head_type='volterra', **kw)
ParallelEventPredictor_DisentangledGated = lambda **kw: ParallelEventPredictor_Unified(event_head_type='disentangled_gated', **kw)
ParallelEventPredictor_Disentangled = lambda **kw: ParallelEventPredictor_Unified(event_head_type='disentangled', **kw)
ParallelEventPredictor_DisentangledNonlinear = lambda **kw: ParallelEventPredictor_Unified(event_head_type='disentangled_nonlinear', **kw)
ParallelEventPredictor_TGN = lambda **kw: ParallelEventPredictor_Unified(event_head_type='tgn', **kw)
ParallelEventPredictor_DisentangledPatch = lambda **kw: ParallelEventPredictor_Unified(event_head_type='disentangled_patch', **kw)

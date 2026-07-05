# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearEventHead(nn.Module):
    """
    Simple linear fusion baseline - no cross-player attention.

    forward(h_players, h_ball, prior_emb, fut_emb) ->
        (tgt_probs, event_logits, time_logits, pos_pred, tgt_logits)
    """
    def __init__(
        self,
        hidden_dim: int,
        n_event_types: int = 5,
        fut_T: int = 64,
        n_players: int = 22,
        d_model: int = 256,        # 保持接口一致，不使用
        nhead: int = 8,            # 保持接口一致，不使用
        num_layers: int = 2,       # 保持接口一致，不使用
        dim_feedforward: int = 512,# 保持接口一致，不使用
        dropout: float = 0.1,     # 保持接口一致，不使用
        volterra_Q: int = 2,       # 保持接口一致，不使用
        volterra_scales: int = 2,  # 保持接口一致，不使用
    ):
        super().__init__()
        self.n_players = n_players
        self.n_event_types = n_event_types
        self.fut_T = fut_T
        self.hidden_dim = hidden_dim

        self.pred_head_tgt = nn.Linear(4 * hidden_dim, 1)
        self.head_event = nn.Linear(3 * hidden_dim, n_event_types)
        self.head_time = nn.Linear(3 * hidden_dim, fut_T)
        self.head_pos = nn.Linear(3 * hidden_dim, 2)

    def forward(self, h_players, h_ball, prior_emb, fut_emb, event_emb=None, context_ball=None):
        B, A, H = h_players.shape

        if context_ball is None:
            attn_score = torch.sum(h_players * h_ball.unsqueeze(1), dim=-1, keepdim=True) / math.sqrt(H)
            attn_weight = torch.softmax(attn_score, dim=1)
            context_ball = (attn_weight * h_players).sum(dim=1, keepdim=True)

        context_ball_expanded = context_ball.expand(-1, A, -1)

        fused = torch.cat([h_players, prior_emb, context_ball_expanded, fut_emb], dim=-1)
        tgt_logits = self.pred_head_tgt(fused).squeeze(-1)
        tgt_probs = F.softmax(tgt_logits, dim=-1)

        ctx_emb = torch.cat([h_players, context_ball_expanded, fut_emb], dim=-1)
        event_logits = self.head_event(ctx_emb)
        time_logits = self.head_time(ctx_emb)
        pos_pred = self.head_pos(ctx_emb)

        return tgt_probs, event_logits, time_logits, pos_pred, tgt_logits

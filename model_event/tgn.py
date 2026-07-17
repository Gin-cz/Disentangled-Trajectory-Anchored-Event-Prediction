"""Implement the temporal graph event-prediction baseline."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TGNEventHead(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        n_players: int = 22,
        n_event_types: int = 5,
        fut_T: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        volterra_Q: int = 2,
        volterra_scales: int = 2,
        n_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_players = int(n_players)
        self.n_event_types = int(n_event_types)
        self.fut_T = int(fut_T)
        self.n_heads = int(n_heads)

        self.input_fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.memory_rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_s = nn.LayerNorm(self.hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_f = nn.LayerNorm(self.hidden_dim)

        self.final_fusion = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.ball_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.prior_bias_scale = nn.Parameter(torch.tensor(1.0))

        self.head_tgt = nn.Linear(self.hidden_dim, 1)
        self.head_event = nn.Linear(self.hidden_dim, self.n_event_types)
        self.head_time = nn.Linear(self.hidden_dim, self.fut_T)
        self.head_pos = nn.Linear(self.hidden_dim, 2)

    def forward(self, h_players, h_ball, prior_emb, fut_emb, event_emb=None, context_ball=None):
        if h_players.dim() == 3: h_players = h_players.unsqueeze(1)
        if h_ball.dim() == 2: h_ball = h_ball.unsqueeze(1)

        B, C, A, H = h_players.shape

        if event_emb is None:
            event_emb = torch.zeros(B, C, H, device=h_players.device, dtype=h_players.dtype)

        edge_ctx_player = event_emb.unsqueeze(2).expand(-1, -1, A, -1)

        tgn_inputs = self.input_fusion(torch.cat([h_players, edge_ctx_player], dim=-1))

        memory = torch.zeros(B * A, H, device=h_players.device, dtype=h_players.dtype)


        for t in range(C):
            msg_t = tgn_inputs[:, t, :, :].reshape(B * A, H)

            memory = self.memory_rnn(msg_t, memory)

            mem_view = memory.view(B, A, H)

            spatial_out, _ = self.spatial_attn(mem_view, mem_view, mem_view)

            embedding_t = self.norm_s(mem_view + spatial_out)

            embedding_t = embedding_t + self.ffn(embedding_t)
            embedding_t = self.norm_f(embedding_t)


        final_node_states = embedding_t

        final_feat = self.final_fusion(torch.cat([final_node_states, prior_emb, fut_emb], dim=-1))

        ball_feat = h_ball.mean(dim=1)
        ball_feat = self.ball_encoder(ball_feat)

        ball_feat_expand = ball_feat.unsqueeze(1).expand(-1, A, -1)
        prior_bias = (final_feat * ball_feat_expand).sum(dim=-1) / math.sqrt(H)

        tgt_logits_main = self.head_tgt(final_feat).squeeze(-1)
        tgt_logits = tgt_logits_main + self.prior_bias_scale * prior_bias
        tgt_probs = F.softmax(tgt_logits, dim=-1)

        event_logits = self.head_event(final_feat)
        time_logits = self.head_time(final_feat)
        pos_pred = self.head_pos(final_feat)

        return tgt_probs, event_logits, time_logits, pos_pred, tgt_logits

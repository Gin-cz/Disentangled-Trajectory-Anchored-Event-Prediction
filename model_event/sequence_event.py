"""Implement sequence-to-event and marked point-process baselines."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pool_event_sequence(event_emb, h_players):
    if event_emb is None:
        return h_players.mean(dim=1, keepdim=True)
    if event_emb.dim() == 2:
        return event_emb.unsqueeze(1)
    if event_emb.dim() == 3:
        return event_emb
    raise ValueError(f"bad event_emb shape: {tuple(event_emb.shape)}")


def _ball_context(h_players, h_ball, context_ball):
    B, A, H = h_players.shape
    if context_ball is None:
        attn_score = torch.sum(h_players * h_ball.unsqueeze(1), dim=-1, keepdim=True) / math.sqrt(H)
        attn_weight = torch.softmax(attn_score, dim=1)
        context_ball = (attn_weight * h_players).sum(dim=1, keepdim=True)
    if context_ball.dim() == 2:
        context_ball = context_ball.unsqueeze(1)
    if context_ball.size(1) == 1:
        context_ball = context_ball.expand(-1, A, -1)
    return context_ball


class Seq2EventHead(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        n_event_types: int = 5,
        fut_T: int = 64,
        n_players: int = 22,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 1,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        volterra_Q: int = 2,
        volterra_scales: int = 2,
    ):
        super().__init__()
        self.n_players = n_players
        self.n_event_types = n_event_types
        self.fut_T = fut_T
        self.hidden_dim = hidden_dim

        gru_layers = max(int(num_layers), 1)
        self.event_rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.player_fusion = nn.Sequential(
            nn.Linear(5 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.event_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head_tgt = nn.Linear(hidden_dim, 1)
        self.head_time = nn.Linear(hidden_dim, fut_T)
        self.head_event = nn.Linear(hidden_dim, n_event_types)
        self.head_pos = nn.Linear(hidden_dim, 2)
        self.prior_bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h_players, h_ball, prior_emb, fut_emb, event_emb=None, context_ball=None):
        B, A, H = h_players.shape
        assert A == self.n_players, f"expected {self.n_players} players, got {A}"

        event_seq = _pool_event_sequence(event_emb, h_players)
        _, h_last = self.event_rnn(event_seq)
        seq_ctx = h_last[-1]
        seq_ctx_player = seq_ctx.unsqueeze(1).expand(-1, A, -1)

        context_ball = _ball_context(h_players, h_ball, context_ball)

        player_ctx = self.player_fusion(
            torch.cat([h_players, prior_emb, context_ball, fut_emb, seq_ctx_player], dim=-1)
        )
        z_player = self.event_fusion(torch.cat([player_ctx, seq_ctx_player], dim=-1))

        tgt_logits_main = self.head_tgt(z_player).squeeze(-1)
        prior_bias = (z_player * context_ball).sum(dim=-1) / math.sqrt(H)
        tgt_logits = tgt_logits_main + self.prior_bias_scale * prior_bias
        tgt_probs = F.softmax(tgt_logits, dim=-1)

        event_logits = self.head_event(z_player)
        time_logits = self.head_time(z_player)
        pos_pred = self.head_pos(z_player)
        return tgt_probs, event_logits, time_logits, pos_pred, tgt_logits


class NMSTPPEventHead(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        n_event_types: int = 5,
        fut_T: int = 64,
        n_players: int = 22,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 1,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        volterra_Q: int = 2,
        volterra_scales: int = 2,
    ):
        super().__init__()
        self.n_players = n_players
        self.n_event_types = n_event_types
        self.fut_T = fut_T
        self.hidden_dim = hidden_dim
        self.d_model = d_model

        self.event_in = nn.Linear(hidden_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.event_encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(int(num_layers), 1))
        self.event_out = nn.Linear(d_model, hidden_dim)

        self.player_fusion = nn.Sequential(
            nn.Linear(5 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mark_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.time_emb = nn.Embedding(fut_T, hidden_dim)
        self.time_intensity = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.head_tgt = nn.Linear(hidden_dim, 1)
        self.head_event = nn.Linear(hidden_dim, n_event_types)
        self.head_pos = nn.Linear(hidden_dim, 2)
        self.prior_bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h_players, h_ball, prior_emb, fut_emb, event_emb=None, context_ball=None):
        B, A, H = h_players.shape
        assert A == self.n_players, f"expected {self.n_players} players, got {A}"

        event_seq = _pool_event_sequence(event_emb, h_players)
        z_seq = self.event_encoder(self.event_in(event_seq))
        seq_ctx = self.event_out(z_seq[:, -1, :])
        seq_ctx_player = seq_ctx.unsqueeze(1).expand(-1, A, -1)

        context_ball = _ball_context(h_players, h_ball, context_ball)

        player_ctx = self.player_fusion(
            torch.cat([h_players, prior_emb, context_ball, fut_emb, seq_ctx_player], dim=-1)
        )
        mark_state = self.mark_fusion(torch.cat([player_ctx, seq_ctx_player], dim=-1))

        tgt_logits_main = self.head_tgt(mark_state).squeeze(-1)
        prior_bias = (mark_state * context_ball).sum(dim=-1) / math.sqrt(H)
        tgt_logits = tgt_logits_main + self.prior_bias_scale * prior_bias
        tgt_probs = F.softmax(tgt_logits, dim=-1)

        t_idx = torch.arange(self.fut_T, device=h_players.device)
        t_emb = self.time_emb(t_idx).view(1, 1, self.fut_T, H).expand(B, A, -1, -1)
        mark_time = mark_state.unsqueeze(2).expand(-1, -1, self.fut_T, -1)
        time_logits = self.time_intensity(torch.cat([mark_time, t_emb], dim=-1)).squeeze(-1)

        event_logits = self.head_event(mark_state)
        pos_pred = self.head_pos(mark_state)
        return tgt_probs, event_logits, time_logits, pos_pred, tgt_logits

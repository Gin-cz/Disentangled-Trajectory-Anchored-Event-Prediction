"""Implement the Transformer event-prediction baseline."""

import math
import torch
import torch.nn as nn


def _event_to_player(event_emb, n_players):
    if event_emb is None:
        return None
    if event_emb.dim() == 3:
        e = event_emb.mean(dim=1)
    elif event_emb.dim() == 2:
        e = event_emb
    else:
        raise ValueError(f"bad event_emb shape: {tuple(event_emb.shape)}")
    return e.unsqueeze(1).expand(-1, n_players, -1)


class TransformerEventHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_event_types: int = 5,
        fut_T: int = 64,
        n_players: int = 22,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        volterra_Q: int = 2,
        volterra_scales: int = 2,
    ):
        super().__init__()
        self.n_players = n_players
        self.n_event_types = n_event_types
        self.fut_T = fut_T
        self.d_model = d_model

        self.player_in = nn.Linear(4 * hidden_dim, d_model)
        self.ball_in = nn.Linear(2 * hidden_dim, d_model)
        self.player_index_emb = nn.Embedding(n_players, d_model)
        self.fusion_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head_tgt = nn.Linear(d_model, 1)
        self.head_event = nn.Linear(d_model, n_event_types)
        self.head_time = nn.Linear(d_model, fut_T)
        self.head_pos = nn.Linear(d_model, 2)
        self.prior_bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h_players, h_ball, prior_emb, fut_emb, event_emb=None, context_ball=None):
        B, A, H = h_players.shape
        assert A == self.n_players, f"expected {self.n_players} players, got {A}"

        e_player = _event_to_player(event_emb, A)
        if e_player is not None:
            fut_emb = self.fusion_proj(torch.cat([fut_emb, e_player], dim=-1))

        if context_ball is None:
            attn_score = torch.sum(h_players * h_ball.unsqueeze(1), dim=-1, keepdim=True) / math.sqrt(H)
            attn_weight = torch.softmax(attn_score, dim=1)
            context_ball = (attn_weight * h_players).sum(dim=1, keepdim=True).expand(-1, A, -1)

        fused_player = torch.cat([h_players, prior_emb, context_ball.expand(-1, A, -1), fut_emb], dim=-1)
        x_player = self.player_in(fused_player)

        gpool_players = h_players.mean(dim=1)
        ball_token = self.ball_in(torch.cat([h_ball, gpool_players], dim=-1)).unsqueeze(1)

        idx = torch.arange(A, device=h_players.device).unsqueeze(0).expand(B, A)
        x_player = x_player + self.player_index_emb(idx)

        tokens = torch.cat([ball_token, x_player], dim=1)
        z = self.encoder(tokens)
        z_ball = z[:, 0:1, :]
        z_player = z[:, 1:, :]

        tgt_logits_main = self.head_tgt(z_player).squeeze(-1)
        prior_bias = (z_player * z_ball.expand_as(z_player)).sum(dim=-1)
        tgt_logits = tgt_logits_main + self.prior_bias_scale * prior_bias
        tgt_probs = torch.softmax(tgt_logits, dim=-1)

        event_logits = self.head_event(z_player)
        time_logits = self.head_time(z_player)
        pos_pred = self.head_pos(z_player)

        return tgt_probs, event_logits, time_logits, pos_pred, tgt_logits

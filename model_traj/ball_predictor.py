"""Implement the optional autoregressive ball trajectory predictor."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BallStepHead(nn.Module):
    def __init__(self, hist_len, d_h, d_ctx, d_attn=128):
        super().__init__()
        self.hist_len = hist_len
        self.d_ctx = d_ctx
        self.d_attn = d_attn

        self.gru = nn.GRU(input_size=2, hidden_size=d_attn, batch_first=True)

        self.q = nn.Linear(d_attn + d_h + d_ctx, d_attn)
        self.k = nn.Linear(d_h, d_attn)
        self.v = nn.Linear(d_h, d_attn)

        self.mlp = nn.Sequential(
            nn.Linear(d_attn + d_attn, d_attn),
            nn.ReLU(inplace=True),
            nn.Linear(d_attn, 2)
        )

    def forward(self, ball_hist, h_ball_roll, context_ball=None, h_players=None, mode=None):
        B, L, _ = ball_hist.shape
        if L > self.hist_len:
            ball_hist = ball_hist[:, -self.hist_len:, :]

        _, h_last = self.gru(ball_hist)
        z_hist = h_last.squeeze(0)

        if context_ball is None:
            ctx = torch.zeros(B, self.d_ctx, device=ball_hist.device, dtype=ball_hist.dtype)
        else:
            ctx = context_ball.squeeze(1)

        z_base = torch.cat([z_hist, h_ball_roll, ctx], dim=-1)

        if h_players is not None and mode != 'simple':
            q = self.q(z_base).unsqueeze(1)
            k = self.k(h_players)
            v = self.v(h_players)
            scores = (q * k).sum(-1) / math.sqrt(self.d_attn)
            attn = F.softmax(scores, dim=1)
            z_pl = (attn.unsqueeze(-1) * v).sum(dim=1)
        else:
            z_pl = torch.zeros(B, self.d_attn, device=ball_hist.device, dtype=ball_hist.dtype)

        delta = self.mlp(torch.cat([self.q(z_base), z_pl], dim=-1))
        ball_pos = ball_hist[:, -1, :] + delta
        return ball_pos

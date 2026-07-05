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

        # players cross-attn (optional)
        self.q = nn.Linear(d_attn + d_h + d_ctx, d_attn)
        self.k = nn.Linear(d_h, d_attn)
        self.v = nn.Linear(d_h, d_attn)

        # predict delta
        self.mlp = nn.Sequential(
            nn.Linear(d_attn + d_attn, d_attn),
            nn.ReLU(inplace=True),
            nn.Linear(d_attn, 2)
        )

    def forward(self, ball_hist, h_ball_roll, context_ball=None, h_players=None, mode=None):
        """
        ball_hist:    [B, L, 2]
        h_ball_roll:  [B, d_h]
        context_ball: [B, 1, d_ctx] or None
        h_players:    [B, 22, d_h] or None
        return: ball_pos [B,2]
        """
        B, L, _ = ball_hist.shape
        if L > self.hist_len:
            ball_hist = ball_hist[:, -self.hist_len:, :]

        _, h_last = self.gru(ball_hist)      # [1,B,d_attn]
        z_hist = h_last.squeeze(0)           # [B,d_attn]

        # ctx vector (always define)
        if context_ball is None:
            ctx = torch.zeros(B, self.d_ctx, device=ball_hist.device, dtype=ball_hist.dtype)
        else:
            ctx = context_ball.squeeze(1)    # [B,d_ctx]

        z_base = torch.cat([z_hist, h_ball_roll, ctx], dim=-1)  # [B, d_attn+d_h+d_ctx]

        # optional cross-attn to players (always define z_pl)
        if h_players is not None and mode != 'simple':
            q = self.q(z_base).unsqueeze(1)      # [B,1,d_attn]
            k = self.k(h_players)                # [B,22,d_attn]
            v = self.v(h_players)                # [B,22,d_attn]
            scores = (q * k).sum(-1) / math.sqrt(self.d_attn)   # [B,22]
            attn = F.softmax(scores, dim=1)                    # [B,22]
            z_pl = (attn.unsqueeze(-1) * v).sum(dim=1)         # [B,d_attn]
        else:
            z_pl = torch.zeros(B, self.d_attn, device=ball_hist.device, dtype=ball_hist.dtype)

        delta = self.mlp(torch.cat([self.q(z_base), z_pl], dim=-1))  # [B,2]
        ball_pos = ball_hist[:, -1, :] + delta
        return ball_pos

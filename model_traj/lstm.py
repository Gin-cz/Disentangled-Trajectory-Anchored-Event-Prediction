"""Implement the LSTM trajectory backbone and prediction head."""

import torch
import torch.nn as nn
import torch.nn.functional as Fnn


class StrictLSTM(nn.Module):
    def __init__(self, in_dim, h_dim, nonlin='tanh'):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim

        self.lstm_cell = nn.LSTMCell(in_dim, h_dim)

        self.act = {'tanh': torch.tanh, 'relu': Fnn.relu, 'identity': (lambda x: x)}[nonlin]

    def step(self, x_t, h_prev, c_prev):

        h_t, c_t = self.lstm_cell(x_t, (h_prev, c_prev))
        return h_t, c_t

    def forward(self, x, h0=None, c0=None):
        B, T, _ = x.shape

        if h0 is None:
            h = x.new_zeros(B, self.h_dim)
        else:
            h = h0.clone()

        if c0 is None:
            c = x.new_zeros(B, self.h_dim)
        else:
            c = c0.clone()

        h_seq, c_seq = [], []
        for t in range(T):
            h_new, c_new = self.step(x[:, t], h, c)
            h = h_new
            c = c_new

            h_seq.append(h.unsqueeze(1))
            c_seq.append(c.unsqueeze(1))

        h_seq = torch.cat(h_seq, dim=1)
        c_seq = torch.cat(c_seq, dim=1)

        return h_seq, h.clone(), c_seq, c.clone()


class AgentLSTM(nn.Module):
    def __init__(self, in_dim, h_dim, d_mem=None, theta=None, dt=None,
                 use_h=True, use_m=True, nonlin='tanh'):
        super().__init__()
        self.h_dim = h_dim
        self.cell = StrictLSTM(in_dim, h_dim, nonlin)

    def forward(self, x, h0=None, m0=None):
        B, T, A, F = x.shape

        x_flat = x.reshape(B*A, T, F).contiguous().clone()

        if h0 is not None:
            h0_flat = h0.reshape(B*A, -1).contiguous().clone()
        else:
            h0_flat = None

        if m0 is not None:
            c0_flat = m0.reshape(B*A, -1).contiguous().clone()
        else:
            c0_flat = None

        h_seq, h_last, c_seq, c_last = self.cell(x_flat, h0_flat, c0_flat)

        return {
            "h_seq":  h_seq.reshape(B, T, A, -1),
            "h_last": h_last.reshape(B, A, -1),
            "m_seq":  c_seq.reshape(B, T, A, -1),
            "m_last": c_last.reshape(B, A, -1),
        }

    def step(self, x_t, h_prev, m_prev):
        B, A, F = x_t.shape

        x_flat = x_t.reshape(B*A, F).contiguous().clone()
        h_flat = h_prev.reshape(B*A, -1).contiguous().clone()
        c_flat = m_prev.reshape(B*A, -1).contiguous().clone()

        h_t, c_t = self.cell.step(x_flat, h_flat, c_flat)
        return (h_t.reshape(B, A, -1), c_t.reshape(B, A, -1))


class LSTMTrajHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, T: int = 64, dropout: float = 0.1):
        super().__init__()
        self.T = T
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2 * T),
        )

    def forward(self, h_agents: torch.Tensor, last_pos: torch.Tensor) -> torch.Tensor:
        B, A, Dh = h_agents.shape
        deltas = self.mlp(h_agents).view(B, A, self.T, 2)
        traj_abs = last_pos.unsqueeze(2) + deltas.cumsum(2)
        return traj_abs.permute(2, 0, 1, 3).contiguous()

class LSTMTrajStepHead(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.1, use_m=True, m_dim=None, max_step=None):
        super().__init__()
        self.use_m = use_m
        f_in = in_dim + 2 + (m_dim if (use_m and m_dim is not None) else 0)
        self.net = nn.Sequential(
            nn.Linear(f_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2)
        )

    def forward(self, h_agents, last_pos, m_t=None,
                h_players=None, prior_emb=None, volterra_features=None):
        if self.use_m and m_t is not None:
            x = torch.cat([h_agents, last_pos, m_t], dim=-1)
        else:
            x = torch.cat([h_agents, last_pos], dim=-1)
        delta = self.net(x)
        return last_pos + delta

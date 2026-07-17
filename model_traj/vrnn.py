"""Implement the variational recurrent trajectory backbone and prediction heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

LOG_2PI = math.log(2 * math.pi)
EPS = 1e-6


class VRNNCell(nn.Module):

    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
        )
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
        )

        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus(),
        )

        self.enc = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus(),
        )

        self.dec = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus(),
        )

        self.rnn = nn.GRUCell(h_dim * 2, h_dim)

    def forward_posterior(self, x_t, h_prev):
        phi_x_t = self.phi_x(x_t)

        prior_t = self.prior(h_prev)
        p_mean = self.prior_mean(prior_t)
        p_std = self.prior_std(prior_t).clamp(min=EPS)

        enc_input = torch.cat([phi_x_t, h_prev], dim=-1)
        enc_t = self.enc(enc_input)
        q_mean = self.enc_mean(enc_t)
        q_std = self.enc_std(enc_t).clamp(min=EPS)

        eps = torch.randn_like(q_std)
        z_t = q_mean + q_std * eps

        phi_z_t = self.phi_z(z_t)

        dec_input = torch.cat([phi_z_t, h_prev], dim=-1)
        dec_t = self.dec(dec_input)
        x_mean = self.dec_mean(dec_t)
        x_std = self.dec_std(dec_t).clamp(min=EPS)

        rnn_input = torch.cat([phi_x_t, phi_z_t], dim=-1)
        h_t = self.rnn(rnn_input, h_prev)

        kld = 0.5 * torch.sum(
            2 * torch.log(p_std) - 2 * torch.log(q_std) +
            (q_std.pow(2) + (q_mean - p_mean).pow(2)) / p_std.pow(2) - 1,
            dim=-1
        )

        var = x_std.pow(2)
        nll = 0.5 * torch.sum(
            ((x_t - x_mean).pow(2) / var) + torch.log(var) + LOG_2PI,
            dim=-1
        )

        return h_t, kld, nll, x_mean

    def forward_prior(self, x_t, h_prev):
        phi_x_t = self.phi_x(x_t)

        prior_t = self.prior(h_prev)
        p_mean = self.prior_mean(prior_t)
        p_std = self.prior_std(prior_t).clamp(min=EPS)

        eps = torch.randn_like(p_std)
        z_t = p_mean + p_std * eps

        phi_z_t = self.phi_z(z_t)

        dec_input = torch.cat([phi_z_t, h_prev], dim=-1)
        dec_t = self.dec(dec_input)
        x_pred_mean = self.dec_mean(dec_t)
        x_pred_std = self.dec_std(dec_t).clamp(min=EPS)

        rnn_input = torch.cat([phi_x_t, phi_z_t], dim=-1)
        h_t = self.rnn(rnn_input, h_prev)

        return h_t, x_pred_mean, x_pred_std


class AgentVRNN(nn.Module):

    def __init__(self,
                 in_dim,
                 h_dim=128,
                 z_dim=32,
                 window_K=8,
                 d_mem=None,
                 n_players=22,
                 dropout=0.1,
                 n_heads=None,
                 n_layers=None,
                 context_dim=None,
                 theta=None,
                 dt=None,
                 use_h=True,
                 use_m=True,
                 nonlin='tanh'):
        super().__init__()

        self.in_dim = in_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.d_mem = d_mem if d_mem is not None else h_dim
        self.window_K = window_K
        self.n_players = n_players

        self.vrnn_cell = VRNNCell(in_dim, h_dim, z_dim)

        if self.d_mem != h_dim:
            self.mem_proj = nn.Linear(h_dim, self.d_mem)
        else:
            self.mem_proj = None

        self._kld_loss = None
        self._nll_loss = None

    def forward(self, x, h0=None, m0=None,
                h_players=None, prior_emb=None, context_ball=None,
                event_emb=None):
        B, T, A, F = x.shape
        device = x.device

        if h0 is not None:
            h = h0.reshape(B * A, self.h_dim)
        else:
            h = torch.zeros(B * A, self.h_dim, device=device)

        h_seq_list = []
        total_kld = 0.0
        total_nll = 0.0

        for t in range(T):
            x_t = x[:, t].reshape(B * A, F)

            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)

            h, kld, nll, _ = self.vrnn_cell.forward_posterior(x_t, h)

            total_kld = total_kld + kld.mean()
            total_nll = total_nll + nll.mean()

            h_seq_list.append(h.reshape(B, A, self.h_dim))

        self._kld_loss = total_kld / T
        self._nll_loss = total_nll / T

        h_seq = torch.stack(h_seq_list, dim=1)
        h_last = h_seq[:, -1]

        if self.mem_proj is not None:
            m_seq = self.mem_proj(h_seq)
            m_last = m_seq[:, -1]
        else:
            m_seq = h_seq
            m_last = h_last

        K = self.window_K
        if T >= K:
            mem_window = h_seq[:, -K:]
        else:
            pad_len = K - T
            pad = torch.zeros(B, pad_len, A, self.h_dim, device=device)
            mem_window = torch.cat([pad, h_seq], dim=1)

        return {
            "h_seq": h_seq,
            "h_last": h_last,
            "m_seq": m_seq,
            "m_last": m_last,
            "mem_window": mem_window,
        }

    def step(self, x_t, h_prev_window,
             h_players=None, prior_emb=None, context_ball=None):
        B, A, F = x_t.shape
        device = x_t.device
        K = h_prev_window.size(1)

        h_prev = h_prev_window[:, -1].reshape(B * A, self.h_dim)

        x_t_flat = x_t.reshape(B * A, F)
        x_t_flat = torch.nan_to_num(x_t_flat, nan=0.0, posinf=0.0, neginf=0.0)

        h_t, _, _ = self.vrnn_cell.forward_prior(x_t_flat, h_prev)

        h_t = h_t.reshape(B, A, self.h_dim)

        if self.mem_proj is not None:
            m_t = self.mem_proj(h_t)
        else:
            m_t = h_t

        new_window = torch.cat([h_prev_window[:, 1:], h_t.unsqueeze(1)], dim=1)

        return {
            "h_t": h_t,
            "m_t": m_t,
            "mem_window": new_window,
        }

    def get_vrnn_losses(self):
        return self._kld_loss, self._nll_loss


class VRNNTrajStepHead(nn.Module):

    def __init__(self,
                 in_dim,
                 hidden=256,
                 dropout=0.1,
                 use_m=True,
                 m_dim=None,
                 context_dim=None,
                 use_volterra=False,
                 max_step=None):
        super().__init__()

        self.use_m = use_m
        self.context_dim = context_dim

        f_in = in_dim + 2
        if use_m and m_dim is not None:
            f_in += m_dim

        self.net = nn.Sequential(
            nn.Linear(f_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self,
                h_agents,
                last_pos,
                m_t=None,
                h_players=None,
                prior_emb=None,
                volterra_features=None):
        features = [h_agents, last_pos]

        if self.use_m and m_t is not None:
            features.append(m_t)

        x = torch.cat(features, dim=-1)
        delta = self.net(x)

        return last_pos + delta



class AgentVRNNWithContext(AgentVRNN):

    def __init__(self,
                 in_dim,
                 h_dim=128,
                 z_dim=32,
                 window_K=8,
                 d_mem=None,
                 n_heads=4,
                 n_layers=2,
                 dropout=0.1,
                 context_dim=128,
                 n_players=22,
                 use_event_as_cls=False):
        super().__init__(
            in_dim=in_dim,
            h_dim=h_dim,
            z_dim=z_dim,
            window_K=window_K,
            d_mem=d_mem,
            n_players=n_players,
            dropout=dropout,
        )

        self.context_dim = context_dim

        self.context_fusion = nn.Sequential(
            nn.Linear(h_dim + context_dim * 2, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _fuse_context(self, h, h_players, prior_emb, B, A):
        device = h.device

        if h_players is None:
            h_players = torch.zeros(B, A, self.context_dim, device=device)
        if prior_emb is None:
            prior_emb = torch.zeros(B, A, self.context_dim, device=device)

        h_reshaped = h.reshape(B, A, self.h_dim)

        if h_players.size(1) < A:
            pad = torch.zeros(B, A - h_players.size(1), self.context_dim, device=device)
            h_players = torch.cat([h_players, pad], dim=1)
            prior_emb = torch.cat([prior_emb, pad], dim=1)

        combined = torch.cat([h_reshaped, h_players, prior_emb], dim=-1)
        h_fused = self.context_fusion(combined)

        return h_fused.reshape(B * A, self.h_dim)

    def forward(self, x, h0=None, m0=None,
                h_players=None, prior_emb=None, context_ball=None,
                event_emb=None):
        B, T, A, F = x.shape
        device = x.device

        if h0 is not None:
            h = h0.reshape(B * A, self.h_dim)
        else:
            h = torch.zeros(B * A, self.h_dim, device=device)

        h_seq_list = []
        total_kld = 0.0
        total_nll = 0.0

        for t in range(T):
            x_t = x[:, t].reshape(B * A, F)
            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)

            h_ctx = self._fuse_context(h, h_players, prior_emb, B, A)

            h, kld, nll, _ = self.vrnn_cell.forward_posterior(x_t, h_ctx)

            total_kld = total_kld + kld.mean()
            total_nll = total_nll + nll.mean()

            h_seq_list.append(h.reshape(B, A, self.h_dim))

        self._kld_loss = total_kld / T
        self._nll_loss = total_nll / T

        h_seq = torch.stack(h_seq_list, dim=1)
        h_last = h_seq[:, -1]

        if self.mem_proj is not None:
            m_seq = self.mem_proj(h_seq)
            m_last = m_seq[:, -1]
        else:
            m_seq = h_seq
            m_last = h_last

        K = self.window_K
        if T >= K:
            mem_window = h_seq[:, -K:]
        else:
            pad_len = K - T
            pad = torch.zeros(B, pad_len, A, self.h_dim, device=device)
            mem_window = torch.cat([pad, h_seq], dim=1)

        return {
            "h_seq": h_seq,
            "h_last": h_last,
            "m_seq": m_seq,
            "m_last": m_last,
            "mem_window": mem_window,
        }

    def step(self, x_t, h_prev_window,
             h_players=None, prior_emb=None, context_ball=None):
        B, A, F = x_t.shape
        device = x_t.device
        K = h_prev_window.size(1)

        h_prev = h_prev_window[:, -1].reshape(B * A, self.h_dim)

        h_ctx = self._fuse_context(h_prev, h_players, prior_emb, B, A)

        x_t_flat = x_t.reshape(B * A, F)
        x_t_flat = torch.nan_to_num(x_t_flat, nan=0.0, posinf=0.0, neginf=0.0)

        h_t, _, _ = self.vrnn_cell.forward_prior(x_t_flat, h_ctx)
        h_t = h_t.reshape(B, A, self.h_dim)

        if self.mem_proj is not None:
            m_t = self.mem_proj(h_t)
        else:
            m_t = h_t

        new_window = torch.cat([h_prev_window[:, 1:], h_t.unsqueeze(1)], dim=1)

        return {
            "h_t": h_t,
            "m_t": m_t,
            "mem_window": new_window,
        }


class VRNNTrajStepHeadWithContext(nn.Module):

    def __init__(self,
                 in_dim,
                 hidden=256,
                 dropout=0.1,
                 use_m=False,
                 m_dim=None,
                 context_dim=128,
                 use_volterra=False):
        super().__init__()

        self.use_m = use_m
        self.context_dim = context_dim

        f_in = in_dim + 2 + context_dim * 2
        if use_m and m_dim is not None:
            f_in += m_dim

        self.feat_fusion = nn.Sequential(
            nn.Linear(f_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, in_dim),
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(in_dim)

        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self,
                h_agents,
                last_pos,
                m_t=None,
                h_players=None,
                prior_emb=None,
                volterra_features=None):
        B, A, Dh = h_agents.shape
        device = h_agents.device

        features = [h_agents, last_pos]

        if self.use_m and m_t is not None:
            features.append(m_t)

        if h_players is not None:
            if h_players.size(1) < A:
                pad = torch.zeros(B, A - h_players.size(1), self.context_dim, device=device)
                h_players = torch.cat([h_players, pad], dim=1)
            features.append(h_players)
        else:
            features.append(torch.zeros(B, A, self.context_dim, device=device))

        if prior_emb is not None:
            if prior_emb.size(1) < A:
                pad = torch.zeros(B, A - prior_emb.size(1), self.context_dim, device=device)
                prior_emb = torch.cat([prior_emb, pad], dim=1)
            features.append(prior_emb)
        else:
            features.append(torch.zeros(B, A, self.context_dim, device=device))

        x = torch.cat(features, dim=-1)
        x_fused = self.feat_fusion(x)

        x_attended, _ = self.self_attn(x_fused, x_fused, x_fused)
        x_fused = self.attn_norm(x_fused + x_attended)

        delta = self.predictor(x_fused)

        return last_pos + delta


VRNNWindowedWithContext = AgentVRNNWithContext
VRNNWindowedNoContext = AgentVRNN
TransformerTrajStepHeadWithContext = VRNNTrajStepHeadWithContext
TransformerTrajStepHeadNoContext = VRNNTrajStepHead

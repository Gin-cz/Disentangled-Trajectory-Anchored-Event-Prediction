"""Implement the STAR trajectory backbone and prediction heads."""

import copy
import torch
import torch.nn as nn


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



class SpatialTransformerBatched(nn.Module):
    def __init__(self, d_model, nhead, nhid, nlayers=1, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.d_model = d_model

    def forward(self, x, adj_mask=None):
        B, T, P, D = x.shape

        x_flat = x.reshape(B * T, P, D)

        attn_mask = None
        if adj_mask is not None:
            A = adj_mask.size(0)
            n_mask = adj_mask + torch.eye(A, device=adj_mask.device)
            attn_mask = n_mask.float().masked_fill(n_mask == 0., float('-inf')).masked_fill(n_mask == 1., 0.0)

        out_flat = self.transformer(x_flat, mask=attn_mask)

        return out_flat.reshape(B, T, P, D)


class TemporalTransformerBatched(nn.Module):
    def __init__(self, d_model, nhead, nhid, nlayers=1, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.d_model = d_model

    def forward(self, x, causal_mask=None):
        B, T, P, D = x.shape

        x_perm = x.permute(0, 2, 1, 3)
        x_flat = x_perm.reshape(B * P, T, D)

        out_flat = self.transformer(x_flat, mask=causal_mask)

        out_perm = out_flat.reshape(B, P, T, D)
        return out_perm.permute(0, 2, 1, 3)



class STARWindowedWithContext(nn.Module):

    def __init__(self,
                 in_dim,
                 h_dim=128,
                 window_K=8,
                 d_mem=None,
                 n_heads=8,
                 n_layers=2,
                 dropout=0.1,
                 context_dim=128,
                 n_players=22,
                 use_event_as_cls=False):
        super().__init__()

        self.in_dim = in_dim
        self.h_dim = h_dim
        self.d_mem = d_mem if d_mem is not None else h_dim
        self.window_K = window_K
        self.context_dim = context_dim
        self.n_players = n_players
        self.n_layers = n_layers

        player_input_dim = in_dim + context_dim * 2
        self.input_embedding_player = nn.Sequential(
            nn.Linear(player_input_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        ball_input_dim = in_dim + context_dim
        self.input_embedding_ball = nn.Sequential(
            nn.Linear(ball_input_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.spatial_layers = nn.ModuleList([
            SpatialTransformerBatched(h_dim, n_heads, h_dim * 4, nlayers=1, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.temporal_layers = nn.ModuleList([
            TemporalTransformerBatched(h_dim, n_heads, h_dim * 4, nlayers=1, dropout=dropout)
            for _ in range(n_layers)
        ])

        if self.d_mem != h_dim:
            self.mem_proj = nn.Linear(h_dim, self.d_mem)
        else:
            self.mem_proj = None

    def _build_full_adj(self, P, device):
        adj = torch.ones(P, P, device=device)
        adj = adj - torch.eye(P, device=device)
        return adj

    def _embed_players_batched(self, x_players, h_players, prior_emb, device):
        if x_players.dim() == 3:
            B, P, _ = x_players.shape
            if h_players is None:
                h_players = torch.zeros(B, P, self.context_dim, device=device)
            if prior_emb is None:
                prior_emb = torch.zeros(B, P, self.context_dim, device=device)
            combined = torch.cat([x_players, h_players, prior_emb], dim=-1)
            return self.input_embedding_player(combined)
        else:
            B, T, P, _ = x_players.shape
            if h_players is None:
                h_players = torch.zeros(B, P, self.context_dim, device=device)
            if prior_emb is None:
                prior_emb = torch.zeros(B, P, self.context_dim, device=device)
            h_players_exp = h_players.unsqueeze(1).expand(B, T, P, -1)
            prior_emb_exp = prior_emb.unsqueeze(1).expand(B, T, P, -1)
            combined = torch.cat([x_players, h_players_exp, prior_emb_exp], dim=-1)
            flat = combined.reshape(B * T * P, -1)
            emb = self.input_embedding_player(flat)
            return emb.reshape(B, T, P, self.h_dim)

    def _embed_ball_batched(self, x_ball, context_ball, device):
        if x_ball.dim() == 3:
            B = x_ball.size(0)
            if context_ball is None:
                context_ball = torch.zeros(B, 1, self.context_dim, device=device)
            combined = torch.cat([x_ball, context_ball], dim=-1)
            return self.input_embedding_ball(combined)
        else:
            B, T = x_ball.shape[:2]
            if context_ball is None:
                context_ball = torch.zeros(B, 1, self.context_dim, device=device)
            context_ball_exp = context_ball.unsqueeze(1).expand(B, T, 1, -1)
            combined = torch.cat([x_ball, context_ball_exp], dim=-1)
            flat = combined.reshape(B * T, -1)
            emb = self.input_embedding_ball(flat)
            return emb.reshape(B, T, 1, self.h_dim)

    def forward(self, x, h0=None, m0=None,
                h_players=None, prior_emb=None, context_ball=None):
        B, T, A_total, F_in = x.shape
        device = x.device
        P = self.n_players

        x_players = x[:, :, :P, :]
        x_ball = x[:, :, P:P+1, :]

        players_emb = self._embed_players_batched(x_players, h_players, prior_emb, device)
        ball_emb = self._embed_ball_batched(x_ball, context_ball, device)
        h = torch.cat([players_emb, ball_emb], dim=2)

        adj = self._build_full_adj(P+1, device)

        for layer_idx in range(self.n_layers):
            h = self.spatial_layers[layer_idx](h, adj)

            h = self.temporal_layers[layer_idx](h)

        h_seq = h
        h_last = h_seq[:, -1, :, :]

        if self.mem_proj is not None:
            m_seq = self.mem_proj(h_seq)
            m_last = m_seq[:, -1, :, :]
        else:
            m_seq = h_seq
            m_last = h_last

        K = self.window_K
        if T >= K:
            mem_window = h_seq[:, -K:, :, :]
        else:
            pad_len = K - T
            pad = torch.zeros(B, pad_len, A_total, self.h_dim, device=device)
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
        B, A_total, F_in = x_t.shape
        device = x_t.device
        P = self.n_players
        K = h_prev_window.size(1)

        x_players = x_t[:, :P, :]
        x_ball = x_t[:, P:P+1, :]

        h_new = self._embed_players_batched(x_players, h_players, prior_emb, device)
        h_new = h_new.unsqueeze(1)
        ball_emb = self._embed_ball_batched(x_ball, context_ball, device)
        ball_emb = ball_emb.unsqueeze(2)
        h_now = torch.cat([h_new, ball_emb], dim=2)

        h_full = torch.cat([h_prev_window, h_now], dim=1)

        adj = self._build_full_adj(P + 1, device)

        for layer_idx in range(self.n_layers):
            h_full = self.spatial_layers[layer_idx](h_full, adj)
            h_full = self.temporal_layers[layer_idx](h_full)

        h_t = h_full[:, -1, :, :]

        if self.mem_proj is not None:
            m_t = self.mem_proj(h_t)
        else:
            m_t = h_t

        new_window = torch.cat([h_prev_window[:, 1:, :, :], h_t.unsqueeze(1)], dim=1)

        return {
            "h_t": h_t,
            "m_t": m_t,
            "mem_window": new_window,
        }



class STARWindowedNoContext(nn.Module):

    def __init__(self,
                 in_dim,
                 h_dim=128,
                 window_K=8,
                 d_mem=None,
                 n_heads=8,
                 n_layers=2,
                 dropout=0.1,
                 n_players=22):
        super().__init__()

        self.in_dim = in_dim
        self.h_dim = h_dim
        self.d_mem = d_mem if d_mem is not None else h_dim
        self.window_K = window_K
        self.n_players = n_players
        self.n_layers = n_layers

        self.input_embedding_player = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.input_embedding_ball = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.spatial_layers = nn.ModuleList([
            SpatialTransformerBatched(h_dim, n_heads, h_dim * 4, nlayers=1, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.temporal_layers = nn.ModuleList([
            TemporalTransformerBatched(h_dim, n_heads, h_dim * 4, nlayers=1, dropout=dropout)
            for _ in range(n_layers)
        ])

        if self.d_mem != h_dim:
            self.mem_proj = nn.Linear(h_dim, self.d_mem)
        else:
            self.mem_proj = None

    def _build_full_adj(self, P, device):
        adj = torch.ones(P, P, device=device)
        adj = adj - torch.eye(P, device=device)
        return adj

    def forward(self, x, h0=None, m0=None,
                h_players=None, prior_emb=None, context_ball=None):
        B, T, A_total, F_in = x.shape
        device = x.device
        P = self.n_players

        x_players = x[:, :, :P, :]
        x_ball = x[:, :, P:P+1, :]

        players_flat = x_players.reshape(B * T * P, F_in)
        players_emb = self.input_embedding_player(players_flat)
        players_emb = players_emb.reshape(B, T, P, self.h_dim)

        ball_flat = x_ball.reshape(B * T, F_in)
        ball_emb = self.input_embedding_ball(ball_flat)
        ball_emb = ball_emb.reshape(B, T, 1, self.h_dim)

        h = torch.cat([players_emb, ball_emb], dim=2)
        adj = self._build_full_adj(P + 1, device)

        for layer_idx in range(self.n_layers):
            h = self.spatial_layers[layer_idx](h, adj)
            h = self.temporal_layers[layer_idx](h)

        h_seq = h
        h_last = h_seq[:, -1, :, :]

        if self.mem_proj is not None:
            m_seq = self.mem_proj(h_seq)
            m_last = m_seq[:, -1, :, :]
        else:
            m_seq = h_seq
            m_last = h_last

        K = self.window_K
        if T >= K:
            mem_window = h_seq[:, -K:, :, :]
        else:
            pad_len = K - T
            pad = torch.zeros(B, pad_len, A_total, self.h_dim, device=device)
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
        B, A_total, F_in = x_t.shape
        device = x_t.device
        P = self.n_players
        K = h_prev_window.size(1)

        x_players = x_t[:, :P, :]
        x_ball = x_t[:, P:P+1, :]

        players_flat = x_players.reshape(B * P, F_in)
        h_new = self.input_embedding_player(players_flat)
        h_new = h_new.reshape(B, 1, P, self.h_dim)

        ball_emb = self.input_embedding_ball(x_ball.reshape(B, F_in))
        ball_emb = ball_emb.reshape(B, 1, self.h_dim)
        ball_emb = ball_emb.unsqueeze(2)
        h_now = torch.cat([h_new, ball_emb], dim=2)

        h_full = torch.cat([h_prev_window, h_now], dim=1)

        adj = self._build_full_adj(P + 1, device)


        for layer_idx in range(self.n_layers):
            h_full = self.spatial_layers[layer_idx](h_full, adj)
            h_full = self.temporal_layers[layer_idx](h_full)

        h_t = h_full[:, -1, :, :]
        if self.mem_proj is not None:
            m_t = self.mem_proj(h_t)
        else:
            m_t = h_t

        new_window = torch.cat([h_prev_window[:, 1:, :, :], h_t.unsqueeze(1)], dim=1)

        return {
            "h_t": h_t,
            "m_t": m_t,
            "mem_window": new_window,
        }



class TransformerTrajStepHeadWithContext(nn.Module):
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
        self.use_volterra = use_volterra
        self.context_dim = context_dim

        f_in = in_dim + 2
        if use_m and m_dim is not None:
            f_in += m_dim
        f_in += context_dim * 2
        if use_volterra:
            f_in += context_dim

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
            features.append(h_players)
        else:
            features.append(torch.zeros(B, A, self.context_dim, device=device))

        if prior_emb is not None:
            features.append(prior_emb)
        else:
            features.append(torch.zeros(B, A, self.context_dim, device=device))

        if self.use_volterra and volterra_features is not None:
            features.append(volterra_features)

        x = torch.cat(features, dim=-1)
        x_fused = self.feat_fusion(x)
        x_attended, _ = self.self_attn(x_fused, x_fused, x_fused)
        x_fused = self.attn_norm(x_fused + x_attended)
        delta = self.predictor(x_fused)

        return last_pos + delta


class TransformerTrajStepHeadNoContext(nn.Module):
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

        f_in = in_dim + 2

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

        x = torch.cat([h_agents, last_pos], dim=-1)
        x_fused = self.feat_fusion(x)
        x_attended, _ = self.self_attn(x_fused, x_fused, x_fused)
        x_fused = self.attn_norm(x_fused + x_attended)
        delta = self.predictor(x_fused)

        return last_pos + delta

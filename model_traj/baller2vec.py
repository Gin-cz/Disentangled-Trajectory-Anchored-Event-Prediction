"""Implement the Baller2Vec trajectory backbone and prediction heads."""

import math
import torch

from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        if x.dim() == 3:
            return x + self.pe[:, :x.size(1)]
        else:
            return x + self.pe[:, :x.size(1)].unsqueeze(2)

class Baller2VecWindowedWithContext(nn.Module):

    def __init__(self,
                 in_dim,
                 h_dim,
                 window_K=8,
                 d_mem=None,
                 n_heads=8,
                 n_layers=4,
                 dropout=0.1,
                 context_dim=None,
                 n_players=22,
                 use_event_as_cls=True):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.d_mem = h_dim if d_mem is None else d_mem
        self.context_dim = context_dim or h_dim
        self.n_players = n_players
        self.use_event_as_cls = use_event_as_cls
        self.window_K = window_K

        initrange = 0.1
        self.ball_embedding = nn.Parameter(torch.Tensor(h_dim))
        nn.init.uniform_(self.ball_embedding, -initrange, initrange)

        if use_event_as_cls:
            self.event_token = nn.Parameter(torch.Tensor(h_dim))
            nn.init.uniform_(self.event_token, -initrange, initrange)

        player_input_dim = in_dim + context_dim * 2
        self.player_mlp = nn.Sequential(
            nn.Linear(player_input_dim, h_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim * 2, h_dim),
            nn.LayerNorm(h_dim),
        )

        ball_input_dim = in_dim + context_dim
        self.ball_mlp = nn.Sequential(
            nn.Linear(ball_input_dim, h_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim * 2, h_dim),
            nn.LayerNorm(h_dim),
        )

        if use_event_as_cls:
            self.event_mlp = nn.Sequential(
                nn.Linear(context_dim * 3, h_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(h_dim * 2, h_dim),
            )

        self.pos_encoder = PositionalEncoding(h_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h_dim,
            nhead=n_heads,
            dim_feedforward=h_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.step_transformer_layer = nn.TransformerEncoderLayer(
            d_model=h_dim,
            nhead=n_heads,
            dim_feedforward=h_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.step_norm = nn.LayerNorm(h_dim)

        if self.d_mem != h_dim:
            self.mem_proj = nn.Linear(h_dim, self.d_mem)
        else:
            self.mem_proj = None

    def generate_baller2vec_mask(self, seq_len, device):
        n_entities = self.n_players + 1
        if self.use_event_as_cls:
            n_entities += 1

        total_tokens = n_entities * seq_len
        mask = torch.zeros(total_tokens, total_tokens, device=device)

        ball_start = self.n_players * seq_len
        event_start = ball_start + seq_len if self.use_event_as_cls else None

        for t in range(seq_len):
            players_start = self.n_players * t
            players_end = players_start + self.n_players
            ball_idx = ball_start + t

            for i in range(players_start, players_end):
                mask[i, :players_end] = 1
                mask[i, ball_start:ball_idx+1] = 1
                if self.use_event_as_cls:
                    mask[i, event_start:event_start+t+1] = 1

            mask[ball_idx, :players_end] = 1
            mask[ball_idx, ball_start:ball_idx+1] = 1
            if self.use_event_as_cls:
                mask[ball_idx, event_start:event_start+t+1] = 1

            if self.use_event_as_cls:
                event_idx = event_start + t
                mask[event_idx, :players_end] = 1
                mask[event_idx, ball_start:ball_idx+1] = 1
                mask[event_idx, event_start:event_idx+1] = 1

        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def _process_sequence(self, x, h_players, prior_emb, context_ball):
        B, T, A_total, F_in = x.shape
        device = x.device

        players_x = x[:, :, :self.n_players, :]
        ball_x = x[:, :, self.n_players:self.n_players+1, :]

        if h_players is None:
            h_players = torch.zeros(B, self.n_players, self.context_dim, device=device)
        if prior_emb is None:
            prior_emb = torch.zeros(B, self.n_players, self.context_dim, device=device)
        if context_ball is None:
            context_ball = torch.zeros(B, 1, self.context_dim, device=device)

        h_players_exp = h_players.unsqueeze(1).expand(B, T, -1, -1)
        prior_emb_exp = prior_emb.unsqueeze(1).expand(B, T, -1, -1)
        players_combined = torch.cat([players_x, h_players_exp, prior_emb_exp], dim=-1)
        players_flat = players_combined.reshape(B * T * self.n_players, -1)
        players_emb = self.player_mlp(players_flat)
        players_emb = players_emb.reshape(B, T * self.n_players, self.h_dim)

        context_ball_exp = context_ball.unsqueeze(1).expand(B, T, -1, -1)
        ball_combined = torch.cat([ball_x, context_ball_exp], dim=-1)
        ball_flat = ball_combined.reshape(B * T, -1)
        ball_base = self.ball_embedding.unsqueeze(0).expand(B * T, -1)
        ball_emb = self.ball_mlp(ball_flat) + ball_base
        ball_emb = ball_emb.reshape(B, T, self.h_dim)

        if self.use_event_as_cls:
            h_players_mean = h_players.mean(dim=1, keepdim=True)
            prior_emb_mean = prior_emb.mean(dim=1, keepdim=True)
            event_context = torch.cat([
                h_players_mean, prior_emb_mean, context_ball
            ], dim=-1).squeeze(1)
            event_emb = self.event_mlp(event_context)
            event_emb = event_emb.unsqueeze(1).expand(B, T, -1)
            event_emb = event_emb + self.event_token

        if self.use_event_as_cls:
            all_tokens = torch.cat([players_emb, ball_emb, event_emb], dim=1)
        else:
            all_tokens = torch.cat([players_emb, ball_emb], dim=1)

        all_tokens = all_tokens * math.sqrt(self.h_dim)
        all_tokens = self.pos_encoder(all_tokens)

        return all_tokens, T

    def forward(self, x, h0=None, m0=None,
                h_players=None, prior_emb=None, context_ball=None):
        B, T, A_total, F_in = x.shape
        device = x.device

        all_tokens, T = self._process_sequence(x, h_players, prior_emb, context_ball)

        mask = self.generate_baller2vec_mask(T, device)

        h_out = self.transformer(all_tokens, mask=mask)

        players_out = h_out[:, :T*self.n_players].reshape(B, T, self.n_players, self.h_dim)
        ball_out = h_out[:, T*self.n_players:T*self.n_players+T].reshape(B, T, 1, self.h_dim)

        h_seq = torch.cat([players_out, ball_out], dim=2)
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
        B2, K, A2, Dh = h_prev_window.shape
        assert B == B2, "Batch size mismatch between x_t and h_prev_window"
        assert A_total == A2 == self.n_players + 1, \
            "A_total must equal n_players + 1 (players plus ball)"
        assert Dh == self.h_dim, "Hidden dimension mismatch"

        device = x_t.device
        seq_len = K + 1

        past_players = h_prev_window[:, :, :self.n_players, :]
        past_ball = h_prev_window[:, :, self.n_players:, :]

        players_x = x_t[:, :self.n_players, :]
        ball_x = x_t[:, self.n_players:self.n_players+1, :]

        if h_players is None:
            h_players = torch.zeros(B, self.n_players, self.context_dim, device=device)
        if prior_emb is None:
            prior_emb = torch.zeros(B, self.n_players, self.context_dim, device=device)
        if context_ball is None:
            context_ball = torch.zeros(B, 1, self.context_dim, device=device)

        players_in = torch.cat([players_x, h_players, prior_emb], dim=-1)
        players_flat = players_in.reshape(B * self.n_players, -1)
        players_new = self.player_mlp(players_flat)
        players_new = players_new.reshape(B, 1, self.n_players, self.h_dim)

        ball_in = torch.cat([ball_x, context_ball], dim=-1)
        ball_flat = ball_in.reshape(B, -1)
        ball_base = self.ball_embedding.unsqueeze(0).expand(B, -1)
        ball_new = (self.ball_mlp(ball_flat) + ball_base).reshape(B, 1, 1, self.h_dim)

        if self.use_event_as_cls:
            h_players_mean = h_players.mean(dim=1, keepdim=True)
            prior_emb_mean = prior_emb.mean(dim=1, keepdim=True)
            event_ctx = torch.cat([h_players_mean, prior_emb_mean, context_ball], dim=-1).squeeze(1)
            event_new = self.event_mlp(event_ctx) + self.event_token
            event_past = event_new.unsqueeze(1).unsqueeze(2).expand(B, K, 1, self.h_dim)
            event_seq = torch.cat(
                [event_past, event_new.unsqueeze(1).unsqueeze(2)], dim=1
            )
        else:
            event_seq = None

        players_time = torch.cat([past_players, players_new], dim=1)
        ball_time = torch.cat([past_ball, ball_new], dim=1)

        players_tokens = players_time.reshape(B, seq_len * self.n_players, self.h_dim)
        ball_tokens = ball_time.reshape(B, seq_len, self.h_dim)

        if self.use_event_as_cls:
            event_tokens = event_seq.reshape(B, seq_len, self.h_dim)
            all_tokens = torch.cat([players_tokens, ball_tokens, event_tokens], dim=1)
        else:
            all_tokens = torch.cat([players_tokens, ball_tokens], dim=1)

        mask = self.generate_baller2vec_mask(seq_len, device)

        h_all = self.step_transformer_layer(all_tokens, src_mask=mask)
        h_all = self.step_norm(h_all)

        players_all = h_all[:, :seq_len * self.n_players].reshape(
            B, seq_len, self.n_players, self.h_dim
        )

        ball_all = h_all[:, seq_len * self.n_players: seq_len * self.n_players + seq_len].reshape(
            B, seq_len, 1, self.h_dim
        )

        h_players_t = players_all[:, -1]
        h_ball_t = ball_all[:, -1]
        h_t = torch.cat([h_players_t, h_ball_t], dim=1)

        if self.mem_proj is not None:
            m_t = self.mem_proj(h_t)
        else:
            m_t = h_t

        if K > 0:
            new_window = torch.cat([h_prev_window[:, 1:], h_t.unsqueeze(1)], dim=1)
        else:
            new_window = h_t.unsqueeze(1)

        return {
            "h_t": h_t,
            "m_t": m_t,
            "mem_window": new_window
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
                volterra_features=None
                ):
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

class Baller2VecWindowedNoContext(nn.Module):

    def __init__(self,
                 in_dim,
                 h_dim,
                 window_K=8,
                 d_mem=None,
                 n_heads=8,
                 n_layers=4,
                 dropout=0.1,
                 n_players=22):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.d_mem = h_dim if d_mem is None else d_mem
        self.n_players = n_players
        self.window_K = window_K

        initrange = 0.1
        self.ball_embedding = nn.Parameter(torch.Tensor(h_dim))
        nn.init.uniform_(self.ball_embedding, -initrange, initrange)

        self.player_mlp = nn.Sequential(
            nn.Linear(in_dim, h_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim * 2, h_dim),
            nn.LayerNorm(h_dim),
        )

        self.ball_mlp = nn.Sequential(
            nn.Linear(in_dim, h_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim * 2, h_dim),
            nn.LayerNorm(h_dim),
        )

        self.pos_encoder = PositionalEncoding(h_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h_dim,
            nhead=n_heads,
            dim_feedforward=h_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.step_transformer_layer = nn.TransformerEncoderLayer(
            d_model=h_dim,
            nhead=n_heads,
            dim_feedforward=h_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.step_norm = nn.LayerNorm(h_dim)

        if self.d_mem != h_dim:
            self.mem_proj = nn.Linear(h_dim, self.d_mem)
        else:
            self.mem_proj = None

    def generate_baller2vec_mask(self, seq_len, device):
        n_entities = self.n_players + 1
        total_tokens = n_entities * seq_len
        mask = torch.zeros(total_tokens, total_tokens, device=device)

        ball_start = self.n_players * seq_len

        for t in range(seq_len):
            players_start = self.n_players * t
            players_end = players_start + self.n_players
            ball_idx = ball_start + t

            for i in range(players_start, players_end):
                mask[i, :players_end] = 1
                mask[i, ball_start:ball_idx+1] = 1

            mask[ball_idx, :players_end] = 1
            mask[ball_idx, ball_start:ball_idx+1] = 1

        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, h0=None, m0=None,
                h_players=None, prior_emb=None, context_ball=None):
        B, T, A_total, F_in = x.shape
        device = x.device

        players_x = x[:, :, :self.n_players, :]
        ball_x = x[:, :, self.n_players:, :]

        players_flat = players_x.reshape(B * T * self.n_players, -1)
        players_emb = self.player_mlp(players_flat)
        players_emb = players_emb.reshape(B, T * self.n_players, self.h_dim)

        ball_flat = ball_x.reshape(B * T, -1)
        ball_base = self.ball_embedding.unsqueeze(0).expand(B * T, -1)
        ball_emb = self.ball_mlp(ball_flat) + ball_base
        ball_emb = ball_emb.reshape(B, T, self.h_dim)

        all_tokens = torch.cat([players_emb, ball_emb], dim=1)

        all_tokens = all_tokens * math.sqrt(self.h_dim)
        all_tokens = self.pos_encoder(all_tokens)

        mask = self.generate_baller2vec_mask(T, device)
        h_out = self.transformer(all_tokens, mask=mask)

        players_out = h_out[:, :T*self.n_players].reshape(B, T, self.n_players, self.h_dim)
        ball_out = h_out[:, T*self.n_players:].reshape(B, T, 1, self.h_dim)

        h_seq = torch.cat([players_out, ball_out], dim=2)
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
        K = h_prev_window.size(1)
        device = x_t.device
        seq_len = K + 1

        past_players = h_prev_window[:, :, :self.n_players, :]
        past_ball = h_prev_window[:, :, self.n_players:, :]

        players_x = x_t[:, :self.n_players, :]
        ball_x = x_t[:, self.n_players:, :]

        players_flat = players_x.reshape(B * self.n_players, -1)
        players_new = self.player_mlp(players_flat)
        players_new = players_new.reshape(B, 1, self.n_players, self.h_dim)

        ball_flat = ball_x.reshape(B, -1)
        ball_base = self.ball_embedding.unsqueeze(0).expand(B, -1)
        ball_new = (self.ball_mlp(ball_flat) + ball_base).reshape(B, 1, 1, self.h_dim)

        players_time = torch.cat([past_players, players_new], dim=1)
        ball_time = torch.cat([past_ball, ball_new], dim=1)

        players_tokens = players_time.reshape(B, seq_len * self.n_players, self.h_dim)
        ball_tokens = ball_time.reshape(B, seq_len, self.h_dim)
        all_tokens = torch.cat([players_tokens, ball_tokens], dim=1)

        mask = self.generate_baller2vec_mask(seq_len, device)
        h_all = self.step_transformer_layer(all_tokens, src_mask=mask)
        h_all = self.step_norm(h_all)

        players_all = h_all[:, :seq_len * self.n_players].reshape(B, seq_len, self.n_players, self.h_dim)
        ball_all = h_all[:, seq_len * self.n_players:].reshape(B, seq_len, 1, self.h_dim)

        h_players_t = players_all[:, -1]
        h_ball_t = ball_all[:, -1]
        h_t = torch.cat([h_players_t, h_ball_t], dim=1)

        if self.mem_proj is not None:
            m_t = self.mem_proj(h_t)
        else:
            m_t = h_t

        if K > 0:
            new_window = torch.cat([h_prev_window[:, 1:], h_t.unsqueeze(1)], dim=1)
        else:
            new_window = h_t.unsqueeze(1)

        return {
            "h_t": h_t,
            "m_t": m_t,
            "mem_window": new_window,
        }

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

"""Implement the Volterra-fusion event-prediction baseline."""

import torch
import torch.nn as nn
import numpy as np


class VolterraFusionBlock(nn.Module):
    def __init__(self, in_dim_flow, in_dim_rgb, out_dim, Q=2, dropout=0.1):
        super().__init__()
        self.Q = Q
        self.out_dim = out_dim

        combined_dim = in_dim_flow + in_dim_rgb

        self.linear_proj = nn.Sequential(
            nn.Linear(combined_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        self.quad_proj = nn.Linear(combined_dim, 2 * Q * out_dim)
        self.quad_norm = nn.LayerNorm(out_dim)

        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.dropout = nn.Dropout(dropout)

    def forward(self, flow_feat, rgb_feat):
        combined = torch.cat([flow_feat, rgb_feat], dim=-1)

        x_linear = self.linear_proj(combined)

        x_quad_raw = self.quad_proj(combined)

        half = self.Q * self.out_dim
        x_quad_a = x_quad_raw[..., :half]
        x_quad_b = x_quad_raw[..., half:]

        x_quad_mul = x_quad_a * x_quad_b

        B, N, _ = x_quad_mul.shape
        x_quad_mul = x_quad_mul.view(B, N, self.Q, self.out_dim)
        x_quadratic = x_quad_mul.sum(dim=2)
        x_quadratic = self.quad_norm(x_quadratic)

        alpha = torch.sigmoid(self.alpha)
        fused = alpha * x_linear + (1 - alpha) * x_quadratic

        return self.dropout(fused)


class MultiScaleVolterraFusion(nn.Module):
    def __init__(self, in_dim_flow, in_dim_rgb, hidden_dim, out_dim,
                 Q=2, n_scales=2, dropout=0.1):
        super().__init__()
        self.n_scales = n_scales

        self.volterra_blocks = nn.ModuleList()

        current_flow_dim = in_dim_flow
        current_rgb_dim = in_dim_rgb

        for i in range(n_scales):
            if i == n_scales - 1:
                block_out = out_dim
            else:
                block_out = hidden_dim

            block = VolterraFusionBlock(
                in_dim_flow=current_flow_dim,
                in_dim_rgb=current_rgb_dim,
                out_dim=block_out,
                Q=Q,
                dropout=dropout
            )
            self.volterra_blocks.append(block)

            current_flow_dim = block_out
            current_rgb_dim = block_out

        self.final_norm = nn.LayerNorm(out_dim)

    def forward(self, flow_feat, rgb_feat):
        f = flow_feat
        r = rgb_feat

        for block in self.volterra_blocks:
            out = block(f, r)
            f = out
            r = out

        return self.final_norm(out)


class VolterraFusionEventHead(nn.Module):
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
        self.hidden_dim = hidden_dim

        self.volterra_fusion = MultiScaleVolterraFusion(
            in_dim_flow=hidden_dim,
            in_dim_rgb=3 * hidden_dim,
            hidden_dim=hidden_dim * 2,
            out_dim=hidden_dim,
            Q=volterra_Q,
            n_scales=volterra_scales,
            dropout=dropout
        )

        self.player_in = nn.Linear(4 * hidden_dim, d_model)
        self.ball_in = nn.Linear(2 * hidden_dim, d_model)

        self.player_index_emb = nn.Embedding(n_players, d_model)

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

        if context_ball is None:
            attn_score = torch.sum(h_players * h_ball.unsqueeze(1), dim=-1, keepdim=True)
            attn_weight = torch.softmax(attn_score, dim=1)
            context_ball = (attn_weight * h_players).sum(dim=1, keepdim=True)

        if context_ball.size(1) == 1:
            context_ball = context_ball.expand(-1, A, -1)

        attn_score = torch.sum(h_players * h_ball.unsqueeze(1), dim=-1, keepdim=True)
        attn_weight = torch.softmax(attn_score, dim=1)
        context_ball = (attn_weight * h_players).sum(dim=1, keepdim=True)
        context_ball = context_ball.expand(-1, A, -1)

        flow_feat = fut_emb

        rgb_feat = torch.cat([h_players, prior_emb, context_ball], dim=-1)

        volterra_out = self.volterra_fusion(flow_feat, rgb_feat)

        player_features = torch.cat([
            volterra_out,
            h_players,
            prior_emb,
            fut_emb,
        ], dim=-1)

        x_player = self.player_in(player_features)

        gpool_players = h_players.mean(dim=1)
        ball_token = self.ball_in(torch.cat([h_ball, gpool_players], dim=-1))
        ball_token = ball_token.unsqueeze(1)

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


class ParallelEventPredictor_VolterraFusion(nn.Module):
    def __init__(
        self,
        context_encoder,
        prior_scorer,
        hidden_dim: int,
        n_players: int = 22,
        fut_T: int = 64,
        n_event_types: int = 5,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        volterra_Q: int = 2,
        volterra_scales: int = 2,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.prior_scorer = prior_scorer
        self.n_players = n_players
        self.hidden_dim = hidden_dim

        self.proj_fut = nn.Linear(2, hidden_dim)

        self.head = VolterraFusionEventHead(
            hidden_dim=hidden_dim,
            n_event_types=n_event_types,
            fut_T=fut_T,
            n_players=n_players,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            volterra_Q=volterra_Q,
            volterra_scales=volterra_scales,
        )

        self.volterra_module = None

    def forward(self, ctx_dyns, ctx_edges, ctx_seqX, ctx_labels_22, cols,
                edge_cols=None, fut_pred=None):
        if isinstance(ctx_labels_22, (list, tuple, np.ndarray)):
            arr = np.array(ctx_labels_22, dtype=object)
            if arr.ndim == 2:
                enc_labels_22 = [ctx_labels_22]
            else:
                enc_labels_22 = ctx_labels_22
        else:
            enc_labels_22 = ctx_labels_22

        h_players, h_ball, team_labels = self.context_encoder(
            ctx_dyns, ctx_edges, enc_labels_22,
            cols=cols, edge_cols=edge_cols,
            return_player_states=True,
            return_team_labels=True,
        )

        labels_last = []
        for b in range(len(enc_labels_22)):
            row = enc_labels_22[b][-1]
            if hasattr(row, "tolist"):
                labels_last.append([str(x) for x in row.tolist()])
            elif isinstance(row, (list, tuple)):
                labels_last.append([str(x) for x in row])
            else:
                labels_last.append([str(row)])

        if self.prior_scorer is not None:
            prior_emb = self.prior_scorer(ctx_dyns[:, -1, :], cols, labels_last)
        else:
            prior_emb = torch.zeros_like(h_players)

        if fut_pred is not None:
            fut_mean = fut_pred.mean(dim=0)
            fut_emb = self.proj_fut(fut_mean)
        else:
            fut_emb = torch.zeros_like(h_players)

        return self.head(h_players, h_ball, prior_emb, fut_emb)



class LightweightVolterraFusion(nn.Module):
    def __init__(self, in_dim_flow, in_dim_rgb, out_dim, Q=2, dropout=0.1):
        super().__init__()
        self.Q = Q
        self.out_dim = out_dim

        combined_dim = in_dim_flow + in_dim_rgb

        self.linear_proj = nn.Linear(combined_dim, out_dim)

        self.quad_proj = nn.Linear(combined_dim, 2 * Q * out_dim)

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, flow_feat, rgb_feat):
        combined = torch.cat([flow_feat, rgb_feat], dim=-1)

        x_linear = self.linear_proj(combined)

        x_quad_raw = self.quad_proj(combined)
        half = self.Q * self.out_dim
        x_quad_mul = x_quad_raw[..., :half] * x_quad_raw[..., half:]

        B, N, _ = x_quad_mul.shape
        x_quad_mul = x_quad_mul.view(B, N, self.Q, self.out_dim)
        x_quadratic = x_quad_mul.sum(dim=2)

        out = self.norm(x_linear + x_quadratic)
        return self.dropout(out)


class LightweightVolterraEventHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_event_types: int = 5,
        fut_T: int = 64,
        n_players: int = 22,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        volterra_Q: int = 2,
    ):
        super().__init__()
        self.n_players = n_players
        self.n_event_types = n_event_types
        self.fut_T = fut_T
        self.d_model = d_model

        self.volterra_fusion = LightweightVolterraFusion(
            in_dim_flow=hidden_dim,
            in_dim_rgb=3 * hidden_dim,
            out_dim=hidden_dim,
            Q=volterra_Q,
            dropout=dropout
        )

        self.player_in = nn.Linear(2 * hidden_dim, d_model)
        self.ball_in = nn.Linear(hidden_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head_tgt = nn.Linear(d_model, 1)
        self.head_event = nn.Linear(d_model, n_event_types)
        self.head_time = nn.Linear(d_model, fut_T)
        self.head_pos = nn.Linear(d_model, 2)

    def forward(self, h_players, h_ball, prior_emb, fut_emb):
        B, A, H = h_players.shape

        attn_score = torch.sum(h_players * h_ball.unsqueeze(1), dim=-1, keepdim=True)
        attn_weight = torch.softmax(attn_score, dim=1)
        context_ball = (attn_weight * h_players).sum(dim=1, keepdim=True).expand(-1, A, -1)

        rgb_feat = torch.cat([h_players, prior_emb, context_ball], dim=-1)
        volterra_out = self.volterra_fusion(fut_emb, rgb_feat)

        x_player = self.player_in(torch.cat([volterra_out, fut_emb], dim=-1))
        ball_token = self.ball_in(h_ball).unsqueeze(1)

        tokens = torch.cat([ball_token, x_player], dim=1)
        z = self.encoder(tokens)
        z_player = z[:, 1:, :]

        tgt_logits = self.head_tgt(z_player).squeeze(-1)
        tgt_probs = torch.softmax(tgt_logits, dim=-1)

        return (
            tgt_probs,
            self.head_event(z_player),
            self.head_time(z_player),
            self.head_pos(z_player),
            tgt_logits
        )



def create_volterra_event_predictor(
    context_encoder,
    prior_scorer,
    hidden_dim: int,
    mode: str = "full",
    **kwargs
):
    if mode == "lightweight":
        class LightweightParallelPredictor(nn.Module):
            def __init__(self, ctx_enc, prior, h_dim, **kw):
                super().__init__()
                self.context_encoder = ctx_enc
                self.prior_scorer = prior
                self.proj_fut = nn.Linear(2, h_dim)
                self.head = LightweightVolterraEventHead(hidden_dim=h_dim, **kw)
                self.volterra_module = None

            def forward(self, ctx_dyns, ctx_edges, ctx_seqX, ctx_labels_22, cols,
                        edge_cols=None, fut_pred=None):
                if isinstance(ctx_labels_22, (list, tuple, np.ndarray)):
                    arr = np.array(ctx_labels_22, dtype=object)
                    enc_labels_22 = ctx_labels_22 if arr.ndim != 2 else [ctx_labels_22]
                else:
                    enc_labels_22 = ctx_labels_22

                h_players, h_ball, _ = self.context_encoder(
                    ctx_dyns, ctx_edges, enc_labels_22,
                    cols=cols, edge_cols=edge_cols,
                    return_player_states=True, return_team_labels=True,
                )

                labels_last = []
                for b in range(len(enc_labels_22)):
                    row = enc_labels_22[b][-1]
                    if hasattr(row, "tolist"):
                        labels_last.append([str(x) for x in row.tolist()])
                    elif isinstance(row, (list, tuple)):
                        labels_last.append([str(x) for x in row])
                    else:
                        labels_last.append([str(row)])

                prior_emb = self.prior_scorer(ctx_dyns[:, -1, :], cols, labels_last) \
                            if self.prior_scorer else torch.zeros_like(h_players)

                fut_emb = self.proj_fut(fut_pred.mean(dim=0)) if fut_pred is not None \
                          else torch.zeros_like(h_players)

                return self.head(h_players, h_ball, prior_emb, fut_emb)

        return LightweightParallelPredictor(
            context_encoder, prior_scorer, hidden_dim, **kwargs
        )
    else:
        return ParallelEventPredictor_VolterraFusion(
            context_encoder=context_encoder,
            prior_scorer=prior_scorer,
            hidden_dim=hidden_dim,
            **kwargs
        )

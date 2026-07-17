"""Implement the main event-guided disentangled event predictor."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class DisentangledTrajProjection(nn.Module):
    def __init__(self,
                 in_dim: int = 2,
                 hidden_dim: int = 128,
                 out_dim: int = 128,
                 n_players: int = 22,
                 dropout: float = 0.1,
                 ):
        super().__init__()
        self.out_dim = out_dim
        self.n_players = n_players

        self.shared_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.intent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        self.dynamic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        self.temporal_weight = nn.Parameter(torch.ones(1))

        self.intent_norm = nn.LayerNorm(out_dim)
        self.dynamic_norm = nn.LayerNorm(out_dim)

    def forward(self, fut_traj: torch.Tensor, return_sequence: bool = False):
        T, B, A, F = fut_traj.shape
        device = fut_traj.device

        x = fut_traj.permute(1, 2, 0, 3).reshape(B * A, T, F)

        shared = self.shared_encoder(x)

        intent_seq = self.intent_head(shared)
        dynamic_seq = self.dynamic_head(shared)

        gamma = torch.sigmoid(self.temporal_weight) * 0.5 + 0.5

        intent_pooled = intent_seq.mean(dim=1)

        decay_weights = gamma ** torch.arange(T-1, -1, -1, device=device, dtype=torch.float)
        decay_weights = decay_weights / decay_weights.sum()
        decay_weights = decay_weights.view(1, T, 1)
        dynamic_pooled = (dynamic_seq * decay_weights).sum(dim=1)

        traj_intent = self.intent_norm(intent_pooled.view(B, A, -1))
        traj_dynamic = self.dynamic_norm(dynamic_pooled.view(B, A, -1))

        if return_sequence:
            intent_seq = intent_seq.view(B, A, T, -1).permute(0, 2, 1, 3)
            dynamic_seq = dynamic_seq.view(B, A, T, -1).permute(0, 2, 1, 3)
            return traj_intent, traj_dynamic, intent_seq, dynamic_seq

        return traj_intent, traj_dynamic


class DisentanglementLoss(nn.Module):
    def __init__(self,
                 ortho_weight: float = 1.0,
                 align_weight: float = 0.5,
                 align_type: str = 'cosine'):

        super().__init__()
        self.ortho_weight = ortho_weight
        self.align_weight = align_weight
        self.align_type = align_type
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self,
                traj_intent: torch.Tensor,
                traj_dynamic: torch.Tensor,
                event_emb: torch.Tensor):
        B, A, D = traj_intent.shape

        intent_norm = F.normalize(traj_intent, dim=-1)
        dynamic_norm = F.normalize(traj_dynamic, dim=-1)

        cos_sim = (intent_norm * dynamic_norm).sum(dim=-1)
        ortho_loss = cos_sim.pow(2).mean()

        if event_emb.dim() == 3:
            event_target = event_emb[:, -1, :]
        else:
            event_target = event_emb

        intent_pooled = traj_intent.mean(dim=1)

        if self.align_type == 'cosine':
            intent_pooled_norm = F.normalize(intent_pooled, dim=-1)
            event_target_norm = F.normalize(event_target, dim=-1)
            align_loss = 1.0 - (intent_pooled_norm * event_target_norm).sum(dim=-1).mean()

        else:
            intent_pooled_norm = F.normalize(intent_pooled, dim=-1)
            event_target_norm = F.normalize(event_target, dim=-1)

            pos_sim = (intent_pooled_norm * event_target_norm).sum(dim=-1) / self.temperature

            neg_sim = torch.mm(intent_pooled_norm, event_target_norm.t()) / self.temperature

            labels = torch.arange(B, device=traj_intent.device)
            align_loss = F.cross_entropy(neg_sim, labels)

        total_loss = self.ortho_weight * ortho_loss + self.align_weight * align_loss

        return {
            'disentangle_loss': total_loss,
            'ortho_loss': ortho_loss,
            'align_loss': align_loss,
        }



class MacroIntentEncoder(nn.Module):
    def __init__(self,
                 hidden_dim: int = 128,
                 n_players: int = 22,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_players = n_players

        self.intent_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.prior_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.event_seq_proj = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim)
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                traj_intent: torch.Tensor,
                event_emb: torch.Tensor,
                prior_emb: torch.Tensor,
                ball_all: torch.Tensor):
        B, A, D = traj_intent.shape

        raw_context = torch.cat([event_emb, ball_all], dim=-1)

        intent = self.intent_proj(traj_intent)
        prior = self.prior_proj(prior_emb)
        event_seq = self.event_seq_proj(raw_context)

        intent_attn, _ = self.cross_attn(
            query=intent,
            key=event_seq,
            value=event_seq,
        )
        intent = self.cross_norm(intent + intent_attn)

        intent = self.self_attn(intent)

        fused = self.fusion(torch.cat([intent, prior], dim=-1))

        intent_features = self.output_norm(fused + traj_intent)

        return intent_features



class MicroDynamicEncoder(nn.Module):
    def __init__(self,
                 hidden_dim: int = 128,
                 n_players: int = 22,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_players = n_players

        self.dynamic_proj = nn.Linear(hidden_dim, hidden_dim)
        self.player_proj = nn.Linear(hidden_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                traj_dynamic: torch.Tensor,
                h_players: torch.Tensor):
        B, A, D = traj_dynamic.shape

        dynamic = self.dynamic_proj(traj_dynamic)
        players = self.player_proj(h_players)

        dynamic_attn, _ = self.cross_attn(
            query=dynamic,
            key=players,
            value=players,
        )
        dynamic = self.cross_norm(dynamic + dynamic_attn)

        dynamic = self.self_attn(dynamic)

        fused = self.fusion(torch.cat([dynamic, h_players], dim=-1))

        dynamic_features = self.output_norm(fused + traj_dynamic)

        return dynamic_features


class HierarchicalFusionEventHead(nn.Module):
    def __init__(self,
                 hidden_dim: int = 128,
                 n_players: int = 22,
                 n_event_types: int = 5,
                 fut_T: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 d_model: int = None,
                 dropout: float = 0.1,
                 dynamic_weight: float = 0.5,
                 **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_players = n_players
        self.n_event_types = n_event_types
        self.fut_T = fut_T
        self.d_model = d_model or hidden_dim * 2

        self.dynamic_weight = dynamic_weight

        self.player_in = nn.Linear(2 * hidden_dim, self.d_model)

        self.ball_in = nn.Linear(2 * hidden_dim, self.d_model)

        self.player_index_emb = nn.Embedding(n_players, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head_tgt = nn.Linear(self.d_model, 1)
        self.head_event = nn.Linear(self.d_model, n_event_types)
        self.head_time = nn.Linear(self.d_model, fut_T)
        self.head_pos = nn.Linear(self.d_model, 2)

        self.prior_bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self,
                h_players: torch.Tensor,
                h_ball: torch.Tensor,
                context_ball: torch.Tensor,
                fut_emb,
                event_emb: torch.Tensor = None,
                prior_emb: torch.Tensor = None):
        B, A, H = h_players.shape

        if isinstance(fut_emb, dict):
            intent_features = fut_emb['intent']
            dynamic_features = fut_emb['dynamic']
        else:
            if fut_emb.dim() == 2:
                intent_features = fut_emb.unsqueeze(1).expand(B, A, H)
                dynamic_features = intent_features
            else:
                intent_features = fut_emb
                dynamic_features = fut_emb

        alpha = self.dynamic_weight
        intent_scale = math.sqrt(1.0 - alpha)
        dynamic_scale = math.sqrt(alpha)

        scaled_intent = intent_scale * intent_features
        scaled_dynamic = dynamic_scale * dynamic_features

        player_feat = torch.cat([scaled_intent, scaled_dynamic], dim=-1)
        x_player = self.player_in(player_feat)

        ball_token = self.ball_in(torch.cat([h_ball, context_ball.squeeze(1)], dim=-1)).unsqueeze(1)

        idx = torch.arange(A, device=h_players.device).unsqueeze(0).expand(B, A)
        x_player = x_player + self.player_index_emb(idx)

        tokens = torch.cat([x_player, ball_token], dim=1)
        z = self.encoder(tokens)

        z_ball = z[:, -1:, :]
        z_player = z[:, :-1, :]

        tgt_logits_main = self.head_tgt(z_player).squeeze(-1)
        prior_bias = (z_player * z_ball.expand_as(z_player)).sum(dim=-1)
        tgt_logits = tgt_logits_main + self.prior_bias_scale * prior_bias
        tgt_probs = F.softmax(tgt_logits, dim=-1)
        time_logits = self.head_time(z_player)

        event_logits = self.head_event(z_player)
        pos_pred = self.head_pos(z_player)

        return tgt_probs, event_logits, time_logits, pos_pred, tgt_logits



class DisentangledEventPredictor(nn.Module):
    def __init__(self,
                 context_encoder,
                 prior_scorer,
                 hidden_dim: int = 128,
                 n_players: int = 22,
                 fut_T: int = 64,
                 n_event_types: int = 5,
                 dropout: float = 0.1,
                 ortho_weight: float = 1.0,
                 align_weight: float = 1.0,
                 align_type: str = 'cosine',
                 n_heads: int = 4,
                 n_layers: int = 2,
                 **kwargs):
        super().__init__()

        self.context_encoder = context_encoder
        self.prior_scorer = prior_scorer
        self.hidden_dim = hidden_dim
        self.n_players = n_players

        self.traj_projection = DisentangledTrajProjection(
            in_dim=2,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_players=n_players,
            dropout=dropout,
        )

        self.disentangle_loss_fn = DisentanglementLoss(
            ortho_weight=ortho_weight,
            align_weight=align_weight,
            align_type=align_type,
        )

        self.macro_encoder = MacroIntentEncoder(
            hidden_dim=hidden_dim,
            n_players=n_players,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.micro_encoder = MicroDynamicEncoder(
            hidden_dim=hidden_dim,
            n_players=n_players,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.event_head = HierarchicalFusionEventHead(
            hidden_dim=hidden_dim,
            n_players=n_players,
            n_event_types=n_event_types,
            fut_T=fut_T,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        self._disentangle_loss_cache = None

    def forward(self,
                ctx_dyns, ctx_edges, ctx_seqX, ctx_labels_22,
                cols, edge_cols=None, fut_pred=None, h_players=None,
                h_ball=None, team_labels=None, event_emb=None,context_ball=None, ball_all=None):
        import numpy as np

        if isinstance(ctx_labels_22, (list, tuple, np.ndarray)):
            arr = np.array(ctx_labels_22, dtype=object)
            enc_labels_22 = [ctx_labels_22] if arr.ndim == 2 else ctx_labels_22
        else:
            enc_labels_22 = ctx_labels_22

        if h_players is None or h_ball is None or event_emb is None:
            if isinstance(ctx_labels_22, (list, tuple, np.ndarray)):
                arr = np.array(ctx_labels_22, dtype=object)
                enc_labels_22 = [ctx_labels_22] if arr.ndim == 2 else ctx_labels_22
            else:
                enc_labels_22 = ctx_labels_22

            h_players, h_ball, team_labels, players_all, ball_all, event_emb = self.context_encoder(
                ctx_dyns, ctx_edges, enc_labels_22,
                cols=cols, edge_cols=edge_cols,
                return_team_labels=True,
                return_decay=True,
            )
        if context_ball is None:
            attn_score = torch.sum(h_players * h_ball.unsqueeze(1), dim=-1, keepdim=True)
            attn_weight = torch.softmax(attn_score, dim=1)
            context_ball = torch.sum(h_players * attn_weight, dim=1, keepdim=True)

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
            fut_pred_players = fut_pred[:, :, :self.n_players, :]
            traj_intent, traj_dynamic = self.traj_projection(fut_pred_players)

            if self.training:
                self._disentangle_loss_cache = self.disentangle_loss_fn(
                    traj_intent, traj_dynamic, event_emb
                )
        else:
            B = h_players.shape[0]
            traj_intent = torch.zeros(B, self.n_players, self.hidden_dim,
                                     device=h_players.device)
            traj_dynamic = torch.zeros_like(traj_intent)
            self._disentangle_loss_cache = None

        if ball_all is None:
            C = event_emb.size(1) if event_emb.dim() == 3 else 10
            ball_all = torch.zeros(h_players.size(0), C, self.hidden_dim, device=h_players.device)

        intent_features = self.macro_encoder(traj_intent, event_emb, prior_emb, ball_all)

        dynamic_features = self.micro_encoder(traj_dynamic, h_players)
        fut_emb = {
            'intent': intent_features,
            'dynamic': dynamic_features
        }
        return self.event_head(
            h_players=h_players,
            h_ball=h_ball,
            fut_emb=fut_emb,
            event_emb=event_emb,
            context_ball=context_ball,
        )

    def get_disentangle_loss(self):
        return self._disentangle_loss_cache

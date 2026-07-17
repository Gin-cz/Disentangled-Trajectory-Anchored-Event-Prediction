"""Implement nonlinear, patch-wise HSIC/GRL, and gated-fusion ablations."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def hsic_loss(x, y):
    B = x.size(0)
    if B < 2: return torch.tensor(0.0, device=x.device)
    def rbf_kernel(mat):
        dists = torch.cdist(mat, mat).pow(2)
        sigma = torch.median(dists[dists > 0]) if dists.max() > 0 else 1.0
        return torch.exp(-dists / (2 * sigma + 1e-7))

    K = rbf_kernel(x)
    L = rbf_kernel(y)
    H = torch.eye(B, device=x.device) - (1.0 / B)
    return torch.trace(K @ H @ L @ H) / ((B - 1) ** 2)


class DisentangledTrajProjection(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, out_dim=128, dropout=0.1):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.intent_linear = nn.Linear(hidden_dim, out_dim)

        self.intent_nonlinear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, fut_traj):
        T, B, A, F = fut_traj.shape
        x = fut_traj.permute(1, 2, 0, 3).reshape(B * A, T, F)
        shared = self.shared_encoder(x)

        z_lin = self.intent_linear(shared).mean(dim=1).view(B, A, -1)
        z_nonlin = self.intent_nonlinear(shared).mean(dim=1).view(B, A, -1)
        traj_residual = self.residual_head(shared).mean(dim=1).view(B, A, -1)

        return z_lin, z_nonlin, traj_residual


class ContrastiveDiscriminator(nn.Module):
    def __init__(self, res_dim=128, event_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(res_dim + event_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x_res, e, alpha=1.0):
        x_res = GradientReversalLayer.apply(x_res, alpha)
        pair = torch.cat([x_res, e], dim=-1)
        return self.net(pair)


class DisentanglementLoss(nn.Module):
    def __init__(self, align_weight=0.5, adv_weight=0.1):
        super().__init__()
        self.align_weight = align_weight
        self.adv_weight = adv_weight
        self.discriminator = ContrastiveDiscriminator(128, 128)

    def forward(self, z_lin, z_nonlin, traj_residual, event_emb):
        B, A, D = z_lin.shape
        if event_emb.dim() == 3:
            event_target = event_emb[:, -1, :]
        else:
            event_target = event_emb

        loss_lin = 1.0 - F.cosine_similarity(z_lin.mean(1), event_target, dim=-1).mean()

        target_exp = event_target.unsqueeze(1).expand(-1, A, -1).reshape(-1, D)
        loss_hsic = -hsic_loss(z_nonlin.view(-1, D), target_exp)

        res_flat = traj_residual.view(-1, D)
        pos_logits = self.discriminator(res_flat, target_exp, alpha=1.0)
        idx = torch.randperm(B, device=z_lin.device)
        neg_target_exp = event_target[idx].unsqueeze(1).expand(-1, A, -1).reshape(-1, D)
        neg_logits = self.discriminator(res_flat, neg_target_exp, alpha=1.0)

        loss_adv = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits)) + \
                   F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))

        with torch.no_grad():
            acc = ((torch.sigmoid(pos_logits) > 0.5).float().mean() + \
                   (torch.sigmoid(neg_logits) < 0.5).float().mean()) / 2

        return {
            'disentangle_loss': loss_lin * self.align_weight + 0.1 * loss_hsic + self.adv_weight * loss_adv,
            'ortho_loss': loss_lin,
            'align_loss': loss_hsic,
            'adv_loss': loss_adv,
            'disc_acc': acc,
            'hsic_score': -loss_hsic,
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


class HierarchicalFusionEventHead_Nonlinear(nn.Module):
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


class DisentangledEventPredictor_Nonlinear(nn.Module):
    def __init__(self, context_encoder, prior_scorer, hidden_dim=128, n_players=22,
                 fut_T=64, n_event_types=5, dropout=0.1, **kwargs):
        super().__init__()

        self.context_encoder = context_encoder
        self.prior_scorer = prior_scorer
        self.hidden_dim = hidden_dim
        self.n_players = n_players

        self.traj_projection = DisentangledTrajProjection(hidden_dim=hidden_dim)

        self.disentangle_loss_fn = DisentanglementLoss(adv_weight=kwargs.get('adv_weight', 0.1))

        self.intent_projector = nn.Linear(2 * hidden_dim, hidden_dim)

        self.macro_encoder = MacroIntentEncoder(hidden_dim=hidden_dim, n_players=n_players, dropout=dropout)
        self.micro_encoder = MicroDynamicEncoder(hidden_dim=hidden_dim, n_players=n_players, dropout=dropout)

        self.event_head = HierarchicalFusionEventHead_Nonlinear(
            hidden_dim=hidden_dim, n_players=n_players, n_event_types=n_event_types,
            fut_T=fut_T, dropout=dropout, **kwargs
        )

        self._disentangle_loss_cache = None

    def forward(self, ctx_dyns, ctx_edges, ctx_seqX, ctx_labels_22, cols,
                edge_cols=None, fut_pred=None, h_players=None, h_ball=None,
                team_labels=None, event_emb=None, context_ball=None, ball_all=None):

        import numpy as np
        if isinstance(ctx_labels_22, (list, tuple, np.ndarray)):
            arr = np.array(ctx_labels_22, dtype=object)
            enc_labels_22 = [ctx_labels_22] if arr.ndim == 2 else ctx_labels_22
        else:
            enc_labels_22 = ctx_labels_22

        if h_players is None or h_ball is None or event_emb is None:
            h_players, h_ball, team_labels, players_all, ball_all, event_emb = self.context_encoder(
                ctx_dyns, ctx_edges, enc_labels_22, cols=cols, edge_cols=edge_cols,
                return_team_labels=True, return_decay=True,
            )

        if context_ball is None:
            attn_score = torch.sum(h_players * h_ball.unsqueeze(1), dim=-1, keepdim=True)
            attn_weight = torch.softmax(attn_score, dim=1)
            context_ball = torch.sum(h_players * attn_weight, dim=1, keepdim=True)

        if ball_all is None:
            ball_all = torch.zeros(h_players.size(0), event_emb.size(1), self.hidden_dim, device=h_players.device)

        if self.prior_scorer is not None:
            labels_last = [[str(x) for x in b[-1]] for b in enc_labels_22]
            prior_emb = self.prior_scorer(ctx_dyns[:, -1, :], cols, labels_last)
        else:
            prior_emb = torch.zeros_like(h_players)

        if fut_pred is not None:
            z_lin, z_nonlin, traj_residual = self.traj_projection(fut_pred[:, :, :self.n_players, :])

            traj_intent = self.intent_projector(torch.cat([z_lin, z_nonlin], dim=-1))

            if self.training:
                self._disentangle_loss_cache = self.disentangle_loss_fn(z_lin, z_nonlin, traj_residual, event_emb)
        else:
            traj_intent = torch.zeros(h_players.size(0), self.n_players, self.hidden_dim, device=h_players.device)
            self._disentangle_loss_cache = None

        intent_features = self.macro_encoder(traj_intent, event_emb, prior_emb, ball_all)

        dynamic_features = self.micro_encoder(h_players, h_players)

        fut_emb = {'intent': intent_features, 'dynamic': dynamic_features}

        return self.event_head(
            h_players=h_players, h_ball=h_ball, context_ball=context_ball,
            fut_emb=fut_emb, event_emb=event_emb, prior_emb=prior_emb
        )

    def get_disentangle_loss(self):
        return self._disentangle_loss_cache



class TrajPatchEmbed(nn.Module):

    def __init__(self, in_dim=2, embed_dim=128, patch_size=32, stride=8):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Conv1d(in_dim, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        _, time_steps, _ = x.shape
        if time_steps < self.patch_size:
            x = F.pad(x, (0, 0, 0, self.patch_size - time_steps))
        x = self.proj(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.norm(x)


class PatchDisentangledTrajProjection(nn.Module):

    def __init__(self, in_dim=2, hidden_dim=128, out_dim=128, dropout=0.1):
        super().__init__()
        self.patch_embed = TrajPatchEmbed(in_dim=in_dim, embed_dim=hidden_dim)
        self.shared_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.intent_linear = nn.Linear(hidden_dim, out_dim)
        self.intent_nonlinear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, fut_traj):
        time_steps, batch_size, n_agents, feature_dim = fut_traj.shape
        x = fut_traj.permute(1, 2, 0, 3).reshape(
            batch_size * n_agents, time_steps, feature_dim
        )
        shared = self.shared_encoder(self.patch_embed(x))
        n_patches = shared.size(1)

        def restore(tensor):
            return tensor.reshape(batch_size, n_agents, n_patches, -1)

        return (
            restore(self.intent_linear(shared)),
            restore(self.intent_nonlinear(shared)),
            restore(self.residual_head(shared)),
        )


class PatchDisentanglementLoss(nn.Module):

    def __init__(self, align_weight=0.5, adv_weight=0.1, decay_rate=0.5):
        super().__init__()
        self.align_weight = align_weight
        self.adv_weight = adv_weight
        self.decay_rate = decay_rate
        self.discriminator = ContrastiveDiscriminator(128, 128)

    def forward(self, z_lin, z_nonlin, traj_residual, event_emb, use_hsic=True):
        batch_size, n_agents, n_patches, hidden_dim = z_nonlin.shape
        event_target = event_emb[:, -1, :] if event_emb.dim() == 3 else event_emb
        target_by_agent = event_target.unsqueeze(1).expand(-1, n_agents, -1).reshape(
            batch_size * n_agents, hidden_dim
        )

        z_lin_mean = z_lin.mean(dim=2)
        loss_lin = 1.0 - F.cosine_similarity(
            z_lin_mean.mean(dim=1), event_target, dim=-1
        ).mean()

        loss_hsic = z_lin.new_zeros(())
        if use_hsic:
            for patch_idx in range(n_patches):
                patch = z_nonlin[:, :, patch_idx, :].reshape(
                    batch_size * n_agents, hidden_dim
                )
                loss_hsic = loss_hsic - (
                    self.decay_rate ** patch_idx
                ) * hsic_loss(patch, target_by_agent)
            loss_hsic = loss_hsic / n_patches
            hsic_score = -loss_hsic.detach()
        else:
            with torch.no_grad():
                first_patch = z_nonlin[:, :, 0, :].reshape(
                    batch_size * n_agents, hidden_dim
                )
                hsic_score = hsic_loss(first_patch, target_by_agent)

        residual_flat = traj_residual.reshape(-1, hidden_dim)
        target_by_patch = event_target.unsqueeze(1).expand(
            -1, n_agents * n_patches, -1
        ).reshape(-1, hidden_dim)
        shuffled_idx = torch.randperm(batch_size, device=z_lin.device)
        shuffled_target = event_target[shuffled_idx].unsqueeze(1).expand(
            -1, n_agents * n_patches, -1
        ).reshape(-1, hidden_dim)

        positive_logits = self.discriminator(residual_flat, target_by_patch, alpha=1.0)
        negative_logits = self.discriminator(residual_flat, shuffled_target, alpha=1.0)
        loss_adv = F.binary_cross_entropy_with_logits(
            positive_logits, torch.ones_like(positive_logits)
        ) + F.binary_cross_entropy_with_logits(
            negative_logits, torch.zeros_like(negative_logits)
        )

        with torch.no_grad():
            discriminator_accuracy = (
                (torch.sigmoid(positive_logits) > 0.5).float().mean()
                + (torch.sigmoid(negative_logits) < 0.5).float().mean()
            ) / 2

        return {
            'disentangle_loss': (
                self.align_weight * loss_lin
                + 0.1 * loss_hsic
                + self.adv_weight * loss_adv
            ),
            'ortho_loss': loss_lin,
            'align_loss': loss_hsic,
            'adv_loss': loss_adv,
            'disc_acc': discriminator_accuracy,
            'hsic_score': hsic_score,
        }



class GatedFusionModule(nn.Module):

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, intent, dynamic):
        gate = self.gate_net(torch.cat([intent, dynamic], dim=-1))
        return gate * intent + (1.0 - gate) * dynamic, gate


class GatedHierarchicalFusionEventHead(nn.Module):

    def __init__(
        self,
        hidden_dim=128,
        n_players=22,
        n_event_types=5,
        fut_T=64,
        n_heads=4,
        n_layers=2,
        d_model=None,
        dropout=0.1,
        use_gated_fusion=True,
        **kwargs,
    ):
        super().__init__()
        self.n_players = n_players
        self.d_model = d_model or hidden_dim * 2
        self.use_gated_fusion = use_gated_fusion

        if use_gated_fusion:
            self.gated_fusion = GatedFusionModule(hidden_dim, dropout)
            self.player_in = nn.Linear(hidden_dim, self.d_model)
        else:
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
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head_tgt = nn.Linear(self.d_model, 1)
        self.head_event = nn.Linear(self.d_model, n_event_types)
        self.head_time = nn.Linear(self.d_model, fut_T)
        self.head_pos = nn.Linear(self.d_model, 2)
        self.prior_bias_scale = nn.Parameter(torch.tensor(1.0))
        self._gate_weights = None

    def forward(
        self,
        h_players,
        h_ball,
        context_ball,
        fut_emb,
        event_emb=None,
        prior_emb=None,
    ):
        batch_size, n_agents, _ = h_players.shape
        if not isinstance(fut_emb, dict):
            raise ValueError("fut_emb must contain 'intent' and 'dynamic' features")
        intent = fut_emb['intent']
        dynamic = fut_emb['dynamic']

        if self.use_gated_fusion:
            player_features, gate = self.gated_fusion(intent, dynamic)
            self._gate_weights = gate.detach()
        else:
            player_features = torch.cat([intent, dynamic], dim=-1)
            self._gate_weights = None

        player_tokens = self.player_in(player_features)
        player_idx = torch.arange(n_agents, device=h_players.device).unsqueeze(0)
        player_tokens = player_tokens + self.player_index_emb(
            player_idx.expand(batch_size, n_agents)
        )
        ball_token = self.ball_in(
            torch.cat([h_ball, context_ball.squeeze(1)], dim=-1)
        ).unsqueeze(1)
        encoded = self.encoder(torch.cat([player_tokens, ball_token], dim=1))
        player_encoded = encoded[:, :-1, :]
        ball_encoded = encoded[:, -1:, :]

        target_logits = self.head_tgt(player_encoded).squeeze(-1)
        target_logits = target_logits + self.prior_bias_scale * (
            player_encoded * ball_encoded.expand_as(player_encoded)
        ).sum(dim=-1)
        target_probs = F.softmax(target_logits, dim=-1)
        event_logits = self.head_event(player_encoded)
        time_logits = self.head_time(player_encoded)
        position_prediction = self.head_pos(player_encoded)
        return (
            target_probs,
            event_logits,
            time_logits,
            position_prediction,
            target_logits,
        )

    def get_gate_weights(self):
        return self._gate_weights

# -*- coding: utf-8 -*-
"""
Volterra Fusion Event Head

Inspired by VNN_F architecture: fuses trajectory embeddings (like optical flow)
with context embeddings (like RGB) using second-order Volterra interactions.

Key idea:
- fut_emb from trajectory predictor = "optical flow" (motion information)
- h_players + prior_emb = "RGB" (appearance/context information)
- Volterra fusion captures their multiplicative interactions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VolterraFusionBlock(nn.Module):
    """
    Single Volterra fusion block inspired by VNN_F.
    
    Computes: output = linear_term + sum_q(quadratic_terms)
    
    Where:
    - linear_term = Conv(concat(flow, rgb))
    - quadratic_terms = elementwise_product of paired convolutions
    
    This captures both first-order (linear) and second-order (multiplicative)
    interactions between the two input streams.
    """
    def __init__(self, in_dim_flow, in_dim_rgb, out_dim, Q=2, dropout=0.1):
        """
        Args:
            in_dim_flow: dimension of flow-like features (trajectory embeddings)
            in_dim_rgb: dimension of RGB-like features (context embeddings)
            out_dim: output dimension
            Q: number of quadratic interaction terms (higher = more capacity)
            dropout: dropout rate
        """
        super().__init__()
        self.Q = Q
        self.out_dim = out_dim
        
        combined_dim = in_dim_flow + in_dim_rgb
        
        # Linear pathway: projects concatenated features
        self.linear_proj = nn.Sequential(
            nn.Linear(combined_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        
        # Quadratic pathway: two parallel projections that get multiplied
        # Output 2*Q*out_dim channels, split into pairs for multiplication
        self.quad_proj = nn.Linear(combined_dim, 2 * Q * out_dim)
        self.quad_norm = nn.LayerNorm(out_dim)
        
        # Learnable mixing weight between linear and quadratic
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, flow_feat, rgb_feat):
        """
        Args:
            flow_feat: [B, N, D_flow] trajectory/motion features
            rgb_feat: [B, N, D_rgb] context/appearance features
            
        Returns:
            fused: [B, N, out_dim] fused features with Volterra interactions
        """
        # Concatenate inputs
        combined = torch.cat([flow_feat, rgb_feat], dim=-1)  # [B, N, D_flow+D_rgb]
        
        # Linear term
        x_linear = self.linear_proj(combined)  # [B, N, out_dim]
        
        # Quadratic term: split into two halves, multiply, then sum across Q
        x_quad_raw = self.quad_proj(combined)  # [B, N, 2*Q*out_dim]
        
        # Split into two halves for multiplication
        half = self.Q * self.out_dim
        x_quad_a = x_quad_raw[..., :half]      # [B, N, Q*out_dim]
        x_quad_b = x_quad_raw[..., half:]      # [B, N, Q*out_dim]
        
        # Element-wise multiplication (captures second-order interactions)
        x_quad_mul = x_quad_a * x_quad_b       # [B, N, Q*out_dim]
        
        # Sum across Q terms
        B, N, _ = x_quad_mul.shape
        x_quad_mul = x_quad_mul.view(B, N, self.Q, self.out_dim)
        x_quadratic = x_quad_mul.sum(dim=2)    # [B, N, out_dim]
        x_quadratic = self.quad_norm(x_quadratic)
        
        # Combine linear and quadratic with learnable weight
        alpha = torch.sigmoid(self.alpha)
        fused = alpha * x_linear + (1 - alpha) * x_quadratic
        
        return self.dropout(fused)


class MultiScaleVolterraFusion(nn.Module):
    """
    Multi-scale Volterra fusion with pooling, similar to VNN architecture.
    
    Applies Volterra fusion at multiple scales and combines them.
    """
    def __init__(self, in_dim_flow, in_dim_rgb, hidden_dim, out_dim, 
                 Q=2, n_scales=2, dropout=0.1):
        super().__init__()
        self.n_scales = n_scales
        
        # First scale: full resolution
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
            
            # Next scale uses output of previous
            current_flow_dim = block_out
            current_rgb_dim = block_out
            
        # Final projection
        self.final_norm = nn.LayerNorm(out_dim)
        
    def forward(self, flow_feat, rgb_feat):
        """
        Args:
            flow_feat: [B, N, D_flow] trajectory features
            rgb_feat: [B, N, D_rgb] context features
            
        Returns:
            fused: [B, N, out_dim]
        """
        f = flow_feat
        r = rgb_feat
        
        for block in self.volterra_blocks:
            out = block(f, r)
            # For subsequent blocks, use the fused output as both inputs
            # This creates hierarchical Volterra interactions
            f = out
            r = out
            
        return self.final_norm(out)


class VolterraFusionEventHead(nn.Module):
    """
    Event prediction head with Volterra fusion of trajectory and context features.
    
    Architecture:
    1. Volterra fusion: fut_emb (flow) + [h_players, prior_emb, context_ball] (RGB)
    2. Transformer encoder for temporal reasoning
    3. Task-specific prediction heads
    
    This replaces the complex GraphVolterra321 with a cleaner VNN-inspired approach.
    """
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
        
        # Volterra fusion module
        # flow_dim = hidden_dim (from fut_emb)
        # rgb_dim = 3 * hidden_dim (h_players + prior_emb + context_ball)
        self.volterra_fusion = MultiScaleVolterraFusion(
            in_dim_flow=hidden_dim,
            in_dim_rgb=3 * hidden_dim,
            hidden_dim=hidden_dim * 2,
            out_dim=hidden_dim,
            Q=volterra_Q,
            n_scales=volterra_scales,
            dropout=dropout
        )
        
        # Input projection (volterra output + original features for residual)
        # Total: volterra_out + h_players + prior_emb + fut_emb = 4 * hidden_dim
        self.player_in = nn.Linear(4 * hidden_dim, d_model)
        self.ball_in = nn.Linear(2 * hidden_dim, d_model)
        
        # Learnable position embeddings
        self.player_index_emb = nn.Embedding(n_players, d_model)
        
        # Transformer encoder
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
        
        # Prediction heads
        self.head_tgt = nn.Linear(d_model, 1)
        self.head_event = nn.Linear(d_model, n_event_types)
        self.head_time = nn.Linear(d_model, fut_T)
        self.head_pos = nn.Linear(d_model, 2)
        
        # Prior bias for target prediction
        self.prior_bias_scale = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, h_players, h_ball, prior_emb, fut_emb, event_emb=None, context_ball=None):
        """
        Forward pass with Volterra fusion.
        
        Args:
            h_players: [B, 22, H] player embeddings from context encoder
            h_ball: [B, H] ball embedding
            prior_emb: [B, 22, H] prior embeddings (importance/position-based)
            fut_emb: [B, 22, H] future trajectory embeddings (from traj predictor)
            
        Returns:
            tgt_probs: [B, 22] target player probabilities
            event_logits: [B, 22, E] event type logits per player
            time_logits: [B, 22, T] time step logits per player
            pos_pred: [B, 22, 2] predicted event position per player
            tgt_logits: [B, 22] raw target logits (for loss computation)
        """
        B, A, H = h_players.shape

        if context_ball is None:
            # Compute ball context through attention
            attn_score = torch.sum(h_players * h_ball.unsqueeze(1), dim=-1, keepdim=True)
            attn_weight = torch.softmax(attn_score, dim=1)
            context_ball = (attn_weight * h_players).sum(dim=1, keepdim=True)
        
        if context_ball.size(1) == 1:
            context_ball = context_ball.expand(-1, A, -1)  # [B, 22, H]

        # Compute ball context through attention
        attn_score = torch.sum(h_players * h_ball.unsqueeze(1), dim=-1, keepdim=True)
        attn_weight = torch.softmax(attn_score, dim=1)
        context_ball = (attn_weight * h_players).sum(dim=1, keepdim=True)
        context_ball = context_ball.expand(-1, A, -1)  # [B, 22, H]
        
        # Prepare inputs for Volterra fusion
        # "Flow" = trajectory embeddings (motion information)
        flow_feat = fut_emb  # [B, 22, H]
        
        # "RGB" = context embeddings (appearance/state information)
        rgb_feat = torch.cat([h_players, prior_emb, context_ball], dim=-1)  # [B, 22, 3H]
        
        # Volterra fusion: captures second-order interactions between motion and context
        volterra_out = self.volterra_fusion(flow_feat, rgb_feat)  # [B, 22, H]
        
        # Combine for final player features (with residuals)
        player_features = torch.cat([
            volterra_out,   # Volterra-fused features
            h_players,      # Original context (residual)
            prior_emb,      # Prior information (residual)
            fut_emb,        # Trajectory features (residual)
        ], dim=-1)  # [B, 22, 4H]
        
        x_player = self.player_in(player_features)  # [B, 22, d_model]
        
        # Ball token
        gpool_players = h_players.mean(dim=1)
        ball_token = self.ball_in(torch.cat([h_ball, gpool_players], dim=-1))
        ball_token = ball_token.unsqueeze(1)  # [B, 1, d_model]
        
        # Add player index embeddings
        idx = torch.arange(A, device=h_players.device).unsqueeze(0).expand(B, A)
        x_player = x_player + self.player_index_emb(idx)
        
        # Transformer: [ball, players]
        tokens = torch.cat([ball_token, x_player], dim=1)  # [B, 23, d_model]
        z = self.encoder(tokens)
        
        z_ball = z[:, 0:1, :]    # [B, 1, d_model]
        z_player = z[:, 1:, :]   # [B, 22, d_model]
        
        # Target player prediction
        tgt_logits_main = self.head_tgt(z_player).squeeze(-1)  # [B, 22]
        
        # Add ball-player similarity bias
        prior_bias = (z_player * z_ball.expand_as(z_player)).sum(dim=-1)
        tgt_logits = tgt_logits_main + self.prior_bias_scale * prior_bias
        tgt_probs = torch.softmax(tgt_logits, dim=-1)
        
        # Other predictions
        event_logits = self.head_event(z_player)  # [B, 22, E]
        time_logits = self.head_time(z_player)    # [B, 22, T]
        pos_pred = self.head_pos(z_player)        # [B, 22, 2]
        
        return tgt_probs, event_logits, time_logits, pos_pred, tgt_logits


class ParallelEventPredictor_VolterraFusion(nn.Module):
    """
    Complete event predictor with VNN-style Volterra fusion.
    
    This is a drop-in replacement for ParallelEventPredictor_TransformerWithVolterra
    but uses the cleaner VNN-inspired fusion instead of GraphVolterra321.
    
    Key differences:
    - Uses VolterraFusionBlock instead of GraphVolterraWithCubic
    - Cleaner architecture with proven VNN design
    - Better gradient flow through residual connections
    """
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
        
        # Future trajectory projection
        self.proj_fut = nn.Linear(2, hidden_dim)
        
        # Main event head with Volterra fusion
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
        
        # Store reference to volterra module for external access (e.g., in step_batch)
        self.volterra_module = None  # Set to None to indicate we use built-in fusion
        
    def forward(self, ctx_dyns, ctx_edges, ctx_seqX, ctx_labels_22, cols,
                edge_cols=None, fut_pred=None):
        """
        Forward pass.
        
        Args:
            ctx_dyns: [B, C, F_dyn] node dynamics
            ctx_edges: [B, C, F_edge] edge features  
            ctx_seqX: [B, T, 23, 6] context sequence (not used directly here)
            ctx_labels_22: player labels
            cols: dynamic feature column names
            edge_cols: edge feature column names
            fut_pred: [T, B, 22, 2] future trajectory predictions from STAR model
            
        Returns:
            tgt_probs, event_logits, time_logits, pos_pred, tgt_logits
        """
        # Prepare labels
        if isinstance(ctx_labels_22, (list, tuple, np.ndarray)):
            arr = np.array(ctx_labels_22, dtype=object)
            if arr.ndim == 2:
                enc_labels_22 = [ctx_labels_22]
            else:
                enc_labels_22 = ctx_labels_22
        else:
            enc_labels_22 = ctx_labels_22
            
        # Get embeddings from context encoder
        h_players, h_ball, team_labels = self.context_encoder(
            ctx_dyns, ctx_edges, enc_labels_22,
            cols=cols, edge_cols=edge_cols,
            return_player_states=True,
            return_team_labels=True,
        )  # h_players: [B, 22, H], h_ball: [B, H]
        
        # Get prior embeddings
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
            
        # Get future embeddings from trajectory predictions
        if fut_pred is not None:
            # fut_pred: [T, B, 22, 2] -> project to hidden_dim
            fut_mean = fut_pred.mean(dim=0)  # [B, 22, 2]
            fut_emb = self.proj_fut(fut_mean)  # [B, 22, H]
        else:
            fut_emb = torch.zeros_like(h_players)
            
        # Forward through Volterra fusion event head
        return self.head(h_players, h_ball, prior_emb, fut_emb)


# ============================================================================
# Alternative: Lightweight version for faster training
# ============================================================================

class LightweightVolterraFusion(nn.Module):
    """
    Lightweight Volterra fusion - single block, fewer parameters.
    Good for initial experiments or when compute is limited.
    """
    def __init__(self, in_dim_flow, in_dim_rgb, out_dim, Q=2, dropout=0.1):
        super().__init__()
        self.Q = Q
        self.out_dim = out_dim
        
        combined_dim = in_dim_flow + in_dim_rgb
        
        # Single linear layer for linear term
        self.linear_proj = nn.Linear(combined_dim, out_dim)
        
        # Quadratic pathway
        self.quad_proj = nn.Linear(combined_dim, 2 * Q * out_dim)
        
        # Output normalization
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, flow_feat, rgb_feat):
        combined = torch.cat([flow_feat, rgb_feat], dim=-1)
        
        # Linear term
        x_linear = self.linear_proj(combined)
        
        # Quadratic term
        x_quad_raw = self.quad_proj(combined)
        half = self.Q * self.out_dim
        x_quad_mul = x_quad_raw[..., :half] * x_quad_raw[..., half:]
        
        B, N, _ = x_quad_mul.shape
        x_quad_mul = x_quad_mul.view(B, N, self.Q, self.out_dim)
        x_quadratic = x_quad_mul.sum(dim=2)
        
        # Add and normalize
        out = self.norm(x_linear + x_quadratic)
        return self.dropout(out)


class LightweightVolterraEventHead(nn.Module):
    """
    Lightweight event head with single-block Volterra fusion.
    """
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
        
        # Lightweight Volterra fusion
        self.volterra_fusion = LightweightVolterraFusion(
            in_dim_flow=hidden_dim,
            in_dim_rgb=3 * hidden_dim,
            out_dim=hidden_dim,
            Q=volterra_Q,
            dropout=dropout
        )
        
        # Simpler input projection
        self.player_in = nn.Linear(2 * hidden_dim, d_model)  # volterra + fut_emb
        self.ball_in = nn.Linear(hidden_dim, d_model)
        
        # Single transformer layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction heads
        self.head_tgt = nn.Linear(d_model, 1)
        self.head_event = nn.Linear(d_model, n_event_types)
        self.head_time = nn.Linear(d_model, fut_T)
        self.head_pos = nn.Linear(d_model, 2)
        
    def forward(self, h_players, h_ball, prior_emb, fut_emb):
        B, A, H = h_players.shape
        
        # Ball context
        attn_score = torch.sum(h_players * h_ball.unsqueeze(1), dim=-1, keepdim=True)
        attn_weight = torch.softmax(attn_score, dim=1)
        context_ball = (attn_weight * h_players).sum(dim=1, keepdim=True).expand(-1, A, -1)
        
        # Volterra fusion
        rgb_feat = torch.cat([h_players, prior_emb, context_ball], dim=-1)
        volterra_out = self.volterra_fusion(fut_emb, rgb_feat)
        
        # Simple feature combination
        x_player = self.player_in(torch.cat([volterra_out, fut_emb], dim=-1))
        ball_token = self.ball_in(h_ball).unsqueeze(1)
        
        # Transformer
        tokens = torch.cat([ball_token, x_player], dim=1)
        z = self.encoder(tokens)
        z_player = z[:, 1:, :]
        
        # Predictions
        tgt_logits = self.head_tgt(z_player).squeeze(-1)
        tgt_probs = torch.softmax(tgt_logits, dim=-1)
        
        return (
            tgt_probs,
            self.head_event(z_player),
            self.head_time(z_player),
            self.head_pos(z_player),
            tgt_logits
        )


# ============================================================================
# Factory function to create the appropriate predictor
# ============================================================================

def create_volterra_event_predictor(
    context_encoder,
    prior_scorer,
    hidden_dim: int,
    mode: str = "full",  # "full", "lightweight"
    **kwargs
):
    """
    Factory function to create Volterra fusion event predictor.
    
    Args:
        context_encoder: ContextEncoder instance
        prior_scorer: TgtPlayerScorer instance (or None)
        hidden_dim: hidden dimension
        mode: "full" for full model, "lightweight" for faster training
        **kwargs: additional arguments for the head
        
    Returns:
        ParallelEventPredictor with Volterra fusion
    """
    if mode == "lightweight":
        # Use lightweight version
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
                # Same logic as ParallelEventPredictor_VolterraFusion
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
        # Use full version
        return ParallelEventPredictor_VolterraFusion(
            context_encoder=context_encoder,
            prior_scorer=prior_scorer,
            hidden_dim=hidden_dim,
            **kwargs
        )
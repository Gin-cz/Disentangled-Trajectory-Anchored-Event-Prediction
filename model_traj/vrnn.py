# -*- coding: utf-8 -*-
"""
VRNN for Multi-Agent Trajectory Prediction
Compatible with existing training framework (LSTM/STAR interface)

Interface:
- forward(x, h0=None, m0=None, **context_kwargs) -> dict with h_seq, h_last, m_seq, m_last, mem_window
- step(x_t, h_prev_window, **context_kwargs) -> dict with h_t, m_t, mem_window
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

LOG_2PI = math.log(2 * math.pi)
EPS = 1e-6


class VRNNCell(nn.Module):
    """
    Single VRNN cell for one timestep.
    
    VRNN combines:
    - Prior: p(z_t | h_{t-1})
    - Posterior/Encoder: q(z_t | x_t, h_{t-1})  
    - Decoder: p(x_t | z_t, h_{t-1})
    - RNN transition: h_t = f(h_{t-1}, x_t, z_t)
    """
    
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        
        # Feature extractors
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
        )
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
        )
        
        # Prior: p(z_t | h_{t-1})
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus(),
        )
        
        # Encoder/Posterior: q(z_t | x_t, h_{t-1})
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
        
        # Decoder: p(x_t | z_t, h_{t-1})
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
        
        # RNN: h_t = f(phi_x, phi_z, h_{t-1})
        self.rnn = nn.GRUCell(h_dim * 2, h_dim)
    
    def forward_posterior(self, x_t, h_prev):
        """
        Forward pass using posterior (for training with teacher forcing).
        
        Args:
            x_t: [B, x_dim] - observed input
            h_prev: [B, h_dim] - previous hidden state
            
        Returns:
            h_t: [B, h_dim] - new hidden state
            kld: [B] - KL divergence
            nll: [B] - negative log likelihood
            x_recon_mean: [B, x_dim] - reconstructed mean
        """
        # Feature extraction
        phi_x_t = self.phi_x(x_t)  # [B, h_dim]
        
        # Prior p(z|h)
        prior_t = self.prior(h_prev)
        p_mean = self.prior_mean(prior_t)
        p_std = self.prior_std(prior_t).clamp(min=EPS)
        
        # Posterior q(z|x,h)
        enc_input = torch.cat([phi_x_t, h_prev], dim=-1)
        enc_t = self.enc(enc_input)
        q_mean = self.enc_mean(enc_t)
        q_std = self.enc_std(enc_t).clamp(min=EPS)
        
        # Reparameterization trick
        eps = torch.randn_like(q_std)
        z_t = q_mean + q_std * eps
        
        # Feature extraction for z
        phi_z_t = self.phi_z(z_t)
        
        # Decoder p(x|z,h)
        dec_input = torch.cat([phi_z_t, h_prev], dim=-1)
        dec_t = self.dec(dec_input)
        x_mean = self.dec_mean(dec_t)
        x_std = self.dec_std(dec_t).clamp(min=EPS)
        
        # RNN update
        rnn_input = torch.cat([phi_x_t, phi_z_t], dim=-1)
        h_t = self.rnn(rnn_input, h_prev)
        
        # KL divergence: KL(q || p)
        kld = 0.5 * torch.sum(
            2 * torch.log(p_std) - 2 * torch.log(q_std) +
            (q_std.pow(2) + (q_mean - p_mean).pow(2)) / p_std.pow(2) - 1,
            dim=-1
        )
        
        # NLL: -log p(x|z,h)
        var = x_std.pow(2)
        nll = 0.5 * torch.sum(
            ((x_t - x_mean).pow(2) / var) + torch.log(var) + LOG_2PI,
            dim=-1
        )
        
        return h_t, kld, nll, x_mean
    
    def forward_prior(self, x_t, h_prev):
        """
        Forward pass using prior (for generation/rollout).
        Uses x_t only for phi_x (to update RNN), samples z from prior.
        
        Args:
            x_t: [B, x_dim] - input (can be predicted from previous step)
            h_prev: [B, h_dim] - previous hidden state
            
        Returns:
            h_t: [B, h_dim] - new hidden state
            x_pred_mean: [B, x_dim] - predicted next position mean
            x_pred_std: [B, x_dim] - predicted next position std
        """
        # Feature extraction
        phi_x_t = self.phi_x(x_t)
        
        # Prior p(z|h) - sample from prior during generation
        prior_t = self.prior(h_prev)
        p_mean = self.prior_mean(prior_t)
        p_std = self.prior_std(prior_t).clamp(min=EPS)
        
        # Sample z from prior
        eps = torch.randn_like(p_std)
        z_t = p_mean + p_std * eps
        
        # Feature extraction for z
        phi_z_t = self.phi_z(z_t)
        
        # Decoder p(x|z,h) - this gives the prediction
        dec_input = torch.cat([phi_z_t, h_prev], dim=-1)
        dec_t = self.dec(dec_input)
        x_pred_mean = self.dec_mean(dec_t)
        x_pred_std = self.dec_std(dec_t).clamp(min=EPS)
        
        # RNN update (use phi_x from input, not prediction)
        rnn_input = torch.cat([phi_x_t, phi_z_t], dim=-1)
        h_t = self.rnn(rnn_input, h_prev)
        
        return h_t, x_pred_mean, x_pred_std


class AgentVRNN(nn.Module):
    """
    Multi-agent VRNN with interface compatible with AgentLSTM/STAR.
    
    Input: x [B, T, A, F_in]
    Output: dict with h_seq, h_last, m_seq, m_last, (mem_window)
    """
    
    def __init__(self,
                 in_dim,
                 h_dim=128,
                 z_dim=32,
                 window_K=8,
                 d_mem=None,
                 n_players=22,
                 dropout=0.1,
                 # Unused but kept for interface compatibility
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
        
        # VRNN cell (shared across all agents)
        self.vrnn_cell = VRNNCell(in_dim, h_dim, z_dim)
        
        # Memory projection (if d_mem != h_dim)
        if self.d_mem != h_dim:
            self.mem_proj = nn.Linear(h_dim, self.d_mem)
        else:
            self.mem_proj = None
        
        # Store losses for training
        self._kld_loss = None
        self._nll_loss = None
    
    def forward(self, x, h0=None, m0=None,
                h_players=None, prior_emb=None, context_ball=None,
                event_emb=None):
        """
        Burn-in: encode sequence using posterior.
        
        Args:
            x: [B, T, A, F_in] - input sequence (A = n_players + 1 for ball)
            h0: optional initial hidden state
            m0: unused, kept for interface compatibility
            
        Returns:
            dict with:
                h_seq: [B, T, A, h_dim]
                h_last: [B, A, h_dim]
                m_seq: [B, T, A, d_mem]
                m_last: [B, A, d_mem]
                mem_window: [B, K, A, h_dim]
        """
        B, T, A, F = x.shape
        device = x.device
        
        # Initialize hidden state
        if h0 is not None:
            h = h0.reshape(B * A, self.h_dim)
        else:
            h = torch.zeros(B * A, self.h_dim, device=device)
        
        h_seq_list = []
        total_kld = 0.0
        total_nll = 0.0
        
        for t in range(T):
            # Get current frame: [B, A, F] -> [B*A, F]
            x_t = x[:, t].reshape(B * A, F)
            
            # Clean NaN values
            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Forward through VRNN cell using posterior
            h, kld, nll, _ = self.vrnn_cell.forward_posterior(x_t, h)
            
            # Accumulate losses
            total_kld = total_kld + kld.mean()
            total_nll = total_nll + nll.mean()
            
            # Store hidden state
            h_seq_list.append(h.reshape(B, A, self.h_dim))
        
        # Store losses for training
        self._kld_loss = total_kld / T
        self._nll_loss = total_nll / T
        
        # Stack sequence
        h_seq = torch.stack(h_seq_list, dim=1)  # [B, T, A, h_dim]
        h_last = h_seq[:, -1]  # [B, A, h_dim]
        
        # Memory projection
        if self.mem_proj is not None:
            m_seq = self.mem_proj(h_seq)
            m_last = m_seq[:, -1]
        else:
            m_seq = h_seq
            m_last = h_last
        
        # Build memory window
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
        """
        Single step prediction (rollout phase).
        Uses prior for sampling z.
        
        Args:
            x_t: [B, A, F_in] - current input
            h_prev_window: [B, K, A, h_dim] - history window
            
        Returns:
            dict with:
                h_t: [B, A, h_dim]
                m_t: [B, A, d_mem]
                mem_window: [B, K, A, h_dim] - updated window
        """
        B, A, F = x_t.shape
        device = x_t.device
        K = h_prev_window.size(1)
        
        # Get previous hidden state (last in window)
        h_prev = h_prev_window[:, -1].reshape(B * A, self.h_dim)
        
        # Flatten input
        x_t_flat = x_t.reshape(B * A, F)
        x_t_flat = torch.nan_to_num(x_t_flat, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Forward using prior (generation mode)
        h_t, _, _ = self.vrnn_cell.forward_prior(x_t_flat, h_prev)
        
        # Reshape
        h_t = h_t.reshape(B, A, self.h_dim)
        
        # Memory projection
        if self.mem_proj is not None:
            m_t = self.mem_proj(h_t)
        else:
            m_t = h_t
        
        # Update window (sliding)
        new_window = torch.cat([h_prev_window[:, 1:], h_t.unsqueeze(1)], dim=1)
        
        return {
            "h_t": h_t,
            "m_t": m_t,
            "mem_window": new_window,
        }
    
    def get_vrnn_losses(self):
        """Get KLD and NLL losses from last forward pass."""
        return self._kld_loss, self._nll_loss


class VRNNTrajStepHead(nn.Module):
    """
    Trajectory prediction head for VRNN.
    Compatible with existing heads (LSTMTrajStepHead, TransformerTrajStepHead).
    """
    
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
        
        # Calculate input dimension
        f_in = in_dim + 2  # h + last_pos
        if use_m and m_dim is not None:
            f_in += m_dim
        
        self.net = nn.Sequential(
            nn.Linear(f_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),  # Predict Δx, Δy
        )
    
    def forward(self,
                h_agents,
                last_pos,
                m_t=None,
                h_players=None,
                prior_emb=None,
                volterra_features=None):
        """
        Predict next position.
        
        Args:
            h_agents: [B, A, h_dim] - agent hidden states
            last_pos: [B, A, 2] - last known positions
            m_t: [B, A, m_dim] - memory state (optional)
            
        Returns:
            next_pos: [B, A, 2] - predicted next positions
        """
        features = [h_agents, last_pos]
        
        if self.use_m and m_t is not None:
            features.append(m_t)
        
        x = torch.cat(features, dim=-1)
        delta = self.net(x)
        
        return last_pos + delta


# ==================== With Context Versions ====================

class AgentVRNNWithContext(AgentVRNN):
    """
    VRNN with context injection (h_players, prior_emb).
    """
    
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
        
        # Context fusion layer
        self.context_fusion = nn.Sequential(
            nn.Linear(h_dim + context_dim * 2, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def _fuse_context(self, h, h_players, prior_emb, B, A):
        """Fuse hidden state with context."""
        device = h.device
        
        if h_players is None:
            h_players = torch.zeros(B, A, self.context_dim, device=device)
        if prior_emb is None:
            prior_emb = torch.zeros(B, A, self.context_dim, device=device)
        
        # h: [B*A, h_dim] -> [B, A, h_dim]
        h_reshaped = h.reshape(B, A, self.h_dim)
        
        # Expand context if needed (context might be for players only, not ball)
        if h_players.size(1) < A:
            # Pad with zeros for ball
            pad = torch.zeros(B, A - h_players.size(1), self.context_dim, device=device)
            h_players = torch.cat([h_players, pad], dim=1)
            prior_emb = torch.cat([prior_emb, pad], dim=1)
        
        # Fuse
        combined = torch.cat([h_reshaped, h_players, prior_emb], dim=-1)
        h_fused = self.context_fusion(combined)
        
        return h_fused.reshape(B * A, self.h_dim)
    
    def forward(self, x, h0=None, m0=None,
                h_players=None, prior_emb=None, context_ball=None,
                event_emb=None):
        """Forward with context fusion."""
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
            
            # Fuse context before VRNN step
            h_ctx = self._fuse_context(h, h_players, prior_emb, B, A)
            
            # Forward through VRNN cell
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
        """Single step with context."""
        B, A, F = x_t.shape
        device = x_t.device
        K = h_prev_window.size(1)
        
        h_prev = h_prev_window[:, -1].reshape(B * A, self.h_dim)
        
        # Fuse context
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
    """
    Trajectory prediction head with context for VRNN.
    """
    
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
        
        f_in = in_dim + 2 + context_dim * 2  # h + pos + h_players + prior_emb
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
            # Pad if needed
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


# Aliases for compatibility
VRNNWindowedWithContext = AgentVRNNWithContext
VRNNWindowedNoContext = AgentVRNN
TransformerTrajStepHeadWithContext = VRNNTrajStepHeadWithContext
TransformerTrajStepHeadNoContext = VRNNTrajStepHead


# ==================== Test ====================
if __name__ == "__main__":
    B, T, A, F = 8, 20, 23, 64  # batch, time, agents (22 players + ball), features
    
    print("Testing AgentVRNN (NoContext)...")
    vrnn = AgentVRNN(in_dim=F, h_dim=128, z_dim=32, window_K=8, n_players=22)
    x = torch.randn(B, T, A, F)
    
    out = vrnn(x)
    print(f"  h_seq: {out['h_seq'].shape}")      # [8, 20, 23, 128]
    print(f"  h_last: {out['h_last'].shape}")    # [8, 23, 128]
    print(f"  m_seq: {out['m_seq'].shape}")      # [8, 20, 23, 128]
    print(f"  mem_window: {out['mem_window'].shape}")  # [8, 8, 23, 128]
    
    kld, nll = vrnn.get_vrnn_losses()
    print(f"  KLD: {kld.item():.4f}, NLL: {nll.item():.4f}")
    
    # Test step
    x_t = torch.randn(B, A, F)
    step_out = vrnn.step(x_t, out['mem_window'])
    print(f"  step h_t: {step_out['h_t'].shape}")  # [8, 23, 128]
    
    print("\nTesting AgentVRNNWithContext...")
    vrnn_ctx = AgentVRNNWithContext(
        in_dim=F, h_dim=128, z_dim=32, window_K=8,
        context_dim=128, n_players=22
    )
    h_players = torch.randn(B, 22, 128)
    prior_emb = torch.randn(B, 22, 128)
    
    out_ctx = vrnn_ctx(x, h_players=h_players, prior_emb=prior_emb)
    print(f"  h_seq: {out_ctx['h_seq'].shape}")
    
    print("\nTesting VRNNTrajStepHead...")
    traj_head = VRNNTrajStepHead(in_dim=128, hidden=256, use_m=True, m_dim=128)
    last_pos = torch.randn(B, A, 2)
    next_pos = traj_head(out['h_last'], last_pos, m_t=out['m_last'])
    print(f"  next_pos: {next_pos.shape}")  # [8, 23, 2]
    
    print("\n✅ All tests passed!")
# model_event/tgn_event_head.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TGNEventHead(nn.Module):
    """
    True TGN Implementation for Event Prediction.
    
    Architecture follows Rossi et al. (2020) Logic:
    1. Input: Per-frame features (h_players + event_emb) act as 'Messages'.
    2. Memory: GRUCell updates player state over time C.
    3. Embedding: Spatial Attention (Graph Transformer style) aggregates teammates at each step.
    
    Inputs:
    - h_players: [B, C, A, H] (Your pre-computed embeddings)
    - h_ball:    [B, C, H]
    - event_emb: [B, C, H]    (Time-aligned context)
    
    Outputs:
    - tgt_probs, event_logits, etc. (Same interface)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_players: int = 22,
        n_event_types: int = 5,
        fut_T: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        # Unused args for compatibility
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        volterra_Q: int = 2,
        volterra_scales: int = 2,
        n_layers: int = 2, # Note: TGN is recurrent, n_layers usually means stacking GNN layers per step
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_players = int(n_players)
        self.n_event_types = int(n_event_types)
        self.fut_T = int(fut_T)
        self.n_heads = int(n_heads)

        # ============ 1. Message/Input Fusion ============
        # Compresses h_players + event_emb into the input for the memory cell
        self.input_fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ============ 2. Memory Module (The Soul of TGN) ============
        # Updates state: s_i(t) = GRU( message(t), s_i(t-1) )
        self.memory_rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        # ============ 3. Graph Embedding Module (Spatial Attention) ============
        # At each step, players look at each other using updated memory
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_s = nn.LayerNorm(self.hidden_dim)
        
        # Simple FFN after attention
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_f = nn.LayerNorm(self.hidden_dim)

        # ============ 4. Output Fusion ============
        # Final aggregation for prediction (Combine with static features)
        self.final_fusion = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),  # [tgn_out, prior, fut]
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Ball Encoder (Simplified for compatibility)
        self.ball_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.prior_bias_scale = nn.Parameter(torch.tensor(1.0))

        # ============ Prediction Heads ============
        self.head_tgt = nn.Linear(self.hidden_dim, 1)
        self.head_event = nn.Linear(self.hidden_dim, self.n_event_types)
        self.head_time = nn.Linear(self.hidden_dim, self.fut_T)
        self.head_pos = nn.Linear(self.hidden_dim, 2)

    def forward(self, h_players, h_ball, prior_emb, fut_emb, event_emb=None, context_ball=None):
        """
        Forward pass implementing the TGN loop.
        """
        # ---- Compatibility & Shapes ----
        if h_players.dim() == 3: h_players = h_players.unsqueeze(1)
        if h_ball.dim() == 2: h_ball = h_ball.unsqueeze(1)
        
        B, C, A, H = h_players.shape
        
        if event_emb is None:
            event_emb = torch.zeros(B, C, H, device=h_players.device, dtype=h_players.dtype)
        
        # Expand event_emb for all players: [B, C, H] -> [B, C, A, H]
        edge_ctx_player = event_emb.unsqueeze(2).expand(-1, -1, A, -1)

        # ============ Pre-compute Inputs for Efficiency ============
        # Instead of running Linear inside the loop, we run it parallel first
        # Input to memory = Fusion(Player_Feat + Event_Context)
        # [B, C, A, H*2] -> [B, C, A, H]
        tgn_inputs = self.input_fusion(torch.cat([h_players, edge_ctx_player], dim=-1))

        # ============ TGN Loop (Sequential) ============
        # Initialize Memory (s_0) for all players: [B*A, H]
        memory = torch.zeros(B * A, H, device=h_players.device, dtype=h_players.dtype)
        
        # We can optionally store all hidden states, but usually we only need the LAST one for prediction
        # or we can perform attention at the last step.
        # However, typical TGN updates and then aggregates.
        
        # Loop over time C
        for t in range(C):
            # 1. Get Message/Input for this step
            # [B, A, H] -> [B*A, H]
            msg_t = tgn_inputs[:, t, :, :].reshape(B * A, H)
            
            # 2. Memory Update (GRU)
            # memory(t) = GRU( message(t), memory(t-1) )
            memory = self.memory_rnn(msg_t, memory) # [B*A, H]
            
            # 3. Embedding Module (Spatial Graph Attention)
            # Players interact based on their updated memory state
            # Reshape for Attention: [B, A, H]
            mem_view = memory.view(B, A, H)
            
            # Self-Attention: Players attend to teammates
            # Q=K=V=Memory (Full graph communication)
            spatial_out, _ = self.spatial_attn(mem_view, mem_view, mem_view)
            
            # Residual + Norm
            embedding_t = self.norm_s(mem_view + spatial_out)
            
            # FFN
            embedding_t = embedding_t + self.ffn(embedding_t)
            embedding_t = self.norm_f(embedding_t)
            
            # Update memory with the spatially enhanced info? 
            # In standard TGN, the embedding is the OUTPUT, memory is INTERNAL.
            # But for next step stability, we often feed embedding back or just keep memory GRU state.
            # Standard TGN: Memory is updated by RAW events, Embedding is computed FROM memory.
            # So 'memory' variable stays as the GRU hidden state. 'embedding_t' is the layer output.

        # ============ Final Prediction Stage ============
        # We use the embedding from the LAST time step (C-1) for prediction
        final_node_states = embedding_t # [B, A, H]

        # Combine with Prior (Static) and Future (Trajectory) embeddings
        # [B, A, H*3] -> [B, A, H]
        final_feat = self.final_fusion(torch.cat([final_node_states, prior_emb, fut_emb], dim=-1))

        # ---- Ball Context (Bias Calculation) ----
        # Simple temporal aggregation for ball
        # [B, C, H] -> [B, H]
        ball_feat = h_ball.mean(dim=1) 
        ball_feat = self.ball_encoder(ball_feat)
        
        # Bias: Ball * Player similarity
        ball_feat_expand = ball_feat.unsqueeze(1).expand(-1, A, -1)
        prior_bias = (final_feat * ball_feat_expand).sum(dim=-1) / math.sqrt(H)

        # ---- Heads ----
        tgt_logits_main = self.head_tgt(final_feat).squeeze(-1)
        tgt_logits = tgt_logits_main + self.prior_bias_scale * prior_bias
        tgt_probs = F.softmax(tgt_logits, dim=-1)

        event_logits = self.head_event(final_feat)
        time_logits = self.head_time(final_feat)
        pos_pred = self.head_pos(final_feat)

        return tgt_probs, event_logits, time_logits, pos_pred, tgt_logits
"""Encode synchronized event history and player-state context."""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
class ContextEncoder(nn.Module):

    def __init__(self, d_dyn: int, d_edge: int, d_h: int,
                 num_layers: int = 1, bidirectional: bool = False, dropout: float = 0.0,
                 n_home: int = 11, n_away: int = 11, wanted_keys: Optional[List[str]] = None,
                 d_pid: int = 16, d_team: int = 8, d_time_idx: int = 16, d_time_sin: int = 16,
                 max_C: int = 256, attn_pool: bool = False, max_side_id: int = 23):
        super().__init__()

        self.n_home, self.n_away = n_home, n_away
        self.n_players = int(n_home) + int(n_away)
        self.ball_idx = self.n_players
        self.A = self.n_players + 1
        self.d_edge = int(d_edge)
        self.attn_pool = bool(attn_pool)

        max_side_id = int(max(max_side_id, n_home, n_away))
        self.max_side_id = max_side_id
        self.name2id = {f"h{i}": i for i in range(1, max_side_id + 1)}
        self.name2id.update({f"a{i}": max_side_id + i for i in range(1, max_side_id + 1)})
        self.name2id["ball"] = 2 * max_side_id + 1
        self.name2id["PAD_h"] = 0
        self.name2id["PAD_a"] = 0

        if wanted_keys is None:
            wanted_keys = [
                "x", "y", "vx", "vy", "speed",
                "gl_dist", "gl_ang", "tl_dist", "tl_ang",
            ]
        self.wanted_keys = list(wanted_keys)

        self.player_emb   = nn.Embedding(2 * max_side_id + 2, d_pid, padding_idx=0)
        self.team_emb     = nn.Embedding(3, d_team, padding_idx=0)
        self.time_idx_emb = nn.Embedding(max_C, d_time_idx)
        self.d_time_sin   = d_time_sin

        d_time_total = d_time_idx + 2 * d_time_sin

        self.player_token_proj = None
        self.player_gru = nn.GRU(
            input_size=d_h,
            hidden_size=d_h,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.event_token_proj = nn.Sequential(
            nn.Linear(d_edge + d_time_total, d_h),
            nn.ReLU(inplace=True),
        )
        self.event_gru = nn.GRU(
            input_size=d_h,
            hidden_size=d_h,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        if self.attn_pool:
            self.attn = nn.Linear(d_h, 1)

        self._cached_layout: Optional[Dict[str, List[int]]] = None
        self._cached_cols_len: int = -1
        self.d_h = d_h
        self.d_pid = d_pid
        self.d_team = d_team
        self.d_time_total = d_time_total

    def _extract_ts(self, ctx_dyns: torch.Tensor, ctx_edges: torch.Tensor,
                    dyn_cols: List[str], edge_cols: Optional[List[str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, _ = ctx_dyns.shape
        if "seconds" in dyn_cols:
            idx = list(dyn_cols).index("seconds")
            ts_val = ctx_dyns[:, :, idx]
        else:
            ts_val = torch.arange(C, device=ctx_dyns.device).view(1, C).expand(B, C).float()
        ts_idx = torch.arange(C, device=ctx_dyns.device).view(1, C).expand(B, C).long()

        ts0 = ts_val[:, :1]
        denom = (ts_val[:, -1:] - ts0).clamp_min(1e-6)
        ts_rel = (ts_val - ts0) / denom
        ts_abs = (ts_val / 5400.0).clamp(0, 1)
        return ts_idx, ts_rel, ts_abs

    def _sinusoidal_time_encoding(self, ts_val: torch.Tensor, dim: int) -> torch.Tensor:
        B, C = ts_val.shape
        pe = ts_val.new_zeros((B, C, dim))
        pos = ts_val.unsqueeze(-1)
        div = torch.exp(torch.arange(0, dim, 2, device=ts_val.device, dtype=ts_val.dtype)
                        * (-math.log(10000.0) / dim))
        pe[..., 0::2] = torch.sin(pos * div)
        pe[..., 1::2] = torch.cos(pos * div)
        return pe


    def _build_layout_from_cols(self, cols: List[str], player_names: List[str]) -> Dict[str, List[int]]:
        name2idx = {str(c): i for i, c in enumerate(cols)}
        layout: Dict[str, List[int]] = {}
        for pid in player_names:
            if pid.startswith("PAD_"):
                layout[pid] = []
                continue

            idxs = []
            for k in self.wanted_keys:
                col = f"{pid}_{k}"
                if col in name2idx:
                    idxs.append(name2idx[col])
            layout[pid] = idxs
        return layout

    def _gather_per_player_dyn(self, ctx_dyns: torch.Tensor, cols: List[str],
                               player_names_batch: List[List[List[str]]]) -> Tuple[torch.Tensor, int]:
        B, C, _ = ctx_dyns.shape
        A = max(len(frame) for sample in player_names_batch for frame in sample)

        if self._cached_layout is None or self._cached_cols_len != len(cols):
            flat_players = [p for sample in player_names_batch for frame in sample for p in frame]
            self._cached_layout = self._build_layout_from_cols(cols, flat_players)
            self._cached_cols_len = len(cols)

        max_len = max(
            (len(self._cached_layout.get(p, []))
             for sample in player_names_batch
             for frame in sample
             for p in frame),
            default=0
        )

        out = ctx_dyns.new_zeros((B, C, A, max_len))
        for b in range(B):
            for c in range(C):
                names = player_names_batch[b][c]
                for j, pid in enumerate(names):
                    idxs = self._cached_layout.get(pid, [])
                    if not idxs:
                        continue
                    sel = ctx_dyns[b, c, idxs]
                    out[b, c, j, :len(idxs)] = sel
        return out, max_len
    def forward(self, ctx_dyns, ctx_edges, ctx_labels_22, *, cols, edge_cols=None, player_names=None, return_team_labels=False, return_decay=True):
        assert ctx_dyns.dim() == 3 and ctx_edges.dim() == 3
        assert ctx_dyns.size(0) == ctx_edges.size(0) and ctx_dyns.size(1) == ctx_edges.size(1)

        if isinstance(ctx_labels_22, list):
            if len(ctx_labels_22) != ctx_dyns.size(0):
                if len(ctx_labels_22) == 1 and isinstance(ctx_labels_22[0], list):
                    ctx_labels_22 = ctx_labels_22[0]
                elif len(ctx_labels_22) == ctx_dyns.size(1):
                    ctx_labels_22 = [ctx_labels_22 for _ in range(ctx_dyns.size(0))]
                elif isinstance(ctx_labels_22[0], np.ndarray) and ctx_labels_22[0].ndim == 1 and len(ctx_labels_22[0]) == self.n_players:
                    ctx_labels_22 = [[frame for frame in ctx_labels_22]]

        B, C, _ = ctx_dyns.shape

        if player_names is None:
            player_names_batch: List[List[List[str]]] = []
            for b in range(B):
                frames = []
                if b >= len(ctx_labels_22):
                    labels_bc = ctx_labels_22[-1]
                else:
                    labels_bc = ctx_labels_22[b]

                for c in range(len(labels_bc)):
                    row = labels_bc[c]
                    if isinstance(row, np.ndarray):
                        if row.ndim == 1:
                            names_22 = [str(x) for x in row.tolist()]
                        elif row.ndim == 0:
                            names_22 = [str(row.item())]
                        else:
                            names_22 = [str(x) for x in row.reshape(-1).tolist()[:self.n_players]]
                    elif isinstance(row, (list, tuple)):
                        names_22 = [str(x) for x in row]
                    elif isinstance(row, str):
                        names_22 = [row]
                    else:
                        names_22 = [str(row)]
                    names_22 = names_22[:self.n_players]
                    if len(names_22) < self.n_players:
                        names_22 = names_22 + ["PAD_h"] * (self.n_players - len(names_22))
                    frames.append(names_22 + ["ball"])
                player_names_batch.append(frames)
        else:
            frames = [player_names for _ in range(C)]
            player_names_batch = [frames for _ in range(B)]

        A = max(len(frame) for sample in player_names_batch for frame in sample)

        cols_list = list(map(str, cols))
        edge_cols_list = list(map(str, edge_cols)) if edge_cols is not None else None

        ts_idx, ts_rel, ts_abs = self._extract_ts(ctx_dyns, ctx_edges, cols_list, edge_cols_list)
        time_emb = torch.cat([
            self.time_idx_emb(ts_idx),
            self._sinusoidal_time_encoding(ts_rel, self.d_time_sin),
            self._sinusoidal_time_encoding(ts_abs, self.d_time_sin),
        ], dim=-1)

        per_dyn, Dp = self._gather_per_player_dyn(ctx_dyns, cols_list, player_names_batch)

        ids = torch.zeros(B, C, A, dtype=torch.long, device=ctx_dyns.device)
        for b in range(B):
            for c in range(C):
                names = player_names_batch[b][c]
                row = torch.tensor([self.name2id.get(str(x), 0) for x in names],
                                dtype=torch.long, device=ctx_dyns.device)
                if row.numel() < A:
                    pad = torch.zeros(A - row.numel(), dtype=torch.long, device=ctx_dyns.device)
                    row = torch.cat([row, pad], dim=0)
                ids[b, c] = row

        player_emb = self.player_emb(ids)

        team_ids = torch.zeros_like(ids)
        team_ids[(ids >= 1) & (ids <= self.max_side_id)] = 1
        team_ids[(ids > self.max_side_id) & (ids <= 2 * self.max_side_id)] = 2
        team_emb = self.team_emb(team_ids)

        team_last = team_ids[:, -1, :self.n_players]
        team_labels = (team_last == 2).float()

        time_emb_expanded = time_emb.unsqueeze(2).expand(B, C, A, -1)
        player_token = torch.cat([per_dyn, player_emb, team_emb, time_emb_expanded], dim=-1)

        if self.player_token_proj is None:
            input_dim = player_token.size(-1)
            self.player_token_proj = nn.Sequential(
                nn.Linear(input_dim, self.d_h),
                nn.ReLU(inplace=True),
            ).to(player_token.device)

        player_token = self.player_token_proj(player_token)
        player_seq = player_token.view(B * A, C, self.d_h)
        h_players, _ = self.player_gru(player_seq)
        h_players = h_players.reshape(B, C, A, self.d_h)

        event_token = torch.cat([ctx_edges, time_emb], dim=-1)
        event_token = self.event_token_proj(event_token)
        event_emb, _ = self.event_gru(event_token)

        players_all = h_players[:, :, :self.n_players, :]
        ball_all = h_players[:, :, self.ball_idx, :]


        decay = torch.exp(-0.1 * torch.arange(C, device=h_players.device, dtype=torch.float))
        decay = decay / decay.sum()
        decay = decay.view(1, C, 1, 1)

        players_decay = (players_all * decay).sum(dim=1)
        ball_decay = (ball_all * decay.squeeze(-1)).sum(dim=1)

        if self.attn_pool:
            alpha = torch.softmax(self.attn(players_decay).squeeze(-1), dim=1)
            h_player = (players_decay * alpha.unsqueeze(-1)).sum(dim=1)
        else:
            h_player = players_decay

        if return_team_labels:
            return h_player, ball_decay, team_labels, players_all, ball_all, event_emb
        else:
            return h_player, ball_decay, players_all, ball_all, event_emb

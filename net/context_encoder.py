import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class ContextEncoder(nn.Module):
    """
    输入:
      - ctx_dyns : [B, C, F_dyn]
      - ctx_edges: [B, C, F_edge]
      - cols     : dyn列名
      - edge_cols: edge列名
      - ctx_labels_22: List[B]，其中每个元素是 [C, 22] 的帧级标签
    输出:
      - h_player / h_all : player states
      - ball_decay : ball state
      - event_emb : event embedding [B,C,d_h]
    """

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

        # Embeddings
        self.player_emb   = nn.Embedding(2 * max_side_id + 2, d_pid, padding_idx=0)
        self.team_emb     = nn.Embedding(3, d_team, padding_idx=0)
        self.time_idx_emb = nn.Embedding(max_C, d_time_idx)
        self.d_time_sin   = d_time_sin
        
        # 计算 time_emb 总维度
        d_time_total = d_time_idx + 2 * d_time_sin  # idx + rel_sin + abs_sin

        # ===== Player 分支 =====
        # 注意：Dp 在运行时才知道，这里先占位，实际在 forward 里动态计算
        # 所以我们用一个 LazyLinear 或者在第一次 forward 时初始化
        # 为了简单，这里假设你知道 Dp，或者我们改用动态方式
        self.player_token_proj = None  # 延迟初始化
        self.player_gru = nn.GRU(
            input_size=d_h,
            hidden_size=d_h,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ===== Event 分支 =====
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

        # Attention pooling (for player)
        if self.attn_pool:
            self.attn = nn.Linear(d_h, 1)

        # 运行时缓存
        self._cached_layout: Optional[Dict[str, List[int]]] = None
        self._cached_cols_len: int = -1
        self.d_h = d_h
        self.d_pid = d_pid
        self.d_team = d_team
        self.d_time_total = d_time_total

    # ---------- time ----------
    def _extract_ts(self, ctx_dyns: torch.Tensor, ctx_edges: torch.Tensor,
                    dyn_cols: List[str], edge_cols: Optional[List[str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, _ = ctx_dyns.shape
        if "seconds" in dyn_cols:
            idx = list(dyn_cols).index("seconds")
            ts_val = ctx_dyns[:, :, idx]  # [B,C]
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

    # ---------- dyn gather ----------
    # def _build_layout_from_cols(self, cols: List[str], player_names: List[str]) -> Dict[str, List[int]]:
    #     name2idx = {str(c): i for i, c in enumerate(cols)}
    #     layout: Dict[str, List[int]] = {}
    #     for pid in player_names:
    #         idxs, missing = [], []
    #         for k in self.wanted_keys:
    #             col = f"{pid}_{k}"
    #             if col in name2idx:
    #                 idxs.append(name2idx[col])
    #             else:
    #                 missing.append(col)
    #         if missing:
    #             print(f"[WARN] {pid} 缺失列: {missing}")
    #         layout[pid] = idxs
    #     return layout

    def _build_layout_from_cols(self, cols: List[str], player_names: List[str]) -> Dict[str, List[int]]:
        name2idx = {str(c): i for i, c in enumerate(cols)}
        layout: Dict[str, List[int]] = {}
        for pid in player_names:
            #  跳过 PAD_xxx，不打印警告
            if pid.startswith("PAD_"):
                layout[pid] = []
                continue

            idxs, missing = [], []
            for k in self.wanted_keys:
                col = f"{pid}_{k}"
                if col in name2idx:
                    idxs.append(name2idx[col])
                else:
                    missing.append(col)
            if missing:
                print(f"[WARN] {pid} 缺失列: {missing}")
            layout[pid] = idxs
        return layout

    def _gather_per_player_dyn(self, ctx_dyns: torch.Tensor, cols: List[str],
                               player_names_batch: List[List[List[str]]]) -> Tuple[torch.Tensor, int]:
        """
        ctx_dyns: [B,C,F_dyn]
        player_names_batch: [B][C][A]
        return: out [B,C,A,Dp], Dp
        """
        B, C, _ = ctx_dyns.shape
        # 计算 A
        A = max(len(frame) for sample in player_names_batch for frame in sample)

        # 缓存 layout
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
                names = player_names_batch[b][c]  # [<=A]
                for j, pid in enumerate(names):
                    idxs = self._cached_layout.get(pid, [])
                    if not idxs:
                        continue
                    sel = ctx_dyns[b, c, idxs]  # [len(idxs)]
                    out[b, c, j, :len(idxs)] = sel
        return out, max_len
    def forward(self, ctx_dyns, ctx_edges, ctx_labels_22, *, cols, edge_cols=None, player_names=None, return_team_labels=False, return_decay=True):
        assert ctx_dyns.dim() == 3 and ctx_edges.dim() == 3
        assert ctx_dyns.size(0) == ctx_edges.size(0) and ctx_dyns.size(1) == ctx_edges.size(1)
        
        # 处理 ctx_labels_22 的各种格式
        if isinstance(ctx_labels_22, list):
            # case1: batch 正常 [B][C][22]
            if len(ctx_labels_22) == ctx_dyns.size(0):
                pass
            # case2: 被包成 [[C][22]]，只有1层
            elif len(ctx_labels_22) == 1 and isinstance(ctx_labels_22[0], list):
                ctx_labels_22 = ctx_labels_22[0]
                print("[INFO][ContextEncoder] Unwrapped ctx_labels_22 from extra list level.")
            # case3: 丢失 batch 维（只有C层），复制B次
            elif len(ctx_labels_22) == ctx_dyns.size(1):
                ctx_labels_22 = [ctx_labels_22 for _ in range(ctx_dyns.size(0))]
                print(f"[FIX] Expanded ctx_labels_22 to batch size {ctx_dyns.size(0)}")
            elif isinstance(ctx_labels_22[0], np.ndarray) and ctx_labels_22[0].ndim == 1 and len(ctx_labels_22[0]) == self.n_players:
                # 当前每个 batch 只有一帧 [n_players]，包装成 [C=1,n_players]
                ctx_labels_22 = [[frame for frame in ctx_labels_22]]
                print("[FIX] Wrapped ctx_labels_22 into time dimension (C=1).")

        B, C, _ = ctx_dyns.shape

        # player_names_batch
        if player_names is None:
            player_names_batch: List[List[List[str]]] = []
            for b in range(B):
                frames = []
                if b >= len(ctx_labels_22):
                    print(f"[WARN][ContextEncoder] Batch mismatch detected! "
                        f"len(ctx_labels_22)={len(ctx_labels_22)}, expected B={B}, diff={B - len(ctx_labels_22)}")
                    for bi, l in enumerate(ctx_labels_22):
                        if isinstance(l, list):
                            print(f"  ctx_labels_22[{bi}] -> len={len(l)} type={type(l[0])}")
                        else:
                            print(f"  ctx_labels_22[{bi}] -> type={type(l)}")
                    labels_bc = ctx_labels_22[-1]
                else:
                    labels_bc = ctx_labels_22[b]

                if b == 0 and (not isinstance(labels_bc[0][0], (str, np.str_, np.string_))):
                    print("[WARN] ctx_labels_22[0][0] type:", type(labels_bc[0][0]))
                
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
        if A != self.A:
            print(f"[WARN] A={A} != configured self.A={self.A}. Using A={A}.")

        cols_list = list(map(str, cols))
        edge_cols_list = list(map(str, edge_cols)) if edge_cols is not None else None

        # Time Embedding
        ts_idx, ts_rel, ts_abs = self._extract_ts(ctx_dyns, ctx_edges, cols_list, edge_cols_list)
        time_emb = torch.cat([
            self.time_idx_emb(ts_idx),
            self._sinusoidal_time_encoding(ts_rel, self.d_time_sin),
            self._sinusoidal_time_encoding(ts_abs, self.d_time_sin),
        ], dim=-1)  # [B, C, d_time_total]

        # Per-player Dynamics
        per_dyn, Dp = self._gather_per_player_dyn(ctx_dyns, cols_list, player_names_batch)  # [B, C, A, Dp]

        # Player & Team IDs 
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
        
        player_emb = self.player_emb(ids)  # [B, C, A, d_pid]
        
        team_ids = torch.zeros_like(ids)
        team_ids[(ids >= 1) & (ids <= self.max_side_id)] = 1
        team_ids[(ids > self.max_side_id) & (ids <= 2 * self.max_side_id)] = 2
        team_emb = self.team_emb(team_ids)  # [B, C, A, d_team]
        
        team_last = team_ids[:, -1, :self.n_players]  # [B, n_players]
        team_labels = (team_last == 2).float()  # [B, n_players] 0=home/offense, 1=away/defense

        # Player (no edge)
        time_emb_expanded = time_emb.unsqueeze(2).expand(B, C, A, -1)  # [B, C, A, d_time_total]
        player_token = torch.cat([per_dyn, player_emb, team_emb, time_emb_expanded], dim=-1)  # [B, C, A, Dp+d_pid+d_team+d_time]
        
        if self.player_token_proj is None:
            input_dim = player_token.size(-1)
            self.player_token_proj = nn.Sequential(
                nn.Linear(input_dim, self.d_h),
                nn.ReLU(inplace=True),
            ).to(player_token.device)
            print(f"[INFO] Initialized player_token_proj with input_dim={input_dim}")
        
        player_token = self.player_token_proj(player_token)  # [B, C, A, d_h]
        player_seq = player_token.view(B * A, C, self.d_h)  # [B*A, C, d_h]
        h_players, _ = self.player_gru(player_seq)  # [B*A, C, d_h]
        h_players = h_players.reshape(B, C, A, self.d_h)  # [B, C, A, d_h]

        # Event 分支（edge + time)
        event_token = torch.cat([ctx_edges, time_emb], dim=-1)  # [B, C, d_edge + d_time_total]
        event_token = self.event_token_proj(event_token)  # [B, C, d_h]
        event_emb, _ = self.event_gru(event_token)  # [B, C, d_h]

        # Player States pooling
        players_all = h_players[:, :, :self.n_players, :]  # [B, C, n_players, d_h]
        ball_all = h_players[:, :, self.ball_idx, :]        # [B, C, d_h]

  
        # ✅ 统一计算池化版本（所有 event head 都需要）
        decay = torch.exp(-0.1 * torch.arange(C, device=h_players.device, dtype=torch.float))
        decay = decay / decay.sum()
        decay = decay.view(1, C, 1, 1)

        players_decay = (players_all * decay).sum(dim=1)  # [B, n_players, d_h]
        ball_decay = (ball_all * decay.squeeze(-1)).sum(dim=1)  # [B, d_h]

        if self.attn_pool:
            alpha = torch.softmax(self.attn(players_decay).squeeze(-1), dim=1)
            h_player = (players_decay * alpha.unsqueeze(-1)).sum(dim=1)
        else:
            h_player = players_decay

        # ✅ 统一返回格式：池化版本 + 完整序列（供 TGN 使用）
        if return_team_labels:
            return h_player, ball_decay, team_labels, players_all, ball_all, event_emb
        else:
            return h_player, ball_decay, players_all, ball_all, event_emb
        
# ------------------------- 测试脚本 -------------------------
# if __name__ == "__main__":
#     import torch
#     import numpy as np
    
#     print("=" * 80)
#     print("ContextEncoder 测试脚本")
#     print("=" * 80)
    
#     # ========== 1. 加载数据 ==========
#     pt_path = "data/processed/dataloader/train/example_batch.pt"
#     print(f"\n[1] 加载数据文件: {pt_path}")
#     data = torch.load(pt_path, map_location="cpu", weights_only=False)
    
#     print(f"    - 数据类型: {type(data)}")
    
#     first_key = list(data.keys())[0]
#     print(f"    - 使用第一个 key: {first_key}")
#     sample = data[first_key]


#     # ========== 3. 详细检查每个字段 ==========
#     print(f"\n[3] 详细检查各个字段:")

#     # data 本身就是字典，直接用
#     sample = data  # ← 加这行

#     # 检查是否包含预期的字段
#     required_fields = ["ctx_dyns", "ctx_edges", "dyn_cols", "ctx_labels_22"]  # ← 删除 edge_cols
#     missing_fields = [f for f in required_fields if f not in data]  # ← 改成 data
#     # 3.1 ctx_dyns
#     if "ctx_dyns" in sample:
#         ctx_dyns = sample["ctx_dyns"]
#         print(f"\n    [ctx_dyns] 上下文动态特征")
#         print(f"      - 类型: {type(ctx_dyns)}")
#         print(f"      - 形状: {ctx_dyns.shape}  # 期望 [C, F_dyn]")
#         print(f"      - 数据类型: {ctx_dyns.dtype}")
#         print(f"      - 值范围: [{ctx_dyns.min().item():.3f}, {ctx_dyns.max().item():.3f}]")
#         if ctx_dyns.dim() == 2:
#             print(f"      - 前3个特征的前2帧:\n{ctx_dyns[:2, :3]}")
#         elif ctx_dyns.dim() == 3:
#             print(f"      - 注意：已经是批次格式 [B, C, F_dyn]")
#             print(f"      - 前3个特征的前2帧:\n{ctx_dyns[0, :2, :3]}")
#             ctx_dyns = ctx_dyns[0]  # 取第一个样本
    
#     # 3.2 ctx_edges
#     if "ctx_edges" in sample:
#         ctx_edges = sample["ctx_edges"]
#         print(f"\n    [ctx_edges] 上下文边特征")
#         print(f"      - 类型: {type(ctx_edges)}")
#         print(f"      - 形状: {ctx_edges.shape}  # 期望 [C, F_edge]")
#         print(f"      - 数据类型: {ctx_edges.dtype}")
#         print(f"      - 值范围: [{ctx_edges.min().item():.3f}, {ctx_edges.max().item():.3f}]")
#         if ctx_edges.dim() == 2:
#             print(f"      - 前2帧:\n{ctx_edges[:2]}")
#         elif ctx_edges.dim() == 3:
#             print(f"      - 注意：已经是批次格式 [B, C, F_edge]")
#             print(f"      - 前2帧:\n{ctx_edges[0, :2]}")
#             ctx_edges = ctx_edges[0]  # 取第一个样本
    
#     # 3.3 dyn_cols
#     if "dyn_cols" in sample:
#         dyn_cols = sample["dyn_cols"]
#         print(f"\n    [dyn_cols] 动态特征列名")
#         print(f"      - 类型: {type(dyn_cols)}")
#         print(f"      - 长度: {len(dyn_cols)}")
#         print(f"      - 前10个列名: {dyn_cols[:10]}")
#         print(f"      - 后10个列名: {dyn_cols[-10:]}")
    
#     # 3.4 edge_cols
#     if "edge_cols" in sample:
#         edge_cols = sample["edge_cols"]
#         print(f"\n    [edge_cols] 边特征列名")
#         print(f"      - 类型: {type(edge_cols)}")
#         print(f"      - 长度: {len(edge_cols)}")
#         print(f"      - 所有列名: {list(edge_cols)}")
    
#     # 3.5 ctx_labels_22
#     if "ctx_labels_22" in sample:
#         ctx_labels_22 = sample["ctx_labels_22"]
#         print(f"\n    [ctx_labels_22] 上下文球员标签")
#         print(f"      - 类型: {type(ctx_labels_22)}")
        
#         if isinstance(ctx_labels_22, torch.Tensor) or isinstance(ctx_labels_22, np.ndarray):
#             print(f"      - 形状: {ctx_labels_22.shape}")
#             print(f"      - 数据类型: {ctx_labels_22.dtype}")
            
#             # 根据维度判断结构
#             if ctx_labels_22.ndim == 2:
#                 print(f"      - 格式: [C, 22]")
#                 print(f"      - 第1帧的球员标签: {ctx_labels_22[0]}")
#                 print(f"      - 第2帧的球员标签: {ctx_labels_22[1]}")
#                 print(f"      - 最后一帧的球员标签: {ctx_labels_22[-1]}")
                
#                 # 统计 home/away 分布
#                 first_frame = ctx_labels_22[0]
#                 home_count = sum(1 for p in first_frame if str(p).startswith('h'))
#                 away_count = sum(1 for p in first_frame if str(p).startswith('a'))
#                 print(f"      - 第1帧统计: home={home_count}, away={away_count}")
            
#             elif ctx_labels_22.ndim == 3:
#                 print(f"      - 格式: [B, C, 22] (已经是批次格式)")
#                 print(f"      - 第1个样本第1帧: {ctx_labels_22[0, 0]}")
#                 ctx_labels_22 = ctx_labels_22[0]  # 取第一个样本
        
#         elif isinstance(ctx_labels_22, list):
#             print(f"      - 这是一个列表，长度: {len(ctx_labels_22)}")
#             print(f"      - 第一个元素类型: {type(ctx_labels_22[0])}")
#             if len(ctx_labels_22) > 0:
#                 print(f"      - 第一个元素: {ctx_labels_22[0]}")
    
#     # 3.6 检查其他可能的字段
#     print(f"\n    [其他字段] 检查样本中的其他数据:")
#     for key in sample.keys():
#         if key not in ['ctx_dyns', 'ctx_edges', 'dyn_cols', 'edge_cols', 'ctx_labels_22']:
#             val = sample[key]
#             if isinstance(val, torch.Tensor):
#                 print(f"      - {key}: shape={val.shape}, dtype={val.dtype}")
#             elif isinstance(val, np.ndarray):
#                 print(f"      - {key}: shape={val.shape}, dtype={val.dtype}")
#             elif isinstance(val, list):
#                 print(f"      - {key}: type=list, len={len(val)}")
#             else:
#                 print(f"      - {key}: type={type(val)}, value={val}")
    
#     # ========== 4. 验证必需字段 ==========
#     if not all(f in sample for f in required_fields):
#         print(f"\n⚠️  缺少必需字段，无法继续测试模型")
#         print(f"需要的字段: {required_fields}")
#         print(f"实际字段: {list(sample.keys())}")
#         exit(1)
    
#     # ========== 5. 创建模型 ==========
#     print(f"\n[4] 创建 ContextEncoder 模型:")
    
#     # 准备批次数据
#     if ctx_dyns.dim() == 2:
#         ctx_dyns_batch = ctx_dyns.unsqueeze(0)      # [1, C, F_dyn]
#     else:
#         ctx_dyns_batch = ctx_dyns  # 已经是批次格式
    
#     if ctx_edges.dim() == 2:
#         ctx_edges_batch = ctx_edges.unsqueeze(0)    # [1, C, F_edge]
#     else:
#         ctx_edges_batch = ctx_edges  # 已经是批次格式
    
#     # 处理 ctx_labels_22
#     if isinstance(ctx_labels_22, (torch.Tensor, np.ndarray)):
#         if ctx_labels_22.ndim == 2:
#             ctx_labels_22_batch = [ctx_labels_22]  # 包装成列表
#         else:
#             ctx_labels_22_batch = list(ctx_labels_22)  # 已经是批次格式
#     else:
#         ctx_labels_22_batch = ctx_labels_22
    
#     print(f"    - ctx_dyns_batch: {ctx_dyns_batch.shape}")
#     print(f"    - ctx_edges_batch: {ctx_edges_batch.shape}")
#     print(f"    - ctx_labels_22_batch: len={len(ctx_labels_22_batch)}")
#     if isinstance(ctx_labels_22_batch[0], (torch.Tensor, np.ndarray)):
#         print(f"      first element shape: {ctx_labels_22_batch[0].shape}")
    
#     model = ContextEncoder(
#         d_dyn=ctx_dyns_batch.size(-1),
#         d_edge=ctx_edges_batch.size(-1),
#         d_h=64,
#         num_layers=2,
#         bidirectional=False,
#         dropout=0.1,
#         n_home=11,
#         n_away=11,
#         attn_pool=False,
#         d_pid=16,
#         d_team=8,
#         d_time_idx=16,
#         d_time_sin=16,
#         max_C=256
#     )
    
#     print(f"    - 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
#     # ========== 6. 前向传播测试 ==========
#     print(f"\n[5] 前向传播测试:")
    
#     try:
#         with torch.no_grad():
#             # 测试1: 基础输出
#             print(f"\n    [测试1] 基础输出 (return_team_labels=False)")
#             h_player, ball_decay, event_emb = model(
#                 ctx_dyns_batch, ctx_edges_batch, ctx_labels_22_batch,
#                 cols=dyn_cols, edge_cols=edge_cols,
#                 return_team_labels=False
#             )
#             print(f"      ✅ - h_player shape: {h_player.shape}  # [B, 22, d_h]")
#             print(f"      ✅ - ball_decay shape: {ball_decay.shape}  # [B, d_h]")
#             print(f"      ✅ - event_emb shape: {event_emb.shape}  # [B, C, d_h]")
#             print(f"      - h_player 值范围: [{h_player.min().item():.3f}, {h_player.max().item():.3f}]")
#             print(f"      - ball_decay 值范围: [{ball_decay.min().item():.3f}, {ball_decay.max().item():.3f}]")
#             print(f"      - event_emb 值范围: [{event_emb.min().item():.3f}, {event_emb.max().item():.3f}]")
            
#             # 测试2: 带 team_labels
#             print(f"\n    [测试2] 带队伍标签 (return_team_labels=True)")
#             h_player2, ball_decay2, team_labels, event_emb2 = model(
#                 ctx_dyns_batch, ctx_edges_batch, ctx_labels_22_batch,
#                 cols=dyn_cols, edge_cols=edge_cols,
#                 return_team_labels=True
#             )
#             print(f"      ✅ - h_player shape: {h_player2.shape}")
#             print(f"      ✅ - ball_decay shape: {ball_decay2.shape}")
#             print(f"      ✅ - team_labels shape: {team_labels.shape}  # [B, 22]")
#             print(f"      ✅ - event_emb shape: {event_emb2.shape}")
#             print(f"      - team_labels 内容: {team_labels[0]}")
#             print(f"      - team_labels 统计: home={(team_labels[0] == 0).sum().item()}, away={(team_labels[0] == 1).sum().item()}")
        
#         # ========== 7. 可视化部分输出 ==========
#         print(f"\n[6] 可视化部分输出:")
        
#         print(f"\n    [球员特征] 前3个球员的特征 (前5维):")
#         for i in range(3):
#             print(f"      - 球员 {i}: {h_player[0, i, :5]}")
        
#         print(f"\n    [球特征] 前10维:")
#         print(f"      {ball_decay[0, :10]}")
        
#         print(f"\n    [Event 特征] 前3帧 (前5维):")
#         for i in range(min(3, event_emb.shape[1])):
#             print(f"      - 帧 {i}: {event_emb[0, i, :5]}")
        
#         print(f"\n{'=' * 80}")
#         print("测试完成！✅")
#         print(f"{'=' * 80}")
    
#     except Exception as e:
#         print(f"\n❌ 前向传播出错: {e}")
#         import traceback
#         traceback.print_exc()
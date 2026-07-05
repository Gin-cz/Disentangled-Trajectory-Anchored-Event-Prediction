# -*- coding: utf-8 -*-
"""
gin6 预处理改为：
- 按 shard 切分保存（避免单个 4GB .pt）
- 训练时懒加载 shard（MatchDataset 缓存最近一个 shard）
- 新增：跳过已存在比赛，便于断点续跑
"""

import os
import argparse
import re
from glob import glob
from typing import List, Tuple, Iterator, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import time
from torch.utils.data import Dataset, get_worker_info
from typing import Optional, Dict
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")

# ================================ 基础工具 ===================================

TOL = 0.3

def _is_pad_label(x: str) -> bool:
    return isinstance(x, str) and x.startswith("PAD_")

def _valid_22_nonpad(lst) -> bool:
    """长度=22 且不含 PAD"""
    if isinstance(lst, np.ndarray):
        arr = lst.tolist()
    else:
        arr = lst
    return isinstance(arr, (list, tuple)) and len(arr) == 22 \
           and not any(_is_pad_label(str(z)) for z in arr)

def id_to_label(pid: int):
    """gin3 的 p2_id 编码 => 'hX'/'aY'；0/1000 表示无目标"""
    if pid in (0, 1000): return None   # 解决target out of range问题
    if 1 <= pid <= 23:   return f"h{pid}"
    if 51 <= pid <= 73:  return f"a{pid-50}"
    return None

def align_by_time_strict(seq_times, graph_times, tol=TOL):
    """严格一一匹配（以少的数组为基准）"""
    seq_times = np.asarray(seq_times, dtype=float)
    graph_times = np.asarray(graph_times, dtype=float)

    if len(seq_times) <= len(graph_times):
        base, other = seq_times, graph_times
        base_name = "seq"
    else:
        base, other = graph_times, seq_times
        base_name = "graph"

    pairs = []
    used_other = set()
    for i, t in enumerate(base):
        j = np.argmin(np.abs(other - t))
        dt = abs(other[j] - t)
        if dt <= tol and j not in used_other:
            if base_name == "seq":
                pairs.append((i, j))
            else:
                pairs.append((j, i))
            used_other.add(j)
    return pairs

def _infer_mid_from_mldir(mldir: str) -> str:
    base = os.path.basename(mldir)  # e.g. ml_3894529_1026512
    assert base.startswith("ml_"), f"ml_dir 名称异常: {base}"
    return base[3:]  # strip 'ml_'

# ============================= 原子数据集（单场） ==============================

class SoccerSeqGraphDataset(Dataset):
    """单场比赛：负责把 sequence.npz 与 ml_dir 内的图数据对齐；提供 aligned_pairs 等原始材料"""
    def __init__(self, seq_npz_path, ml_dir, tol=TOL):
        # --- sequence ---
        seq = np.load(seq_npz_path, allow_pickle=True)
        self.X_all     = seq["X_all"]               # (N_seq, 84, 23, 6)
        self.Y_all     = seq["Y_all"]               # (N_seq, 84, 23, 6)
        self.labels_22 = seq["all_labels_22"]       # (N_seq, 22)
        self.seq_times = seq["window_start_event"]  # (N_seq,)

        # --- graph (gin3输出, 现在是 npz 带列名) ---
        edge_npz = np.load(os.path.join(ml_dir, "edges_raw_2d.npz"), allow_pickle=True)
        self.edge_raw = edge_npz["values"]          # (N_graph, feat_dim)
        # columns 可能是 numpy 数组（dtype=object），转成干净的 list[str]
        _edge_cols = edge_npz["columns"]
        self.edge_cols = [str(x) for x in (_edge_cols.tolist() if hasattr(_edge_cols, "tolist") else list(_edge_cols))]

        dyn_npz = np.load(os.path.join(ml_dir, "nodes_dynamic_2d.npz"), allow_pickle=True)
        self.node_dyn = dyn_npz["values"]           # (N_graph, dyn_dim)
        _dyn_cols = dyn_npz["columns"]
        self.dyn_cols = np.array([str(x) for x in (_dyn_cols.tolist() if hasattr(_dyn_cols, "tolist") else list(_dyn_cols))], dtype=object)

        # graph 时间轴来自 CSV
        csv_name = [f for f in os.listdir(ml_dir) if f.endswith(".csv")]
        assert len(csv_name) >= 1, f"{ml_dir} 未找到 CSV"
        df = pd.read_csv(os.path.join(ml_dir, csv_name[0]))
        if "adjusted_seconds" in df.columns:
            self.graph_times = df["adjusted_seconds"].to_numpy(dtype=float)
        elif "ts" in df.columns:
            self.graph_times = df["ts"].to_numpy(dtype=float)
        else:
            raise KeyError(f"CSV 没有时间列, columns={list(df.columns)}")
        # 事件位置（这里沿用 ha_ 的聚合点）
        if all(x in df.columns for x in ["ha_x","ha_y"]):
            self.event_position = df[["ha_x","ha_y"]].to_numpy(dtype=float)  # (N_graph, 2)
        else:
            # 若缺失，退化为 0
            self.event_position = np.zeros((len(self.graph_times), 2), dtype=float)

        # u: p1_id（动作主体）
        if "u" in df.columns:
            self.action_pid = df["u"].to_numpy(dtype=int)
        else:
            # 兼容性退化
            self.action_pid = np.zeros((len(self.graph_times),), dtype=int)

        # --- 时间对齐 ---
        self.aligned_pairs = align_by_time_strict(self.seq_times, self.graph_times, tol=tol)

        # trackable id 映射
        mid = _infer_mid_from_mldir(ml_dir)
        SYN_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(ml_dir)), "syn_data")
        aligned_csv_path = os.path.join(SYN_DATA_DIR, f"{mid}_aligned_df.csv")
        aligned_df = pd.read_csv(aligned_csv_path)

        self.label_to_trackable = {}
        for side in ['h', 'a']:
            for num in range(1, 24):
                col = f"{side}{num}_trackable_id"
                if col in aligned_df.columns:
                    vals = aligned_df[col].dropna()
                    vals = vals[vals != 0]
                    if len(vals) > 0:
                        self.label_to_trackable[f"{side}{num}"] = int(vals.iloc[0])

        def _labels22_to_tids(labels22, include_ball=True):
            labs = labels22.tolist() if not isinstance(labels22, list) else labels22
            tids = [int(self.label_to_trackable.get(str(lab), -1)) for lab in labs]
            if include_ball:
                tids.append(55)  # 球固定用 id=55
            return np.asarray(tids, dtype=np.int64)
        self._labels22_to_tids = _labels22_to_tids

        # ---------- 事件 onehot：自适应 7 类 / 15 类 ----------
        expected7 = ["OP_OP","OP_BR","OP_G","DP_DB","OP_BC","DP_BTPM","Other"]

        # 判定是否已是 7类：列数≥10 且 3~9 列名顺序等于 expected7
        is_already_7 = (len(self.edge_cols) >= 10 and self.edge_cols[3:3+7] == expected7)

        if is_already_7:
            # 直接使用 7 类
            onehot7 = self.edge_raw[:, 3:3+7].astype(np.float32)         # [N,7]
            self.edge_raw7 = self.edge_raw.astype(np.float32)            # [N,10]: u,i,ts + 7onehot
            self.event_id_7 = np.argmax(onehot7, axis=1).astype(np.int64)
            self.event7_names = expected7
        else:
            # 老 15 类 -> 7 类
            onehot15 = self.edge_raw[:, 3:].astype(np.float32)
            K = onehot15.shape[1]
            if K < 15:
                onehot15 = np.pad(onehot15, ((0,0),(0,15-K)), mode="constant")
            elif K > 15:
                onehot15 = onehot15[:, :15]

            IDX = {
                "OP_OP":0,"OP_BR":1,"OP_DP":2,"OP_B":3,"OP_G":4,"OP_C":5,
                "DP_B":6,"DP_T":7,"B_T":8,"B_G":9,"DP_DB":10,"DP_P":11,"DP_M":12,"DP_F":13,"Other":14
            }
            T = onehot15.shape[0]
            eid = np.full((T,), 6, dtype=np.int64)  # 默认 Other=6
            eid[onehot15[:, IDX["OP_OP"]] == 1] = 0
            eid[onehot15[:, IDX["OP_BR"]] == 1] = 1
            eid[onehot15[:, IDX["OP_G"]]  == 1] = 2
            eid[onehot15[:, IDX["DP_DB"]] == 1] = 3
            mask_OP_BC = (onehot15[:, IDX["OP_B"]] == 1) | (onehot15[:, IDX["OP_C"]] == 1)
            eid[mask_OP_BC] = 4
            mask_DP_grp = (onehot15[:, IDX["DP_B"]] == 1) | (onehot15[:, IDX["DP_T"]] == 1) | \
                          (onehot15[:, IDX["DP_P"]] == 1) | (onehot15[:, IDX["DP_M"]] == 1)
            eid[mask_DP_grp] = 5

            onehot7 = np.eye(7, dtype=np.float32)[eid]                 # [N,7]
            meta = self.edge_raw[:, :3]                                # [N,3]  p1_id, p2_id, ts
            self.edge_raw7  = np.concatenate([meta, onehot7], axis=1).astype(np.float32)  # [N,10]
            self.event_id_7 = eid.astype(np.int64)
            self.event7_names = expected7

    def __len__(self): return len(self.aligned_pairs)

# ============================= 辅助函数 ======================================

_MID_RE = re.compile(r'(\d+)_(\d+)_sequence\.npz$')

def discover_match_pairs(seqs_dir: str, ml_root: str):
    match_pairs = []
    for npz_path in glob(os.path.join(seqs_dir, "*_sequence.npz")):
        m = _MID_RE.search(os.path.basename(npz_path))
        if not m:
            continue
        mid = f"{m.group(1)}_{m.group(2)}"
        ml_dir = os.path.join(ml_root, f"ml_{mid}")
        need = [
            os.path.join(ml_dir, f"{mid}.csv"),
            os.path.join(ml_dir, "nodes_dynamic_2d.npz"),
            os.path.join(ml_dir, "edges_raw_2d.npz"),
            os.path.join(ml_dir, "target_players.npy"),
        ]
        if os.path.isdir(ml_dir) and all(os.path.exists(p) for p in need):
            match_pairs.append((npz_path, ml_dir))
    return match_pairs


def split_pairs_by_match(pairs, train_ratio=0.7, val_ratio=0.15, seed=2025):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]
    return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx], [pairs[i] for i in test_idx]

# ============================= 窗口生成器 ====================================
# ============================= 窗口生成器 ====================================
def iter_windows_from_subset(
    sub: SoccerSeqGraphDataset,
    C: int = 10,
    H: int = 3,
    y_frames=None,
    stats: Optional[Dict[str, int]] = None,
) -> Iterator[Dict]:
    """
    若传入 stats，将在其中累计：
      stats = {
        'total_centers': 0,
        'skip_ctx_pad': 0,
        'skip_fut_pad': 0,
        'skip_u_invalid': 0,
        'skip_fut_idx_neg': 0,
        'skip_label_not_in_ctx': 0,
        'skip_epos_nan': 0,        # ← 新增：fut 第一帧 ha_x/ha_y 含 NaN
        'kept': 0,
      }
    """
    local = {
        'total_centers': 0,
        'skip_ctx_pad': 0,
        'skip_fut_pad': 0,
        'skip_u_invalid': 0,
        'skip_fut_idx_neg': 0,
        'skip_label_not_in_ctx': 0,
        'skip_epos_nan': 0,        # ← 新增
        'kept': 0,
    }


    def _bump(key): 
        local[key] += 1

    pairs_sorted = sorted(sub.aligned_pairs, key=lambda x: x[1])
    for center in range(C-1, len(pairs_sorted)-H):
        local['total_centers'] += 1

        ctx_pairs = pairs_sorted[center-C+1 : center+1]
        fut_pairs = pairs_sorted[center+1 : center+H+1]

        # ---- 0) ctx_last 必须是 22 且无 PAD ----
        i_ctx_last = ctx_pairs[-1][0]
        ctx_last_labels = sub.labels_22[i_ctx_last]
        if not _valid_22_nonpad(ctx_last_labels):
            _bump('skip_ctx_pad'); 
            continue
        ctx_last_labels_set = set(map(str, (ctx_last_labels.tolist() if not isinstance(ctx_last_labels, list) else ctx_last_labels)))

        # ---- 1) fut 的前 need_k 帧必须全是 22 且无 PAD，且 ha_x/ha_y 都非 NaN ----
        need_k = min(3, len(fut_pairs))
        ok = True
        nan_flag = False
        for k in range(need_k):
            i_k, j_k = fut_pairs[k]
            fut_k_labels = sub.labels_22[i_k]
            if not _valid_22_nonpad(fut_k_labels):
                ok = False
                break
            # 新增：事件位置 NaN 检查
            pos_k = sub.event_position[j_k]  # numpy array shape (2,)
            if np.isnan(pos_k).any():
                ok = False
                nan_flag = True
                break

        if not ok:
            if nan_flag:
                _bump('skip_epos_nan')
            else:
                _bump('skip_fut_pad')
            continue


        # ---- 2) 严格事件目标校验（仅检查 fut 第一帧）----
        i0, j0 = fut_pairs[0]
        fut_first_labels = sub.labels_22[i0]
        fut_first_labels_list = (fut_first_labels.tolist() 
                                 if not isinstance(fut_first_labels, list) else fut_first_labels)
        fut_first_labels_list = list(map(str, fut_first_labels_list))

        fut_pid0 = int(sub.action_pid[j0])              # u（动作主体）
        fut_tlabel0 = id_to_label(fut_pid0)             # hX/aY 或 None
        if fut_tlabel0 is None:
            _bump('skip_u_invalid'); 
            continue

        # fut 第一帧目标在 fut_first22 里必须能找到索引
        try:
            fut_tindex0 = fut_first_labels_list.index(fut_tlabel0)
        except ValueError:
            _bump('skip_fut_idx_neg'); 
            continue

        # fut 第一帧的目标 label 必须出现在 ctx_last 的 22 人里
        if fut_tlabel0 not in ctx_last_labels_set:
            _bump('skip_label_not_in_ctx'); 
            continue

        # ================== 通过所有过滤，正式构建 window ==================
        # ----------------- 上下文 -----------------
        ctx_dyns, ctx_edges, ctx_seqX = [], [], []
        ctx_traj_trackable_ids = []
        ctx_time, ctx_events, ctx_players, ctx_positions = [], [], [], []
        ctx_labels_22 = []

        for i, j in ctx_pairs:
            ctx_dyns.append(torch.tensor(sub.node_dyn[j], dtype=torch.float32))
            ctx_edges.append(torch.tensor(sub.edge_raw7[j], dtype=torch.float32))
            ctx_time.append(torch.tensor(sub.graph_times[j], dtype=torch.float32))
            ctx_events.append(torch.tensor(sub.event_id_7[j], dtype=torch.long))

            if j+1 < len(sub.event_position):
                ctx_positions.append(torch.tensor(sub.event_position[j+1], dtype=torch.float32))
            else:
                ctx_positions.append(torch.zeros(2, dtype=torch.float32))

            pid = int(sub.action_pid[j])
            tlabel = id_to_label(pid)
            tindex = -1
            lbls = sub.labels_22[i].tolist() if not isinstance(sub.labels_22[i], list) else sub.labels_22[i]
            lbls = list(map(str, lbls))
            if tlabel is not None and tlabel in lbls:
                try:
                    tindex = lbls.index(tlabel)
                except ValueError:
                    tindex = -1
            ctx_players.append(torch.tensor(tindex, dtype=torch.long))

            tids_23 = sub._labels22_to_tids(sub.labels_22[i], include_ball=True)
            ctx_traj_trackable_ids.append(torch.tensor(tids_23, dtype=torch.long))

            ctx_labels_22.append(sub.labels_22[i])

        ctx_seqX = torch.tensor(sub.X_all[ctx_pairs[-1][0]], dtype=torch.float32)

        # ----------------- 未来 -----------------
        fut_seqX = []
        fut_traj_trackable_ids = []
        fut_times, fut_events, fut_players, fut_positions = [], [], [], []
        fut_next_times, fut_next_events, fut_next_players, fut_next_positions = [], [], [], []
        fut_labels_22 = []

        for i, j in fut_pairs:
            evt_id = int(sub.event_id_7[j])
            fut_events.append(torch.tensor(evt_id, dtype=torch.long))

            fut_pid = int(sub.action_pid[j])
            fut_tlabel = id_to_label(fut_pid)
            fut_tindex = -1
            fut_lbls = sub.labels_22[i].tolist() if not isinstance(sub.labels_22[i], list) else sub.labels_22[i]
            fut_lbls = list(map(str, fut_lbls))
            if fut_tlabel is not None and fut_tlabel in fut_lbls:
                try:
                    fut_tindex = fut_lbls.index(fut_tlabel)
                except ValueError:
                    fut_tindex = -1
            fut_players.append(torch.tensor(fut_tindex, dtype=torch.long))

            if j + 1 < len(sub.event_id_7):
                fut_next_events.append(torch.tensor(int(sub.event_id_7[j+1]), dtype=torch.long))
            else:
                fut_next_events.append(torch.tensor(-1, dtype=torch.long))

            if j + 1 < len(sub.action_pid):
                fut_next_pid = int(sub.action_pid[j+1])
            else:
                fut_next_pid = 0
            fut_next_tlabel = id_to_label(fut_next_pid)
            fut_next_tindex = -1
            if fut_next_tlabel is not None and fut_next_tlabel in fut_lbls:
                try:
                    fut_next_tindex = fut_lbls.index(fut_next_tlabel)
                except ValueError:
                    fut_next_tindex = -1
            fut_next_players.append(torch.tensor(fut_next_tindex, dtype=torch.long))

            if j + 1 < len(sub.graph_times):
                fut_next_times.append(torch.tensor(sub.graph_times[j+1], dtype=torch.float32))
            else:
                fut_next_times.append(torch.tensor(-1.0, dtype=torch.float32))

            if j + 1 < len(sub.event_position):
                fut_next_positions.append(torch.tensor(sub.event_position[j+1], dtype=torch.float32))
            else:
                fut_next_positions.append(torch.zeros(2, dtype=torch.float32))

            fut_tids_23 = sub._labels22_to_tids(sub.labels_22[i], include_ball=True)
            fut_traj_trackable_ids.append(torch.tensor(fut_tids_23, dtype=torch.long))

            fut_times.append(torch.tensor(sub.graph_times[j], dtype=torch.float32))
            fut_positions.append(torch.tensor(sub.event_position[j], dtype=torch.float32))

            x = sub.X_all[i]
            if y_frames is not None:
                x = x[:y_frames]
            fut_seqX.append(torch.tensor(x, dtype=torch.float32))

            fut_labels_22.append(sub.labels_22[i])

        local['kept'] += 1

        yield dict(
            ctx_dyns=torch.stack(ctx_dyns, 0),
            ctx_edges=torch.stack(ctx_edges, 0),
            ctx_seqX=ctx_seqX,
            ctx_traj_trackable_ids=torch.stack(ctx_traj_trackable_ids, 0),
            ctx_time=torch.stack(ctx_time, 0),
            ctx_events=torch.stack(ctx_events, 0),
            ctx_players=torch.stack(ctx_players, 0),
            ctx_positions=torch.stack(ctx_positions, 0),
            ctx_labels_22=ctx_labels_22,

            fut_seqX=torch.stack(fut_seqX, 0),
            fut_traj_trackable_ids=torch.stack(fut_traj_trackable_ids, 0),
            fut_times=torch.stack(fut_times, 0),
            fut_events=torch.stack(fut_events, 0),
            fut_players=torch.stack(fut_players, 0),
            fut_positions=torch.stack(fut_positions, 0),
            fut_next_times=torch.stack(fut_next_times, 0),
            fut_next_events=torch.stack(fut_next_events, 0),
            fut_next_players=torch.stack(fut_next_players, 0),
            fut_next_positions=torch.stack(fut_next_positions, 0),
            fut_labels_22=fut_labels_22,

            dyn_cols=sub.dyn_cols,
            edge_cols=sub.edge_cols,
        )

    # 把本地统计累加到外部 stats
    if stats is not None:
        for k, v in local.items():
            stats[k] = stats.get(k, 0) + v


# ============================= collate 函数 ==================================

def window_collate(batch):
    out = {}
    for k in ["ctx_dyns","ctx_edges","ctx_seqX","fut_seqX"]:
        out[k] = torch.stack([b[k] for b in batch], 0)

    out["ctx_time"] = torch.stack([b["ctx_time"] for b in batch], 0)
    out["ctx_events"] = torch.stack([b["ctx_events"] for b in batch], 0)
    out["ctx_players"] = torch.stack([b["ctx_players"] for b in batch], 0)
    out["ctx_positions"] = torch.stack([b["ctx_positions"] for b in batch], 0)

    out["fut_times"] = torch.stack([b["fut_times"] for b in batch], 0)
    out["fut_events"] = torch.stack([b["fut_events"] for b in batch], 0)
    out["fut_players"] = torch.stack([b["fut_players"] for b in batch], 0)
    out["fut_positions"] = torch.stack([b["fut_positions"] for b in batch], 0)

    out["fut_next_times"] = torch.stack([b["fut_next_times"] for b in batch], 0)
    out["fut_next_events"] = torch.stack([b["fut_next_events"] for b in batch], 0)
    out["fut_next_players"] = torch.stack([b["fut_next_players"] for b in batch], 0)
    out["fut_next_positions"] = torch.stack([b["fut_next_positions"] for b in batch], 0)

    out["ctx_traj_trackable_ids"] = torch.stack([b["ctx_traj_trackable_ids"] for b in batch], 0)
    out["fut_traj_trackable_ids"] = torch.stack([b["fut_traj_trackable_ids"] for b in batch], 0)

    # labels_22 是字符串 list，这里保持原样
    out["ctx_labels_22"] = [b["ctx_labels_22"] for b in batch]
    out["fut_labels_22"] = [b["fut_labels_22"] for b in batch]

    # 严格检查嵌套层级，只有在「最外层是 list 且 内部全是 list of np.array(22)」时才 flatten
    if len(out["ctx_labels_22"]) == 1 and all(
        isinstance(x, (list, np.ndarray)) for x in out["ctx_labels_22"][0]
    ):
        print("[FIX] ctx_labels_22 was nested once; flattening.")
        out["ctx_labels_22"] = out["ctx_labels_22"][0]
    else:
        pass
        # print(f"[DEBUG] ctx_labels_22 shape kept: {len(out['ctx_labels_22'])}×{len(out['ctx_labels_22'][0])}")

    #  dyn 列名（用第一个样本的）
    out["dyn_cols"] = batch[0].get("dyn_cols", None)

    return out
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _make_batches_with_wrap(all_wins, batch_size: int):
    """
    输入: all_wins = [window_dict, ...]
    输出: [ [window_dict × batch_size], ... ]  最后一组不足则从头部循环借满
    """
    N = len(all_wins)
    if N == 0:
        return []
    batches = []
    i = 0
    while i < N:
        j = min(i + batch_size, N)
        chunk = all_wins[i:j]
        if len(chunk) < batch_size:
            need = batch_size - len(chunk)
            # 从开头循环借满
            chunk = chunk + all_wins[:need]
        batches.append(chunk)
        i += batch_size
    return batches

def preprocess_and_save_skip_existing_permatch(
    pairs, out_dir, split, C=10, H=3, y_frames=None, batch_size=64, tol=TOL
):
    _ensure_dir(out_dir)

    # 全局统计
    global_stats = {
        'total_centers': 0, 'skip_ctx_pad': 0, 'skip_fut_pad': 0,
        'skip_u_invalid': 0, 'skip_fut_idx_neg': 0, 'skip_label_not_in_ctx': 0,
        'skip_epos_nan': 0,
        'kept': 0,
    }

    for (seq_path, ml_dir) in pairs:
        mid = os.path.basename(seq_path).replace("_sequence.npz", "")
        match_dir = os.path.join(out_dir, mid)
        _ensure_dir(match_dir)

        # existing_pt = [f for f in os.listdir(match_dir) if f.endswith(".pt")]
        # if len(existing_pt) > 0:
        #     print(f"⏩ skip {mid}, already has {len(existing_pt)} file(s) in {match_dir}")
        #     continue

        print(f"[{split}] building (per-match batches): {mid}")
        sub = SoccerSeqGraphDataset(seq_path, ml_dir, tol=tol)

        # 本场统计
        match_stats = {}

        # 收集本场所有window (同时累计统计)
        all_wins = []
        for w in iter_windows_from_subset(sub, C=C, H=H, y_frames=y_frames, stats=match_stats):
            all_wins.append(w)

        # 打印本场丢弃比例
        tc = match_stats.get('total_centers', 0)
        kp = match_stats.get('kept', 0)
        if tc > 0:
            def pct(x): return f"{(x/tc)*100:.1f}%"
            print(f"  [stats] centers={tc}, kept={kp} ({pct(kp)}) | "
                  f"ctx_pad={match_stats.get('skip_ctx_pad',0)} ({pct(match_stats.get('skip_ctx_pad',0))}), "
                  f"fut_pad={match_stats.get('skip_fut_pad',0)} ({pct(match_stats.get('skip_fut_pad',0))}), "
                  f"u_invalid={match_stats.get('skip_u_invalid',0)} ({pct(match_stats.get('skip_u_invalid',0))}), "
                  f"fut_idx_neg={match_stats.get('skip_fut_idx_neg',0)} ({pct(match_stats.get('skip_fut_idx_neg',0))}), "
                  f"label_not_in_ctx={match_stats.get('skip_label_not_in_ctx',0)} ({pct(match_stats.get('skip_label_not_in_ctx',0))})")
        else:
            print("  [stats] centers=0 (no windows after alignment)")

        # 累计到全局
        for k, v in match_stats.items():
            global_stats[k] = global_stats.get(k, 0) + v

        if len(all_wins) == 0:
            print(f"⚠️ {mid} has 0 valid windows after filtering, skip.")
            continue

        batches = _make_batches_with_wrap(all_wins, batch_size=batch_size)

        for bi, win_list in enumerate(batches):
            batch_dict = window_collate(win_list)
            save_path = os.path.join(match_dir, f"{split}_{mid}_batch{bi:04d}.pt")
            torch.save(batch_dict, save_path)
            print(f"  saved {save_path}  (batch_size={batch_size})")

        print(f"✅ done {mid}, windows={len(all_wins)}, batches={len(batches)}, out={match_dir}")

    # 全局汇总打印
    TC = global_stats.get('total_centers', 0)
    KP = global_stats.get('kept', 0)
    if TC > 0:
        def P(x): return f"{(x/TC)*100:.1f}%"
        print("\n====== GLOBAL FILTER STATS ======")
        print(f"centers={TC}, kept={KP} ({P(KP)})")
        print(f" - ctx_pad:           {global_stats['skip_ctx_pad']} ({P(global_stats['skip_ctx_pad'])})")
        print(f" - fut_pad(need_k):   {global_stats['skip_fut_pad']} ({P(global_stats['skip_fut_pad'])})")
        print(f" - epos_nan(need_k):  {global_stats['skip_epos_nan']} ({P(global_stats['skip_epos_nan'])})")  # ← 放到这里
        print(f" - u_invalid(0/1000): {global_stats['skip_u_invalid']} ({P(global_stats['skip_u_invalid'])})")
        print(f" - fut_idx_neg:       {global_stats['skip_fut_idx_neg']} ({P(global_stats['skip_fut_idx_neg'])})")
        print(f" - label_not_in_ctx:  {global_stats['skip_label_not_in_ctx']} ({P(global_stats['skip_label_not_in_ctx'])})")
        print("=================================\n")
    else:
        print("\n====== GLOBAL FILTER STATS ======\ncenters=0\n=================================\n")


# ============================== 训练用 Dataset ================================

# class MatchDataset(Dataset):
#     """训练用：懒加载 shard"""
#     def __init__(self, shard_files: List[str]):
#         self.shard_files = sorted(shard_files)
#         self.index: List[Tuple[int,int]] = []
#         self.shard_lengths: List[int] = []
#         for fi, f in enumerate(self.shard_files):
#             data = torch.load(f, map_location="cpu", weights_only=True)
#             n = len(data)
#             self.shard_lengths.append(n)
#             for wi in range(n):
#                 self.index.append((fi, wi))
#         print(f"[MatchDataset] shards={len(self.shard_files)}, total windows={len(self.index)}")
#         self._cache_file, self._cache_data = "", []

#     def __len__(self): return len(self.index)
#     def __getitem__(self, idx):
#         fi, wi = self.index[idx]
#         fpath = self.shard_files[fi]
#         if self._cache_file != fpath:
#             self._cache_data = torch.load(fpath, map_location="cpu", weights_only=True)
#             self._cache_file = fpath
#         return self._cache_data[wi]
# import time
# import os
# from typing import List, Tuple
# from torch.utils.data import Dataset
# import torch
# from torch.utils.data import get_worker_info
class MatchDataset(Dataset):
    """
    每个 .pt 文件是一个“已 collate 的 batch 字典”（固定 64）。
    __getitem__ 返回整个 batch 字典；是否裁切由 collate_fn 决定。
    """
    def __init__(self, paths: List[str], *, recursive: bool = True, debug: bool = False, rank: int = 0):
        self._debug = bool(debug and rank == 0)
        pt_files = []
        for p in paths:
            if os.path.isdir(p):
                walker = os.walk(p) if recursive else [(p, [], os.listdir(p))]
                for root, _, files in walker:
                    for f in files:
                        if f.endswith(".pt"):
                            pt_files.append(os.path.join(root, f))
            elif os.path.isfile(p) and p.endswith(".pt"):
                pt_files.append(p)
        self.pt_files = sorted(pt_files)
        if self._debug:
            print(f"[MatchDataset] total pt files={len(self.pt_files)}")
        if not self.pt_files:
            raise RuntimeError("No .pt files found under given paths.")

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        fpath = self.pt_files[idx]
        batch = torch.load(fpath, map_location="cpu", weights_only=False)
        return batch  # 整包（64）
    
def slice_batch_dict(batch: Dict, n: int, *, random_pick: bool = False) -> Dict:
    ref_key = next((k for k, v in batch.items() if torch.is_tensor(v)), None)
    if ref_key is None:
        return batch
    B = batch[ref_key].shape[0]
    if n >= B:
        return batch

    idx = torch.randperm(B)[:n] if random_pick else torch.arange(n)

    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.shape[0] == B:
            out[k] = v.index_select(0, idx)
        elif isinstance(v, list) and len(v) == B:
            # 用相同的 idx 对 list 取样
            out[k] = [v[i] for i in idx.tolist()]
        else:
            out[k] = v
    return out

def make_adaptive_collate(target_bs: int, random_pick: bool = False):
    def _collate(batch_list):
        # batch_list 长度应为1（每次取一个.pt文件）
        batch = batch_list[0]
        return slice_batch_dict(batch, target_bs, random_pick=random_pick)
    return _collate

# ============================== 主入口 =======================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build DisTraj-Event dataloader shards from sequence and graph features.")
    parser.add_argument("--seqs_dir", required=True, help="Directory containing *_sequence.npz files.")
    parser.add_argument("--ml_root", required=True, help="Directory containing per-match ml_* feature folders.")
    parser.add_argument("--output_dir", required=True, help="Output dataloader root with train/val/test subdirectories.")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    pairs = discover_match_pairs(args.seqs_dir, args.ml_root)
    print("Total match pairs:", len(pairs))
    if not pairs:
        raise RuntimeError("No match pairs found. Check --seqs_dir and --ml_root.")

    train_pairs, val_pairs, test_pairs = split_pairs_by_match(
        pairs, args.train_ratio, args.val_ratio, args.seed
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for sub in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    preprocess_and_save_skip_existing_permatch(
        train_pairs, os.path.join(args.output_dir, "train"), split="train", batch_size=args.batch_size
    )
    preprocess_and_save_skip_existing_permatch(
        val_pairs, os.path.join(args.output_dir, "val"), split="val", batch_size=args.batch_size
    )
    preprocess_and_save_skip_existing_permatch(
        test_pairs, os.path.join(args.output_dir, "test"), split="test", batch_size=args.batch_size
    )

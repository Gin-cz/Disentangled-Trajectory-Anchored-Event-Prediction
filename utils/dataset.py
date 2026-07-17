"""Build sharded datasets and lazily load prepared DisTraj batches."""

import argparse
import os
import re
import warnings
from glob import glob
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")


TOL = 0.3

def _is_pad_label(x: str) -> bool:
    return isinstance(x, str) and x.startswith("PAD_")

def _valid_22_nonpad(lst) -> bool:
    if isinstance(lst, np.ndarray):
        arr = lst.tolist()
    else:
        arr = lst
    return isinstance(arr, (list, tuple)) and len(arr) == 22 \
           and not any(_is_pad_label(str(z)) for z in arr)

def id_to_label(pid: int):
    if pid in (0, 1000): return None
    if 1 <= pid <= 23:   return f"h{pid}"
    if 51 <= pid <= 73:  return f"a{pid-50}"
    return None

def align_by_time_strict(seq_times, graph_times, tol=TOL):
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
    base = os.path.basename(mldir)
    assert base.startswith("ml_"), f"Invalid ml_dir name: {base}"
    return base[3:]


class SoccerSeqGraphDataset(Dataset):
    def __init__(self, seq_npz_path, ml_dir, tol=TOL):
        seq = np.load(seq_npz_path, allow_pickle=True)
        self.X_all     = seq["X_all"]
        self.Y_all     = seq["Y_all"]
        self.labels_22 = seq["all_labels_22"]
        self.seq_times = seq["window_start_event"]

        edge_npz = np.load(os.path.join(ml_dir, "edges_raw_2d.npz"), allow_pickle=True)
        self.edge_raw = edge_npz["values"]
        _edge_cols = edge_npz["columns"]
        self.edge_cols = [str(x) for x in (_edge_cols.tolist() if hasattr(_edge_cols, "tolist") else list(_edge_cols))]

        dyn_npz = np.load(os.path.join(ml_dir, "nodes_dynamic_2d.npz"), allow_pickle=True)
        self.node_dyn = dyn_npz["values"]
        _dyn_cols = dyn_npz["columns"]
        self.dyn_cols = np.array([str(x) for x in (_dyn_cols.tolist() if hasattr(_dyn_cols, "tolist") else list(_dyn_cols))], dtype=object)

        csv_name = [f for f in os.listdir(ml_dir) if f.endswith(".csv")]
        assert len(csv_name) >= 1, f"No CSV file found in {ml_dir}"
        df = pd.read_csv(os.path.join(ml_dir, csv_name[0]))
        if "adjusted_seconds" in df.columns:
            self.graph_times = df["adjusted_seconds"].to_numpy(dtype=float)
        elif "ts" in df.columns:
            self.graph_times = df["ts"].to_numpy(dtype=float)
        else:
            raise KeyError(f"CSV has no time column; columns={list(df.columns)}")
        if all(x in df.columns for x in ["ha_x","ha_y"]):
            self.event_position = df[["ha_x","ha_y"]].to_numpy(dtype=float)
        else:
            self.event_position = np.zeros((len(self.graph_times), 2), dtype=float)

        if "u" in df.columns:
            self.action_pid = df["u"].to_numpy(dtype=int)
        else:
            self.action_pid = np.zeros((len(self.graph_times),), dtype=int)

        self.aligned_pairs = align_by_time_strict(self.seq_times, self.graph_times, tol=tol)

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
                tids.append(55)
            return np.asarray(tids, dtype=np.int64)
        self._labels22_to_tids = _labels22_to_tids

        expected7 = ["OP_OP","OP_BR","OP_G","DP_DB","OP_BC","DP_BTPM","Other"]

        is_already_7 = (len(self.edge_cols) >= 10 and self.edge_cols[3:3+7] == expected7)

        if is_already_7:
            onehot7 = self.edge_raw[:, 3:3+7].astype(np.float32)
            self.edge_raw7 = self.edge_raw.astype(np.float32)
            self.event_id_7 = np.argmax(onehot7, axis=1).astype(np.int64)
            self.event7_names = expected7
        else:
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
            eid = np.full((T,), 6, dtype=np.int64)
            eid[onehot15[:, IDX["OP_OP"]] == 1] = 0
            eid[onehot15[:, IDX["OP_BR"]] == 1] = 1
            eid[onehot15[:, IDX["OP_G"]]  == 1] = 2
            eid[onehot15[:, IDX["DP_DB"]] == 1] = 3
            mask_OP_BC = (onehot15[:, IDX["OP_B"]] == 1) | (onehot15[:, IDX["OP_C"]] == 1)
            eid[mask_OP_BC] = 4
            mask_DP_grp = (onehot15[:, IDX["DP_B"]] == 1) | (onehot15[:, IDX["DP_T"]] == 1) | \
                          (onehot15[:, IDX["DP_P"]] == 1) | (onehot15[:, IDX["DP_M"]] == 1)
            eid[mask_DP_grp] = 5

            onehot7 = np.eye(7, dtype=np.float32)[eid]
            meta = self.edge_raw[:, :3]
            self.edge_raw7  = np.concatenate([meta, onehot7], axis=1).astype(np.float32)
            self.event_id_7 = eid.astype(np.int64)
            self.event7_names = expected7

    def __len__(self): return len(self.aligned_pairs)


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

def iter_windows_from_subset(
    sub: SoccerSeqGraphDataset,
    C: int = 10,
    H: int = 3,
    y_frames=None,
    stats: Optional[Dict[str, int]] = None,
) -> Iterator[Dict]:
    local = {
        'total_centers': 0,
        'skip_ctx_pad': 0,
        'skip_fut_pad': 0,
        'skip_u_invalid': 0,
        'skip_fut_idx_neg': 0,
        'skip_label_not_in_ctx': 0,
        'skip_epos_nan': 0,
        'kept': 0,
    }


    def _bump(key):
        local[key] += 1

    pairs_sorted = sorted(sub.aligned_pairs, key=lambda x: x[1])
    for center in range(C-1, len(pairs_sorted)-H):
        local['total_centers'] += 1

        ctx_pairs = pairs_sorted[center-C+1 : center+1]
        fut_pairs = pairs_sorted[center+1 : center+H+1]

        i_ctx_last = ctx_pairs[-1][0]
        ctx_last_labels = sub.labels_22[i_ctx_last]
        if not _valid_22_nonpad(ctx_last_labels):
            _bump('skip_ctx_pad')
            continue
        ctx_last_labels_set = set(map(str, (ctx_last_labels.tolist() if not isinstance(ctx_last_labels, list) else ctx_last_labels)))

        need_k = min(3, len(fut_pairs))
        ok = True
        nan_flag = False
        for k in range(need_k):
            i_k, j_k = fut_pairs[k]
            fut_k_labels = sub.labels_22[i_k]
            if not _valid_22_nonpad(fut_k_labels):
                ok = False
                break
            pos_k = sub.event_position[j_k]
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


        i0, j0 = fut_pairs[0]
        fut_first_labels = sub.labels_22[i0]
        fut_first_labels_list = (fut_first_labels.tolist()
                                 if not isinstance(fut_first_labels, list) else fut_first_labels)
        fut_first_labels_list = list(map(str, fut_first_labels_list))

        fut_pid0 = int(sub.action_pid[j0])
        fut_tlabel0 = id_to_label(fut_pid0)
        if fut_tlabel0 is None:
            _bump('skip_u_invalid')
            continue

        try:
            fut_tindex0 = fut_first_labels_list.index(fut_tlabel0)
        except ValueError:
            _bump('skip_fut_idx_neg')
            continue

        if fut_tlabel0 not in ctx_last_labels_set:
            _bump('skip_label_not_in_ctx')
            continue

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

    if stats is not None:
        for k, v in local.items():
            stats[k] = stats.get(k, 0) + v



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

    out["ctx_labels_22"] = [b["ctx_labels_22"] for b in batch]
    out["fut_labels_22"] = [b["fut_labels_22"] for b in batch]

    if len(out["ctx_labels_22"]) == 1 and all(
        isinstance(x, (list, np.ndarray)) for x in out["ctx_labels_22"][0]
    ):
        out["ctx_labels_22"] = out["ctx_labels_22"][0]

    out["dyn_cols"] = batch[0].get("dyn_cols", None)

    return out
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _make_batches_with_wrap(all_wins, batch_size: int):
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
            chunk = chunk + all_wins[:need]
        batches.append(chunk)
        i += batch_size
    return batches

def preprocess_and_save_skip_existing_permatch(
    pairs, out_dir, split, C=10, H=3, y_frames=None, batch_size=64, tol=TOL
):
    _ensure_dir(out_dir)

    for (seq_path, ml_dir) in pairs:
        mid = os.path.basename(seq_path).replace("_sequence.npz", "")
        match_dir = os.path.join(out_dir, mid)
        _ensure_dir(match_dir)

        sub = SoccerSeqGraphDataset(seq_path, ml_dir, tol=tol)
        all_wins = list(iter_windows_from_subset(sub, C=C, H=H, y_frames=y_frames))

        if len(all_wins) == 0:
            continue

        batches = _make_batches_with_wrap(all_wins, batch_size=batch_size)

        for bi, win_list in enumerate(batches):
            batch_dict = window_collate(win_list)
            save_path = os.path.join(match_dir, f"{split}_{mid}_batch{bi:04d}.pt")
            torch.save(batch_dict, save_path)

class MatchDataset(Dataset):
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
        if not self.pt_files:
            raise RuntimeError("No .pt files found under given paths.")

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        fpath = self.pt_files[idx]
        batch = torch.load(fpath, map_location="cpu", weights_only=False)
        return batch

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
            out[k] = [v[i] for i in idx.tolist()]
        else:
            out[k] = v
    return out

def make_adaptive_collate(target_bs: int, random_pick: bool = False):
    def _collate(batch_list):
        batch = batch_list[0]
        return slice_batch_dict(batch, target_bs, random_pick=random_pick)
    return _collate


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

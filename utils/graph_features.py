"""Build graph tensors from synchronized node and edge features."""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path


FOLDER_PATH = os.environ.get("DISTRAJ_NODE_EDGE_DIR", os.path.join("data", "interim", "syn_node_edge_data"))
BASE_ROOT = os.environ.get("DISTRAJ_ML_ROOT", os.path.join("data", "processed", "ml"))
ALIGNED_DIR = os.environ.get("DISTRAJ_ALIGNED_DIR", os.path.join("data", "interim", "syn_data_weighted"))

EVENT_COLS_15 = [
    "OP_OP", "OP_BR", "OP_DP", "OP_B", "OP_G", "OP_C",
    "DP_B", "DP_T", "B_T", "B_G",
    "DP_DB", "DP_P", "DP_M", "DP_F", "Other"
]

EVENT7_NAMES = ["OP_OP", "OP_BR", "OP_G", "DP_DB", "OP_BC", "DP_BTPM", "Other"]

EXCLUDE_DYNAMIC_SUFFIXES = ("_tl_dir_x", "_tl_dir_y", "_gl_dir_x", "_gl_dir_y")
EXCLUDE_DYNAMIC_EXACT = {"p2_dist"}

def adjust_timestamps(ts_list, epsilon=1e-6):
    out, prev = [], None
    for t in ts_list:
        t = float(t)
        if prev is not None and t <= prev:
            t = prev + epsilon
        out.append(t)
        prev = t
    return out

def id_to_label(id_num: int) -> str:
    if id_num in (0, 1000): return "0"
    if id_num < 50:         return f"h{id_num}"
    return f"a{id_num-50}"

def map_id_to_trackable(id_num: int, label_to_trackable: dict) -> int:
    return label_to_trackable.get(id_to_label(id_num), -1)

def get_label_to_trackable_map(aligned_csv_path: str) -> dict:
    if not os.path.exists(aligned_csv_path):
        return {}
    df_aligned = pd.read_csv(aligned_csv_path)
    m = {}
    for side in ['h','a']:
        for num in range(1,24):
            col = f"{side}{num}_trackable_id"
            if col in df_aligned.columns:
                vals = df_aligned[col].dropna()
                vals = vals[vals != 0]
                if len(vals) > 0:
                    m[f"{side}{num}"] = int(vals.iloc[0])
    return m


def preprocess(dataf: pd.DataFrame):
    dataf = dataf.copy()

    dataf["p1_id"] = dataf["p1_id"].fillna("1000").replace("", "1000").astype(str)
    dataf["p2_id"] = dataf["p2_id"].fillna("1000").replace("", "1000").astype(str)

    def _encode_pid(series: pd.Series) -> pd.Series:
        s = series.astype(str)
        out = pd.Series(np.zeros(len(s), dtype=np.int32))
        mask_0 = s.isin(["0", "1000"])
        out.loc[mask_0] = s.loc[mask_0].astype(int)
        mask_h = (~mask_0) & (s.str[0] == "h")
        mask_a = (~mask_0) & (s.str[0] == "a")
        out.loc[mask_h] = s.loc[mask_h].str[1:].astype(int)
        out.loc[mask_a] = s.loc[mask_a].str[1:].astype(int) + 50
        return out.astype(int)

    dataf["p1_enc"] = _encode_pid(dataf["p1_id"])
    dataf["p2_enc"] = _encode_pid(dataf["p2_id"])

    for c in ["seconds", "start_x", "start_y", "end_x", "end_y", "ha_x", "ha_y"]:
        if c in dataf.columns:
            dataf[c] = pd.to_numeric(dataf[c], errors="coerce")

    if "seconds" not in dataf.columns:
        raise ValueError("Input data must contain a seconds column")
    dataf["ts"] = adjust_timestamps(dataf["seconds"].to_numpy(dtype=float))

    dataf["td"] = pd.to_numeric(dataf.get("timedelta", 0.0), errors="coerce").fillna(0.0)

    if "event_type" not in dataf.columns:  dataf["event_type"]  = "unknown"
    if "event_type_2" not in dataf.columns: dataf["event_type_2"] = "unknown"

    for c in EVENT_COLS_15:
        if c not in dataf.columns:
            dataf[c] = 0
        dataf[c] = pd.to_numeric(dataf[c], errors="coerce").fillna(0).astype(np.float32)

    onehot15 = dataf[EVENT_COLS_15].to_numpy(dtype=np.float32)
    idx = {name: EVENT_COLS_15.index(name) for name in EVENT_COLS_15}
    T = onehot15.shape[0]
    eid = np.full((T,), 6, dtype=np.int64)

    eid[onehot15[:, idx["OP_OP"]] == 1] = 0
    eid[onehot15[:, idx["OP_BR"]] == 1] = 1
    eid[onehot15[:, idx["OP_G"]]  == 1] = 2
    eid[onehot15[:, idx["DP_DB"]] == 1] = 3

    mask_OP_BC = (onehot15[:, idx["OP_B"]] == 1) | (onehot15[:, idx["OP_C"]] == 1)
    eid[mask_OP_BC] = 4
    mask_DP_grp = (onehot15[:, idx["DP_B"]] == 1) | (onehot15[:, idx["DP_T"]] == 1) | \
                  (onehot15[:, idx["DP_P"]] == 1) | (onehot15[:, idx["DP_M"]] == 1)
    eid[mask_DP_grp] = 5

    onehot7 = np.eye(7, dtype=np.float32)[eid]

    u = dataf["p1_enc"].to_numpy(dtype=np.float32).reshape(-1,1)
    i = dataf["p2_enc"].to_numpy(dtype=np.float32).reshape(-1,1)
    ts = np.asarray(dataf["ts"], dtype=np.float32).reshape(-1,1)
    e_feat = np.concatenate([u, i, ts, onehot7], axis=1)

    edge_columns = np.array(["u", "i", "ts"] + EVENT7_NAMES, dtype=object)

    if "home_attack_dir" in dataf.columns:
        vals = dataf["home_attack_dir"].astype(str).str.lower().str.strip()
        dir_map = {
            "left to right": 1.0,
            "right to left": -1.0
        }
        dataf["home_attack_dir_num"] = vals.map(dir_map).fillna(0.0).astype(np.float32)
    else:
        dataf["home_attack_dir_num"] = 0.0


    drop_meta = {
        "p1_id","p2_id","p1_enc","p2_enc",
        "ts","td","timedelta",
        "event_type","event_type_2",
        *EVENT_COLS_15,
        "idx","relation_label","relation_label_str","ha_name"
    }

    numeric_cols_all = [c for c in dataf.columns if pd.api.types.is_numeric_dtype(dataf[c])]
    numeric_cols = []
    for c in numeric_cols_all:
        if c in drop_meta:
            continue
        if c in EXCLUDE_DYNAMIC_EXACT:
            continue
        if any(c.endswith(suf) for suf in EXCLUDE_DYNAMIC_SUFFIXES):
            continue
        numeric_cols.append(c)

    if "seconds" in dataf.columns and "seconds" not in numeric_cols:
        numeric_cols = ["seconds"] + numeric_cols

    n_feat = dataf[numeric_cols].to_numpy(dtype=np.float32)
    dyn_columns = np.array(numeric_cols, dtype=object)

    base_cols = [
        "p1_enc","p2_enc","td","ts",
        "event_type","event_type_2",
        "ha_x","ha_y","start_x","start_y"
    ]
    if "end_x" in dataf.columns and "end_y" in dataf.columns:
        base_cols += ["end_x","end_y"]

    df_small = dataf[base_cols].copy()
    df_small = df_small.rename(columns={
        "p1_enc":"u",
        "p2_enc":"i",
        "event_type":"label",
        "event_type_2":"label2"
    })
    df_small["idx"] = np.arange(len(df_small), dtype=int)

    return df_small, e_feat, n_feat, dataf, dyn_columns, edge_columns

def build_node_dynamic_tensor_sorted(df_small: pd.DataFrame, n_feat: np.ndarray, num_nodes: int):
    df_sorted = df_small.sort_values(["u","ts"]).reset_index(drop=True)
    sorted_idx = df_sorted.index.to_numpy()
    n_sorted = n_feat[sorted_idx]
    T, D = n_sorted.shape
    node_dyn = np.zeros((num_nodes, T, D), dtype=np.float32)
    for pos, row in df_sorted.iterrows():
        uid = int(row["u"])
        if 0 <= uid < num_nodes:
            node_dyn[uid, pos, :] = n_sorted[pos]
    return node_dyn, df_sorted

def build_edge_tensor_sorted(df_small: pd.DataFrame, e_feat: np.ndarray, num_nodes: int):
    df_sorted = df_small.sort_values(["u","ts"]).reset_index(drop=True)
    sorted_idx = df_sorted.index.to_numpy()
    e_sorted = e_feat[sorted_idx]
    T, D = e_sorted.shape
    edge_tensor = np.zeros((num_nodes, T, D), dtype=np.float32)
    for pos, row in df_sorted.iterrows():
        uid = int(row["u"])
        if 0 <= uid < num_nodes:
            edge_tensor[uid, pos, :] = e_sorted[pos]
    return edge_tensor, df_sorted

def run_updated(dataf: pd.DataFrame, data_name: str, label_to_trackable: dict):
    out_dir = os.path.join(BASE_ROOT, f"ml_{data_name}")
    os.makedirs(out_dir, exist_ok=True)
    OUT_DF                = os.path.join(out_dir, f"{data_name}.csv")
    OUT_EDGE_2D_NPZ       = os.path.join(out_dir, "edges_raw_2d.npz")
    OUT_NODE_DYN_2D_NPZ   = os.path.join(out_dir, "nodes_dynamic_2d.npz")
    OUT_NODE_FEAT         = os.path.join(out_dir, "nodes_static.npy")
    OUT_NODE_DYNAMIC_FULL = os.path.join(out_dir, "nodes_dynamic_3d.npz")
    OUT_EDGE_TENSOR_FULL  = os.path.join(out_dir, "edges_dynamic_3d.npz")
    OUT_TARGET_PLAYERS    = os.path.join(out_dir, "target_players.npy")

    df_small, e_feat, n_feat, df_full, dyn_colnames, edge_colnames = preprocess(dataf)

    df_small["u"] = df_small["u"].astype(int).replace(1000, 0)
    df_small["i"] = df_small["i"].astype(int).replace(1000, 0)

    event_type_set = sorted(df_small['label'].unique())
    event_type_to_id = {etype: idx for idx, etype in enumerate(event_type_set)}
    df_small['event_label'] = df_small['label'].map(event_type_to_id)

    num_nodes = int(max(df_small["u"].max(), df_small["i"].max())) + 1
    np.save(OUT_NODE_FEAT, np.zeros((num_nodes, 9), dtype=np.float32))

    np.savez(OUT_EDGE_2D_NPZ, values=e_feat.astype(np.float32), columns=edge_colnames)
    np.savez(OUT_NODE_DYN_2D_NPZ, values=n_feat.astype(np.float32), columns=dyn_colnames)

    node_dynamic_full, dynamic_df_sorted = build_node_dynamic_tensor_sorted(df_small, n_feat, num_nodes)
    np.savez(OUT_NODE_DYNAMIC_FULL,
             node_dynamic_adjusted=node_dynamic_full,
             dynamic_sources=dynamic_df_sorted["u"].to_numpy(),
             dynamic_destinations=dynamic_df_sorted["i"].to_numpy())

    edge_tensor_full, edge_df_sorted = build_edge_tensor_sorted(df_small, e_feat, num_nodes)
    np.savez(OUT_EDGE_TENSOR_FULL,
             edge_tensor_adjusted=edge_tensor_full,
             edge_sources=edge_df_sorted["u"].to_numpy(),
             edge_destinations=edge_df_sorted["i"].to_numpy())

    np.save(OUT_TARGET_PLAYERS, df_small["i"].astype(int).to_numpy())

    df_small['trackable_p1'] = df_small['u'].apply(lambda x: map_id_to_trackable(x, label_to_trackable))
    df_small['trackable_p2'] = df_small['i'].apply(lambda x: map_id_to_trackable(x, label_to_trackable))

    df_small.to_csv(OUT_DF, index=False)

    return {
        "event_type_map": event_type_to_id,
        "num_nodes": num_nodes,
        "df": df_small
    }

def build_graph_features():
    Path(BASE_ROOT).mkdir(parents=True, exist_ok=True)
    csv_files = glob.glob(os.path.join(FOLDER_PATH, "*_node_edges.csv"))
    csv_files.sort()

    for csv_path in csv_files:
        fname = os.path.basename(csv_path)
        try:
            name_part = fname.replace("_node_edges.csv", "")
            statsbomb_id, skillcorner_id = name_part.split("_")
        except ValueError:
            continue
        mid = f"{statsbomb_id}_{skillcorner_id}"

        aligned_path = os.path.join(ALIGNED_DIR, f"{mid}_aligned_df.csv")
        label_to_trackable = get_label_to_trackable_map(aligned_path)

        df = pd.read_csv(csv_path)
        run_updated(df, mid, label_to_trackable)


if __name__ == "__main__":
    build_graph_features()

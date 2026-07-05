import pandas as pd
import numpy as np
import sys
import os
import glob
import json

def _safe_float(v, default=0.0):
    """NaN/None/空字符串都当0,无法转就返回default。"""
    try:
        if v is None: return default
        if isinstance(v, str) and v.strip() == "": return default
        f = float(v)
        if np.isnan(f): return default
        return f
    except Exception:
        return default

def _present_xy(x, y, eps=1e-9):
    """是否在场:x、y任一非0 且不是NaN/空。"""
    return (abs(x) > eps) or (abs(y) > eps)

def pick_22_labels_from_row(row, num_players=23, sort_by_ball=True):
    """
    直接在当前tracking行(row)中挑在场球员：
      - home/away各最多11个,不足用 PAD_h / PAD_a 补齐
      - sort_by_ball=True 时,按离球距离从近到远选前11；否则按编号顺序
    返回长度22的标签列表['h*11, 'a*11 或 PAD_* 补齐]
    """
    bx = _safe_float(row.get("ball_x", 0.0))
    by = _safe_float(row.get("ball_y", 0.0))

    home, away = [], []
    for i in range(1, num_players+1):
        # home
        hx = _safe_float(row.get(f"h{i}_x", 0.0))
        hy = _safe_float(row.get(f"h{i}_y", 0.0))
        if _present_xy(hx, hy):
            d2 = (hx - bx)**2 + (hy - by)**2
            home.append((f"h{i}", d2))
        # away
        ax = _safe_float(row.get(f"a{i}_x", 0.0))
        ay = _safe_float(row.get(f"a{i}_y", 0.0))
        if _present_xy(ax, ay):
            d2 = (ax - bx)**2 + (ay - by)**2
            away.append((f"a{i}", d2))

    if sort_by_ball:
        home.sort(key=lambda x: x[1]); away.sort(key=lambda x: x[1])
        home_labels = [lab for lab,_ in home][:11]
        away_labels = [lab for lab,_ in away][:11]
    else:
        # 按编号顺序（已是按i循环）
        home_labels = [lab for lab,_ in home][:11]
        away_labels = [lab for lab,_ in away][:11]

    while len(home_labels) < 11: home_labels.append("PAD_h")
    while len(away_labels) < 11: away_labels.append("PAD_a")

    return home_labels + away_labels

def features_from_labels(df_seg, labels_22, k_near=5, ball_flag=1.0):
    """
    返回: (T, 23, 6) = T * [22球员+球] * [x,y,vx,vy,speed,is_topK_near_ball]
    """
    mats = []
    for _, row in df_seg.iterrows():
        # 取球坐标
        bx  = _safe_float(row.get("ball_x", 0.0))
        by  = _safe_float(row.get("ball_y", 0.0))

        # 先收集 22 名球员的 (x,y) 和是否在场，用于统一排序
        px, py, present = [], [], []
        for lab in labels_22:
            if lab.startswith(("h","a")):
                x = _safe_float(row.get(f"{lab}_x", 0.0))
                y = _safe_float(row.get(f"{lab}_y", 0.0))
                px.append(x); py.append(y)
                present.append(_present_xy(x, y))
            else:
                # PAD 位
                px.append(0.0); py.append(0.0)
                present.append(False)

        # 计算到球的距离；对缺席/PAD 置为 +inf，避免入选
        dists = np.hypot(np.array(px) - bx, np.array(py) - by)
        dists[~np.array(present, dtype=bool)] = np.inf

        # 选出前 k_near
        idx_sorted = np.argsort(dists)
        k = min(k_near, np.isfinite(dists).sum())  # 有效人数不足时自动减小
        topk_idx = set(idx_sorted[:k].tolist())

        # 逐人填特征（最后一维是 is_topK 标志）
        feats = []
        for j, lab in enumerate(labels_22):
            if lab.startswith(("h","a")) and present[j]:
                vx = _safe_float(row.get(f"{lab}_vx", 0.0))
                vy = _safe_float(row.get(f"{lab}_vy", 0.0))
                sp = float(np.hypot(vx, vy))
                is_near = 1.0 if j in topk_idx else 0.0
                feats.append([px[j], py[j], vx, vy, sp, is_near])
            else:
                # PAD 或当帧缺席：全零，标志=0
                feats.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # 球的特征：标志位固定0（只对球员做“最近5人”）
        bvx = _safe_float(row.get("ball_vx", 0.0))
        bvy = _safe_float(row.get("ball_vy", 0.0))
        bsd = float(np.hypot(bvx, bvy))
        feats.append([bx, by, bvx, bvy, bsd, ball_flag])

        mats.append(np.asarray(feats, dtype=float))

    return np.stack(mats, axis=0)  # (T, 23, 6)

def get_window_by_start_time(df_track, secs_arr, start_time, frames_needed, dt=0.1):
    idx = np.searchsorted(secs_arr, start_time, side='left')

    # ---- 1) 前补零 ----
    pad_front = 0
    if start_time < secs_arr[0]:
        pad_front = int(np.ceil((secs_arr[0] - start_time) / dt))

    # ---- 2) 正常取 ----
    slice_df = df_track.iloc[max(idx,0): idx + frames_needed - pad_front].copy()

    # ---- 3) 前补零拼接 ----
    if pad_front > 0:
        pad_df_front = pd.DataFrame(0, index=range(pad_front), columns=df_track.columns)
        pad_df_front["seconds"] = [start_time + i*dt for i in range(pad_front)]
        slice_df = pd.concat([pad_df_front, slice_df], ignore_index=True)

    # ---- 4) 后补零 ----
    if len(slice_df) < frames_needed:
        pad_back = frames_needed - len(slice_df)
        pad_df_back = pd.DataFrame(0, index=range(pad_back), columns=df_track.columns)
        last_t = float(slice_df["seconds"].iloc[-1]) if len(slice_df) > 0 else start_time
        pad_df_back["seconds"] = [last_t + (i+1)*dt for i in range(pad_back)]
        slice_df = pd.concat([slice_df, pad_df_back], ignore_index=True)

    return slice_df

def id_to_label(pid: int):
    """把 target_players.npy 里的 pid 映射成 'hX'/'aY' 标签"""
    if pid in (0, 1000): return None
    if 1 <= pid <= 23:   return f"h{pid}"
    if 51 <= pid <= 73:  return f"a{pid-50}"
    return None

aligned_dir = os.environ.get("DISTRAJ_ALIGNED_DIR", os.path.join("data", "interim", "syn_data_weighted"))
tracking_dir = os.environ.get("DISTRAJ_TRACKING_DIR", os.path.join("data", "interim", "tracking_for_sequence"))

id_mapping_path = os.environ.get("DISTRAJ_ID_MAPPING", os.path.join("data", "id_matching.csv"))
# 建立 dict：{statsbomb_id: skillcorner_id}
id_map_df = pd.read_csv(id_mapping_path)
id_map = dict(zip(id_map_df["match_id_statsbomb"].astype(str), id_map_df["match_id_skillcorner"].astype(str)))

csv_files = glob.glob(f"{aligned_dir}/*.csv")
for csv_file in csv_files:
    trajectories = []
    seg_start_times = []
    df_aligned = pd.read_csv(csv_file)
    statsbomb_id = df_aligned["match_id"].iloc[0]
    skillcorner_id = id_map.get(str(statsbomb_id), None)

    # adjusted_seconds 就是匹配的tracking时间，seconds_event 是原始事件时间  ## 按seconds_event 做排序比人为匹配的tracking会更好！
    df_aligned = df_aligned.sort_values("seconds_event").reset_index(drop=True)
    event_times = df_aligned["adjusted_seconds"].tolist()
    event_times_mapping = df_aligned["seconds_event"].tolist()

    df_tracking = pd.read_csv(f"{tracking_dir}/{skillcorner_id}_tracking_clean.csv")
    df_tracking = df_tracking.sort_values("seconds").reset_index(drop=True)
    df_tracking = df_tracking.fillna(0)  # 填充 NaN 为 0
    valid_rows = 0

    secs = df_tracking["seconds"].to_numpy()
    sequence_length = 84                   # 84 帧作为 X，后面再对齐出 Y（下一帧起）
    need_frames     = sequence_length + 1  # 为了能构造 X(84) 与 Y(84) 的逐帧对齐

    DT = 0.1
    PRE, POST = 20, 64
    X_windows, Y_windows, window_times, all_labels_22 = [], [], [], []
    window_start_event, window_start_tracking = [], []
    next_event_snapshots, event_in_window_flags = [], []

    for j in range(len(event_times)):
        t_evt = event_times[j]   # 当前事件时间 (事件对齐后的 adjusted_seconds)

        # --- 1) 截窗口: 事件前20帧 + 后64帧
        start_time = t_evt - PRE * DT
        window_df = get_window_by_start_time(df_tracking, secs,
                                            start_time=start_time,
                                            frames_needed=PRE+POST+1)

        # --- 2) 锁定 22 人
        labels_22 = pick_22_labels_from_row(window_df.iloc[0], num_players=23, sort_by_ball=True)
        all_labels_22.append(labels_22)

        # --- 3) 构造特征
        feat_full = features_from_labels(window_df, labels_22)   # (84,23,6)

        # --- 4) 切 X, Y
        X_w = feat_full[:84]
        Y_w = feat_full[1:85] if feat_full.shape[0] > 84 else feat_full[:84]
        X_windows.append(X_w)
        Y_windows.append(Y_w)

        # --- 5) 记录时间戳
        window_times.append(float(window_df["seconds"].iloc[83]))   # 窗口最后一帧
        window_start_event.append(event_times_mapping[j])           # 窗口对应的事件时间
        window_start_tracking.append(start_time)                   # 窗口起点对应的 tracking 时间

        # --- 6) 下一事件快照 (最后一个事件没有下一事件)
        if j < len(event_times) - 1:
            t_next = event_times[j+1]
            idx_next = np.searchsorted(secs, t_next, side="left")
            if idx_next < len(df_tracking):
                row_next = df_tracking.iloc[idx_next]
                feat_next = features_from_labels(pd.DataFrame([row_next]), labels_22)[0]  # (23,6)
                next_event_snapshots.append(feat_next)
                # flag: 下一个事件是否在窗口内
                event_in_window_flags.append(
                    1 if t_next <= window_df["seconds"].iloc[-1] else 0
                )
            else:
                next_event_snapshots.append(np.zeros((23,6), dtype=float))
                event_in_window_flags.append(0)
        else:
            # 对最后一个事件，快照为空
            next_event_snapshots.append(np.zeros((23,6), dtype=float))
            event_in_window_flags.append(0)

    # --- stack & save ---
    X_all        = np.stack(X_windows, axis=0)
    Y_all        = np.stack(Y_windows, axis=0)
    window_times = np.array(window_times)
    window_start_event = np.array(window_start_event)
    window_start_tracking = np.array(window_start_tracking)
    next_event_snapshots = np.array(next_event_snapshots)
    event_in_window_flags = np.array(event_in_window_flags)
    all_labels_22 = np.array(all_labels_22, dtype=object)

    print("最终得到：")
    print("  X_all.shape =", X_all.shape)
    print("  Y_all.shape =", Y_all.shape)
    print("  window_times.shape =", window_times.shape)

    save_dir = os.environ.get("DISTRAJ_SEQUENCE_DIR", os.path.join("data", "processed", "sequences"))
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{statsbomb_id}_{skillcorner_id}_sequence.npz"
    np.savez_compressed(
        save_path,
        X_all=X_all,
        Y_all=Y_all,
        window_times=window_times,
        window_start_event=window_start_event,
        window_start_tracking=window_start_tracking,
        next_event_snapshots=next_event_snapshots,
        event_in_window_flags=event_in_window_flags,
        all_labels_22=all_labels_22
    )
    print(f"✅ 已保存到: {save_path}")

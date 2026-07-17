"""Build fixed-length multi-agent trajectory sequences."""

import glob
import os

import numpy as np
import pandas as pd


def _safe_float(value, default=0.0):
    try:
        if value is None or isinstance(value, str) and value.strip() == "":
            return default
        value = float(value)
        return default if np.isnan(value) else value
    except Exception:
        return default


def _present_xy(x, y, eps=1e-9):
    return abs(x) > eps or abs(y) > eps


def pick_22_labels_from_row(row, num_players=23, sort_by_ball=True):
    ball_x = _safe_float(row.get("ball_x", 0.0))
    ball_y = _safe_float(row.get("ball_y", 0.0))
    home = []
    away = []

    for index in range(1, num_players + 1):
        home_x = _safe_float(row.get(f"h{index}_x", 0.0))
        home_y = _safe_float(row.get(f"h{index}_y", 0.0))
        if _present_xy(home_x, home_y):
            home.append((f"h{index}", (home_x - ball_x) ** 2 + (home_y - ball_y) ** 2))

        away_x = _safe_float(row.get(f"a{index}_x", 0.0))
        away_y = _safe_float(row.get(f"a{index}_y", 0.0))
        if _present_xy(away_x, away_y):
            away.append((f"a{index}", (away_x - ball_x) ** 2 + (away_y - ball_y) ** 2))

    if sort_by_ball:
        home.sort(key=lambda item: item[1])
        away.sort(key=lambda item: item[1])

    home_labels = [label for label, _ in home][:11]
    away_labels = [label for label, _ in away][:11]
    home_labels.extend(["PAD_h"] * (11 - len(home_labels)))
    away_labels.extend(["PAD_a"] * (11 - len(away_labels)))
    return home_labels + away_labels


def features_from_labels(dataframe, labels_22, k_near=5, ball_flag=1.0):
    matrices = []
    for _, row in dataframe.iterrows():
        ball_x = _safe_float(row.get("ball_x", 0.0))
        ball_y = _safe_float(row.get("ball_y", 0.0))
        player_x = []
        player_y = []
        present = []

        for label in labels_22:
            if label.startswith(("h", "a")):
                x = _safe_float(row.get(f"{label}_x", 0.0))
                y = _safe_float(row.get(f"{label}_y", 0.0))
                player_x.append(x)
                player_y.append(y)
                present.append(_present_xy(x, y))
            else:
                player_x.append(0.0)
                player_y.append(0.0)
                present.append(False)

        distances = np.hypot(np.array(player_x) - ball_x, np.array(player_y) - ball_y)
        distances[~np.array(present, dtype=bool)] = np.inf
        nearest_count = min(k_near, np.isfinite(distances).sum())
        nearest = set(np.argsort(distances)[:nearest_count].tolist())
        features = []

        for index, label in enumerate(labels_22):
            if label.startswith(("h", "a")) and present[index]:
                velocity_x = _safe_float(row.get(f"{label}_vx", 0.0))
                velocity_y = _safe_float(row.get(f"{label}_vy", 0.0))
                speed = float(np.hypot(velocity_x, velocity_y))
                features.append(
                    [
                        player_x[index],
                        player_y[index],
                        velocity_x,
                        velocity_y,
                        speed,
                        1.0 if index in nearest else 0.0,
                    ]
                )
            else:
                features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        ball_velocity_x = _safe_float(row.get("ball_vx", 0.0))
        ball_velocity_y = _safe_float(row.get("ball_vy", 0.0))
        features.append(
            [
                ball_x,
                ball_y,
                ball_velocity_x,
                ball_velocity_y,
                float(np.hypot(ball_velocity_x, ball_velocity_y)),
                ball_flag,
            ]
        )
        matrices.append(np.asarray(features, dtype=float))

    return np.stack(matrices, axis=0)


def get_window_by_start_time(dataframe, seconds, start_time, frames_needed, dt=0.1):
    index = np.searchsorted(seconds, start_time, side="left")
    front_padding = 0
    if start_time < seconds[0]:
        front_padding = int(np.ceil((seconds[0] - start_time) / dt))

    window = dataframe.iloc[
        max(index, 0) : index + frames_needed - front_padding
    ].copy()
    if front_padding > 0:
        padding = pd.DataFrame(0, index=range(front_padding), columns=dataframe.columns)
        padding["seconds"] = [start_time + offset * dt for offset in range(front_padding)]
        window = pd.concat([padding, window], ignore_index=True)

    if len(window) < frames_needed:
        back_padding = frames_needed - len(window)
        padding = pd.DataFrame(0, index=range(back_padding), columns=dataframe.columns)
        last_time = float(window["seconds"].iloc[-1]) if len(window) > 0 else start_time
        padding["seconds"] = [last_time + (offset + 1) * dt for offset in range(back_padding)]
        window = pd.concat([window, padding], ignore_index=True)
    return window


def id_to_label(player_id: int):
    if player_id in (0, 1000):
        return None
    if 1 <= player_id <= 23:
        return f"h{player_id}"
    if 51 <= player_id <= 73:
        return f"a{player_id - 50}"
    return None


def build_sequences(aligned_dir=None, tracking_dir=None, id_mapping_path=None, save_dir=None):
    aligned_dir = aligned_dir or os.environ.get(
        "DISTRAJ_ALIGNED_DIR", os.path.join("data", "interim", "syn_data_weighted")
    )
    tracking_dir = tracking_dir or os.environ.get(
        "DISTRAJ_TRACKING_DIR", os.path.join("data", "interim", "tracking_for_sequence")
    )
    id_mapping_path = id_mapping_path or os.environ.get(
        "DISTRAJ_ID_MAPPING", os.path.join("data", "id_matching.csv")
    )
    save_dir = save_dir or os.environ.get(
        "DISTRAJ_SEQUENCE_DIR", os.path.join("data", "processed", "sequences")
    )

    mapping = pd.read_csv(id_mapping_path)
    match_ids = dict(
        zip(
            mapping["match_id_statsbomb"].astype(str),
            mapping["match_id_skillcorner"].astype(str),
        )
    )
    saved_paths = []

    for csv_file in glob.glob(os.path.join(aligned_dir, "*.csv")):
        aligned = pd.read_csv(csv_file)
        statsbomb_id = aligned["match_id"].iloc[0]
        skillcorner_id = match_ids.get(str(statsbomb_id), None)
        aligned = aligned.sort_values("seconds_event").reset_index(drop=True)
        event_times = aligned["adjusted_seconds"].tolist()
        mapped_event_times = aligned["seconds_event"].tolist()

        tracking_path = os.path.join(tracking_dir, f"{skillcorner_id}_tracking_clean.csv")
        tracking = pd.read_csv(tracking_path).sort_values("seconds").reset_index(drop=True)
        tracking = tracking.fillna(0)
        seconds = tracking["seconds"].to_numpy()

        x_windows = []
        y_windows = []
        window_times = []
        window_start_event = []
        window_start_tracking = []
        next_event_snapshots = []
        event_in_window_flags = []
        all_labels_22 = []

        for event_index, event_time in enumerate(event_times):
            start_time = event_time - 2.0
            window = get_window_by_start_time(
                tracking, seconds, start_time=start_time, frames_needed=85
            )
            labels_22 = pick_22_labels_from_row(window.iloc[0], num_players=23, sort_by_ball=True)
            all_labels_22.append(labels_22)
            full_features = features_from_labels(window, labels_22)
            x_windows.append(full_features[:84])
            y_windows.append(
                full_features[1:85] if full_features.shape[0] > 84 else full_features[:84]
            )
            window_times.append(float(window["seconds"].iloc[83]))
            window_start_event.append(mapped_event_times[event_index])
            window_start_tracking.append(start_time)

            if event_index < len(event_times) - 1:
                next_event_time = event_times[event_index + 1]
                next_index = np.searchsorted(seconds, next_event_time, side="left")
                if next_index < len(tracking):
                    next_row = tracking.iloc[next_index]
                    next_features = features_from_labels(pd.DataFrame([next_row]), labels_22)[0]
                    next_event_snapshots.append(next_features)
                    event_in_window_flags.append(
                        1 if next_event_time <= window["seconds"].iloc[-1] else 0
                    )
                    continue

            next_event_snapshots.append(np.zeros((23, 6), dtype=float))
            event_in_window_flags.append(0)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"{statsbomb_id}_{skillcorner_id}_sequence.npz"
        )
        np.savez_compressed(
            save_path,
            X_all=np.stack(x_windows, axis=0),
            Y_all=np.stack(y_windows, axis=0),
            window_times=np.array(window_times),
            window_start_event=np.array(window_start_event),
            window_start_tracking=np.array(window_start_tracking),
            next_event_snapshots=np.array(next_event_snapshots),
            event_in_window_flags=np.array(event_in_window_flags),
            all_labels_22=np.array(all_labels_22, dtype=object),
        )
        saved_paths.append(save_path)

    return saved_paths


if __name__ == "__main__":
    build_sequences()

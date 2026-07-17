"""Align event and tracking data and derive synchronized node and edge features."""

import glob
import json
import os
from math import sqrt

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process as rapidfuzz_process

FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0
GOAL_WIDTH = 7.32
PENALTY_X = 105.0/2-16.5
PENALTY_Y = 40.32

base_dir = os.environ.get("DISTRAJ_RAW_DATA_ROOT", "data/raw/laliga_23")
event_dir = os.path.join(base_dir, "statsbomb/events_test")
tracking_dir = os.path.join(base_dir, "skillcorner_v2/tracking_test")
match_dir = os.path.join(base_dir, "skillcorner_v2/match_test")
metadata_dir = os.path.join(base_dir, "skillcorner_v2/metadata")
id_mapping_path = os.environ.get("DISTRAJ_ID_MAPPING", os.path.join("data", "id_matching.csv"))
aligned_output_dir = os.environ.get("DISTRAJ_ALIGNED_DIR", os.path.join("data", "interim", "syn_data_weighted"))


def load_id_map(path):
    dataframe = pd.read_csv(path)
    return dict(
        zip(
            dataframe["match_id_statsbomb"].astype(str),
            dataframe["match_id_skillcorner"].astype(str),
        )
    )


id_map = load_id_map(id_mapping_path) if __name__ == "__main__" else {}

team_name_dict = {
    'UD Almería': 'Almería', 'Real Sociedad': 'Real Sociedad', 'Athletic Club de Bilbao': 'Athletic Club',
    'Villarreal CF': 'Villarreal', 'RC Celta de Vigo': 'Celta Vigo', 'Getafe CF': 'Getafe',
    'UD Las Palmas': 'Las Palmas', 'Sevilla FC': 'Sevilla', 'Cadiz CF': 'Cádiz',
    'Atlético Madrid': 'Atlético Madrid', 'RCD Mallorca': 'Mallorca', 'Valencia CF': 'Valencia',
    'CA Osasuna': 'Osasuna', 'Girona FC': 'Girona', 'Real Betis Balompié': 'Real Betis',
    'FC Barcelona': 'Barcelona', 'Deportivo Alavés': 'Deportivo Alavés', 'Granada CF': 'Granada',
    'Rayo Vallecano': 'Rayo Vallecano', 'Real Madrid CF': 'Real Madrid'
}
def interpolate_adjusted_seconds_between_receipts(df_event, receipt_matches, eps=1e-6):
    if not receipt_matches:
        df_event["adjusted_seconds_interp_weighted"] = np.nan
        return df_event

    df_event = df_event.sort_values("seconds").reset_index(drop=True).copy()
    receipt_times = sorted(receipt_matches.keys())

    df_event["adjusted_seconds_interp_weighted"] = np.nan

    for i in range(len(receipt_times) - 1):
        t1, t2 = receipt_times[i], receipt_times[i + 1]
        a1, a2 = receipt_matches[t1], receipt_matches[t2]

        mask = (df_event["seconds"] >= t1) & (df_event["seconds"] <= t2)
        sub = df_event.loc[mask].copy().sort_values("seconds")

        if sub.empty:
            continue

        if t2 - t1 <= eps:
            df_event.loc[mask, "adjusted_seconds_interp_weighted"] = a1
            continue

        dt = sub["seconds"].diff().fillna(0.0)
        dt = dt.clip(lower=0.0)

        cum_w = dt.cumsum()

        total_w = float(cum_w.iloc[-1])

        if total_w <= eps:
            ratio = (sub["seconds"] - t1) / (t2 - t1)
        else:
            ratio = cum_w / total_w

        adj_vals = a1 + (a2 - a1) * ratio

        df_event.loc[sub.index, "adjusted_seconds_interp_weighted"] = adj_vals

    first_t, last_t = receipt_times[0], receipt_times[-1]
    df_event.loc[df_event["seconds"] < first_t, "adjusted_seconds_interp_weighted"] = receipt_matches[first_t]
    df_event.loc[df_event["seconds"] > last_t, "adjusted_seconds_interp_weighted"] = receipt_matches[last_t]

    return df_event


def rotate_around_center(x, y, center_x=105/2, center_y=34):

    translated_x = x - center_x
    translated_y = y - center_y

    rotated_x = -translated_x
    rotated_y = -translated_y

    new_x = rotated_x + center_x
    new_y = rotated_y + center_y

    return new_x, new_y

def preprocess_name(name):
    if isinstance(name, str):
        parts = name.split()
        if len(parts) > 1:
            return parts[0] + parts[-1]
        return name
    return name

def normalize_name(name):
    if isinstance(name, str):
        name = name.lower()
        name = name.replace(" ", "").replace("-", "").replace("_", "").replace(".", "")
        replacements = {
            "é": "e", "è": "e", "ê": "e", "ë": "e",
            "á": "a", "à": "a", "â": "a", "ä": "a",
            "í": "i", "ì": "i", "î": "i", "ï": "i",
            "ó": "o", "ò": "o", "ô": "o", "ö": "o",
            "ú": "u", "ù": "u", "û": "u", "ü": "u",
            "ç": "c", "ñ": "n", "ß": "ss"
        }
        for original, replacement in replacements.items():
            name = name.replace(original, replacement)
        name = name.strip()
        return name
    return name

def match_names(event_name, trackable_names):
    result = rapidfuzz_process.extractOne(event_name, trackable_names, scorer=fuzz.ratio)
    if result:
        match, score = result[0], result[1]
        return match if score > 80 else None
    return None

def get_kickoff_data(kickoff_name, home_or_away, df_tracking, id_to_info, period):

    if not kickoff_name or kickoff_name not in {info["normalized_name"] for info in id_to_info.values()}:
        return []

    matching_id = [
        info["id"] for tid, info in id_to_info.items() if info["normalized_name"] == kickoff_name
    ]
    matching_name = [
        info["normalized_name"] for tid, info in id_to_info.items() if info["normalized_name"] == kickoff_name
    ]
    matching_object = [
        info["trackable_object"] for tid, info in id_to_info.items() if info["normalized_name"] == kickoff_name
    ]

    if not matching_id:
        return []

    matching_id_value = matching_id[0]

    prefix = "h" if home_or_away == 1 else "a"
    x_column = f"{prefix}{int(matching_id_value)}_x"
    y_column = f"{prefix}{int(matching_id_value)}_y"
    if x_column in df_tracking.columns and y_column in df_tracking.columns:
        matching_rows = df_tracking[df_tracking["period"]== period][[x_column, y_column, "seconds"]].dropna()
        return [
            [(row[x_column], row[y_column]), row["seconds"]]
            for _, row in matching_rows.iterrows()
        ]
    else:
        return []


def calculate_acceleration_and_max_seconds(data, player_position,have_name):
    m_seconds = [entry[0] for entry in data]
    m_x = np.array([entry[1] for entry in data])
    m_y = np.array([entry[2] for entry in data])
    m_z = np.array([entry[3] for entry in data])

    delta_x = np.diff(m_x)
    delta_y = np.diff(m_y)
    delta_z = np.diff(m_z)
    delta_t = np.diff(m_seconds)

    delta_t[delta_t == 0] = 1e-6

    vx = delta_x / delta_t
    vy = delta_y / delta_t
    vz = delta_z / delta_t
    delta_vx = np.diff(vx) / delta_t[1:]
    delta_vy = np.diff(vy) / delta_t[1:]
    delta_vz = np.diff(vz) / delta_t[1:]
    acceleration_magnitude = np.sqrt(delta_vx**2 + delta_vy**2 + delta_vz**2)

    max_acceleration = None
    max_acceleration_seconds = None

    for index, m_second in enumerate(m_seconds):
        px = m_x[index]
        py = m_y[index]
        distances = []
        for p_frame in player_position:
            p_seconds = p_frame[1]
            p_position = p_frame[0]
            if p_seconds == m_second:
                if have_name:
                    kickoff_x = p_position[0]
                    kickoff_y = p_position[1]
                    distance = np.sqrt((px - kickoff_x)**2 + (py - kickoff_y)**2)
                    distances.append(distance)

                else:
                    for p in p_position:
                        if p[0] is not None and p[1] is not None and px is not None and py is not None:
                            kickoff_x = p[0]
                            kickoff_y = p[1]
                            distance = np.sqrt((px - kickoff_x)**2 + (py - kickoff_y)**2)
                            distances.append(distance)

        distances = np.array(distances)
        valid_indices = np.where((distances <= 2) & (~np.isnan(distances)))[0]
        if len(valid_indices) > 0 and len(acceleration_magnitude) > 0:
            max_acceleration_index_in_valid = np.argmax(acceleration_magnitude)
            max_acceleration = acceleration_magnitude[max_acceleration_index_in_valid]
            max_acceleration_seconds = m_seconds[max_acceleration_index_in_valid + 2]
            break

    return max_acceleration_seconds if max_acceleration_seconds is not None else 0.0, max_acceleration if max_acceleration is not None else 0.0, distances


def distance_score(event_pass_x, event_pass_y, df_window, event_player, player_map,
                   weight_player=0.7, weight_event=0.3, distance_cap=10.0):
    dfw = df_window.copy()
    bx = pd.to_numeric(dfw.get('ball_x', 0), errors='coerce').fillna(0)
    by = pd.to_numeric(dfw.get('ball_y', 0), errors='coerce').fillna(0)
    dfw['ball_event_distance'] = np.sqrt((bx - event_pass_x) ** 2 + (by - event_pass_y) ** 2)

    seconds_col = 'adjusted_seconds' if 'adjusted_seconds' in dfw.columns else 'seconds'

    norm_name = normalize_name(preprocess_name(event_player))
    row = player_map[player_map['normalized_name'] == norm_name]

    if row.empty:
        dfw['score'] = np.exp(-dfw['ball_event_distance'] / distance_cap) * weight_event * 100.0
        best_idx = dfw['score'].idxmax()
        best = dfw.loc[best_idx]
        return {
            'adjusted_seconds': float(best.get(seconds_col, np.nan)),
            'score': float(best['score']),
            'ball_to_event_dist': float(best['ball_event_distance']),
        }

    label = row.iloc[0]['label']
    px_col, py_col = f"{label}_x", f"{label}_y"

    if px_col in dfw.columns and py_col in dfw.columns:
        px = pd.to_numeric(dfw[px_col], errors='coerce').fillna(0)
        py = pd.to_numeric(dfw[py_col], errors='coerce').fillna(0)
        ball_to_player = np.sqrt((bx - px)**2 + (by - py)**2)
        score_player = np.exp(-ball_to_player / distance_cap)
        score_event  = np.exp(-dfw['ball_event_distance'] / distance_cap)
        dfw['score'] = (weight_player * score_player + weight_event * score_event) * 100.0
    else:
        dfw['score'] = np.exp(-dfw['ball_event_distance'] / distance_cap) * weight_event * 100.0

    best_idx = dfw['score'].idxmax()
    best = dfw.loc[best_idx]
    return {
        'adjusted_seconds': float(best.get(seconds_col, np.nan)),
        'score': float(best['score']),
        'ball_to_event_dist': float(best['ball_event_distance']),
    }


def get_window_of_frames_around(action_second, df_tracking, ta):
    return df_tracking[
        (df_tracking["adjusted_seconds"].between(action_second - ta, action_second + ta)) &
        df_tracking["period"].notna()
    ]

def compute_offsets(df_m, df_e):
    df_m = df_m.copy()

    df_m['offset'] = 0

    for i in range(len(df_m)):
        if i == 0:
            start = df_e['seconds'].min()

        else:
            start = df_m.iloc[i - 1]['e_seconds']


        end = df_m.iloc[i]['e_seconds']
        adjusted_end = df_m.iloc[i]['adjusted_seconds']


        event_rows = df_e[(df_e['seconds'] > start) & (df_e['seconds'] <= end)]
        if not event_rows.empty:
            time_ratio = (event_rows['seconds'] - start) / (end - start)

            df_e.loc[event_rows.index, 'adjusted_seconds_1'] = event_rows['seconds'] + time_ratio * (adjusted_end- end)
    return df_m, df_e

def align_events(df_event11, df_tracking11, df_m11):
    df_m12, df_event12 = compute_offsets(df_m11, df_event11)
    df_event12['adjusted_seconds_rounded'] = df_event12['adjusted_seconds_1'].round(1)
    aligned_df_1 = pd.merge(
        df_event12,
        df_tracking11,
        left_on='adjusted_seconds_rounded',
        right_on='adjusted_seconds',
        how='left',
        suffixes=('_event', '_tracking')
    )

    return aligned_df_1

def take_first_and_last_with_interp_for_aligned(df, df_tracking, time_col="adjusted_seconds"):
    df = df.sort_values(time_col).reset_index(drop=True)
    unique_times = df[time_col].dropna().unique()
    new_rows = []
    for idx, t1 in enumerate(unique_times):
        sub = df[df[time_col] == t1]
        first = sub.iloc[0].copy()
        new_rows.append(first)

        if len(sub) > 1:
            last = sub.iloc[-1].copy()
            if idx + 1 < len(unique_times):
                t2 = unique_times[idx + 1]
                mid_t = (t1 + t2) / 2.0
                next_seconds_event = df[df[time_col] == t2]["seconds_event"].min()
                mid_e_t = (sub.iloc[-1]["seconds_event"] + next_seconds_event) / 2.0
            else:
                mid_t = t1
                mid_e_t = sub.iloc[-1]["seconds_event"]

            last[time_col] = round(mid_t, 1)
            last["seconds_event"] = round(mid_e_t, 1)

            tracking_cols = [f"{t}{i}_{axis}"
                 for t in ["h", "a"]
                 for i in range(1, 24)
                 for axis in ["x", "y"]] + ["ball_x", "ball_y", "ball_z"]
            tracking_row = df_tracking.iloc[(df_tracking[time_col] - mid_t).abs().argmin()]
            for c in tracking_cols:
                if c in tracking_row:
                    last[c] = tracking_row[c]

            new_rows.append(last)

    return pd.DataFrame(new_rows).reset_index(drop=True)

def create_player_map(df_tracking, trackable_objects):
    rows = []
    for team_prefix in ["h","a"]:
        for i in range(1,24):
            tid_col = f"{team_prefix}{i}_trackable_id"
            if tid_col not in df_tracking.columns:
                continue
            first_tid = df_tracking[tid_col].dropna().iloc[0] if df_tracking[tid_col].notna().any() else None
            if first_tid is None:
                continue
            info = trackable_objects.get(first_tid)
            if not info:
                continue
            rows.append({
                "label": f"{team_prefix}{i}",
                "trackable_id": first_tid,
                "normalized_name": normalize_name(preprocess_name(info["name"])),
                "name": info["name"],
                "team": info["team"],
                "role": info["role"],
                "id": info["id"]
            })
    return pd.DataFrame(rows)


event_files = os.listdir(event_dir) if __name__ == "__main__" else []
for fname in event_files:
    if not fname.endswith(".csv"):
        continue

    statsbomb_id = fname.replace(".csv", "")
    event_path = os.path.join(event_dir, fname)

    skillcorner_id = id_map.get(statsbomb_id)
    if skillcorner_id is None:
        continue
    tracking_path = os.path.join(tracking_dir, f"{skillcorner_id}.json")
    match_path = os.path.join(match_dir, f"{skillcorner_id}.json")
    metadata_path = os.path.join(metadata_dir, "players.json")
    if not os.path.exists(tracking_path) or not os.path.exists(match_path):
        continue


    event_df = pd.read_csv(event_path)

    with open(tracking_path, "r", encoding="utf-8") as f:
        tracking_data = json.load(f)

    with open(match_path, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_data = json.load(f)


    if not os.path.exists(event_path):
        raise FileNotFoundError(f"Statsbomb event file not found: {event_path}")
    if not os.path.exists(tracking_path):
        raise FileNotFoundError(f"Skillcorner tracking file not found: {tracking_path}")
    if not os.path.exists(match_path):
        raise FileNotFoundError(f"Skillcorner match file not found: {match_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Skillcorner metadata file not found: {metadata_path}")

    home_team_name = team_name_dict[match_data['home_team']['name']]
    away_team_name = team_name_dict[match_data['away_team']['name']]

    team_dict = {
        match_data['home_team']['id']: {'role': 'home', 'name': home_team_name},
        match_data['away_team']['id']: {'role': 'away', 'name': away_team_name}
    }
    trackable_objects = {}
    home_count = away_count = 0
    home_id_mapping = {}
    away_id_mapping = {}
    for player in match_data['players']:
        role = team_dict[player['team_id']]['role']
        position = player['player_role']['name']
        if role == 'home':
            home_id_mapping[home_count] = len(home_id_mapping) + 1
            trackable_objects[player['trackable_object']] = {
                'name': f"{player['first_name']} {player['last_name']}".strip(),
                'team': team_dict[player['team_id']]['name'],
                'role': role,
                'id': home_count,
                'position': position
            }
            home_count += 1
        elif role == 'away':
            away_id_mapping[away_count] = len(away_id_mapping) + 1
            trackable_objects[player['trackable_object']] = {
                'name': f"{player['first_name']} {player['last_name']}".strip(),
                'team': team_dict[player['team_id']]['name'],
                'role': role,
                'id': away_count,
                'position': position
            }
            away_count += 1

    trackable_objects[match_data['ball']['trackable_object']] = {'name': 'ball', 'team': 'ball', 'role': 'ball', 'position': 'ball'}
    ball_id = match_data['ball']['trackable_object']

    event_list = []
    for _, event in event_df.iterrows():
        match_id = statsbomb_id
        event_period = event['period']
        event_time = event['timestamp']
        event_minute = event['minute']
        event_second = event['second']
        event_type = event['type']
        event_type_2 = None
        end_x = end_y = None
        pass_type = None
        pass_height = None
        event_possession_team = event["possession_team"]
        pass_outcome = event.get("pass_outcome")
        pass_possibility = event.get("pass_pass_success_probability")
        pass_cluster_label = event.get("pass_pass_cluster_label")
        pass_cluster_possibility = event.get("pass_pass_cluster_probability")


        if event_type == "Pass":
            end_location = event.get('pass_end_location')
            if isinstance(end_location, str):
                end_location = [float(x) for x in end_location[1:-1].split(",")]
                end_x = round(end_location[0] * (1.05 / 1.2), 2)
                end_y = round(end_location[1] * (68 / 80), 2)
            cross = event.get('pass_cross')
            pass_height = event.get('pass_height')
            pass_type = event.get('pass_type')

            event_type_2_mapping = {
                "Corner": "Corner",
                "Cross": "Cross",
                "High Pass": pass_height
            }

            if pass_type in event_type_2_mapping:
                event_type_2 = event_type_2_mapping[pass_type]
            elif cross and not pd.isna(cross):
                event_type_2 = "Cross"
            elif pass_height:
                event_type_2 = pass_height

        elif event_type == "Shot":
            event_type_2 = event.get('shot_outcome')

        event_team = event['team']
        home_team = 1 if event_team == home_team_name else 0
        event_player = event['player']
        event_location = event.get('location')

        if isinstance(event_location, str):
            event_location = [float(x) for x in event_location[1:-1].split(",")]
            start_x, start_y = round(event_location[0] * (1.05 / 1.2), 2), round(event_location[1] * (68 / 80), 2)
        else:
            start_x = start_y = None

        time_components = event_time.split(':')
        event_seconds = round(float(time_components[0]) * 3600 + float(time_components[1]) * 60 + float(time_components[2]), 4)
        if event_period == 2:
            event_seconds += 45 * 60
        elif event_period == 3:
            event_seconds += 90 * 60
        elif event_period == 4:
            event_seconds += (90 + 15) * 60

        event_list.append([match_id, event_period, event_time, event_minute, event_second, event_seconds, event_type, event_type_2, event_team, home_team, event_player, start_x, start_y, end_x, end_y, pass_type, pass_height, pass_outcome, pass_possibility, pass_cluster_label, pass_cluster_possibility,event_possession_team])

    event_list_sorted = sorted(event_list, key=lambda x: x[5])

    event_columns = [
        "match_id", "period", "time", "minute", "second", "seconds", "event_type", "event_type_2", "team", "home_team",
        "player", "start_x", "start_y", "end_x", "end_y", "pass_type", "pass_height", "pass_outcome",
        "pass_possibility", "pass_cluster_label", "pass_cluster_possibility", "possession_team"
    ]

    df_event = pd.DataFrame(event_list_sorted, columns=event_columns)
    reverse_events = []

    for i in range(len(df_event) - 2):
        pt1 = df_event.loc[i, 'possession_team']
        pt2 = df_event.loc[i + 1, 'possession_team']
        pt3 = df_event.loc[i + 2, 'possession_team']

        if pt1 == pt2:
            if pt2 == pt3:
                df_event.loc[i, 'needs_reverse'] = True
            else:
                df_event.loc[i, 'needs_reverse'] = True
                reverse_events.append(df_event.loc[i, 'seconds'])
            if df_event.loc[i, 'needs_reverse']:

                if not pd.isna(df_event.loc[i, 'start_x']) and not pd.isna(df_event.loc[i, 'start_y']):
                    start_x, start_y = rotate_around_center(
                        df_event.loc[i, 'start_x'], df_event.loc[i, 'start_y']
                    )
                    df_event.loc[i, 'start_x'] = start_x
                    df_event.loc[i, 'start_y'] = start_y


                if not pd.isna(df_event.loc[i, 'end_x']) and not pd.isna(df_event.loc[i, 'end_y']):
                    end_x, end_y = rotate_around_center(
                        df_event.loc[i, 'end_x'], df_event.loc[i, 'end_y']
                    )
                    df_event.loc[i, 'end_x'] = end_x
                    df_event.loc[i, 'end_y'] = end_y
        else:

            df_event.loc[i, 'needs_reverse'] = False

    df_event.loc[len(df_event) - 2:, 'needs_reverse'] = False


    home_side = None
    home_gk_x = away_gk_x = None
    df_tracking_list = []


    for frame in tracking_data:
        home_tracking = [None] * 2 * 23
        away_tracking = [None] * 2 * 23
        home_trackable_ids = [None] * 23
        away_trackable_ids = [None] * 23
        ball_x = ball_y = ball_z = None
        tracking_possession = frame.get("possession", {})
        possession_time = frame.get("timestamp", "")
        possession_period = frame.get("period", "")
        if possession_time:
            try:
                time_parts = possession_time.split(':')
                possession_second = round(float(time_parts[0]) * 3600 + float(time_parts[1]) * 60 + float(time_parts[2]),4)
            except (ValueError, IndexError):
                possession_second = 0
        else:
            possession_second = 0
        possession_team = tracking_possession.get("group")
        tracking_d = frame["data"]

        if tracking_d:
            for obj in tracking_d:
                track_obj = trackable_objects.get(obj['trackable_object'])
                if not track_obj:
                    continue


                if track_obj['role'] == 'home':
                    id = track_obj["id"]
                    home_tracking[2 * track_obj['id']] = round(obj['x'] + FIELD_LENGTH / 2,2)
                    home_tracking[2 * track_obj['id'] + 1] = round(-obj['y'] + FIELD_WIDTH / 2 ,2)
                    home_trackable_ids[id] = obj['trackable_object']
                elif track_obj['role'] == 'away':
                    id = track_obj["id"]
                    away_tracking[2 * track_obj['id']] = round(obj['x'] + FIELD_LENGTH / 2,2)
                    away_tracking[2 * track_obj['id'] + 1] = round(-obj['y'] + FIELD_WIDTH / 2 ,2)
                    away_trackable_ids[id] = obj['trackable_object']

                if track_obj['position'] == "Goalkeeper":
                    if track_obj['role'] == 'home' and home_gk_x is None:
                        home_gk_x = obj['x']
                    elif track_obj['role'] == 'away' and away_gk_x is None:
                        away_gk_x = obj['x']


                if track_obj["role"] == "ball":
                    ball_x = round(obj['x'] + FIELD_LENGTH / 2,2)
                    ball_y = round(-obj['y'] + FIELD_WIDTH / 2 ,2)
                    ball_z = round(obj["z"],2)

            if home_gk_x is not None and away_gk_x is not None:
                home_side = 'left' if home_gk_x < away_gk_x else 'right'


        df_tracking_list.append([possession_second, possession_period, possession_team, *home_tracking, *away_tracking, *home_trackable_ids, *away_trackable_ids, home_side, ball_x, ball_y, ball_z])



    home_tracking_columns = []
    away_tracking_columns = []
    home_trackable_columns = []
    away_trackable_columns = []

    for i in range(1, 24):
        home_tracking_columns.extend([f"h{i}_x", f"h{i}_y"])
        away_tracking_columns.extend([f"a{i}_x", f"a{i}_y"])
        home_trackable_columns.append(f"h{i}_trackable_id")
        away_trackable_columns.append(f"a{i}_trackable_id")

    columns1 = ["seconds", "period","possession_team"] + home_tracking_columns + away_tracking_columns + home_trackable_columns + away_trackable_columns + ["home_side", "ball_x", "ball_y", "ball_z"]

    df_tracking_list_filtered = [x for x in df_tracking_list if x[0] is not None and x[1] is not None]
    df_tracking_list_sorted = sorted(df_tracking_list_filtered, key=lambda x: (x[1], x[0]))

    df_tracking = pd.DataFrame(df_tracking_list_sorted, columns=columns1)
    df_tracking["seconds"] = df_tracking["seconds"].round(1)


    label_to_info = {}
    for obj_id, info in trackable_objects.items():
        if 'id' not in info:
            continue
        role = info['role']
        idx  = info['id']
        if role == 'home':
            label = f"h{idx}"
        elif role == 'away':
            label = f"a{idx}"
        else:
            continue
        label_to_info[label] = {
            'name': info['name'],
            'team': info['team']
        }

    df_player = (
        pd.DataFrame([
            {'label': lbl, 'name': inf['name'], 'team': inf['team']}
            for lbl, inf in label_to_info.items()
        ])
        .assign(order=lambda d: d['label'].str[1:].astype(int))
        .sort_values('order')
        .drop(columns='order')
        .reset_index(drop=True)
    )


    df_event["preprocessed_name"] = df_event["player"].apply(preprocess_name)
    df_event["normalized_name"] = df_event["preprocessed_name"].apply(normalize_name)
    trackable_names = [value["name"] for key, value in trackable_objects.items()]
    processed_trackable_names = [normalize_name(preprocess_name(name)) for name in trackable_names]
    df_event["matched_name"] = df_event["normalized_name"].apply(lambda x: match_names(x, processed_trackable_names))
    unique_df = df_event[["player", "matched_name"]].drop_duplicates()

    trackable_name_map = {tid: normalize_name(preprocess_name(info['name'])) for tid, info in trackable_objects.items()}


    id_to_info = {
        tid: {
            "trackable_object": tid,
            "name": info["name"],
            "normalized_name": normalize_name(preprocess_name(info["name"])),
            "team": info["role"],
            "id": (info.get("id") + 1) if info.get("id") is not None else None
        }
        for tid, info in trackable_objects.items()
    }

    matching_id111 = [
        info["id"] for tid, info in id_to_info.items()]

    filtered = df_event[df_event["pass_type"] == "Kick Off"]
    filter1 = filtered[filtered["period"]== 1]
    filter2 = filtered[filtered["period"]== 2]

    home_or_away1 = filter1["home_team"].iloc[0]
    home_or_away2 = filter2["home_team"].iloc[0]
    ball_frames = []

    for _, ball_frame in df_tracking.iterrows():
        if np.isnan(ball_frame['ball_x']) or np.isnan(ball_frame['ball_y']):
            continue
        b_seconds = ball_frame["seconds"]
        b_x = ball_frame["ball_x"]
        b_y = ball_frame["ball_y"]
        b_z = ball_frame["ball_z"]
        ball_frames.append([b_seconds, b_x, b_y, b_z])
        if b_seconds > 5:
            break

    ball_frames_secondhalf = []
    filtered_rows = df_tracking[df_tracking["period"]==2]

    for _, second_frame in filtered_rows.iterrows():
        if np.isnan(second_frame['ball_x']) or np.isnan(second_frame['ball_y']):
            continue
        sb_seconds = second_frame["seconds"]
        sb_x = second_frame["ball_x"]
        sb_y = second_frame["ball_y"]
        sb_z = second_frame["ball_z"]
        ball_frames_secondhalf.append([sb_seconds,sb_x,sb_y,sb_z,possession_team])
        if sb_seconds > 45*60+10:
            break

    kickoff_names1 = filter1["matched_name"].iloc[0]
    kickoff_names2= filter2["matched_name"].iloc[0]

    kickoff_data1 = get_kickoff_data(kickoff_names1, home_or_away1, df_tracking, id_to_info,1)
    kickoff_data2 = get_kickoff_data(kickoff_names2, home_or_away2, df_tracking, id_to_info,2)

    max_seconds1,max_acceleration1, distances_data1 = calculate_acceleration_and_max_seconds(ball_frames,kickoff_data1,True)
    max_seconds2,max_acceleration2, distances_data2= calculate_acceleration_and_max_seconds(ball_frames_secondhalf,kickoff_data2,True)

    h_columns_x = [col for col in df_tracking.columns if col.startswith("h") and col.endswith("_x")]
    h_columns_y = [col.replace("_x", "_y") for col in h_columns_x]

    h_players = []
    for _, h_row in df_tracking.iterrows():
        if all(pd.isna(h_row[x]) for x in h_columns_x):
            continue
        h_player = [(h_row[x], h_row[y]) for x, y in zip(h_columns_x, h_columns_y)]
        h_seconds = h_row["seconds"]
        h_players.append([h_player, h_seconds])

    a_columns_x = [col for col in df_tracking.columns if col.startswith("a") and col.endswith("_x")]
    a_columns_y = [col.replace("_x", "_y") for col in a_columns_x]

    a_players = []
    for _, a_row in df_tracking.iterrows():
        if all(pd.isna(a_row[x]) for x in a_columns_x):
            continue
        a_player = [(a_row[x], a_row[y]) for x, y in zip(a_columns_x, a_columns_y)]
        a_seconds = a_row["seconds"]
        a_players.append([a_player, a_seconds])

    data1_flag = all(distance_data1 >= 10 for distance_data1 in distances_data1) or not kickoff_data1
    data2_flag = all(distance_data2 >= 10 for distance_data2 in distances_data2) or not kickoff_data2

    if data1_flag:
        if home_or_away1==1:
            max_seconds1,max_acceleration1,distances_1 = calculate_acceleration_and_max_seconds(ball_frames,h_players,False)
        else:
            max_seconds1,max_acceleration1,distances_1= calculate_acceleration_and_max_seconds(ball_frames,a_players,False)
    if data2_flag:
        if home_or_away2==1:
            max_seconds2,max_acceleration2,distances_2= calculate_acceleration_and_max_seconds(ball_frames_secondhalf,h_players,False)
        else:
            max_seconds2,max_acceleration2,distances_2= calculate_acceleration_and_max_seconds(ball_frames_secondhalf,a_players,False)


    adjusted_max_seconds2 = round(max_seconds2 - 45*60,1)
    df_tracking["raw_seconds"] = df_tracking["seconds"]

    df_tracking["adjusted_seconds"] = df_tracking["seconds"]
    df_tracking.loc[df_tracking["period"] == 1, "adjusted_seconds"] -= max_seconds1
    df_tracking.loc[df_tracking["period"] == 2, "adjusted_seconds"] -= adjusted_max_seconds2
    df_tracking.loc[df_tracking["period"] == 1, "adjusted_seconds"] = round(df_tracking.loc[df_tracking["period"] == 1, "adjusted_seconds"],1)
    df_tracking.loc[df_tracking["period"] == 2, "adjusted_seconds"] = round(df_tracking.loc[df_tracking["period"] == 2, "adjusted_seconds"],1)
    pass_events = df_event[(df_event['event_type'] == 'Pass') | (df_event['event_type'] == 'Ball Receipt*')]
    pass_events_seconds = pass_events["seconds"]
    ta = 5
    df_player_map = create_player_map(df_tracking, trackable_objects)

    windows=[]
    for action_second in pass_events_seconds:
        if action_second is not None and not isinstance(action_second, pd.DataFrame):
            window = get_window_of_frames_around(action_second, df_tracking, ta)
            windows.append([action_second,window])
        elif action_second is not None and isinstance(action_second, pd.DataFrame):
            action_second = action_second.iloc[0]
            window = get_window_of_frames_around(action_second, df_tracking, ta)
            windows.append([action_second,window])


    results = []

    try:
        for dfff_window in windows:
            dff_window = pd.DataFrame(dfff_window[1])
            e_second = dfff_window[0]

            if e_second not in df_event["seconds"].values:
                raise ValueError(f"Event with seconds={e_second} not found in df_event.")

            pass_x = df_event.loc[df_event["seconds"] == e_second, "start_x"].iloc[0]
            pass_y = df_event.loc[df_event["seconds"] == e_second, "start_y"].iloc[0]
            e_player = df_event.loc[df_event["seconds"] == e_second, "player"].iloc[0]

            window_score = distance_score(pass_x, pass_y, dff_window, e_player, df_player_map)

            results.append({"e_seconds": e_second, "score": window_score})

    except Exception:
        pass

    results_df = pd.DataFrame(results)
    ball_receipts = df_event[df_event["event_type"].str.contains("Ball Receipt", case=False)]

    receipt_matches = {
        row["e_seconds"]: row["score"]["adjusted_seconds"]
        for _, row in results_df.iterrows()
        if row["e_seconds"] in ball_receipts["seconds"].values
    }

    df_event_interp = interpolate_adjusted_seconds_between_receipts(df_event, receipt_matches)
    df_event_interp["adjusted_seconds_interp"] = df_event_interp["adjusted_seconds_interp_weighted"]
    merged_rows = []
    for _, e_row in df_event_interp.iterrows():
        e_seconds = e_row["seconds"]
        adj_sec = e_row.get("adjusted_seconds_interp", np.nan)

        if pd.isna(adj_sec):
            continue

        nearest_tracking = df_tracking.iloc[(df_tracking["adjusted_seconds"] - adj_sec).abs().argmin()]
        merged_row = {**e_row.to_dict(), **nearest_tracking.to_dict()}
        merged_row["e_seconds"] = e_seconds
        merged_row["adjusted_seconds"] = adj_sec
        merged_rows.append(merged_row)

    merged_df = pd.DataFrame(merged_rows)
    columns_to_keep = [
        "e_seconds", "adjusted_seconds", "match_id", "period", "time", "minute", "second",
        "event_type", "event_type_2", "team", "home_team", "player", "start_x", "start_y",
        "end_x", "end_y", "pass_type", "pass_height", "pass_outcome", "possession_team",
        "needs_reverse", "preprocessed_name", "normalized_name", "matched_name",
        *[f"h{i}_x" for i in range(1, 24)], *[f"h{i}_y" for i in range(1, 24)],
        *[f"a{i}_x" for i in range(1, 24)], *[f"a{i}_y" for i in range(1, 24)],
        "home_side", "ball_x", "ball_y", "ball_z", "raw_seconds"
    ]

    simplified_df = merged_df[columns_to_keep]

    aligned_df = align_events(df_event, df_tracking, simplified_df)

    int_cols = [f"h{i}_trackable_id" for i in range(1,24)] + \
            [f"a{i}_trackable_id" for i in range(1,24)]

    aligned_df[int_cols] = (
        aligned_df[int_cols]
        .fillna(0)
        .astype("Int64")
    )

    aligned_df_final = take_first_and_last_with_interp_for_aligned(aligned_df, df_tracking, time_col="adjusted_seconds")
    aligned_df_final = aligned_df_final.sort_values(by="seconds_event", ascending=True).reset_index(drop=True)

    os.makedirs(aligned_output_dir, exist_ok=True)
    save_aligned = os.path.join(aligned_output_dir, f"{statsbomb_id}_{skillcorner_id}_aligned_df.csv")
    os.makedirs(os.path.dirname(save_aligned), exist_ok=True)
    aligned_df_final.to_csv(save_aligned, index=False)

folder_path = aligned_output_dir
match_dir = os.path.join(base_dir, "skillcorner_v2/match")
output_dir = os.environ.get("DISTRAJ_NODE_EDGE_DIR", os.path.join("data", "interim", "syn_node_edge_data"))


def select_target_matches(aligned_dir, matches_dir, match_ids, limit=140):
    candidates = []
    for csv_file in sorted(glob.glob(os.path.join(aligned_dir, "*.csv"))):
        try:
            dataframe = pd.read_csv(csv_file, nrows=1)
            if "match_id" not in dataframe.columns:
                continue
            statsbomb_id = str(dataframe["match_id"].iloc[0])
        except Exception:
            continue

        skillcorner_id = match_ids.get(statsbomb_id)
        if not skillcorner_id:
            continue
        match_json = os.path.join(matches_dir, f"{skillcorner_id}.json")
        if os.path.exists(match_json):
            candidates.append((f"{statsbomb_id}_{skillcorner_id}", csv_file))

    return {match_key for match_key, _ in candidates[:limit]}

def distance_pos(x1, y1, x2, y2):
    if pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2):
        return np.nan
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calc_goalline_features(x, y, home_attack_dir, is_home,
                           field_length=105, field_width=68, goal_width=7.32):
    if pd.isna(x) or pd.isna(y):
        return np.nan, np.nan, np.nan, np.nan

    if (home_attack_dir == "left to right" and is_home == 1) or \
       (home_attack_dir == "right to left" and is_home == 0):
        goal_center = (field_length, field_width / 2)
        goal_left   = (field_length, (field_width / 2) - goal_width / 2)
        goal_right  = (field_length, (field_width / 2) + goal_width / 2)
    elif (home_attack_dir == "right to left" and is_home == 1) or \
         (home_attack_dir == "left to right" and is_home == 0):
        goal_center = (0, field_width / 2)
        goal_left   = (0, (field_width / 2) - goal_width / 2)
        goal_right  = (0, (field_width / 2) + goal_width / 2)
    else:
        return np.nan, np.nan, np.nan, np.nan

    dx, dy = goal_center[0] - x, goal_center[1] - y
    dist_to_goal = np.sqrt(dx**2 + dy**2)

    goal_line_vec = np.array([goal_right[0] - goal_left[0],
                              goal_right[1] - goal_left[1]])

    dir_vec = np.array([dx, dy])

    dot = np.dot(dir_vec, goal_line_vec)
    norms = np.linalg.norm(dir_vec) * np.linalg.norm(goal_line_vec) + 1e-8
    ang_to_goal_line = np.arccos(np.clip(dot / norms, -1.0, 1.0))

    norm_c = np.linalg.norm(dir_vec) + 1e-6
    dir_x, dir_y = dx / norm_c, dy / norm_c

    return dist_to_goal, ang_to_goal_line, dir_x, dir_y

def calc_touchline_features(x, y, field_length=105, field_width=68):
    if pd.isna(x) or pd.isna(y):
        return np.nan, np.nan, np.nan, np.nan

    edges = [(0, y), (field_length, y), (x, 0), (x, field_width)]
    dists = [np.hypot(ex - x, ey - y) for ex, ey in edges]
    min_idx = int(np.argmin(dists))
    dist_to_tl = dists[min_idx]
    edge_x, edge_y = edges[min_idx]

    dx, dy = edge_x - x, edge_y - y
    norm = np.hypot(dx, dy) + 1e-6
    dir_x, dir_y = dx / norm, dy / norm
    ang_to_tl = np.arctan2(dy, dx)

    return dist_to_tl, ang_to_tl, dir_x, dir_y

def apply_all_features_per_row(row):
    results = {}
    home_attack_dir = row["home_attack_dir"]
    is_home = row["home_team"]

    for team in ["h", "a"]:
        for num in range(1, 24):
            x = row.get(f"{team}{num}_x", np.nan)
            y = row.get(f"{team}{num}_y", np.nan)

            g_dist, g_ang, g_dx, g_dy = calc_goalline_features(x, y, home_attack_dir, is_home)
            results[f"{team}{num}_gl_dist"]   = g_dist
            results[f"{team}{num}_gl_ang"]    = g_ang
            results[f"{team}{num}_gl_dir_x"]  = g_dx
            results[f"{team}{num}_gl_dir_y"]  = g_dy

            t_dist, t_ang, t_dx, t_dy = calc_touchline_features(x, y)
            results[f"{team}{num}_tl_dist"]   = t_dist
            results[f"{team}{num}_tl_ang"]    = t_ang
            results[f"{team}{num}_tl_dir_x"]  = t_dx
            results[f"{team}{num}_tl_dir_y"]  = t_dy

    bx, by = row.get("ball_x", np.nan), row.get("ball_y", np.nan)
    g_dist, g_ang, g_dx, g_dy = calc_goalline_features(bx, by, home_attack_dir, is_home)
    results["ball_gl_dist"]   = g_dist
    results["ball_gl_ang"]    = g_ang
    results["ball_gl_dir_x"]  = g_dx
    results["ball_gl_dir_y"]  = g_dy

    t_dist, t_ang, t_dx, t_dy = calc_touchline_features(bx, by)
    results["ball_tl_dist"]   = t_dist
    results["ball_tl_ang"]    = t_ang
    results["ball_tl_dir_x"]  = t_dx
    results["ball_tl_dir_y"]  = t_dy

    label = row.get("ha_name", "0")
    if isinstance(label, str) and label != "0":
        hx = row.get(f"{label}_x", np.nan)
        hy = row.get(f"{label}_y", np.nan)
        ha_g_dist, ha_g_ang, _, _ = calc_goalline_features(hx, hy, home_attack_dir, is_home)
        results["ha_dist"] = ha_g_dist
        results["ha_ang"]  = ha_g_ang
    else:
        results["ha_dist"] = np.nan
        results["ha_ang"]  = np.nan

    results["bc_dist"] = distance_pos(row.get("ha_x", np.nan), row.get("ha_y", np.nan), bx, by)

    return pd.Series(results)

def get_ha_xy(row):
    label = row.get("ha_name", "0")
    if isinstance(label, str) and label != "0":
        x = row.get(f"{label}_x", np.nan)
        y = row.get(f"{label}_y", np.nan)
        vx = row.get(f"{label}_vx", np.nan)
        vy = row.get(f"{label}_vy", np.nan)
        speed = row.get(f"{label}_speed", np.nan)
    else:
        x, y, vx, vy, speed = 0, 0, 0, 0, 0
    return pd.Series([x, y, vx, vy, speed])

def take_first_and_last_with_interp(df, time_col='seconds_event'):
    df = df.sort_values(time_col).reset_index(drop=True)
    unique_times = df[time_col].unique()
    new_rows = []
    for idx, t1 in enumerate(unique_times):
        sub = df[df[time_col] == t1]
        first = sub.iloc[0].copy()
        new_rows.append(first)

        if len(sub) > 1:
            last = sub.iloc[-1].copy()
            if idx + 1 < len(unique_times):
                t2 = unique_times[idx + 1]
                mid_t = (t1 + t2) / 2.0
            else:
                mid_t = t1
            last[time_col] = mid_t
            new_rows.append(last)

    return pd.DataFrame(new_rows)

def get_pass_reciever(df, current_index):
    frame = df.iloc[current_index]
    pX, pY = frame['end_x'], frame['end_y']
    min_dist = 10000
    rec_tr_id = 0
    rec_send_dist = 0

    for i in range(current_index + 1, len(df)):
        next_row = df.iloc[i]
        if isinstance(next_row.get("event_type"), str) and "ball receipt*" in str(next_row["event_type"]).lower():
            return next_row.get("ha_name", "0"), 0

    for num in range(1, 24):
        if isinstance(frame['ha_name'], str):
            if "h" in frame['ha_name']:
                plX, plY = frame.get(f'h{num}_x', np.nan), frame.get(f'h{num}_y', np.nan)
            else:
                plX, plY = frame.get(f'a{num}_x', np.nan), frame.get(f'a{num}_y', np.nan)
            dist = distance_pos(pX, pY, plX, plY)
            if (not pd.isna(dist)) and dist < min_dist:
                min_dist = dist
                rec_tr_id = ("h" if "h" in frame['ha_name'] else "a") + str(num)
                rec_send_dist = dist
    return rec_tr_id, rec_send_dist

def create_edge_features(df1):
    interactions = []
    timedelta = list(((df1['seconds_event'].shift(1) - df1['seconds_event']) * -1).fillna(0))
    OP_OP = []; OP_BR = []; OP_DP = []; OP_B = []; OP_G = []; OP_C = []
    DP_B = []; DP_T = []; DP_P = []; DP_DB = []; DP_F = []
    B_T = []; B_G = []; DP_M = []; Other = []
    p1_ids = []; p2_ids = []; p2_dists = []

    for ev_idx, event in df1.iterrows():
        p1_id = event['ha_name']
        p2_id = 0
        p2_dist = 0

        etype = event.get('event_type', '')
        if etype == "Pass":
            interactions.append('pass')
            p2_id, p2_dist = get_pass_reciever(df1, ev_idx)
            flags = dict(OP_OP=1, OP_BR=0, OP_DP=0, OP_B=0, OP_G=0, OP_C=0,
                         DP_B=0, DP_T=0, B_T=0, B_G=0, DP_DB=0, DP_P=0, DP_M=0, DP_F=0, Other=0)

        elif etype == "Ball Receipt*":
            interactions.append('ball_receipt')
            flags = dict(OP_OP=0, OP_BR=1, OP_DP=0, OP_B=0, OP_G=0, OP_C=0,
                         DP_B=0, DP_T=0, B_T=0, B_G=0, DP_DB=0, DP_P=0, DP_M=0, DP_F=0, Other=0)

        elif etype == "Shot":
            interactions.append('shot')
            if isinstance(event["ha_name"], str) and "h" in event["ha_name"]:
                p2_id = event['a_goalkeeper_id']
            else:
                p2_id = event['h_goalkeeper_id']
            p2_dist = event.get('ha_dist', 0)
            flags = dict(OP_OP=0, OP_BR=0, OP_DP=0, OP_B=0, OP_G=1, OP_C=0,
                         DP_B=0, DP_T=0, B_T=0, B_G=0, DP_DB=0, DP_P=0, DP_M=0, DP_F=0, Other=0)

        elif etype == "Dribble":
            interactions.append('dribble')
            p2_id = df1.iloc[ev_idx-1]["ha_name"] if ev_idx > 0 else "0"
            if p2_id == "0" or p1_id == "0":
                p2_dist = 0
            else:
                p2_dist = distance_pos(event.get(f"{p1_id}_x", np.nan), event.get(f"{p1_id}_y", np.nan),
                                       event.get(f"{p2_id}_x", np.nan), event.get(f"{p2_id}_y", np.nan))
            flags = dict(OP_OP=0, OP_BR=0, OP_DP=0, OP_B=1, OP_G=0, OP_C=0,
                         DP_B=0, DP_T=0, B_T=0, B_G=0, DP_DB=0, DP_P=0, DP_M=0, DP_F=0, Other=0)

        elif etype in ["Interception", "Block", "Dispossessed"]:
            interactions.append('interception')
            p2_id = df1.iloc[ev_idx-1]["ha_name"] if ev_idx > 0 else "0"
            if p2_id == "0" or p1_id == "0":
                p2_dist = 0
            else:
                p2_dist = distance_pos(event.get(f"{p1_id}_x", np.nan), event.get(f"{p1_id}_y", np.nan),
                                       event.get(f"{p2_id}_x", np.nan), event.get(f"{p2_id}_y", np.nan))
            flags = dict(OP_OP=0, OP_BR=0, OP_DP=0, OP_B=0, OP_G=0, OP_C=0,
                         DP_B=1, DP_T=0, B_T=0, B_G=0, DP_DB=0, DP_P=0, DP_M=0, DP_F=0, Other=0)

        elif etype == "Clearance":
            interactions.append('clearence')
            p2_id = df1.iloc[ev_idx-1]["ha_name"] if ev_idx > 0 else "0"
            if p2_id == "0" or p1_id == "0":
                p2_dist = 0
            else:
                p2_dist = distance_pos(event.get(f"{p1_id}_x", np.nan), event.get(f"{p1_id}_y", np.nan),
                                       event.get(f"{p2_id}_x", np.nan), event.get(f"{p2_id}_y", np.nan))
            flags = dict(OP_OP=0, OP_BR=0, OP_DP=0, OP_B=0, OP_G=0, OP_C=0,
                         DP_B=0, DP_T=1, B_T=0, B_G=0, DP_DB=0, DP_P=0, DP_M=0, DP_F=0, Other=0)

        elif etype in ["Duel", "50/50"]:
            interactions.append('duel')
            p2_id = df1.iloc[ev_idx-1]["ha_name"] if ev_idx > 0 else "0"
            if p2_id == "0" or p1_id == "0":
                p2_dist = 0
            else:
                p2_dist = distance_pos(event.get(f"{p1_id}_x", np.nan), event.get(f"{p1_id}_y", np.nan),
                                       event.get(f"{p2_id}_x", np.nan), event.get(f"{p2_id}_y", np.nan))
            flags = dict(OP_OP=0, OP_BR=0, OP_DP=0, OP_B=0, OP_G=0, OP_C=0,
                         DP_B=0, DP_T=0, B_T=0, B_G=0, DP_DB=1, DP_P=0, DP_M=0, DP_F=0, Other=0)

        elif etype in ["Carry", "Ball recovery"]:
            interactions.append('carry')
            p2_id = 0
            p2_dist = 0
            flags = dict(OP_OP=0, OP_BR=0, OP_DP=0, OP_B=0, OP_G=0, OP_C=1,
                         DP_B=0, DP_T=0, B_T=0, B_G=0, DP_DB=0, DP_P=0, DP_M=0, DP_F=0, Other=0)

        elif etype == "Pressure":
            interactions.append('pressure')
            p2_id = df1.iloc[ev_idx-1]["ha_name"] if ev_idx > 0 else "0"
            if p1_id == "0" or p2_id == "0":
                p2_dist = 0
            else:
                p2_dist = distance_pos(event.get(f"{p1_id}_x", np.nan), event.get(f"{p1_id}_y", np.nan),
                                       event.get(f"{p2_id}_x", np.nan), event.get(f"{p2_id}_y", np.nan))
            flags = dict(OP_OP=0, OP_BR=0, OP_DP=0, OP_B=0, OP_G=0, OP_C=0,
                         DP_B=0, DP_T=0, B_T=0, B_G=0, DP_DB=0, DP_P=1, DP_M=0, DP_F=0, Other=0)

        elif etype == "Miscontrol":
            interactions.append('miscontrol')
            p2_id = 0
            p2_dist = 0
            flags = dict(OP_OP=0, OP_BR=0, OP_DP=0, OP_B=0, OP_G=0, OP_C=0,
                         DP_B=0, DP_T=0, B_T=0, B_G=0, DP_DB=0, DP_P=0, DP_M=1, DP_F=0, Other=0)

        elif etype == "Foul Committed":
            interactions.append('foul')
            p2_id = df1.iloc[ev_idx+1]["ha_name"] if ev_idx + 1 < len(df1) else "0"
            if p1_id == "0" or p2_id == "0":
                p2_dist = 0
            else:
                p2_dist = distance_pos(event.get(f"{p1_id}_x", np.nan), event.get(f"{p1_id}_y", np.nan),
                                       event.get(f"{p2_id}_x", np.nan), event.get(f"{p2_id}_y", np.nan))
            flags = dict(OP_OP=0, OP_BR=0, OP_DP=0, OP_B=0, OP_G=0, OP_C=0,
                         DP_B=0, DP_T=0, B_T=0, B_G=0, DP_DB=0, DP_P=0, DP_M=0, DP_F=1, Other=0)

        else:
            interactions.append('other')
            p2_id = 0
            p2_dist = 0
            flags = dict(OP_OP=0, OP_BR=0, OP_DP=0, OP_B=0, OP_G=0, OP_C=0,
                         DP_B=0, DP_T=0, B_T=0, B_G=0, DP_DB=0, DP_P=0, DP_M=0, DP_F=0, Other=1)

        p1_ids.append(p1_id); p2_ids.append(p2_id); p2_dists.append(p2_dist)
        OP_OP.append(flags["OP_OP"]); OP_BR.append(flags["OP_BR"]); OP_DP.append(flags["OP_DP"]); OP_B.append(flags["OP_B"])
        OP_G.append(flags["OP_G"]); OP_C.append(flags["OP_C"]); DP_B.append(flags["DP_B"]); DP_T.append(flags["DP_T"])
        B_T.append(flags["B_T"]); B_G.append(flags["B_G"]); DP_DB.append(flags["DP_DB"]); DP_P.append(flags["DP_P"])
        DP_M.append(flags["DP_M"]); DP_F.append(flags["DP_F"]); Other.append(flags["Other"])

    return (interactions, timedelta, p1_ids, p2_ids, p2_dists, OP_OP, OP_BR, OP_DP, OP_B,
            OP_G, OP_C, DP_B, DP_T, B_T, B_G, DP_DB, DP_P, DP_M, DP_F, Other)

def create_datasets(df1):
    (interactions, timedelta, p1_ids, p2_ids, p2_dists, OP_OP, OP_BR, OP_DP, OP_B, OP_G, OP_C,
     DP_B, DP_T, B_T, B_G, DP_DB, DP_P, DP_M, DP_F, Other) = create_edge_features(df1)

    df1['interactions'] = interactions
    df1['timedelta'] = timedelta
    df1['p1_id'] = p1_ids
    df1['p2_id'] = p2_ids
    df1['p2_dist'] = p2_dists
    df1['OP_OP'] = OP_OP; df1['OP_BR'] = OP_BR; df1['OP_DP'] = OP_DP; df1['OP_B'] = OP_B
    df1['OP_G'] = OP_G; df1['OP_C'] = OP_C; df1['DP_B'] = DP_B; df1['DP_T'] = DP_T
    df1['B_T'] = B_T; df1['B_G'] = B_G; df1['DP_DB'] = DP_DB; df1['DP_P'] = DP_P
    df1['DP_M'] = DP_M; df1['DP_F'] = DP_F; df1['Other'] = Other

    column_names = [
        "p1_id", "p2_id", "p2_dist", "timedelta", "timestamp", "seconds",
        "event_type", "event_type_2", "home_attack_dir",
        "OP_OP", "OP_BR", "OP_DP", "OP_B", "OP_G", "OP_C",
        "DP_B", "DP_T", "B_T", "B_G", "DP_DB", "DP_P", "DP_M", "DP_F", "Other"
    ]

    for num in range(1, 24):
        column_names.extend([
            f"h{num}_x", f"a{num}_x", f"h{num}_y", f"a{num}_y",
            f"h{num}_speed", f"a{num}_speed", f"h{num}_vx", f"a{num}_vx", f"h{num}_vy", f"a{num}_vy",
            f"h{num}_gl_dist", f"h{num}_gl_ang", f"h{num}_gl_dir_x", f"h{num}_gl_dir_y",
            f"a{num}_gl_dist", f"a{num}_gl_ang", f"a{num}_gl_dir_x", f"a{num}_gl_dir_y",
            f"h{num}_tl_dist", f"h{num}_tl_ang", f"h{num}_tl_dir_x", f"h{num}_tl_dir_y",
            f"a{num}_tl_dist", f"a{num}_tl_ang", f"a{num}_tl_dir_x", f"a{num}_tl_dir_y",
        ])

    column_names.extend([
        "ball_x", "ball_y", "ball_speed", "ball_vx", "ball_vy",
        "ball_gl_dist", "ball_gl_ang", "ball_gl_dir_x", "ball_gl_dir_y",
        "ball_tl_dist", "ball_tl_ang", "ball_tl_dir_x", "ball_tl_dir_y",
        "bc_dist"
    ])

    column_names.extend(["ha_x", "ha_y", "ha_vx", "ha_vy", "ha_speed", "ha_dist", "ha_ang",
                         "start_x", "start_y", "end_x", "end_y", "ha_name"])

    structured_data = []
    for _, event in df1.iterrows():
        row_out = []
        row_out.append(event['p1_id'])
        row_out.append(event['p2_id'])
        row_out.append(event['p2_dist'])
        row_out.append(event['timedelta'])
        row_out.append(event['time'])
        row_out.append(event['seconds_event'])
        row_out.append(event['event_type'])
        row_out.append(event['event_type_2'])
        row_out.append(event['home_attack_dir'])
        row_out.extend([
            event['OP_OP'], event['OP_BR'], event['OP_DP'], event['OP_B'], event['OP_G'], event['OP_C'],
            event['DP_B'], event['DP_T'], event['B_T'], event['B_G'], event['DP_DB'], event['DP_P'], event['DP_M'], event['DP_F'], event['Other']
        ])

        for num in range(1, 24):
            row_out.extend([
                event.get(f"h{num}_x", np.nan), event.get(f"a{num}_x", np.nan),
                event.get(f"h{num}_y", np.nan), event.get(f"a{num}_y", np.nan),
                event.get(f"h{num}_speed", np.nan), event.get(f"a{num}_speed", np.nan),
                event.get(f"h{num}_vx", np.nan), event.get(f"a{num}_vx", np.nan),
                event.get(f"h{num}_vy", np.nan), event.get(f"a{num}_vy", np.nan),

                event.get(f"h{num}_gl_dist", np.nan), event.get(f"h{num}_gl_ang", np.nan),
                event.get(f"h{num}_gl_dir_x", np.nan), event.get(f"h{num}_gl_dir_y", np.nan),

                event.get(f"a{num}_gl_dist", np.nan), event.get(f"a{num}_gl_ang", np.nan),
                event.get(f"a{num}_gl_dir_x", np.nan), event.get(f"a{num}_gl_dir_y", np.nan),

                event.get(f"h{num}_tl_dist", np.nan), event.get(f"h{num}_tl_ang", np.nan),
                event.get(f"h{num}_tl_dir_x", np.nan), event.get(f"h{num}_tl_dir_y", np.nan),

                event.get(f"a{num}_tl_dist", np.nan), event.get(f"a{num}_tl_ang", np.nan),
                event.get(f"a{num}_tl_dir_x", np.nan), event.get(f"a{num}_tl_dir_y", np.nan),
            ])

        row_out.extend([
            event.get("ball_x", np.nan), event.get("ball_y", np.nan),
            event.get("ball_speed", np.nan), event.get("ball_vx", np.nan), event.get("ball_vy", np.nan),
            event.get("ball_gl_dist", np.nan), event.get("ball_gl_ang", np.nan),
            event.get("ball_gl_dir_x", np.nan), event.get("ball_gl_dir_y", np.nan),
            event.get("ball_tl_dist", np.nan), event.get("ball_tl_ang", np.nan),
            event.get("ball_tl_dir_x", np.nan), event.get("ball_tl_dir_y", np.nan),
            event.get("bc_dist", np.nan)
        ])

        row_out.extend([
            event.get("ha_x", np.nan), event.get("ha_y", np.nan),
            event.get("ha_vx", np.nan), event.get("ha_vy", np.nan),
            event.get("ha_speed", np.nan),
            event.get("ha_dist", np.nan), event.get("ha_ang", np.nan),
            event.get("start_x", np.nan), event.get("start_y", np.nan),
            event.get("end_x", np.nan), event.get("end_y", np.nan),
            event.get("ha_name", "0")
        ])

        structured_data.append(row_out)

    return pd.DataFrame(structured_data, columns=column_names)

if __name__ == "__main__":
    from fuzzywuzzy import process as fuzzywuzzy_process

    os.makedirs(output_dir, exist_ok=True)
    target_matches = select_target_matches(folder_path, match_dir, id_map)
    csv_files = glob.glob(f"{folder_path}/*.csv")

    for csv_file in csv_files:
        df1 = pd.read_csv(csv_file)

        statsbomb_id = str(df1["match_id"].iloc[0])
        skillcorner_id = id_map.get(statsbomb_id, None)
        match_key = f"{statsbomb_id}_{skillcorner_id}"
        if match_key not in target_matches:
            continue


        target_filename = f"{skillcorner_id}.json"
        target_path = os.path.join(match_dir, target_filename)
        if os.path.exists(target_path):
            with open(target_path, "r", encoding="utf-8") as f:
                match = json.load(f)
        else:
            continue

        players = match["players"]

        goalkeeper_ids = []
        for player in players:
            prole = player.get("player_role", {})
            if isinstance(prole, dict) and prole.get("name") == "Goalkeeper":
                pid = player.get("trackable_object")
                if pid is not None:
                    goalkeeper_ids.append(pid)

        obj_to_label = {}
        for i in range(1, 24):
            h_col = f"h{i}_trackable_id"; a_col = f"a{i}_trackable_id"
            if h_col in df1.columns:
                vals = [v for v in df1[h_col].dropna().unique() if v != 0]
                if len(vals) > 0:
                    obj_to_label[int(vals[0])] = f"h{i}"
            if a_col in df1.columns:
                vals = [v for v in df1[a_col].dropna().unique() if v != 0]
                if len(vals) > 0:
                    obj_to_label[int(vals[0])] = f"a{i}"

        name_to_label = {}
        for p in match["players"]:
            t_obj = p["trackable_object"]
            if t_obj not in obj_to_label:
                continue
            label = obj_to_label[t_obj]
            full_name = f"{p['first_name']} {p['last_name']}".strip()
            clean_name = full_name.lower().replace(" ", "")
            name_to_label[clean_name] = label

        valid_names = list(name_to_label.keys())

        def fuzzy_match(name):
            if pd.isna(name) or str(name).strip() == "":
                return "0"
            name_ = str(name).lower().replace(" ", "")
            res = fuzzywuzzy_process.extractOne(name_, valid_names)
            if res is not None:
                match_name, score = res
                if score > 80:
                    return name_to_label.get(match_name, "0")
            return "0"

        df1["player_clean"] = df1["player"].fillna("").apply(lambda x: str(x).lower().replace(" ", ""))
        df1["ha_name"] = df1["player_clean"].apply(fuzzy_match)
        df1[["ha_x", "ha_y", "ha_vx", "ha_vy", "ha_speed"]] = df1.apply(get_ha_xy, axis=1)

        df1["needs_flipping"] = np.nan
        ha_x = df1["ha_x"].astype(float)
        ha_y = df1["ha_y"].astype(float)
        sx   = df1["start_x"].astype(float)
        sy   = df1["start_y"].astype(float)

        d_raw = np.sqrt((ha_x - sx)**2 + (ha_y - sy)**2)
        ha_xf = 105 - ha_x; ha_yf = 68 - ha_y
        d_flip = np.sqrt((ha_xf - sx)**2 + (ha_yf - sy)**2)
        df1["needs_flipping"] = (d_flip < d_raw).astype("boolean")

        event_x_cols = ["start_x", "end_x"]
        event_y_cols = ["start_y", "end_y"]
        mask_flip = df1["needs_flipping"]
        df1.loc[mask_flip, event_x_cols] = 105 - df1.loc[mask_flip, event_x_cols]
        df1.loc[mask_flip, event_y_cols] = 68 - df1.loc[mask_flip, event_y_cols]

        h_goalkeeper_id, a_goalkeeper_id = None, None
        for keeper_obj in goalkeeper_ids:
            label = obj_to_label.get(keeper_obj)
            if not label:
                continue
            if label.startswith('h'):
                h_goalkeeper_id = label
            else:
                a_goalkeeper_id = label
        df1["h_goalkeeper_id"] = h_goalkeeper_id
        df1["a_goalkeeper_id"] = a_goalkeeper_id

        h_gk_x = df1.get(f"{h_goalkeeper_id}_x") if h_goalkeeper_id else pd.Series([np.nan] * len(df1))
        a_gk_x = df1.get(f"{a_goalkeeper_id}_x") if a_goalkeeper_id else pd.Series([np.nan] * len(df1))
        df1["h_goalkeeper_x"] = h_gk_x
        df1["a_goalkeeper_x"] = a_gk_x

        df1["home_attack_dir"] = np.nan
        df1.loc[(df1["h_goalkeeper_x"] - 105).abs() < 45, "home_attack_dir"] = "right to left"
        df1.loc[(df1["h_goalkeeper_x"] - 105).abs() > 60, "home_attack_dir"] = "left to right"
        df1.loc[((df1["h_goalkeeper_x"] - 105).abs() >= 45) & ((df1["h_goalkeeper_x"] - 105).abs() <= 60), "home_attack_dir"] = "other"

        df1['dT'] = df1['seconds_event'] - df1['seconds_event'].shift(1)
        epsilon = 1e-8
        MAXSPEED = 12

        new_cols_dict = {}
        for p_num in range(1, 24):
            Vhx = df1[f'h{p_num}_x'].diff() / (df1['dT'] + epsilon)
            Vhy = df1[f'h{p_num}_y'].diff() / (df1['dT'] + epsilon)
            Vax = df1[f'a{p_num}_x'].diff() / (df1['dT'] + epsilon)
            Vay = df1[f'a{p_num}_y'].diff() / (df1['dT'] + epsilon)

            rawh_speed = np.sqrt(Vhx**2 + Vhy**2)
            rawa_speed = np.sqrt(Vax**2 + Vay**2)

            if MAXSPEED > 0:
                Vhx[rawh_speed > MAXSPEED] = np.nan
                Vhy[rawh_speed > MAXSPEED] = np.nan
                Vax[rawa_speed > MAXSPEED] = np.nan
                Vay[rawa_speed > MAXSPEED] = np.nan

            new_cols_dict[f'h{p_num}_vx'] = np.round(Vhx, 2)
            new_cols_dict[f'h{p_num}_vy'] = np.round(Vhy, 2)
            new_cols_dict[f'a{p_num}_vx'] = np.round(Vax, 2)
            new_cols_dict[f'a{p_num}_vy'] = np.round(Vay, 2)
            new_cols_dict[f'h{p_num}_speed'] = np.round(np.sqrt(Vhx**2 + Vhy**2), 2)
            new_cols_dict[f'a{p_num}_speed'] = np.round(np.sqrt(Vax**2 + Vay**2), 2)

        ball_vx = df1["ball_x"].diff() / (df1['dT'] + epsilon)
        ball_vy = df1["ball_y"].diff() / (df1['dT'] + epsilon)
        ball_vz = df1["ball_z"].diff() / (df1['dT'] + epsilon) if "ball_z" in df1.columns else 0
        ball_speed = np.sqrt((ball_vx ** 2) + (ball_vy ** 2) + (ball_vz ** 2))

        new_cols_dict["ball_vx"] = np.round(ball_vx, 2)
        new_cols_dict["ball_vy"] = np.round(ball_vy, 2)
        new_cols_dict["ball_speed"] = np.round(ball_speed, 2)

        df1 = pd.concat([df1, pd.DataFrame(new_cols_dict)], axis=1)

        perrow_feats = df1.apply(apply_all_features_per_row, axis=1)
        df1 = pd.concat([df1, perrow_feats], axis=1)

        df1["needs_flipping"] = df1["needs_flipping"].where(df1["home_attack_dir"].notna(), pd.NA)
        df1["needs_flipping"] = df1["needs_flipping"].astype("boolean")

        df1_pruned = take_first_and_last_with_interp(df1, time_col='seconds_event').reset_index(drop=True)

        node_edges_df = create_datasets(df1_pruned)
        filename = f"{statsbomb_id}_{skillcorner_id}_node_edges.csv"
        filepath = os.path.join(output_dir, filename)
        node_edges_df.to_csv(filepath, index=False)

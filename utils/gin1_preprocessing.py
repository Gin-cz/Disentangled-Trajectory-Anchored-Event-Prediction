import json
import pandas as pd
import numpy as np
import os
import math
from rapidfuzz import process, fuzz
import pandas as pd
import tqdm
print("pandas version:", pd.__version__)
FIELD_LENGTH = 105.0  # unit: meters
FIELD_WIDTH = 68.0  # unit: meters
GOAL_WIDTH = 7.32  # unit: meters
PENALTY_X = 105.0/2-16.5 # left point (unit: meters)
PENALTY_Y = 40.32 # upper point (unit: meters)

base_dir = os.environ.get("DISTRAJ_RAW_DATA_ROOT", "data/raw/laliga_23")
event_dir = os.path.join(base_dir, "statsbomb/events_test")
tracking_dir = os.path.join(base_dir, "skillcorner_v2/tracking_test")
match_dir = os.path.join(base_dir, "skillcorner_v2/match_test")
metadata_dir = os.path.join(base_dir, "skillcorner_v2/metadata")
id_mapping_path = os.environ.get("DISTRAJ_ID_MAPPING", os.path.join("data", "id_matching.csv"))
aligned_output_dir = os.environ.get("DISTRAJ_ALIGNED_DIR", os.path.join("data", "interim", "syn_data_weighted"))

# 读取 id 映射表
id_map_df = pd.read_csv(id_mapping_path)

# 建立 dict：{statsbomb_id: skillcorner_id}
id_map = dict(zip(id_map_df["match_id_statsbomb"].astype(str), id_map_df["match_id_skillcorner"].astype(str)))

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
    """
    基于相邻事件的时间差(Δt)做加权插值：
    在两次 Ball Receipt 之间，用区间内事件的累计Δt占比，映射到 [a1, a2]。
    
    参数
    ----
    df_event : DataFrame
        至少包含 ['seconds'] 列，并已是比赛全局时间(半场已折算)。
    receipt_matches : dict
        {receipt_seconds: adjusted_seconds}，接球事件的匹配结果。
    eps : float
        防止除零的小常数。
    返回
    ----
    df_event : DataFrame
        新增列 'adjusted_seconds_interp_weighted'
    """
    if not receipt_matches:
        # 没有锚点时，不插值
        df_event["adjusted_seconds_interp_weighted"] = np.nan
        return df_event

    df_event = df_event.sort_values("seconds").reset_index(drop=True).copy()
    receipt_times = sorted(receipt_matches.keys())

    # 先置空
    df_event["adjusted_seconds_interp_weighted"] = np.nan

    # 逐段处理 [t1, t2]
    for i in range(len(receipt_times) - 1):
        t1, t2 = receipt_times[i], receipt_times[i + 1]
        a1, a2 = receipt_matches[t1], receipt_matches[t2]

        # 本段内事件（含端点）
        mask = (df_event["seconds"] >= t1) & (df_event["seconds"] <= t2)
        sub = df_event.loc[mask].copy().sort_values("seconds")

        if sub.empty:
            continue

        if t2 - t1 <= eps:
            # 两个锚点时间几乎相等：整段都给 a1
            df_event.loc[mask, "adjusted_seconds_interp_weighted"] = a1
            continue

        # 计算相邻事件时间差Δt（段内）
        # 第一个事件Δt设为0，表示从t1到第一个事件没有“经历”的段内时间
        dt = sub["seconds"].diff().fillna(0.0)
        # 数值稳定
        dt = dt.clip(lower=0.0)

        # 累计权重（累计“经历”的段内时间）
        cum_w = dt.cumsum()

        total_w = float(cum_w.iloc[-1])

        if total_w <= eps:
            # 如果段内所有事件seconds都相同（或几乎相同），退化到线性比例插值
            ratio = (sub["seconds"] - t1) / (t2 - t1)
        else:
            # 按累计Δt占比做非均匀插值
            ratio = cum_w / total_w

        # 将 ratio 映射到 [a1, a2]
        adj_vals = a1 + (a2 - a1) * ratio

        df_event.loc[sub.index, "adjusted_seconds_interp_weighted"] = adj_vals

    # 段外填充：首段前、末段后
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

# 规范化名字
def preprocess_name(name):
    if isinstance(name, str):
        parts = name.split()  # 按空格拆分
        if len(parts) > 1:
            # 只保留第一个和最后一个名字
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
    result = process.extractOne(event_name, trackable_names, scorer=fuzz.ratio)
    if result:
        match, score = result[0], result[1]  # 解包匹配项和分数
        return match if score > 80 else None  # 设定相似度阈值
    return None

# kickoff
def get_kickoff_data(kickoff_name, home_or_away, df_tracking, id_to_info, period):

    if not kickoff_name or kickoff_name not in {info["normalized_name"] for info in id_to_info.values()}:
        return []

    # 获取球员对应的 trackable_object ID
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
    
    # 根据主客队选择列名
    prefix = "h" if home_or_away == 1 else "a"
    x_column = f"{prefix}{int(matching_id_value)}_x"
    y_column = f"{prefix}{int(matching_id_value)}_y"
    # 检查列是否存在，并提取数据
    if x_column in df_tracking.columns and y_column in df_tracking.columns:
        matching_rows = df_tracking[df_tracking["period"]== period][[x_column, y_column, "seconds"]].dropna()
        return [
            [(row[x_column], row[y_column]), row["seconds"]]
            for _, row in matching_rows.iterrows()
        ]
    else:
        return []


# 计算加速度和最大加速度发生的时间戳
def calculate_acceleration_and_max_seconds(data, player_position,have_name):
    """
    Calculate acceleration and find the timestamp with the maximum acceleration.

    Parameters:
    - data: List of lists containing [timestamp, x, y, z] coordinates of the ball.
    - player_position: List of lists containing player positions and corresponding timestamps.

    Returns:
    - max_acceleration_seconds: The timestamp at which maximum acceleration occurs.
    - max_acceleration: The maximum acceleration value.
    """
    # Extract seconds, x, y, z coordinates
    m_seconds = [entry[0] for entry in data]
    m_x = np.array([entry[1] for entry in data])
    m_y = np.array([entry[2] for entry in data])
    m_z = np.array([entry[3] for entry in data])

    # Calculate differences
    delta_x = np.diff(m_x)
    delta_y = np.diff(m_y)
    delta_z = np.diff(m_z)
    delta_t = np.diff(m_seconds)

    # Avoid division by zero
    delta_t[delta_t == 0] = 1e-6  # Use a small epsilon value to avoid NaN

    # Calculate velocity components and magnitude
    vx = delta_x / delta_t
    vy = delta_y / delta_t
    vz = delta_z / delta_t
    velocity_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)

    # Calculate acceleration
    delta_vx = np.diff(vx) / delta_t[1:]  # Acceleration in x direction
    delta_vy = np.diff(vy) / delta_t[1:]  # Acceleration in y direction
    delta_vz = np.diff(vz) / delta_t[1:]  # Acceleration in z direction
    acceleration_magnitude = np.sqrt(delta_vx**2 + delta_vy**2 + delta_vz**2)

    max_acceleration = None
    max_acceleration_seconds = None

    # Calculate distances between ball and player
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

        distances = np.array(distances)  # Convert to numpy array
        valid_indices = np.where((distances <= 2) & (~np.isnan(distances)))[0]
        if len(valid_indices) > 0 and len(acceleration_magnitude) > 0:
            min_index = valid_indices[np.argmin(distances[valid_indices])]  # 找出最小值的索引
            min_distance = distances[min_index]
            max_acceleration_index_in_valid = np.argmax(acceleration_magnitude)
            max_acceleration = acceleration_magnitude[max_acceleration_index_in_valid]
            max_acceleration_seconds = m_seconds[max_acceleration_index_in_valid + 2]  # +2 due to second order difference
            break

    return max_acceleration_seconds if max_acceleration_seconds is not None else 0.0, max_acceleration if max_acceleration is not None else 0.0, distances

# score function for distance

# 计算距离的函数
def calculate_distance(df, x_column, y_column, ref_x, ref_y):
    # 确保列为数值型并填充 NaN
    df[x_column] = pd.to_numeric(df[x_column], errors="coerce").fillna(0)
    df[y_column] = pd.to_numeric(df[y_column], errors="coerce").fillna(0)

    # 矢量化计算距离
    return np.sqrt(
        (df[x_column] - ref_x) ** 2 +
        (df[y_column] - ref_y) ** 2
    )

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
    """
    Gets a window of frames around the given action_second using vectorized operations.
    """
    return df_tracking[
        (df_tracking["adjusted_seconds"].between(action_second - ta, action_second + ta)) &
        df_tracking["period"].notna()
    ]

# aligned df 
def compute_offsets(df_m, df_e):
    df_m = df_m.copy()  # 确保对原始 DataFrame 的副本操作

    # 初始化偏移列
    df_m['offset'] = 0

    # 遍历 df_m 的每一段，计算分段偏移
    for i in range(len(df_m)):
        if i == 0:
            start = df_e['seconds'].min()

        else:
            start = df_m.iloc[i - 1]['e_seconds']


        end = df_m.iloc[i]['e_seconds']
        adjusted_end = df_m.iloc[i]['adjusted_seconds']


        # 计算时间占比
        event_rows = df_e[(df_e['seconds'] > start) & (df_e['seconds'] <= end)]
        if not event_rows.empty:
            time_ratio = (event_rows['seconds'] - start) / (end - start)

            df_e.loc[event_rows.index, 'adjusted_seconds_1'] = event_rows['seconds'] + time_ratio * (adjusted_end- end)
    return df_m, df_e

def align_events(df_event11, df_tracking11, df_m11):
    df_m12, df_event12 = compute_offsets(df_m11, df_event11)
    # 对 df_event 和 df_tracking 进行对齐
    df_event12['adjusted_seconds_rounded'] = df_event12['adjusted_seconds_1'].round(1)
    print(df_event12["adjusted_seconds_rounded"].iloc[5:15])
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
        # 第一条永远保留
        first = sub.iloc[0].copy()
        new_rows.append(first)

        # 如果这个时间点有多条事件
        if len(sub) > 1:
            last = sub.iloc[-1].copy()
            # 找下一个 unique 时间
            if idx + 1 < len(unique_times):
                t2 = unique_times[idx + 1]
                mid_t = (t1 + t2) / 2.0
                next_seconds_event = df[df[time_col] == t2]["seconds_event"].min()
                mid_e_t = (sub.iloc[-1]["seconds_event"] + next_seconds_event) / 2.0
            else:
                mid_t = t1
                mid_e_t = sub.iloc[-1]["seconds_event"]
            
            # 修改最后一条事件的 adjusted_seconds 和 seconds_event
            last[time_col] = round(mid_t, 1)
            last["seconds_event"] = round(mid_e_t, 1)    

            #  同步 tracking 部分 (h1..a23, ball_x 等) —— 找最接近 mid_t 的 tracking 帧
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
                "normalized_name": normalize_name(preprocess_name(info["name"])),  # 统一！
                "name": info["name"],
                "team": info["team"],
                "role": info["role"],
                "id": info["id"]
            })
    return pd.DataFrame(rows)

print("Starting preprocessing...")

for fname in os.listdir(event_dir):
    if not fname.endswith(".csv"):
        continue
    
    # 提取 statsbomb match id
    statsbomb_id = fname.replace(".csv", "")
    event_path = os.path.join(event_dir, fname)

    skillcorner_id = id_map.get(statsbomb_id)
    if skillcorner_id is None:
        print(f"No matching SkillCorner ID for StatsBomb ID {statsbomb_id}")
        continue
    # 构造 skillcorner 路径
    tracking_path = os.path.join(tracking_dir, f"{skillcorner_id}.json")
    match_path = os.path.join(match_dir, f"{skillcorner_id}.json")
    metadata_path = os.path.join(metadata_dir, "players.json")
    if not os.path.exists(tracking_path) or not os.path.exists(match_path):
        print(f"Missing skillcorner files for ID {skillcorner_id}")
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
    print(f"Processing match {statsbomb_id} (SkillCorner ID: {skillcorner_id}): {home_team_name} vs {away_team_name}")
    # trackable_objects
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

    # event DataFrame
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
                # end_x, end_y = rotate_around_center(end_x, end_y)
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
            # start_x, start_y = rotate_around_center(start_x, start_y)
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
    # reverse the event location
    reverse_events = []

    # 遍历每一个事件（从第一个到倒数第三个，因为我们需要后两个事件进行判断）
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
        # 执行反转
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

    # 对最后两行标记为不需要反转
    df_event.loc[len(df_event) - 2:, 'needs_reverse'] = False
    # event_output_csv_path = os.path.join("data", "debug", "reverse_events.csv")
    # df_event.to_csv(event_output_csv_path, index=False)
    # print(f"事件 DataFrame 已保存为 CSV 文件：{event_output_csv_path}")


    # tracking dataframe
    home_side = None
    home_gk_x = away_gk_x = None
    df_tracking_list = []


    for frame in tracking_data:
        home_tracking = [None] * 2 * 23
        away_tracking = [None] * 2 * 23
        home_trackable_ids = [None] * 23  # 存储主队球员的 trackable_object
        away_trackable_ids = [None] * 23  # 存储客队球员的 trackable_object
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

    # tracking_output_csv_path = os.path.join("data", "debug", "tracking.csv")
    # df_tracking.to_csv(tracking_output_csv_path, index=False)
    # print(f"追踪数据 DataFrame 已保存为 CSV 文件：{tracking_output_csv_path}")

    # 构建映射字典，包含 label, name, team
    label_to_info = {}
    for obj_id, info in trackable_objects.items():
        # 跳过没有 'id' 字段的（比如 ball）
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

    # 创建一个 trackable_object 到标准化名字的映射
    trackable_name_map = {tid: normalize_name(preprocess_name(info['name'])) for tid, info in trackable_objects.items()}
    # print(trackable_name_map)    # 为每个球员生成名字列
    # for i in range(1, 24):
    #     # 处理主队球员名字
    #     trackable_column = f"h{i}_trackable_id"
    #     name_column = f"h{i}_name"
    #     df_tracking[name_column] = df_tracking[trackable_column].map(trackable_name_map)
        
    #     # 处理客队球员名字
    #     trackable_column = f"a{i}_trackable_id"
    #     name_column = f"a{i}_name"
    #     df_tracking[name_column] = df_tracking[trackable_column].map(trackable_name_map)

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
        # if sb_seconds > 45*60+5:
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

    # Determine home or away and calculate max seconds
    if data1_flag:
        if home_or_away1==1:
            max_seconds1,max_acceleration1,distances_1 = calculate_acceleration_and_max_seconds(ball_frames,h_players,False)
            print(f"distances_1 : {distances_1}")
        else:
            max_seconds1,max_acceleration1,distances_1= calculate_acceleration_and_max_seconds(ball_frames,a_players,False)
            print(f"distances_1 : {distances_1}")
    if data2_flag:
        if home_or_away2==1:
            max_seconds2,max_acceleration2,distances_2= calculate_acceleration_and_max_seconds(ball_frames_secondhalf,h_players,False)
        else:
            max_seconds2,max_acceleration2,distances_2= calculate_acceleration_and_max_seconds(ball_frames_secondhalf,a_players,False)


    adjusted_max_seconds2 = round(max_seconds2 - 45*60,1)
    # 保存原来的 seconds 列到 raw_seconds
    df_tracking["raw_seconds"] = df_tracking["seconds"]

    # 构造新的 adjusted_seconds 列
    df_tracking["adjusted_seconds"] = df_tracking["seconds"]
    df_tracking.loc[df_tracking["period"] == 1, "adjusted_seconds"] -= max_seconds1
    df_tracking.loc[df_tracking["period"] == 2, "adjusted_seconds"] -= adjusted_max_seconds2
    df_tracking.loc[df_tracking["period"] == 1, "adjusted_seconds"] = round(df_tracking.loc[df_tracking["period"] == 1, "adjusted_seconds"],1)
    df_tracking.loc[df_tracking["period"] == 2, "adjusted_seconds"] = round(df_tracking.loc[df_tracking["period"] == 2, "adjusted_seconds"],1)
    # Filter pass events
    pass_events = df_event[(df_event['event_type'] == 'Pass') | (df_event['event_type'] == 'Ball Receipt*')]
    pass_events_seconds = pass_events["seconds"]
    ta = 5
        # Build player map for name->label lookup
    df_player_map = create_player_map(df_tracking, trackable_objects)

    windows=[]
    for action_second in pass_events_seconds:
        if action_second is not None and not isinstance(action_second, pd.DataFrame):
            window = get_window_of_frames_around(action_second, df_tracking, ta)
            windows.append([action_second,window])
        elif action_second is not None and isinstance(action_second, pd.DataFrame):
            action_second = action_second.iloc[0]
            print(f"df event second: {action_second}")
            window = get_window_of_frames_around(action_second, df_tracking, ta)
            windows.append([action_second,window])
        else:
            print(f"No window for the event second: {action_second}")

        
    results = []  # 存储所有窗口的计算结果

    try:
        for dfff_window in windows:
            # 转为 DataFrame
            dff_window = pd.DataFrame(dfff_window[1])
            e_second = dfff_window[0]  # 提取事件时间

            # 检查事件是否存在于 df_event
            if e_second not in df_event["seconds"].values:
                raise ValueError(f"Event with seconds={e_second} not found in df_event.")

            # 提取事件信息
            pass_x = df_event.loc[df_event["seconds"] == e_second, "start_x"].iloc[0]
            pass_y = df_event.loc[df_event["seconds"] == e_second, "start_y"].iloc[0]
            e_team = df_event.loc[df_event["seconds"] == e_second, "home_team"].iloc[0]
            e_player = df_event.loc[df_event["seconds"] == e_second, "player"].iloc[0]

            # 计算窗口得分（两距离版）
            window_score = distance_score(pass_x, pass_y, dff_window, e_player, df_player_map)

            # 将结果加入到 results 中
            results.append({"e_seconds": e_second, "score": window_score})

    except IndexError as ie:
        print("IndexError: Please ensure that 'windows' and 'df_event' contain the expected structure.", ie)
    except KeyError as ke:
        print("KeyError: Ensure all required columns exist in 'df_event' and 'dff_window'.", ke)
    except ValueError as ve:
        print("ValueError:", ve)
    except Exception as e:
        print("An unexpected error occurred:", e)

    results_df = pd.DataFrame(results)
    ball_receipts = df_event[df_event["event_type"].str.contains("Ball Receipt", case=False)]

    # results_df["score_value"] = results_df["score"].apply(lambda x: x["score"] if isinstance(x, dict) else None)
    # filtered_results_1 = results_df[results_df["score_value"] > 60]
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
            continue  # 没有插值结果就跳过

        # 找 tracking 中最接近的帧
        nearest_tracking = df_tracking.iloc[(df_tracking["adjusted_seconds"] - adj_sec).abs().argmin()]
        merged_row = {**e_row.to_dict(), **nearest_tracking.to_dict()}
        merged_row["e_seconds"] = e_seconds
        merged_row["adjusted_seconds"] = adj_sec
        merged_rows.append(merged_row)

    merged_df = pd.DataFrame(merged_rows)
    # print(f"✅ merged_df 已生成，共 {len(merged_df)} 行")
    columns_to_keep = [
        "e_seconds", "adjusted_seconds", "match_id", "period", "time", "minute", "second",
        "event_type", "event_type_2", "team", "home_team", "player", "start_x", "start_y",
        "end_x", "end_y", "pass_type", "pass_height", "pass_outcome", "possession_team",
        "needs_reverse", "preprocessed_name", "normalized_name", "matched_name",
        # 所有的 h 列的 x y 和 a 列的 x y
        *[f"h{i}_x" for i in range(1, 24)], *[f"h{i}_y" for i in range(1, 24)],
        *[f"a{i}_x" for i in range(1, 24)], *[f"a{i}_y" for i in range(1, 24)],
        "home_side", "ball_x", "ball_y", "ball_z", "raw_seconds"
    ]

    # 保留指定列
    simplified_df = merged_df[columns_to_keep]
    # simplified_df.to_csv(output_file, index=False)

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

    # # 额外生成一个球员映射表
    # player_mapping_dir = os.path.join("data", "interim", "player_mapping")
    # os.makedirs(player_mapping_dir, exist_ok=True)

    # player_rows = []
    # for team_prefix in ["h", "a"]:
    #     for i in range(1, 24):
    #         tid_col = f"{team_prefix}{i}_trackable_id"
    #         if tid_col not in df_tracking.columns:
    #             continue

    #         # 获取该 trackable_id 在全场第一次出现时的值（避免 NaN）
    #         first_tid = df_tracking[tid_col].dropna().iloc[0] if df_tracking[tid_col].notna().any() else None
    #         if first_tid is None:
    #             continue

    #         # 获取球员信息
    #         info = trackable_objects.get(first_tid)
    #         if not info:
    #             continue
            
    #         # 生成 normalized 名字
    #         normalized_name = normalize_name(preprocess_name(info["name"]))
    #         player_rows.append({
    #             "label": f"{team_prefix}{i}",
    #             "trackable_id": first_tid,
    #             "normalized_name": normalized_name,
    #             "name": info["name"],
    #             "team": info["team"],
    #             "role": info["role"]
    #         })

    # # 转为 DataFrame 并保存
    # df_player_map = pd.DataFrame(player_rows).sort_values("label").reset_index(drop=True)
    # player_map_path = os.path.join(player_mapping_dir, f"{skillcorner_id}_player_mapping.csv")
    # df_player_map.to_csv(player_map_path, index=False)
    # print(f"✅ 已保存球员映射表: {player_map_path}")

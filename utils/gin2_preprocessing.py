import pandas as pd
import numpy as np
import glob
from math import sqrt
import os
import json
from fuzzywuzzy import process
import re


folder_path = os.environ.get("DISTRAJ_ALIGNED_DIR", os.path.join("data", "interim", "syn_data_weighted"))
base_dir = os.environ.get("DISTRAJ_RAW_DATA_ROOT", os.path.join("data", "raw", "laliga_23"))
metadata_dir = os.path.join(base_dir, "skillcorner_v2/metadata")
match_dir = os.path.join(base_dir, "skillcorner_v2/match")
id_mapping_path = os.environ.get("DISTRAJ_ID_MAPPING", os.path.join("data", "id_matching.csv"))
output_dir = os.environ.get("DISTRAJ_NODE_EDGE_DIR", os.path.join("data", "interim", "syn_node_edge_data"))
os.makedirs(output_dir, exist_ok=True)


id_map_df = pd.read_csv(id_mapping_path)
id_map = dict(zip(id_map_df["match_id_statsbomb"].astype(str),
                  id_map_df["match_id_skillcorner"].astype(str)))

csv_files = sorted(glob.glob(f"{folder_path}/*.csv"))
candidates = []  # [(match_key, csv_file)]

for csv_file in csv_files:
    # 读一行拿到 statsbomb_id
    try:
        df_head = pd.read_csv(csv_file, nrows=1)
        if "match_id" not in df_head.columns:
            continue
        statsbomb_id = str(df_head["match_id"].iloc[0])
    except Exception:
        continue

    skillcorner_id = id_map.get(statsbomb_id)
    if not skillcorner_id:
        # 映射缺失的直接跳过
        continue

    match_json = os.path.join(match_dir, f"{skillcorner_id}.json")
    if not os.path.exists(match_json):
        # 没有 match.json 的跳过（你的后续流程需要它）
        continue

    match_key = f"{statsbomb_id}_{skillcorner_id}"
    candidates.append((match_key, csv_file))

# 取前 140 场
pick = candidates[:140]
target_matches = {mk for mk, _ in pick}
# print("需要重新生成的比赛列表:")
# for match in sorted(target_matches):
#     print("  ", match)

# ------------------------------------------------------------
# 基础工具函数
# ------------------------------------------------------------
def distance_pos(x1, y1, x2, y2):
    if pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2):
        return np.nan
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def distance_time(time_from, time_to):
    return abs(time_to - time_from)

# ------------------------------------------------------------
# 几何特征：球门线 / 边线
# ------------------------------------------------------------
def calc_goalline_features(x, y, home_attack_dir, is_home,
                           field_length=105, field_width=68, goal_width=7.32):
    """
    返回: (dist_to_goal, ang_vs_goal_line, dir_x, dir_y)
      - dist_to_goal: 到“对方球门中心”的距离
      - ang_vs_goal_line: “指向球门中心的方向” 与 “球门线（两门柱连线）”的夹角（弧度，0~pi/2附近更小更对准）
      - dir_x, dir_y: 面向球门中心的单位向量
    """
    if pd.isna(x) or pd.isna(y):
        return np.nan, np.nan, np.nan, np.nan

    # 判定对方球门在哪边
    if (home_attack_dir == "left to right" and is_home == 1) or \
       (home_attack_dir == "right to left" and is_home == 0):
        # 对方球门在右侧
        goal_center = (field_length, field_width / 2)
        goal_left   = (field_length, (field_width / 2) - goal_width / 2)
        goal_right  = (field_length, (field_width / 2) + goal_width / 2)
    elif (home_attack_dir == "right to left" and is_home == 1) or \
         (home_attack_dir == "left to right" and is_home == 0):
        # 对方球门在左侧
        goal_center = (0, field_width / 2)
        goal_left   = (0, (field_width / 2) - goal_width / 2)
        goal_right  = (0, (field_width / 2) + goal_width / 2)
    else:
        # home_attack_dir 为 "other" 等异常
        return np.nan, np.nan, np.nan, np.nan

    # 到球门中心距离
    dx, dy = goal_center[0] - x, goal_center[1] - y
    dist_to_goal = np.sqrt(dx**2 + dy**2)

    # 球门线向量（两门柱连线）
    goal_line_vec = np.array([goal_right[0] - goal_left[0],
                              goal_right[1] - goal_left[1]])  # 基本是 [0, 7.32]

    # 指向球门中心的向量
    dir_vec = np.array([dx, dy])

    # 方向与球门线夹角
    dot = np.dot(dir_vec, goal_line_vec)
    norms = np.linalg.norm(dir_vec) * np.linalg.norm(goal_line_vec) + 1e-8
    ang_to_goal_line = np.arccos(np.clip(dot / norms, -1.0, 1.0))

    # 单位向量
    norm_c = np.linalg.norm(dir_vec) + 1e-6
    dir_x, dir_y = dx / norm_c, dy / norm_c

    return dist_to_goal, ang_to_goal_line, dir_x, dir_y

def calc_touchline_features(x, y, field_length=105, field_width=68):
    """
    返回: (dist_to_tl, ang_to_tl, dir_x, dir_y)
      - 最近边线距离、指向最近边线的方向与水平轴夹角、该方向单位向量
    """
    if pd.isna(x) or pd.isna(y):
        return np.nan, np.nan, np.nan, np.nan

    # 四条边线的最近点
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

# ------------------------------------------------------------
# 事件级：为每行计算「每个球员 + 球」的 GL/TL 特征 + ha_dist/ha_ang/bc_dist
# ------------------------------------------------------------
def apply_all_features_per_row(row):
    """
    为一行（一个事件时刻）计算：
      - 每个 h1..h23 / a1..a23: gl_* / tl_*
      - ball: gl_* / tl_*
      - ha_dist / ha_ang（朝对方球门中心的距离与“对球门线夹角”）
      - bc_dist（ha 与球的欧氏距离）
    """
    results = {}
    home_attack_dir = row["home_attack_dir"]
    is_home = row["home_team"]  # 你原数据：1/0

    # 每个球员
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

    # 球
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

    # ha_*（操作者）——用 ha_name 去取坐标
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

    # ha 与 ball 的距离
    results["bc_dist"] = distance_pos(row.get("ha_x", np.nan), row.get("ha_y", np.nan), bx, by)

    return pd.Series(results)

# ------------------------------------------------------------
# ball-carrier（ha_*）坐标提取
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 合并重复时间（保留首 + 尾(插值时间)）
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 传球接球、对手等（你原来的边/交互标签逻辑）
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 组装最终数据集（列名与内容严格对齐）
# ------------------------------------------------------------
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

    # 列名：先边/交互元数据
    column_names = [
        "p1_id", "p2_id", "p2_dist", "timedelta", "timestamp", "seconds",
        "event_type", "event_type_2", "home_attack_dir",
        "OP_OP", "OP_BR", "OP_DP", "OP_B", "OP_G", "OP_C",
        "DP_B", "DP_T", "B_T", "B_G", "DP_DB", "DP_P", "DP_M", "DP_F", "Other"
    ]

    # 每个球员：基础 + 速度 + GL + TL
    for num in range(1, 24):
        column_names.extend([
            f"h{num}_x", f"a{num}_x", f"h{num}_y", f"a{num}_y",
            f"h{num}_speed", f"a{num}_speed", f"h{num}_vx", f"a{num}_vx", f"h{num}_vy", f"a{num}_vy",
            f"h{num}_gl_dist", f"h{num}_gl_ang", f"h{num}_gl_dir_x", f"h{num}_gl_dir_y",
            f"a{num}_gl_dist", f"a{num}_gl_ang", f"a{num}_gl_dir_x", f"a{num}_gl_dir_y",
            f"h{num}_tl_dist", f"h{num}_tl_ang", f"h{num}_tl_dir_x", f"h{num}_tl_dir_y",
            f"a{num}_tl_dist", f"a{num}_tl_ang", f"a{num}_tl_dir_x", f"a{num}_tl_dir_y",
        ])

    # 球
    column_names.extend([
        "ball_x", "ball_y", "ball_speed", "ball_vx", "ball_vy",
        "ball_gl_dist", "ball_gl_ang", "ball_gl_dir_x", "ball_gl_dir_y",
        "ball_tl_dist", "ball_tl_ang", "ball_tl_dir_x", "ball_tl_dir_y",
        "bc_dist"
    ])

    # ha 与事件起止
    column_names.extend(["ha_x", "ha_y", "ha_vx", "ha_vy", "ha_speed", "ha_dist", "ha_ang",
                         "start_x", "start_y", "end_x", "end_y", "ha_name"])

    # 逐行写入
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

# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    # id 映射：statsbomb_id -> skillcorner_id
    id_map_df = pd.read_csv(id_mapping_path)
    id_map = dict(zip(id_map_df["match_id_statsbomb"].astype(str), id_map_df["match_id_skillcorner"].astype(str)))

    csv_files = glob.glob(f"{folder_path}/*.csv")

    for csv_file in csv_files:
        df1 = pd.read_csv(csv_file)

        # 赛事实体标识
        statsbomb_id = str(df1["match_id"].iloc[0])
        skillcorner_id = id_map.get(statsbomb_id, None)
        match_key = f"{statsbomb_id}_{skillcorner_id}"
        if match_key not in target_matches:
            continue

        print(f"处理比赛 {match_key} -> {csv_file}")

        # ---------- 读取 match JSON ----------
        target_filename = f"{skillcorner_id}.json"
        target_path = os.path.join(match_dir, target_filename)
        if os.path.exists(target_path):
            with open(target_path, "r", encoding="utf-8") as f:
                match = json.load(f)
        else:
            print(f"match文件不存在: {target_path}")
            continue

        players = match["players"]

        # 守门员 trackable_object id
        goalkeeper_ids = []
        for player in players:
            prole = player.get("player_role", {})
            if isinstance(prole, dict) and prole.get("name") == "Goalkeeper":
                pid = player.get("trackable_object")
                if pid is not None:
                    goalkeeper_ids.append(pid)

        # trackable_object -> h_i / a_i
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

        # name -> label
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
            res = process.extractOne(name_, valid_names)
            if res is not None:
                match_name, score = res
                if score > 80:
                    return name_to_label.get(match_name, "0")
            return "0"

        # ha_name / ha 坐标
        df1["player_clean"] = df1["player"].fillna("").apply(lambda x: str(x).lower().replace(" ", ""))
        df1["ha_name"] = df1["player_clean"].apply(fuzzy_match)
        df1[["ha_x", "ha_y", "ha_vx", "ha_vy", "ha_speed"]] = df1.apply(get_ha_xy, axis=1)

        # ---------- needs_flipping（只翻事件的 start/end） ----------
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

        # ---------- 守门员 label ----------
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

        # GK 坐标
        h_gk_x = df1.get(f"{h_goalkeeper_id}_x") if h_goalkeeper_id else pd.Series([np.nan] * len(df1))
        a_gk_x = df1.get(f"{a_goalkeeper_id}_x") if a_goalkeeper_id else pd.Series([np.nan] * len(df1))
        df1["h_goalkeeper_x"] = h_gk_x
        df1["a_goalkeeper_x"] = a_gk_x

        # ---------- 推断 home_attack_dir ----------
        df1["home_attack_dir"] = np.nan
        df1.loc[(df1["h_goalkeeper_x"] - 105).abs() < 45, "home_attack_dir"] = "right to left"
        df1.loc[(df1["h_goalkeeper_x"] - 105).abs() > 60, "home_attack_dir"] = "left to right"
        df1.loc[((df1["h_goalkeeper_x"] - 105).abs() >= 45) & ((df1["h_goalkeeper_x"] - 105).abs() <= 60), "home_attack_dir"] = "other"

        mask_other = df1["home_attack_dir"] == "other"
        if mask_other.any():
            print("⚠️ 出现了 'other' 的行(示例前20行):")
            print(df1.loc[mask_other, ["seconds_event", "h_goalkeeper_x", "home_attack_dir"]].head(20))
            print(f"总共有 {int(mask_other.sum())} 行是 'other'")

        # ---------- 玩家/球速度 ----------
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

        # ---------- 计算每个球员+球 的 GL/TL 特征 + ha_dist/ha_ang + bc_dist ----------
        perrow_feats = df1.apply(apply_all_features_per_row, axis=1)
        df1 = pd.concat([df1, perrow_feats], axis=1)

        # ---------- 保证 needs_flipping 的 dtype ----------
        df1["needs_flipping"] = df1["needs_flipping"].where(df1["home_attack_dir"].notna(), pd.NA)
        df1["needs_flipping"] = df1["needs_flipping"].astype("boolean")

        # ---------- 压缩同秒数据 ----------
        df1_pruned = take_first_and_last_with_interp(df1, time_col='seconds_event').reset_index(drop=True)

        # ---------- 组装并写盘 ----------
        node_edges_df = create_datasets(df1_pruned)
        filename = f"{statsbomb_id}_{skillcorner_id}_node_edges.csv"
        filepath = os.path.join(output_dir, filename)
        node_edges_df.to_csv(filepath, index=False)
        print("已输出：", filepath)

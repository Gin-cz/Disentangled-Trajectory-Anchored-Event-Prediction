import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# shapely 相关
try:
    from shapely import vectorized
    from shapely.geometry import Polygon, MultiPoint
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False
    vectorized = None
    Polygon = MultiPoint = None

# 正态分布
from scipy.stats import norm


def precompute_heatmap(field_length=105, field_width=68, step=3, goal_side="right"):
    """
    goal_side: "right" or "left" （对方球门在哪边）
    """
    xs = np.linspace(0, field_length, int(max(1, field_length/step)))
    ys = np.linspace(0, field_width,  int(max(1, field_width/step)))
    xv, yv = np.meshgrid(xs, ys)

    # 球门位置
    goal_x = field_length if goal_side == "right" else 0

    # x 方向的权重: 离对方球门越近越大
    x_weight = np.exp(-0.05 * np.abs(xs - goal_x))

    # y 方向的权重: 中路更高
    y_weight = 50 * norm.pdf(ys, loc=field_width/2, scale=field_width/4)

    heatmap = np.outer(y_weight, x_weight)
    return xs, ys, xv, yv, heatmap

xs_r, ys_r, xv_r, yv_r, heatmap_right = precompute_heatmap(goal_side="right")
xs_l, ys_l, xv_l, yv_l, heatmap_left  = precompute_heatmap(goal_side="left")


def _weighted_voronoi_scores_batch(positions_batch, grid, field_length=105, field_width=68):
    """
    positions_batch: [B,22,2] in 0..105 x 0..68
    返回 [B,22]，每个样本按加权 Voronoi 面积归一化的占比
    """
    if not HAS_SHAPELY:
        B, N, _ = positions_batch.shape
        return np.full((B, N), 1.0/N, dtype=np.float32)

    xs, ys, xv, yv, heatmap = grid
    B, N, _ = positions_batch.shape
    out = np.zeros((B, N), dtype=np.float32)

    field_poly = Polygon([(0,0),(field_length,0),(field_length,field_width),(0,field_width)])

    for b in range(B):
        pts = positions_batch[b]
        mpoints = MultiPoint([tuple(p) for p in pts])
        vor = mpoints.voronoi_diagram(envelope=field_poly)

        for i, p in enumerate(mpoints.geoms):
            cell = None
            for poly in vor.geoms:
                if poly.contains(p) or poly.touches(p):
                    cell = poly.intersection(field_poly)
                    break
            if cell is None:
                continue
            mask = vectorized.contains(cell, xv, yv)  # bool [H,W]
            out[b, i] = float((heatmap * mask).sum())

        s = out[b].sum()
        if s > 1e-6:
            out[b] /= s
        else:
            out[b, :] = 1.0 / N
    return out

class TgtPlayerScorer(nn.Module):
    def __init__(self, hidden_dim=128, field_L=105.0, field_W=68.0, vmax=7.0, bmax=20.0,
                 use_voronoi=True, voronoi_step=3, n_players=22):
        super().__init__()

        self.field_L = field_L
        self.field_W = field_W
        self.vmax = vmax
        self.bmax = bmax
        self.n_players = int(n_players)
        self.geo_proj = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # # 这些权重你可以在训练中学习；Voronoi 默认只有在开启时给一个非零初值
        # self.w_time     = nn.Parameter(torch.tensor(1.0))
        # self.w_dist     = nn.Parameter(torch.tensor(0.1))
        # self.w_voronoi  = nn.Parameter(torch.tensor(0.5 if (use_voronoi and HAS_SHAPELY) else 0.0))
        # self.w_sideline = nn.Parameter(torch.tensor(0.5))
        # self.w_dist_goal= nn.Parameter(torch.tensor(1.0))
        # self.w_ang_goal = nn.Parameter(torch.tensor(1.0))

        self.use_voronoi = bool(use_voronoi and HAS_SHAPELY)

        self.grid_right = precompute_heatmap(field_length=field_L, field_width=field_W, step=voronoi_step, goal_side="right")
        self.grid_left  = precompute_heatmap(field_length=field_L, field_width=field_W, step=voronoi_step, goal_side="left")

        if use_voronoi and not HAS_SHAPELY and (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
            print("[WARN] shapely 未安装或不可用，Voronoi 被自动关闭。", flush=True)

    def _normalize_labels(self, labels):
        labels = [str(pid) for pid in labels][:self.n_players]
        if len(labels) < self.n_players:
            labels.extend(["PAD_h"] * (self.n_players - len(labels)))
        return labels

    def forward(self, ctx_dyns_BF, cols, labels_22_batch):

        device = ctx_dyns_BF.device
        B = ctx_dyns_BF.size(0)
        col2idx = {str(c): i for i, c in enumerate(list(cols))}

        # --------- 组装球员位置（用于 Voronoi）---------
        positions = []
       
        for b in range(B):
            last = ctx_dyns_BF[b]
            pts = []
            labels_b = self._normalize_labels(labels_22_batch[b])
            for pid in labels_b:
                xk, yk = f"{pid}_x", f"{pid}_y"
                px = last[col2idx[xk]] if xk in col2idx else torch.tensor(0.0, device=device)
                py = last[col2idx[yk]] if yk in col2idx else torch.tensor(0.0, device=device)
                pts.append(torch.stack([px, py]))
            positions.append(torch.stack(pts))
        positions_batch = torch.stack(positions)

        # --------- 读取每个样本的 home_attack_dir_num（>0: home 左->右；<0: 右->左）---------
        if "home_attack_dir_num" in col2idx:
            attack_dir_nums = ctx_dyns_BF[:, col2idx["home_attack_dir_num"]].detach().cpu().numpy()  # [B]
        else:
            attack_dir_nums = np.zeros(B, dtype=np.float32)

        # --------- 整批计算 Voronoi 权重（每个样本算两次：右门热图 / 左门热图）---------
        if self.use_voronoi:
            vor_scores_right_list = []
            vor_scores_left_list  = []
            pos_np = positions_batch.detach().cpu().numpy()
            for b in range(B):
                try:
                    # 同一份 Voronoi 分区，用不同热图积分 -> 简单起见这里各自跑一遍
                    v_right = _weighted_voronoi_scores_batch(
                        pos_np[b:b+1], self.grid_right, self.field_L, self.field_W
                    )[0]
                except Exception:
                    v_right = np.full(self.n_players, 1.0/self.n_players, dtype=np.float32)

                try:
                    v_left = _weighted_voronoi_scores_batch(
                        pos_np[b:b+1], self.grid_left, self.field_L, self.field_W
                    )[0]
                except Exception:
                    v_left = np.full(self.n_players, 1.0/self.n_players, dtype=np.float32)

                vor_scores_right_list.append(v_right)
                vor_scores_left_list.append(v_left)

            vor_scores_right = torch.tensor(np.stack(vor_scores_right_list), dtype=torch.float32, device=device)  # [B,n_players]
            vor_scores_left  = torch.tensor(np.stack(vor_scores_left_list),  dtype=torch.float32, device=device)  # [B,n_players]

            # 为每个样本、每个球员选择正确的“对方球门方向”的权重：
            #   home 球员：对方=away → 方向 = attack_dir_num
            #   away 球员：对方=home → 方向 = -attack_dir_num
            vor_scores = torch.zeros((B, self.n_players), dtype=torch.float32, device=device)
            for b in range(B):
                ad = float(attack_dir_nums[b])  # >0: 对方(对 home 来说)在右门；<0: 在左门
                labels_b = self._normalize_labels(labels_22_batch[b])
                for i, pid in enumerate(labels_b):
                    is_home = str(pid).startswith("h")
                    opp_dir = ad if is_home else -ad
                    # 选择该球员对应的热图方向
                    vor_scores[b, i] = vor_scores_right[b, i] if opp_dir > 0 else vor_scores_left[b, i]
        else:
            vor_scores = torch.zeros((B, self.n_players), dtype=torch.float32, device=device)

        # --------- 其余几何项（球、边线、球门距离/角度）---------
        priors = []
        geo_embs = []  
        for b in range(B):
            last = ctx_dyns_BF[b]

            def _get(key, default=0.0):
                if key in col2idx:
                    v = last[col2idx[key]]
                    return v if torch.isfinite(v) else torch.tensor(default, device=device)
                return torch.tensor(default, device=device)

            bx = _get("ball_x", 0.0)
            by = _get("ball_y", 0.0)
            ball_speed = _get("ball_speed", self.bmax)

            feats = []
            labels_b = self._normalize_labels(labels_22_batch[b])
            for i, pid in enumerate(labels_b):
                pid = str(pid)
                px = _get(f"{pid}_x", 0.0)
                py = _get(f"{pid}_y", 0.0)

                # 到球的距离/时间
                ## 之前我们做过归一化处理，这就不是正确的物理意义的时间了，现在更改
                # d_ball = torch.norm(torch.stack([px - bx, py - by]), p=2) / self.field_L
                d_ball = torch.norm(torch.stack([px - bx, py - by]), p=2)
                t_ball = d_ball / (ball_speed/2 + 1e-6)

                # 边线距离（坐标 0..W）：min(y, W-y)
                sideline_dist = torch.minimum(py.clamp(min=0.0), (self.field_W - py).clamp(min=0.0)) \
                                / (self.field_W/2 + 1e-6)

                # 球门距离/角度（预处理时已按“各自对方球门”计算 *_gl_*）
                gd_k, ga_k = f"{pid}_gl_dist", f"{pid}_gl_ang"
                goal_dist = _get(gd_k, 0.0) / self.field_L
                goal_ang  = _get(ga_k, 0.0)

                # Voronoi 权重（已按队别与方向挑选）
                vscore = vor_scores[b, i]

                feat = torch.stack([t_ball, d_ball, vscore, sideline_dist, goal_dist, goal_ang])
                feats.append(feat)
                # score = (
                #     self.w_time     * t_ball +
                #     self.w_dist     * d_ball +
                #     self.w_voronoi  * vscore +
                #     self.w_sideline * sideline_dist +
                #     self.w_dist_goal* goal_dist +
                #     self.w_ang_goal * goal_ang
                # )
                # scores.append(score)

            # scores = torch.stack(scores)                 # [22]
            # scores = scores / (scores.sum() + 1e-6)      # 归一化到 1
            # probs = F.softmax(-scores, dim=0)            # 代价越小 -> 概率越大
            # priors.append(probs)

            feats = torch.stack(feats, dim=0)    # [n_players,6]
            emb = self.geo_proj(feats)              # [n_players, 128]
            priors.append(emb)
        priors_emb = torch.stack(priors, dim=0)   # [B, n_players, 128]
        return priors_emb
    
# if __name__ == "__main__":
#     # 1. 读取你的 pt 文件
#     pt_path = "data/processed/dataloader_example/train/example_batch.pt"
#     data = torch.load(pt_path, map_location="cpu")

#     # 2. 取第一个样本
#     sample = data[0]
#     ctx_dyns     = sample["ctx_dyns"]      # [C,F_dyn]
#     ctx_edges    = sample["ctx_edges"]     # [C,F_edge]
#     ctx_labels_22= sample["ctx_labels_22"] # [C,22]  每帧22个label
#     dyn_cols     = sample["dyn_cols"]
#     edge_cols    = sample["edge_cols"]

#     # 我们只取最后一帧做 target scoring
#     ctx_dyns_last = ctx_dyns[-1].unsqueeze(0)  # [B=1,F_dyn]
#     labels_last   = [ctx_labels_22[-1]]        # List[B][22]

#     # 3. 初始化 scorer
#     scorer = TgtPlayerScorer()

#     # 4. 前向
#     with torch.no_grad():
#         probs = scorer(ctx_dyns_last, dyn_cols, labels_last)  # [B,n_players]

#     print("输出 shape:", probs.shape)
#     print("归一化概率和:", probs.sum(dim=1))
#     print("前几个球员概率:", probs[0, :10])
"""Compute the Voronoi-based spatial prior for target-player prediction."""

import numpy as np

import torch
import torch.nn as nn

try:
    from shapely import vectorized
    from shapely.geometry import Polygon, MultiPoint
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False
    vectorized = None
    Polygon = MultiPoint = None

from scipy.stats import norm


def precompute_heatmap(field_length=105, field_width=68, step=3, goal_side="right"):
    xs = np.linspace(0, field_length, int(max(1, field_length/step)))
    ys = np.linspace(0, field_width,  int(max(1, field_width/step)))
    xv, yv = np.meshgrid(xs, ys)

    goal_x = field_length if goal_side == "right" else 0

    x_weight = np.exp(-0.05 * np.abs(xs - goal_x))

    y_weight = 50 * norm.pdf(ys, loc=field_width/2, scale=field_width/4)

    heatmap = np.outer(y_weight, x_weight)
    return xs, ys, xv, yv, heatmap

xs_r, ys_r, xv_r, yv_r, heatmap_right = precompute_heatmap(goal_side="right")
xs_l, ys_l, xv_l, yv_l, heatmap_left  = precompute_heatmap(goal_side="left")


def _weighted_voronoi_scores_batch(positions_batch, grid, field_length=105, field_width=68):
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
            mask = vectorized.contains(cell, xv, yv)
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


        self.use_voronoi = bool(use_voronoi and HAS_SHAPELY)

        self.grid_right = precompute_heatmap(field_length=field_L, field_width=field_W, step=voronoi_step, goal_side="right")
        self.grid_left  = precompute_heatmap(field_length=field_L, field_width=field_W, step=voronoi_step, goal_side="left")

    def _normalize_labels(self, labels):
        labels = [str(pid) for pid in labels][:self.n_players]
        if len(labels) < self.n_players:
            labels.extend(["PAD_h"] * (self.n_players - len(labels)))
        return labels

    def forward(self, ctx_dyns_BF, cols, labels_22_batch):

        device = ctx_dyns_BF.device
        B = ctx_dyns_BF.size(0)
        col2idx = {str(c): i for i, c in enumerate(list(cols))}

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

        if "home_attack_dir_num" in col2idx:
            attack_dir_nums = ctx_dyns_BF[:, col2idx["home_attack_dir_num"]].detach().cpu().numpy()
        else:
            attack_dir_nums = np.zeros(B, dtype=np.float32)

        if self.use_voronoi:
            vor_scores_right_list = []
            vor_scores_left_list  = []
            pos_np = positions_batch.detach().cpu().numpy()
            for b in range(B):
                try:
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

            vor_scores_right = torch.tensor(np.stack(vor_scores_right_list), dtype=torch.float32, device=device)
            vor_scores_left  = torch.tensor(np.stack(vor_scores_left_list),  dtype=torch.float32, device=device)

            vor_scores = torch.zeros((B, self.n_players), dtype=torch.float32, device=device)
            for b in range(B):
                ad = float(attack_dir_nums[b])
                labels_b = self._normalize_labels(labels_22_batch[b])
                for i, pid in enumerate(labels_b):
                    is_home = str(pid).startswith("h")
                    opp_dir = ad if is_home else -ad
                    vor_scores[b, i] = vor_scores_right[b, i] if opp_dir > 0 else vor_scores_left[b, i]
        else:
            vor_scores = torch.zeros((B, self.n_players), dtype=torch.float32, device=device)

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

                d_ball = torch.norm(torch.stack([px - bx, py - by]), p=2)
                t_ball = d_ball / (ball_speed/2 + 1e-6)

                sideline_dist = torch.minimum(py.clamp(min=0.0), (self.field_W - py).clamp(min=0.0)) \
                                / (self.field_W/2 + 1e-6)

                gd_k, ga_k = f"{pid}_gl_dist", f"{pid}_gl_ang"
                goal_dist = _get(gd_k, 0.0) / self.field_L
                goal_ang  = _get(ga_k, 0.0)

                vscore = vor_scores[b, i]

                feat = torch.stack([t_ball, d_ball, vscore, sideline_dist, goal_dist, goal_ang])
                feats.append(feat)


            feats = torch.stack(feats, dim=0)
            emb = self.geo_proj(feats)
            priors.append(emb)
        priors_emb = torch.stack(priors, dim=0)
        return priors_emb

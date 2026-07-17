"""Implement the constant-velocity trajectory baseline."""

import torch
import torch.nn as nn


class ConstantVelocityPredictor(nn.Module):

    def __init__(self, in_dim, h_dim, d_mem=None, theta=None, dt=None,
                 use_h=True, use_m=True, nonlin='tanh',
                 velocity_window=5, use_acceleration=False, use_last_n_frames=True):
        super().__init__()
        self.h_dim = h_dim
        self.velocity_window = velocity_window
        self.use_acceleration = use_acceleration
        self.use_last_n_frames = use_last_n_frames

        self.feature_proj = nn.Linear(2, h_dim)

    def estimate_velocity(self, positions, dt=0.1):
        B, T, A, _ = positions.shape

        if T < 2:
            return torch.zeros(B, A, 2, device=positions.device)

        if self.use_last_n_frames:
            window = min(self.velocity_window, T - 1)
            start_idx = max(0, T - window - 1)
            end_idx = T

            selected_positions = positions[:, start_idx:end_idx, :, :]
        else:
            window = min(self.velocity_window, T - 1)
            selected_positions = positions[:, -(window+1):, :, :]

        pos_diffs = selected_positions[:, 1:, :, :] - selected_positions[:, :-1, :, :]

        avg_displacement = pos_diffs.mean(dim=1)
        velocities = avg_displacement / dt

        if self.use_acceleration and pos_diffs.shape[1] >= 2:
            vel_diffs = pos_diffs[:, 1:, :, :] - pos_diffs[:, :-1, :, :]
            avg_accel = vel_diffs.mean(dim=1) / dt

            current_vel = pos_diffs[:, -1, :, :] / dt
            velocities = current_vel + avg_accel * dt

        return velocities

    def forward(self, x, h0=None, m0=None, history_traj=None):
        B, T, A, F = x.shape

        if history_traj is not None:
            positions = history_traj
        else:
            if F >= 2:
                positions = x[..., :2]
            else:
                positions = torch.zeros(B, T, A, 2, device=x.device)

        velocities = self.estimate_velocity(positions, dt=0.1)

        h_seq = self.feature_proj(positions)
        h_last = h_seq[:, -1]

        m_last = self.feature_proj(velocities)
        m_seq = m_last.unsqueeze(1).expand(B, T, A, self.h_dim)

        return {
            "h_seq": h_seq,
            "h_last": h_last,
            "m_seq": m_seq,
            "m_last": m_last,
            "velocities": velocities,
        }

    def step(self, x_t, h_prev, m_prev):
        B, A, F = x_t.shape

        pos_t = x_t[..., :2]

        h_t = self.feature_proj(pos_t)

        m_t = m_prev

        return h_t, m_t


class ConstantVelocityTrajHead(nn.Module):

    def __init__(self, in_dim: int, hidden: int = 256, T: int = 64,
                 dropout: float = 0.1, dt: float = 0.1):
        super().__init__()
        self.T = T
        self.dt = dt

        self.velocity_decoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, h_agents: torch.Tensor, last_pos: torch.Tensor,
                velocities: torch.Tensor = None) -> torch.Tensor:
        B, A, _ = h_agents.shape

        if velocities is None:
            velocities = self.velocity_decoder(h_agents)

        time_steps = torch.arange(1, self.T + 1, device=last_pos.device,
                                 dtype=last_pos.dtype) * self.dt
        time_steps = time_steps.view(self.T, 1, 1, 1)

        velocities_expanded = velocities.unsqueeze(0)
        last_pos_expanded = last_pos.unsqueeze(0)

        traj_pred = last_pos_expanded + velocities_expanded * time_steps

        return traj_pred


class ConstantVelocityTrajStepHead(nn.Module):

    def __init__(self, in_dim, hidden=256, dropout=0.1,
                 use_m=True, m_dim=None, max_step=None, dt=0.1):
        super().__init__()
        self.use_m = use_m
        self.dt = dt

        f_in = in_dim + 2
        if use_m and m_dim is not None:
            f_in += m_dim

        self.velocity_net = nn.Sequential(
            nn.Linear(f_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, h_agents, last_pos, m_t=None,
                h_players=None, prior_emb=None, volterra_features=None):
        if self.use_m and m_t is not None:
            x = torch.cat([h_agents, last_pos, m_t], dim=-1)
        else:
            x = torch.cat([h_agents, last_pos], dim=-1)

        velocity = self.velocity_net(x)

        next_pos = last_pos + velocity * self.dt

        return next_pos



class ConstantVelocityWrapper(nn.Module):

    def __init__(self, in_dim, h_dim, d_mem=None, theta=None, dt=None,
                 use_h=True, use_m=True, nonlin='tanh',
                 velocity_window=5, use_acceleration=False):
        super().__init__()
        self.cv_model = ConstantVelocityPredictor(
            in_dim=in_dim,
            h_dim=h_dim,
            d_mem=d_mem,
            theta=theta,
            dt=dt,
            use_h=use_h,
            use_m=use_m,
            nonlin=nonlin,
            velocity_window=velocity_window,
            use_acceleration=use_acceleration
        )
        self._cached_history = None
        self._warned_no_history = False

    def set_history(self, history_traj):
        self._cached_history = history_traj

    def forward(self, x, h0=None, m0=None):
        if self._cached_history is None and not self._warned_no_history:
            self._warned_no_history = True

        result = self.cv_model(x, h0, m0, history_traj=self._cached_history)

        self._cached_history = None

        return result

    def step(self, x_t, h_prev, m_prev):
        return self.cv_model.step(x_t, h_prev, m_prev)

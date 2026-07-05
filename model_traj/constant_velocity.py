# -*- coding: utf-8 -*-
"""
Constant Velocity Baseline for Trajectory Prediction

简单的线性外推模型，基于球员当前速度进行未来轨迹预测。
与 LSTM 模型保持相同的接口，便于直接替换。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstantVelocityPredictor(nn.Module):
    """
    恒定速度轨迹预测器
    
    原理：
    - 使用最近的位置变化估计速度 (vx, vy)
    - 假设速度恒定，线性外推未来轨迹
    - position(t) = position(t0) + velocity * (t - t0)
    
    特点：
    - 无需训练参数（non-parametric baseline）
    - 计算简单高效
    - 适合作为其他模型的性能下界
    """
    
    def __init__(self, in_dim, h_dim, d_mem=None, theta=None, dt=None,
                 use_h=True, use_m=True, nonlin='tanh', 
                 velocity_window=5, use_acceleration=False, use_last_n_frames=True):
        """
        参数：
        - in_dim: 输入特征维度（需要包含位置信息）
        - h_dim: 输出特征维度（为了接口一致性）
        - d_mem: 为了兼容保留
        - velocity_window: 用于估计速度的历史窗口大小（默认5帧，对应 burn_in 的最后5帧）
        - use_acceleration: 是否考虑加速度（二阶导数）
        - use_last_n_frames: 是否使用最后 N 帧（True: 帧16-20；False: 使用全部历史）
        """
        super().__init__()
        self.h_dim = h_dim
        self.velocity_window = velocity_window
        self.use_acceleration = use_acceleration
        self.use_last_n_frames = use_last_n_frames
        
        # 虽然是 non-parametric，但为了接口一致，添加一个线性层作为特征提取
        # 这样可以保持与其他模型相同的输出维度
        self.feature_proj = nn.Linear(2, h_dim)  # 从位置 (x, y) 映射到 h_dim
        
    def estimate_velocity(self, positions, dt=0.1):
        """
        从位置序列估计速度（使用最后 velocity_window 帧）
        
        对于 burn_in=20, velocity_window=5，使用第16-20帧（最后5帧）
        计算4个位置差分，取平均得到稳定的速度估计
        
        输入:
          positions: [B, T, A, 2]  (x, y 坐标，单位：米)
          dt: 时间步长（秒），默认 0.1s (10Hz)
        输出:
          velocities: [B, A, 2]  (vx, vy，单位：米/秒)
        
        示例（T=20, window=5）:
          帧索引:  16   17   18   19   20
          差分:      Δ1   Δ2   Δ3   Δ4
          velocity = mean(Δ1, Δ2, Δ3, Δ4) / dt
        """
        B, T, A, _ = positions.shape
        
        if T < 2:
            # 历史太短，返回零速度
            return torch.zeros(B, A, 2, device=positions.device)
        
        # 确定使用哪些帧
        if self.use_last_n_frames:
            # 使用最后 velocity_window 帧
            window = min(self.velocity_window, T - 1)  # 至少需要2帧才能计算差分
            start_idx = max(0, T - window - 1)  # 倒数第 (window+1) 帧
            end_idx = T  # 到最后一帧
            
            # 提取这些帧的位置
            selected_positions = positions[:, start_idx:end_idx, :, :]  # [B, window+1, A, 2]
        else:
            # 使用全部历史（兼容模式）
            window = min(self.velocity_window, T - 1)
            selected_positions = positions[:, -(window+1):, :, :]
        
        # 计算相邻帧的位置差分
        # selected_positions: [B, N+1, A, 2] → pos_diffs: [B, N, A, 2]
        pos_diffs = selected_positions[:, 1:, :, :] - selected_positions[:, :-1, :, :]
        
        # 平均速度（所有差分的平均，然后除以时间步长）
        avg_displacement = pos_diffs.mean(dim=1)  # [B, A, 2] 平均位移
        velocities = avg_displacement / dt  # [B, A, 2] 速度 (m/s)
        
        if self.use_acceleration and pos_diffs.shape[1] >= 2:
            # 考虑加速度进行二阶外推
            # 计算速度的变化率（加速度）
            vel_diffs = pos_diffs[:, 1:, :, :] - pos_diffs[:, :-1, :, :]  # [B, N-1, A, 2]
            avg_accel = vel_diffs.mean(dim=1) / dt  # [B, A, 2] 平均加速度 (m/s²)
            
            # 外推：v(t+1) = v(t) + a * dt
            # 使用最后一个差分作为当前速度，加上加速度修正
            current_vel = pos_diffs[:, -1, :, :] / dt  # [B, A, 2]
            velocities = current_vel + avg_accel * dt
        
        return velocities
    
    def forward(self, x, h0=None, m0=None, history_traj=None):
        """
        输入:
          x: [B, T, A, F_in]  投影后的特征（不包含原始位置）
          h0, m0: 为了接口一致，实际不使用
          history_traj: [B, T, A, 2] 历史轨迹（真实位置，单位：米）
                       如果为 None，尝试从 x 的前2维提取
        
        输出:
          dict:
            'h_seq':  [B, T, A, h_dim]  特征序列
            'h_last': [B, A, h_dim]     最后时刻特征
            'm_seq':  [B, T, A, h_dim]  "记忆"序列（用速度信息代替）
            'm_last': [B, A, h_dim]     最后时刻"记忆"
            'velocities': [B, A, 2]     估计的速度 (m/s)
        """
        B, T, A, F = x.shape
        
        # 提取位置信息
        if history_traj is not None:
            # 使用提供的历史轨迹
            positions = history_traj  # [B, T, A, 2]
        else:
            # 假设 x 的前2维是位置（兼容性处理）
            if F >= 2:
                positions = x[..., :2]
            else:
                # 无法获取位置，返回零
                print("[WARN] ConstantVelocityPredictor: No position info, returning zeros")
                positions = torch.zeros(B, T, A, 2, device=x.device)
        
        # 估计速度 (m/s)
        velocities = self.estimate_velocity(positions, dt=0.1)  # [B, A, 2]
        
        # 生成特征序列（将位置映射到 h_dim 空间）
        h_seq = self.feature_proj(positions)  # [B, T, A, h_dim]
        h_last = h_seq[:, -1]  # [B, A, h_dim]
        
        # "记忆"状态使用速度信息（扩展到 h_dim 维度）
        m_last = self.feature_proj(velocities)  # [B, A, h_dim]
        m_seq = m_last.unsqueeze(1).expand(B, T, A, self.h_dim)  # [B, T, A, h_dim]
        
        return {
            "h_seq": h_seq,
            "h_last": h_last,
            "m_seq": m_seq,
            "m_last": m_last,
            "velocities": velocities,  # 额外返回速度，供 TrajHead 使用
        }
    
    def step(self, x_t, h_prev, m_prev):
        """
        单步预测（为了接口一致）
        
        输入:
          x_t: [B, A, F]  当前时刻输入
          h_prev: [B, A, h_dim]  前一时刻隐状态
          m_prev: [B, A, h_dim]  前一时刻记忆状态
        
        输出:
          (h_t, m_t): 当前时刻的隐状态和记忆状态
        """
        B, A, F = x_t.shape
        
        # 提取当前位置
        pos_t = x_t[..., :2]  # [B, A, 2]
        
        # 映射到特征空间
        h_t = self.feature_proj(pos_t)  # [B, A, h_dim]
        
        # 保持记忆状态
        m_t = m_prev
        
        return h_t, m_t


class ConstantVelocityTrajHead(nn.Module):
    """
    恒定速度轨迹预测头
    
    使用线性外推生成未来轨迹：
    future_pos(t) = last_pos + velocity * t
    """
    
    def __init__(self, in_dim: int, hidden: int = 256, T: int = 64, 
                 dropout: float = 0.1, dt: float = 0.1):
        """
        参数：
        - in_dim: 输入特征维度（h_dim）
        - hidden: 为了接口一致保留，实际不使用
        - T: 预测的未来时间步数
        - dt: 时间步长（秒），默认 0.1s (10Hz)
        """
        super().__init__()
        self.T = T
        self.dt = dt
        
        # 从特征空间解码速度（如果需要学习的话）
        self.velocity_decoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),  # 输出 (vx, vy)
        )
        
    def forward(self, h_agents: torch.Tensor, last_pos: torch.Tensor, 
                velocities: torch.Tensor = None) -> torch.Tensor:
        """
        输入:
          h_agents: [B, A, h_dim]  球员特征（包含速度信息）
          last_pos: [B, A, 2]      最后时刻位置
          velocities: [B, A, 2]    估计的速度（如果提供）
        
        输出:
          traj_pred: [T, B, A, 2]  未来轨迹预测
        """
        B, A, _ = h_agents.shape
        
        # 解码速度（如果没有直接提供）
        if velocities is None:
            velocities = self.velocity_decoder(h_agents)  # [B, A, 2]
        
        # 生成时间步
        time_steps = torch.arange(1, self.T + 1, device=last_pos.device, 
                                 dtype=last_pos.dtype) * self.dt  # [T]
        time_steps = time_steps.view(self.T, 1, 1, 1)  # [T, 1, 1, 1]
        
        # 线性外推: pos(t) = pos(0) + v * t
        velocities_expanded = velocities.unsqueeze(0)  # [1, B, A, 2]
        last_pos_expanded = last_pos.unsqueeze(0)      # [1, B, A, 2]
        
        traj_pred = last_pos_expanded + velocities_expanded * time_steps  # [T, B, A, 2]
        
        return traj_pred


class ConstantVelocityTrajStepHead(nn.Module):
    """
    单步恒定速度预测头（用于自回归生成）
    """
    
    def __init__(self, in_dim, hidden=256, dropout=0.1, 
                 use_m=True, m_dim=None, max_step=None, dt=0.1):
        """
        参数：
        - in_dim: 输入特征维度
        - hidden: 隐藏层维度
        - dropout: dropout 比例
        - use_m: 是否使用记忆状态（速度信息）
        - m_dim: 记忆维度
        - dt: 时间步长
        """
        super().__init__()
        self.use_m = use_m
        self.dt = dt
        
        # 速度解码器
        f_in = in_dim + 2  # h + last_pos
        if use_m and m_dim is not None:
            f_in += m_dim  # + memory (velocity features)
        
        self.velocity_net = nn.Sequential(
            nn.Linear(f_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),  # 输出 (vx, vy)
        )
    
    def forward(self, h_agents, last_pos, m_t=None,
                h_players=None, prior_emb=None, volterra_features=None):
        """
        输入:
          h_agents: [B, A, Dh]  球员特征
          last_pos: [B, A, 2]   当前位置
          m_t: [B, A, Dm]       记忆状态（速度特征）
        
        输出:
          next_pos: [B, A, 2]   下一时刻位置
        """
        # 拼接特征
        if self.use_m and m_t is not None:
            x = torch.cat([h_agents, last_pos, m_t], dim=-1)
        else:
            x = torch.cat([h_agents, last_pos], dim=-1)
        
        # 解码速度
        velocity = self.velocity_net(x)  # [B, A, 2]
        
        # 线性外推
        next_pos = last_pos + velocity * self.dt
        
        return next_pos


# ==================== Wrapper for Easy Integration ====================

class ConstantVelocityWrapper(nn.Module):
    """
    包装器：自动处理历史轨迹的传递
    
    使用方式：
        # 1. 创建模型
        agent_lmu = ConstantVelocityWrapper(in_dim=F_LMU, h_dim=128)
        
        # 2. 在 forward 之前设置历史轨迹
        agent_lmu.set_history(ctx_seqX[:, :burn_in, :22, :2])
        
        # 3. 正常调用（会自动使用缓存的历史轨迹）
        x_proj = lmu_proj_agent(ctx_seqX[:, :burn_in, :22, :2])
        lmu_out = agent_lmu(x_proj)
    """
    
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
            velocity_window=velocity_window,  # 默认5帧（burn_in 最后5帧）
            use_acceleration=use_acceleration
        )
        self._cached_history = None
        self._warned_no_history = False
    
    def set_history(self, history_traj):
        """
        设置历史轨迹（在 forward 之前调用）
        
        参数:
          history_traj: [B, T, A, 2] 历史位置（米）
        """
        self._cached_history = history_traj
    
    def forward(self, x, h0=None, m0=None):
        """
        前向传播（自动使用缓存的历史轨迹）
        """
        if self._cached_history is None and not self._warned_no_history:
            print("[WARN] ConstantVelocityWrapper: No history set! "
                  "Call set_history() before forward(). Using fallback.")
            self._warned_no_history = True
        
        result = self.cv_model(x, h0, m0, history_traj=self._cached_history)
        
        # 清空缓存（避免下次 forward 误用旧数据）
        self._cached_history = None
        
        return result
    
    def step(self, x_t, h_prev, m_prev):
        """单步预测"""
        return self.cv_model.step(x_t, h_prev, m_prev)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    print("=" * 70)
    print("Constant Velocity Baseline 测试")
    print("=" * 70)
    
    # 参数设置
    B, T, A, F = 64, 20, 22, 4  # batch_size, time_steps, agents, features
    fut_T = 64
    
    # 1. 测试 ConstantVelocityPredictor
    print("\n[1] 测试 ConstantVelocityPredictor")
    print("-" * 70)
    
    model = ConstantVelocityPredictor(
        in_dim=F, 
        h_dim=128,
        velocity_window=3,
        use_acceleration=False
    )
    
    # 模拟输入数据（前2维是位置，后面是其他特征）
    x = torch.randn(B, T, A, F)
    # 让前2维包含位置信息（模拟真实位置变化）
    x[:, :, :, 0] = torch.arange(T).float().view(1, T, 1).expand(B, T, A)  # x 坐标
    x[:, :, :, 1] = torch.arange(T).float().view(1, T, 1).expand(B, T, A) * 0.5  # y 坐标
    
    output = model(x)
    
    print(f"✅ h_seq shape: {output['h_seq'].shape}")      # [B, T, A, 128]
    print(f"✅ h_last shape: {output['h_last'].shape}")    # [B, A, 128]
    print(f"✅ m_seq shape: {output['m_seq'].shape}")      # [B, T, A, 128]
    print(f"✅ m_last shape: {output['m_last'].shape}")    # [B, A, 128]
    print(f"✅ velocities shape: {output['velocities'].shape}")  # [B, A, 2]
    
    # 打印速度统计
    vel = output['velocities']
    print(f"\n速度统计:")
    print(f"  - vx 范围: [{vel[..., 0].min():.3f}, {vel[..., 0].max():.3f}]")
    print(f"  - vy 范围: [{vel[..., 1].min():.3f}, {vel[..., 1].max():.3f}]")
    
    # 2. 测试 ConstantVelocityTrajHead
    print("\n[2] 测试 ConstantVelocityTrajHead")
    print("-" * 70)
    
    traj_head = ConstantVelocityTrajHead(in_dim=128, hidden=256, T=fut_T, dt=0.1)
    last_pos = x[:, -1, :, :2]  # [B, A, 2] 最后时刻位置
    
    fut_pred = traj_head(output['h_last'], last_pos, output['velocities'])
    print(f"✅ fut_pred shape: {fut_pred.shape}")  # [T, B, A, 2]
    
    # 验证线性外推特性
    print(f"\n验证线性外推:")
    print(f"  - 第1步位移: {(fut_pred[0] - last_pos).abs().mean():.4f}")
    print(f"  - 第10步位移: {(fut_pred[9] - last_pos).abs().mean():.4f}")
    print(f"  - 第64步位移: {(fut_pred[63] - last_pos).abs().mean():.4f}")
    
    # 3. 测试 ConstantVelocityTrajStepHead
    print("\n[3] 测试 ConstantVelocityTrajStepHead")
    print("-" * 70)
    
    step_head = ConstantVelocityTrajStepHead(
        in_dim=128, 
        hidden=256, 
        use_m=True, 
        m_dim=128,
        dt=0.1
    )
    
    next_pos = step_head(output['h_last'], last_pos, output['m_last'])
    print(f"✅ next_pos shape: {next_pos.shape}")  # [B, A, 2]
    
    # 自回归生成测试
    print(f"\n自回归生成测试（生成10步）:")
    pos = last_pos
    positions = [pos]
    h = output['h_last']
    m = output['m_last']
    
    for t in range(10):
        pos = step_head(h, pos, m)
        positions.append(pos)
    
    positions = torch.stack(positions, dim=0)  # [11, B, A, 2]
    print(f"✅ 生成轨迹形状: {positions.shape}")
    print(f"  - 平均位移/步: {(positions[1:] - positions[:-1]).abs().mean():.4f}")
    
    # 4. 与 LSTM 接口对比
    print("\n[4] 接口兼容性验证")
    print("-" * 70)
    
    from lstm import AgentLSTM, LSTMTrajHead, LSTMTrajStepHead
    
    # LSTM 模型
    lstm_model = AgentLSTM(in_dim=F, h_dim=128)
    lstm_output = lstm_model(x)
    
    # 对比输出格式
    print("✅ ConstantVelocity 输出键:", list(output.keys()))
    print("✅ LSTM 输出键:", list(lstm_output.keys()))
    print("✅ 形状匹配:")
    for key in ['h_seq', 'h_last', 'm_seq', 'm_last']:
        cv_shape = output[key].shape
        lstm_shape = lstm_output[key].shape
        match = "✓" if cv_shape == lstm_shape else "✗"
        print(f"  {match} {key}: CV{cv_shape} vs LSTM{lstm_shape}")
    
    print("\n" + "=" * 70)
    print("测试完成！✅")
    print("=" * 70)
    
    # 5. 性能对比示例
    print("\n[5] 简单性能对比")
    print("-" * 70)
    
    # 生成真实轨迹（线性运动 + 噪声）
    true_vel = torch.tensor([1.0, 0.5]).view(1, 1, 2).expand(B, A, 2)
    true_traj = last_pos.unsqueeze(0) + true_vel.unsqueeze(0) * \
                torch.arange(1, fut_T + 1, dtype=torch.float32).view(fut_T, 1, 1, 1) * 0.1
    true_traj = true_traj + torch.randn_like(true_traj) * 0.1  # 添加噪声
    
    # CV 预测
    cv_pred = traj_head(output['h_last'], last_pos, output['velocities'])
    
    # 计算 ADE 和 FDE
    ade_cv = (cv_pred - true_traj).norm(dim=-1).mean().item()
    fde_cv = (cv_pred[-1] - true_traj[-1]).norm(dim=-1).mean().item()
    
    print(f"Constant Velocity Baseline:")
    print(f"  - ADE: {ade_cv:.4f} m")
    print(f"  - FDE: {fde_cv:.4f} m")
    
    print("\n提示：这是一个 non-parametric baseline，性能取决于：")
    print("  1. 球员运动的线性程度")
    print("  2. 速度估计的准确性")
    print("  3. 预测时间跨度的长短")
import torch
import torch.nn as nn
import torch.nn.functional as Fnn 


class StrictLSTM(nn.Module):
    """
    标准 LSTM cell，对应 StrictLMU 的接口
    """
    def __init__(self, in_dim, h_dim, nonlin='tanh'):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        
        # LSTM gates: input, forget, output, cell
        self.lstm_cell = nn.LSTMCell(in_dim, h_dim)
        
        # nonlin 其实 LSTM 用不到，但保留接口
        self.act = {'tanh': torch.tanh, 'relu': Fnn.relu, 'identity': (lambda x: x)}[nonlin]
    
    def step(self, x_t, h_prev, c_prev):
        """
        x_t: [B, in_dim]
        h_prev: [B, h_dim]
        c_prev: [B, h_dim]  (cell state, 对应 LMU 的 m)
        """
        # # 确保输入都是独立的张量
        # x_t = x_t.detach().requires_grad_(x_t.requires_grad)

        h_t, c_t = self.lstm_cell(x_t, (h_prev, c_prev))
        return h_t, c_t
    
    def forward(self, x, h0=None, c0=None):
        """
        x:  [B, T, in_dim]
        h0: [B, h_dim] or None
        c0: [B, h_dim] or None (cell state, 对应 LMU 的 m0)
        
        return:
          h_seq [B, T, h_dim], h_last [B, h_dim]
          c_seq [B, T, h_dim], c_last [B, h_dim]
        """
        B, T, _ = x.shape
        
        # 初始化状态
        if h0 is None:
            h = x.new_zeros(B, self.h_dim)
        else:
            h = h0.clone()
        
        if c0 is None:
            c = x.new_zeros(B, self.h_dim)
        else:
            c = c0.clone()
        
        h_seq, c_seq = [], []
        for t in range(T):
            h_new, c_new = self.step(x[:, t], h, c)
            h = h_new
            c = c_new
            
            h_seq.append(h.unsqueeze(1))
            c_seq.append(c.unsqueeze(1))
        
        h_seq = torch.cat(h_seq, dim=1)  # [B, T, h_dim]
        c_seq = torch.cat(c_seq, dim=1)  # [B, T, h_dim]
        
        return h_seq, h.clone(), c_seq, c.clone()


class AgentLSTM(nn.Module):
    """
    对应 AgentLMU，共享 LSTM 参数，对 22 名球员并行维护状态
    
    输入:  x [B, T, A, F_in]
    输出:  dict:
       'h_seq'  [B, T, A, h_dim], 'h_last'  [B, A, h_dim]
       'm_seq'  [B, T, A, h_dim], 'm_last'  [B, A, h_dim]
       (这里 'm' 对应 LSTM 的 cell state 'c')
    """
    def __init__(self, in_dim, h_dim, d_mem=None, theta=None, dt=None, 
                 use_h=True, use_m=True, nonlin='tanh'):
        """
        参数顺序调整，重要的放前面：
        - in_dim: 输入特征维度
        - h_dim: hidden state 维度
        - d_mem: 为了兼容保留，LSTM中不使用（cell state维度=h_dim）
        - theta, dt, use_h, use_m: 为了兼容保留，LSTM中不使用
        - nonlin: 激活函数类型
        """
        super().__init__()
        self.h_dim = h_dim
        # LSTM 的 cell state 维度等于 hidden 维度
        self.cell = StrictLSTM(in_dim, h_dim, nonlin)
    
    def forward(self, x, h0=None, m0=None):
        """
        m0 在这里对应 LSTM 的 c0 (cell state)
        """
        B, T, A, F = x.shape
        
        x_flat = x.reshape(B*A, T, F).contiguous().clone()
        
        if h0 is not None:
            h0_flat = h0.reshape(B*A, -1).contiguous().clone()
        else:
            h0_flat = None
        
        if m0 is not None:
            c0_flat = m0.reshape(B*A, -1).contiguous().clone()
        else:
            c0_flat = None
        
        h_seq, h_last, c_seq, c_last = self.cell(x_flat, h0_flat, c0_flat)
        
        return {
            "h_seq":  h_seq.reshape(B, T, A, -1),
            "h_last": h_last.reshape(B, A, -1),
            "m_seq":  c_seq.reshape(B, T, A, -1),  # cell state 作为 'm'
            "m_last": c_last.reshape(B, A, -1),
        }
    
    def step(self, x_t, h_prev, m_prev):
        """
        x_t: [B, A, F]
        h_prev: [B, A, h_dim]
        m_prev: [B, A, h_dim] (cell state)
        """
        B, A, F = x_t.shape
        
        x_flat = x_t.reshape(B*A, F).contiguous().clone()
        h_flat = h_prev.reshape(B*A, -1).contiguous().clone()
        c_flat = m_prev.reshape(B*A, -1).contiguous().clone()
        
        h_t, c_t = self.cell.step(x_flat, h_flat, c_flat)
        return (h_t.reshape(B, A, -1), c_t.reshape(B, A, -1))


class LSTMTrajHead(nn.Module):
    """
    对应 LMUTrajHead，使用相同的架构
    
    输入:
      h_agents: [B, 22, h_dim]
      last_pos: [B, 22, 2]
    输出:
      fut_pred: [T, B, 22, 2]
    """
    def __init__(self, in_dim: int, hidden: int = 256, T: int = 64, dropout: float = 0.1):
        super().__init__()
        self.T = T
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2 * T),  # Δx, Δy
        )
    
    def forward(self, h_agents: torch.Tensor, last_pos: torch.Tensor) -> torch.Tensor:
        """
        h_agents: [B, A, h_dim]
        last_pos: [B, A, 2]
        return: [T, B, A, 2]
        """
        B, A, Dh = h_agents.shape
        deltas = self.mlp(h_agents).view(B, A, self.T, 2)          # [B, A, T, 2]
        traj_abs = last_pos.unsqueeze(2) + deltas.cumsum(2)        # [B, A, T, 2]
        return traj_abs.permute(2, 0, 1, 3).contiguous()           # [T, B, A, 2]


# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 测试 AgentLSTM
    B, T, A, F = 64, 20, 22, 4  # batch_size, time_steps, agents, features
    
    # 正确的调用方式（参数顺序已调整）
    lstm = AgentLSTM(in_dim=F, h_dim=128)  # 简化调用
    x = torch.randn(B, T, A, F)
    
    output = lstm(x)
    print("h_seq shape:", output['h_seq'].shape)   # [64, 20, 22, 128]
    print("h_last shape:", output['h_last'].shape) # [64, 22, 128]
    print("m_seq shape:", output['m_seq'].shape)   # [64, 20, 22, 128]
    print("m_last shape:", output['m_last'].shape) # [64, 22, 128]
    
    # 测试 LSTMTrajHead
    traj_head = LSTMTrajHead(in_dim=128, hidden=256, T=64)
    last_pos = torch.randn(B, A, 2)
    
    fut_pred = traj_head(output['h_last'], last_pos)
    print("fut_pred shape:", fut_pred.shape)  # [64, 64, 22, 2]

class LSTMTrajStepHead(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.1, use_m=True, m_dim=None, max_step=None):
        super().__init__()
        self.use_m = use_m
        f_in = in_dim + 2 + (m_dim if (use_m and m_dim is not None) else 0)  # h + last_pos + (opt) m
        self.net = nn.Sequential(
            nn.Linear(f_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2)  # 预测 Δx, Δy
        )

    def forward(self, h_agents, last_pos, m_t=None,
                h_players=None, prior_emb=None, volterra_features=None):
        # h_agents: [B, A, Dh], last_pos: [B, A, 2], m_t: [B, A, Dm] or None
        if self.use_m and m_t is not None:
            x = torch.cat([h_agents, last_pos, m_t], dim=-1)  # [B,A,Dh+2+Dm]
        else:
            x = torch.cat([h_agents, last_pos], dim=-1)       # [B,A,Dh+2]
        delta = self.net(x)                                    # [B,A,2]
        return last_pos + delta                                # 返回绝对位置 next_pos

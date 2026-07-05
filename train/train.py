# -*- coding: utf-8 -*-
"""
Training step with Curriculum Learning support - WITH SHORT-TERM TRAJECTORY

Key features:
1. ✅ Multi-scale trajectory prediction: short-term (3s) + long-term (10s)
2. ✅ Shared traj_step_head for both predictions
3. ✅ Complete Top-3 and Top-5 metrics for tgt, eidx, and epos
4. ✅ Curriculum learning compatible
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


def step_batch_curriculum(
    batch, 
    head, 
    cfg, 
    mode: str = 'train', 
    optimizer=None,
    curriculum_weights: Optional[Dict[str, float]] = None,
    dyn_cols=None,
    edge_cols=None,
):
    """
    Training step with curriculum learning and multi-scale trajectory prediction.
    
    Args:
        batch: Data batch
        head: Model (DDP wrapped)
        cfg: Configuration
        mode: 'train', 'val_tf', or 'test'
        optimizer: Optimizer (train mode only)
        curriculum_weights: Dynamic weights from CurriculumScheduler
        dyn_cols: Dynamic column names
        edge_cols: Edge column names
        
    Returns:
        logs: Dictionary of metrics including short-term and long-term trajectory metrics
    """
    # Import helpers
    try:
        from train.metrics import reindex_by_labels, _compute_ADE_FDE
        from train.interp import interpolate_traj
        from train.ddp import is_dist_avail_and_initialized, get_rank
    except ImportError:
        import torch.distributed as dist
        def is_dist_avail_and_initialized():
            return dist.is_available() and dist.is_initialized()
        def get_rank():
            return dist.get_rank() if is_dist_avail_and_initialized() else 0
    
    device = next(head.parameters()).device
    real_head = head.module if hasattr(head, "module") else head
    
    # Get model components
    agent_lmu = getattr(real_head, "agent_lmu", None)
    lmu_proj = getattr(real_head, "lmu_proj_agent", None)
    traj_step_head = getattr(real_head, "traj_step_head", None)
    lmu_gamma = getattr(real_head, "lmu_gamma", 0.9)
    
    traj_model = getattr(cfg, 'traj_model', 'star').lower()
    use_context = getattr(cfg, 'use_context', True)
    n_agents = int(getattr(cfg, 'n_agents', 22))
    ball_idx = int(getattr(cfg, 'ball_idx', n_agents))
    n_total_agents = int(getattr(cfg, 'n_total_agents', max(n_agents + 1, ball_idx + 1)))
    
    # =========== Multi-scale configuration ===========
    short_T = getattr(cfg, 'short_fut_T', 30)  # 3s @ 10Hz
    long_T = cfg.fut_T  # 100 frames (10s)
    w_short = getattr(cfg, 'w_short_traj', 0.5)  # Short-term weight
    
    # =========== Get curriculum weights ===========
    if curriculum_weights is None:
        curriculum_weights = {
            'traj': 1.0,
            'tgt': cfg.w_tgt,
            'eidx': cfg.w_eidx,
            'epos': 0.0,
            'disentangle': getattr(cfg, 'w_disentangle', 0.1),
            'w_short_traj': getattr(cfg, 'w_short_traj', 0.5),
            'phase': 'full',
        }
    
    w_traj = curriculum_weights.get('traj', 1.0)
    w_tgt = curriculum_weights.get('tgt', cfg.w_tgt)
    w_eidx = curriculum_weights.get('eidx', cfg.w_eidx)
    w_disentangle = curriculum_weights.get('disentangle', getattr(cfg, 'w_disentangle', 0.1))
    w_short = curriculum_weights.get('w_short_traj', getattr(cfg, 'w_short_traj', 0.5))
    phase = curriculum_weights.get('phase', 'full')
    is_traj_only = (w_tgt == 0.0 and w_eidx == 0.0 and w_disentangle == 0.0)
    
    # =========== Helper functions ===========
    def _exp_decay_pool(m_seq_btad, gamma=0.9, n_agents=22):
        """Exponentially weighted pooling over time"""
        B, T, A, d = m_seq_btad.shape
        w = (gamma ** torch.arange(T-1, -1, -1, device=m_seq_btad.device, dtype=m_seq_btad.dtype))
        w = (w / (w.sum() + 1e-8)).view(1, T, 1, 1)
        pooled = (m_seq_btad * w).sum(dim=1)
        return pooled[:, :n_agents, :]

    def _pack_players_ball(seq):
        if seq.size(-2) <= ball_idx:
            raise ValueError(f"ball_idx={ball_idx} is outside sequence agent axis size {seq.size(-2)}")
        players = seq[..., :n_agents, :]
        ball = seq[..., ball_idx:ball_idx + 1, :]
        return torch.cat([players, ball], dim=-2)

    # =========== Unpack batch ===========
    ctx_dyns = batch['ctx_dyns'].to(device)
    ctx_edges = batch['ctx_edges'].to(device)
    ctx_seqX = batch['ctx_seqX'][..., :2].to(device)
    fut_seqX = batch['fut_seqX'][..., :2].to(device)

    fut_players = batch['fut_players']
    if not torch.is_tensor(fut_players):
        fut_players = torch.as_tensor(fut_players)
    if fut_players.dtype != torch.long:
        fut_players = torch.nan_to_num(fut_players, nan=-1.0).to(torch.long)
    fut_players[(fut_players < 0) | (fut_players >= n_agents)] = -1
    fut_players = fut_players.to(device)

    fut_positions = batch['fut_positions'].to(device)

    # Interpolate and clean
    ctx_seqX = interpolate_traj(ctx_seqX.unsqueeze(1))
    fut_seqX = interpolate_traj(fut_seqX)
    ctx_dyns = torch.nan_to_num(ctx_dyns, nan=0.0)
    ctx_edges = torch.nan_to_num(ctx_edges, nan=0.0)
    ctx_seqX = torch.nan_to_num(ctx_seqX, nan=0.0, posinf=0.0, neginf=0.0)
    fut_seqX = torch.nan_to_num(fut_seqX, nan=0.0, posinf=0.0, neginf=0.0)
    fut_positions = torch.nan_to_num(fut_positions, nan=0.0)

    B = ctx_seqX.size(0)

    # =========== Context encoding ===========
    dyn_in = ctx_dyns if not getattr(cfg, 'disable_dyn_ctx', False) else torch.zeros_like(ctx_dyns)
    edge_in = ctx_edges if not getattr(cfg, 'disable_edge_ctx', False) else torch.zeros_like(ctx_edges)
    enc_labels_22 = batch["ctx_labels_22"]
    
    h_players, h_ball, team_labels, players_all, ball_all, event_emb = real_head.context_encoder(
        dyn_in, edge_in, enc_labels_22,
        cols=dyn_cols, edge_cols=edge_cols,
        return_team_labels=True,
    )

    # Context ball
    attn_score = torch.sum(h_players * h_ball.unsqueeze(1), dim=-1, keepdim=True)
    attn_weight = torch.softmax(attn_score, dim=1)
    context_ball = (attn_weight * h_players).sum(dim=1, keepdim=True)

    use_event_emb = getattr(cfg, 'use_event_emb', False)
    if use_context:
        event_ctx_pooled = event_emb.mean(dim=1)

    # =========== Trajectory prediction ===========
    ctx_seqX_all = ctx_seqX[:, 0]  # [B, T_ctx, 23, 2]
    
    # Prior embeddings
    if hasattr(real_head, 'prior_scorer') and real_head.prior_scorer is not None:
        labels_last = [list(map(str, seq[-1])) for seq in batch["ctx_labels_22"]]
        prior_emb = real_head.prior_scorer(ctx_dyns[:, -1, :], dyn_cols, labels_last)
    else:
        prior_emb = None
    
    ball_pred_list = []
    predict_ball = bool(getattr(cfg, 'predict_ball', False))
    ball_hist = ctx_seqX_all[:, :cfg.burn_in, ball_idx, :].contiguous()

    # =========== Model-specific trajectory prediction ===========
    if traj_model in ['baller2vec', 'star']:
        # Baller2Vec / STAR path
        x_lmu_in = _pack_players_ball(ctx_seqX_all[:, :cfg.burn_in, :, :])
        x_lmu_proj = lmu_proj(x_lmu_in)

        if use_context:
            lmu_out = agent_lmu(
                x_lmu_proj,
                h_players=h_players,
                prior_emb=prior_emb,
                context_ball=context_ball,
            )
        else:
            lmu_out = agent_lmu(x_lmu_proj)
        
        h_roll = lmu_out["h_last"]
        h_burn_seq = lmu_out.get("h_seq", None)
        mem_window = lmu_out.get("mem_window", None)

        last_pos = ctx_seqX_all[:, cfg.burn_in-1, :n_agents, :]
        fut_steps = []
        m_future_list = []

        # ✅ Predict to long_T (includes short_T)
        for t in range(long_T):
            if use_context:
                next_pos_players = traj_step_head(
                    h_roll[:, :n_agents, :],
                    last_pos,
                    m_t=None,
                    h_players=h_players,
                    prior_emb=prior_emb,
                    volterra_features=None
                )
            else:
                next_pos_players = traj_step_head(
                    h_roll[:, :n_agents, :],
                    last_pos,
                    m_t=None,
                    h_players=None,
                    prior_emb=None,
                    volterra_features=None
                )
            
            # Ball prediction
            if predict_ball:
                h_ball_roll = h_roll[:, ball_idx, :]
                ball_pos = real_head.ball_step_head(
                    ball_hist=ball_hist,
                    h_ball_roll=h_ball_roll,
                    context_ball=context_ball if use_context else None
                )
            else:
                ball_pos = ball_hist[:, -1, :]
            
            ball_pred_list.append(ball_pos)
            ball_hist = torch.cat([ball_hist[:, 1:, :], ball_pos.unsqueeze(1)], dim=1)

            ball_t = ball_pos.unsqueeze(1)
            next_pos_23 = torch.cat([next_pos_players, ball_t], dim=1)
            fut_steps.append(next_pos_players.unsqueeze(0))
            
            x_t_proj = lmu_proj(next_pos_23.unsqueeze(1))
            x_t_proj = x_t_proj.squeeze(1)
            
            if use_context:
                step_out = agent_lmu.step(
                    x_t_proj,
                    mem_window,
                    h_players=h_players,
                    prior_emb=prior_emb,
                    context_ball=context_ball
                )
            else:
                step_out = agent_lmu.step(x_t_proj, mem_window)
            
            h_roll = step_out["h_t"]
            mem_window = step_out["mem_window"]
            m_future_list.append(h_roll[:, :n_agents, :].unsqueeze(1))
            last_pos = next_pos_players

        fut_pred_long = torch.cat(fut_steps, dim=0).contiguous()  # [T_long, B, 22, 2]
        fut_pred_short = fut_pred_long[:short_T]  # [T_short, B, 22, 2]
        
        if h_burn_seq is not None and m_future_list:
            m_future_seq = torch.cat(m_future_list, dim=1).contiguous()
            m_seq = h_burn_seq[:, :, :n_agents, :]
            m_full_seq = torch.cat([m_seq, m_future_seq], dim=1)
            agents_mem_for_head = _exp_decay_pool(m_full_seq, gamma=lmu_gamma, n_agents=n_agents)
        else:
            agents_mem_for_head = None

    elif traj_model == 'lstm':
        # LSTM path
        x_lmu_in = _pack_players_ball(ctx_seqX_all[:, :cfg.burn_in, :, :])
        x_lmu_proj = lmu_proj(x_lmu_in)

        # ✅ FIX: Handle dict return
        lmu_out = agent_lmu(x_lmu_proj)
        if isinstance(lmu_out, dict):
            h_roll = lmu_out["h_last"]  # [B, 23, h_dim]
            m_roll = lmu_out.get("m_last", None)  # cell state
        else:
            h_roll = lmu_out
            m_roll = None

        last_pos = _pack_players_ball(ctx_seqX_all[:, cfg.burn_in-1, :, :])
        fut_steps = []

        for t in range(long_T):
            # ✅ h_roll is now tensor [B, 23, h_dim]
            if use_context:
                next_pos_23 = traj_step_head(
                    h_roll,
                    last_pos,
                    m_t=m_roll,  # Pass cell state
                    h_players=h_players,
                )
            else:
                next_pos_23 = traj_step_head(
                    h_roll,
                    last_pos,
                    m_t=m_roll,
                )
            
            if predict_ball:
                h_ball_roll = h_roll[:, ball_idx, :]
                ball_pos = real_head.ball_step_head(
                    ball_hist=ball_hist,
                    h_ball_roll=h_ball_roll,
                    context_ball=context_ball if use_context else None
                )
            else:
                ball_pos = next_pos_23[:, ball_idx, :]

            ball_pred_list.append(ball_pos)
            ball_hist = torch.cat([ball_hist[:, 1:, :], ball_pos.unsqueeze(1)], dim=1)

            next_pos_23 = torch.cat([next_pos_23[:, :n_agents, :], ball_pos.unsqueeze(1)], dim=1)
            next_pos_players = next_pos_23[:, :n_agents, :]
            fut_steps.append(next_pos_players.unsqueeze(0))
            
            # ✅ FIX: Handle step() return
            x_t_proj = lmu_proj(next_pos_23.unsqueeze(1)).squeeze(1)
            step_out = agent_lmu.step(x_t_proj, h_roll, m_roll)
            
            if isinstance(step_out, tuple):
                h_roll, m_roll = step_out  # (h_t, c_t)
            else:
                h_roll = step_out
                m_roll = None
            
            last_pos = next_pos_23

        fut_pred_long = torch.cat(fut_steps, dim=0).contiguous()
        fut_pred_short = fut_pred_long[:short_T]
        agents_mem_for_head = None
    elif traj_model == 'constant_velocity':
        # 1. 提取 burn_in 轨迹
        x_lmu_in = _pack_players_ball(ctx_seqX_all[:, :cfg.burn_in, :, :])  # [B, 20, 23, 2]
        
        # 2. ✅ 设置历史轨迹（必须！）
        if hasattr(agent_lmu, 'set_history'):
            agent_lmu.set_history(x_lmu_in[:, :, :n_agents, :])  # [B, 20, 22, 2]
        
        # 3. 投影特征
        x_lmu_proj = lmu_proj(x_lmu_in)  # [B, 20, 23, F_LMU]
        
        # 4. Forward：获取速度
        lmu_out = agent_lmu(x_lmu_proj[:, :, :n_agents, :])
        h_roll = lmu_out["h_last"]  # [B, 22, h_dim]
        m_roll = lmu_out["m_last"]  # [B, 22, h_dim] 速度信息
        velocities = lmu_out.get("velocities", None)  # [B, 22, 2] m/s
        
        # 5. 初始位置
        last_pos = ctx_seqX_all[:, cfg.burn_in-1, :n_agents, :]  # [B, 22, 2]
        ball_pos = ctx_seqX_all[:, cfg.burn_in-1, ball_idx, :]   # [B, 2]
        
        # 6. 自回归生成
        fut_steps = []
        for t in range(long_T):
            next_pos_players = traj_step_head(h_roll, last_pos, m_t=m_roll)
            
            if predict_ball:
                ball_vel = (ball_hist[:, -1, :] - ball_hist[:, -2, :]) / 0.1
                ball_pos = ball_pos + ball_vel * 0.1
            else:
                ball_pos = ball_hist[:, -1, :]
            
            ball_pred_list.append(ball_pos)
            ball_hist = torch.cat([ball_hist[:, 1:, :], ball_pos.unsqueeze(1)], dim=1)
            
            fut_steps.append(next_pos_players.unsqueeze(0))
            last_pos = next_pos_players
        
        # 7. 输出
        fut_pred_long = torch.cat(fut_steps, dim=0).contiguous()
        fut_pred_short = fut_pred_long[:short_T]
        agents_mem_for_head = None

    elif traj_model == 'vrnn':
        # VRNN path - similar to baller2vec/star
        x_lmu_in = _pack_players_ball(ctx_seqX_all[:, :cfg.burn_in, :, :])
        x_lmu_proj = lmu_proj(x_lmu_in)

        if use_context:
            lmu_out = agent_lmu(
                x_lmu_proj,
                h_players=h_players,
                prior_emb=prior_emb,
                context_ball=context_ball,
            )
        else:
            lmu_out = agent_lmu(x_lmu_proj)
        
        h_roll = lmu_out["h_last"]
        h_burn_seq = lmu_out.get("h_seq", None)
        mem_window = lmu_out.get("mem_window", None)
        
        # ✅ 获取 VRNN 特有的 KLD 和 NLL 损失
        vrnn_kld, vrnn_nll = agent_lmu.get_vrnn_losses()
        kld = vrnn_kld if vrnn_kld is not None else kld
        nll = vrnn_nll if vrnn_nll is not None else nll

        last_pos = ctx_seqX_all[:, cfg.burn_in-1, :n_agents, :]
        fut_steps = []
        m_future_list = []

        for t in range(long_T):
            # 1. Predict next position for players
            if use_context:
                next_pos_players = traj_step_head(
                    h_roll[:, :n_agents, :],
                    last_pos,
                    m_t=None,
                    h_players=h_players,
                    prior_emb=prior_emb,
                    volterra_features=None
                )
            else:
                next_pos_players = traj_step_head(
                    h_roll[:, :n_agents, :],
                    last_pos,
                    m_t=None,
                )
            
            # 2. Ball prediction logic [Remains the same]
            if predict_ball:
                h_ball_roll = h_roll[:, ball_idx, :]
                ball_pos = real_head.ball_step_head(
                    ball_hist=ball_hist,
                    h_ball_roll=h_ball_roll,
                    context_ball=context_ball if use_context else None
                )
            else:
                ball_pos = ball_hist[:, -1, :]
            
            ball_pred_list.append(ball_pos)
            ball_hist = torch.cat([ball_hist[:, 1:, :], ball_pos.unsqueeze(1)], dim=1)

            # 3. Prepare input for next step (Concatenate ball)
            ball_t = ball_pos.unsqueeze(1)
            next_pos_23 = torch.cat([next_pos_players, ball_t], dim=1)
            fut_steps.append(next_pos_players.unsqueeze(0))
            
            # 4. Update VRNN state (step)
            x_t_proj = lmu_proj(next_pos_23.unsqueeze(1))
            x_t_proj = x_t_proj.squeeze(1)
            
            if use_context:
                step_out = agent_lmu.step(
                    x_t_proj,
                    mem_window,
                    h_players=h_players,      # <--- Ensure these are passed
                    prior_emb=prior_emb,      # <--- Ensure these are passed
                    context_ball=context_ball # <--- Ensure these are passed
                )
            else:
                step_out = agent_lmu.step(x_t_proj, mem_window)
            
            h_roll = step_out["h_t"]
            mem_window = step_out["mem_window"]
            m_future_list.append(h_roll[:, :n_agents, :].unsqueeze(1))
            last_pos = next_pos_players

        fut_pred_long = torch.cat(fut_steps, dim=0).contiguous()
        fut_pred_short = fut_pred_long[:short_T]
        
        if h_burn_seq is not None and m_future_list:
            m_future_seq = torch.cat(m_future_list, dim=1).contiguous()
            m_seq = h_burn_seq[:, :, :n_agents, :]
            m_full_seq = torch.cat([m_seq, m_future_seq], dim=1)
            agents_mem_for_head = _exp_decay_pool(m_full_seq, gamma=lmu_gamma, n_agents=n_agents)
        else:
            agents_mem_for_head = None

    else:
        raise NotImplementedError(f"Model {traj_model} not implemented in curriculum version")
    
    # Placeholder losses
    kld = fut_pred_long.sum() * 0.0
    nll = fut_pred_long.sum() * 0.0
    
    # =========== Ground truth preparation ===========
    ctx_seqX_players = ctx_seqX[..., :n_agents, :]
    fut_seqX_players = fut_seqX[..., :n_agents, :]

    labels_ctx_last = [list(map(str, seq[-1])) for seq in batch["ctx_labels_22"]]
    labels_fut0 = [list(map(str, seq[0])) for seq in batch["fut_labels_22"]]

    fut_gt_traj_long = ctx_seqX_players[:, 0, cfg.burn_in:cfg.burn_in + long_T, :, :]
    fut_gt_traj_long = fut_gt_traj_long.permute(1, 0, 2, 3).contiguous()  # [T_long, B, 22, 2]
    fut_gt_traj_short = fut_gt_traj_long[:short_T]  # [T_short, B, 22, 2]

    next_fut_gt = fut_seqX_players[:, 0, :long_T, :, :]
    next_fut_gt = next_fut_gt.permute(1, 0, 2, 3).contiguous()

    fut_gt_ctx = reindex_by_labels(
        next_fut_gt, src_labels_BA=labels_fut0, dst_labels_BA=labels_ctx_last
    )

    # Target player alignment
    tgt_index_fut = fut_players[:, 0]
    tgt_index_ctx = torch.full_like(tgt_index_fut, -1)
    for b in range(B):
        fi = int(tgt_index_fut[b].item())
        if 0 <= fi < n_agents:
            lbl = labels_fut0[b][fi]
            tgt_index_ctx[b] = labels_ctx_last[b].index(lbl) if lbl in labels_ctx_last[b] else -1

    # Event timing (ground truth)
    epos = fut_positions[:, 0, :]
    ar = torch.arange(B, device=device)
    tgt_traj_future = fut_gt_traj_long[:, ar, tgt_index_ctx.clamp(0, n_agents - 1), :]
    sqdist = ((tgt_traj_future - epos.unsqueeze(0)) ** 2).sum(dim=-1)
    valid_tgt = tgt_index_ctx >= 0
    sqdist = torch.where(valid_tgt.unsqueeze(0), sqdist, torch.full_like(sqdist, 1e9))
    eidx = sqdist.argmin(dim=0).clamp(0, long_T - 1)

    # =========== Multi-scale trajectory loss ===========
    loss_traj_short = F.mse_loss(fut_pred_short, fut_gt_traj_short)
    loss_traj_long = F.mse_loss(fut_pred_long, fut_gt_traj_long)
        # ✅ Use curriculum-controlled w_short
    # Epoch 0-10:  w_short=1.0, so only loss_short matters
    # Epoch 10-20: w_short: 1.0→0.5, gradually introduce loss_long
    # Epoch 20+:   w_short=0.5, balanced
    loss_traj_all = w_short * loss_traj_short + (1.0 - w_short) * loss_traj_long

    # Metrics for both scales
    ADE_short, FDE_short = _compute_ADE_FDE(fut_pred_short, fut_gt_traj_short)
    ADE_long, FDE_long = _compute_ADE_FDE(fut_pred_long, fut_gt_traj_long)
    
    loss_ball = torch.zeros((), device=device)
    if predict_ball:
        ball_pred_TB2 = torch.stack(ball_pred_list, dim=0).contiguous()
        ball_gt_BT2 = ctx_seqX[:, 0, cfg.burn_in:cfg.burn_in + long_T, ball_idx, :]
        ball_gt_TB2 = ball_gt_BT2.permute(1, 0, 2).contiguous()
        loss_ball = F.mse_loss(ball_pred_TB2, ball_gt_TB2)

    # =========== Event prediction (skip if traj_only) ===========
    if is_traj_only:
        # Dummy outputs during traj_only phase
        tgt_probs = torch.zeros(B, n_agents, device=device)
        event_logits = torch.zeros(B, n_agents, int(getattr(cfg, "n_event_types", 5)), device=device)
        time_logits = torch.zeros(B, n_agents, long_T, device=device)
        pos_pred = torch.zeros(B, n_agents, 2, device=device)
        tgt_logits = torch.zeros(B, n_agents, device=device)
        
        loss_tgt = torch.zeros((), device=device)
        loss_eidx = torch.zeros((), device=device)
        disentangle_loss = torch.zeros((), device=device)
        ortho_loss_val = 0.0
        align_loss_val = 0.0
        adv_loss_val = 0.0
        disc_acc_val = 0.5
        hsic_score_val = 0.0
        
    else:
        # Event head returns 5 values
        tgt_probs, event_logits, time_logits, pos_pred, tgt_logits = real_head(
            dyn_in, edge_in, ctx_seqX[:, 0].to(dyn_in.device), enc_labels_22,
            cols=dyn_cols, edge_cols=edge_cols,
            fut_pred=fut_pred_long,  # Use long-term for event prediction
            h_players=h_players,
            h_ball=h_ball,
            team_labels=team_labels,
            event_emb=event_emb,
            context_ball=context_ball,
            ball_all=ball_all,
            players_all=players_all,
        )

        # Event losses (only tgt and eidx)
        if valid_tgt.any().item():
            loss_tgt = F.cross_entropy(tgt_logits[valid_tgt], tgt_index_ctx[valid_tgt])
            
            tl = time_logits[valid_tgt]
            ti = tgt_index_ctx[valid_tgt].view(-1, 1, 1)
            tl_tgt = tl.gather(1, ti.expand(-1, 1, tl.size(-1))).squeeze(1)
            loss_eidx = F.cross_entropy(tl_tgt, eidx[valid_tgt])
        else:
            loss_tgt = torch.zeros((), device=device)
            loss_eidx = torch.zeros((), device=device)

        # Disentangle loss
        disentangle_loss = torch.zeros((), device=device)
        ortho_loss_val = 0.0
        align_loss_val = 0.0
        adv_loss_val = 0.0
        disc_acc_val = 0.5
        hsic_score_val = 0.0

        if hasattr(real_head, 'get_disentangle_loss'):
            disentangle_losses = real_head.get_disentangle_loss()
            if disentangle_losses is not None:
                disentangle_loss = disentangle_losses['disentangle_loss']
                ortho_loss_val = float(disentangle_losses['ortho_loss'].item())
                align_loss_val = float(disentangle_losses['align_loss'].item())
                adv_loss_val = float(disentangle_losses.get('adv_loss', 0.0))
                disc_acc_val = float(disentangle_losses.get('disc_acc', 0.5))
                hsic_score_val = float(disentangle_losses.get('hsic_score', 0.0))

    # =========== Compute metrics with Top-K ===========
    valid_count = int(valid_tgt.sum().item())
    total_count = int(B)

    # Initialize all metrics
    acc_tgt_sum = mae_eidx_sum = mse_epos_sum = 0.0
    acc_tgt_top3_sum = mae_eidx_top3_sum = mse_epos_top3_sum = 0.0
    acc_tgt_top5_sum = mae_eidx_top5_sum = mse_epos_top5_sum = 0.0
    
    if not is_traj_only and valid_tgt.any().item():
        b_idx = torch.arange(B, device=device)
        
        # ========== Top-1 Metrics ==========
        pred_tgt_top1 = tgt_logits.argmax(dim=-1)  # [B]
        acc_tgt_sum = (pred_tgt_top1[valid_tgt] == tgt_index_ctx[valid_tgt]).float().sum().item()

        tl_all = time_logits  # [B, 22, T]
        tl_top1 = tl_all[b_idx, pred_tgt_top1, :]  # [B, T]
        pred_eidx_top1 = tl_top1.argmax(dim=-1)  # [B]
        epos_hat_top1 = fut_pred_long[pred_eidx_top1.clamp(0, long_T-1), b_idx, pred_tgt_top1.clamp(0, n_agents - 1), :]

        mae_eidx_sum = (pred_eidx_top1[valid_tgt].float() - eidx[valid_tgt].float()).abs().sum().item()
        mse_epos_sum = F.mse_loss(epos_hat_top1[valid_tgt], epos[valid_tgt], reduction="sum").item()
        
        # ========== Top-3 Metrics ==========
        k3 = min(3, tgt_logits.size(-1))
        _, topk3_indices = torch.topk(tgt_logits, k=k3, dim=-1)  # [B, k3]
        
        gt_in_top3 = (topk3_indices == tgt_index_ctx.unsqueeze(-1)).any(dim=-1)  # [B]
        acc_tgt_top3_sum = gt_in_top3[valid_tgt].float().sum().item()

        best_mae_top3 = torch.full((B,), float('inf'), device=device)
        best_mse_top3 = torch.full((B,), float('inf'), device=device)

        for k in range(k3):
            cand_tgt = topk3_indices[:, k]
            cand_tl = tl_all[b_idx, cand_tgt, :]
            cand_eidx = cand_tl.argmax(dim=-1)
            cand_epos = fut_pred_long[cand_eidx.clamp(0, long_T-1), b_idx, cand_tgt.clamp(0, n_agents - 1), :]
            
            cand_mae = (cand_eidx.float() - eidx.float()).abs()
            cand_mse = ((cand_epos - epos) ** 2).sum(dim=-1)
            
            # ✅ MAE 和 MSE 独立选最小
            best_mae_top3 = torch.minimum(best_mae_top3, cand_mae)
            best_mse_top3 = torch.minimum(best_mse_top3, cand_mse)

        mae_eidx_top3_sum = best_mae_top3[valid_tgt].sum().item()
        mse_epos_top3_sum = best_mse_top3[valid_tgt].sum().item()
        
        # ========== Top-5 Metrics ==========
        k5 = min(5, tgt_logits.size(-1))
        _, topk5_indices = torch.topk(tgt_logits, k=k5, dim=-1)  # [B, k5]
        
        gt_in_top5 = (topk5_indices == tgt_index_ctx.unsqueeze(-1)).any(dim=-1)  # [B]
        acc_tgt_top5_sum = gt_in_top5[valid_tgt].float().sum().item()
        
        best_mae_top5 = torch.full((B,), float('inf'), device=device)
        best_mse_top5 = torch.full((B,), float('inf'), device=device)
        
        for k in range(k5):
            cand_tgt = topk5_indices[:, k]
            cand_tl = tl_all[b_idx, cand_tgt, :]
            cand_eidx = cand_tl.argmax(dim=-1)
            cand_epos = fut_pred_long[cand_eidx.clamp(0, long_T-1), b_idx, cand_tgt.clamp(0, n_agents - 1), :]
            
            cand_mae = (cand_eidx.float() - eidx.float()).abs()
            cand_mse = ((cand_epos - epos) ** 2).sum(dim=-1)
            
            # ✅ 修改：MAE 和 MSE 独立选最小（和 Top-3 保持一致）
            best_mae_top5 = torch.minimum(best_mae_top5, cand_mae)
            best_mse_top5 = torch.minimum(best_mse_top5, cand_mse)
        
        mae_eidx_top5_sum = best_mae_top5[valid_tgt].sum().item()
        mse_epos_top5_sum = best_mse_top5[valid_tgt].sum().item()

    # =========== Total loss with curriculum weights ===========
    w_ball = float(getattr(cfg, "w_ball", 1.0))
    
    loss = (
        nll + cfg.beta_kl * kld +
        w_traj * cfg.w_traj * loss_traj_all +  # Already weighted short+long
        w_ball * loss_ball +
        w_tgt * loss_tgt +
        w_eidx * loss_eidx +
        w_disentangle * disentangle_loss
    )

    # =========== Backward ===========
    if mode == 'train' and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(head.parameters()), 1.0)
        optimizer.step()

    # =========== Complete logs with multi-scale metrics ===========
    logs = {
        'total_sum': float(loss.item() * total_count),
        'nll_sum': float(nll.item() * total_count),
        'kld_sum': float(kld.item() * total_count),
        
        # ✅ Short-term trajectory metrics
        'traj_loss_short_sum': float(loss_traj_short.item() * total_count),
        'ADE_short_sum': float(ADE_short * total_count),
        'FDE_short_sum': float(FDE_short * total_count),
        
        # Long-term trajectory metrics (reported as main metrics)
        'traj_loss_sum': float(loss_traj_long.item() * total_count),
        'ADE_sum': float(ADE_long * total_count),
        'FDE_sum': float(FDE_long * total_count),
        
        'ball_loss_sum': float(loss_ball.item() * total_count),
        'tgt_loss_sum': float(loss_tgt.item() * max(valid_count, 1)),
        'eidx_loss_sum': float(loss_eidx.item() * max(valid_count, 1)),
        
        # Top-1 metrics
        'acc_tgt_sum': float(acc_tgt_sum),
        'mae_eidx_sum': float(mae_eidx_sum),
        'mse_epos_sum': float(mse_epos_sum),
        
        # Top-3 metrics
        'acc_tgt_top3_sum': float(acc_tgt_top3_sum),
        'mae_eidx_top3_sum': float(mae_eidx_top3_sum),
        'mse_epos_top3_sum': float(mse_epos_top3_sum),
        
        # Top-5 metrics
        'acc_tgt_top5_sum': float(acc_tgt_top5_sum),
        'mae_eidx_top5_sum': float(mae_eidx_top5_sum),
        'mse_epos_top5_sum': float(mse_epos_top5_sum),
        
        # Disentangle losses
        'disentangle_loss_sum': float(disentangle_loss.item() * total_count),
        'ortho_loss_sum': float(ortho_loss_val * total_count),
        'align_loss_sum': float(align_loss_val * total_count),
        # ✅ 新增：
        'adv_loss_sum': float(adv_loss_val * total_count),
        'disc_acc_sum': float(disc_acc_val * total_count),
        'hsic_score_sum': float(hsic_score_val * total_count),     

        'aux_epos_tf_sum': 0.0,
        'aux_win_sum': 0.0,
        'count_total': float(total_count),
        'count_valid': float(valid_count),
        'phase': phase,
    }
    return logs


def evaluate_curriculum(
    loader, 
    head, 
    cfg, 
    mode='test',
    curriculum_weights=None,
    dyn_cols=None, 
    edge_cols=None
):
    """
    Evaluation with curriculum support (uses full weights)
    
    Returns result with both short-term and long-term metrics
    """
    from train.ddp import is_dist_avail_and_initialized
    import torch.distributed as dist
    
    if curriculum_weights is None:
        curriculum_weights = {
            'traj': 1.0,
            'tgt': cfg.w_tgt,
            'eidx': cfg.w_eidx,
            'epos': 0.0,
            'disentangle': getattr(cfg, 'w_disentangle', 0.1),
            'phase': 'full',
        }
    
    device = next(head.parameters()).device
    
    SUM_KEYS = [
        'total_sum', 'nll_sum', 'kld_sum',
        'traj_loss_short_sum', 'ADE_short_sum', 'FDE_short_sum',  # ✅ Short-term
        'traj_loss_sum', 'ADE_sum', 'FDE_sum', 'ball_loss_sum',  # Long-term
        'tgt_loss_sum', 'eidx_loss_sum',
        'acc_tgt_sum', 'mae_eidx_sum', 'mse_epos_sum',
        'acc_tgt_top3_sum', 'mae_eidx_top3_sum', 'mse_epos_top3_sum',
        'acc_tgt_top5_sum', 'mae_eidx_top5_sum', 'mse_epos_top5_sum',
        'disentangle_loss_sum', 'ortho_loss_sum', 'align_loss_sum',
        'adv_loss_sum', 
        'disc_acc_sum', 
        'hsic_score_sum',
        'aux_epos_tf_sum', 'aux_win_sum',
    ]
    COUNT_KEYS = ['count_total', 'count_valid']
    
    sums = {k: 0.0 for k in SUM_KEYS}
    counts = {k: 0.0 for k in COUNT_KEYS}
    
    head.eval()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            
            logs = step_batch_curriculum(
                batch, head, cfg,
                mode=mode,
                optimizer=None,
                curriculum_weights=curriculum_weights,
                dyn_cols=dyn_cols,
                edge_cols=edge_cols,
            )
            
            for k in SUM_KEYS:
                sums[k] += float(logs.get(k, 0.0))
            for k in COUNT_KEYS:
                counts[k] += float(logs.get(k, 0.0))
    
    head.train()
    
    # Aggregate across ranks
    sums_vec = torch.tensor([sums[k] for k in SUM_KEYS], device=device)
    counts_vec = torch.tensor([counts[k] for k in COUNT_KEYS], device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(sums_vec, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts_vec, op=dist.ReduceOp.SUM)
    
    sums = {k: float(v) for k, v in zip(SUM_KEYS, sums_vec.tolist())}
    counts = {k: float(v) for k, v in zip(COUNT_KEYS, counts_vec.tolist())}
    
    ct = max(counts['count_total'], 1.0)
    cv = max(counts['count_valid'], 1.0)
    
    result = {
        # ✅ Short-term trajectory metrics
        'ADE_short': float(sums['ADE_short_sum'] / ct),
        'FDE_short': float(sums['FDE_short_sum'] / ct),
        'traj_loss_short': float(sums['traj_loss_short_sum'] / ct),
        
        # Long-term trajectory metrics (main metrics)
        'ADE': float(sums['ADE_sum'] / ct),
        'FDE': float(sums['FDE_sum'] / ct),
        'traj_loss': float(sums['traj_loss_sum'] / ct),
        'ball_loss': float(sums['ball_loss_sum'] / ct),
        
        # Event losses
        'tgt_loss': float(sums['tgt_loss_sum'] / cv),
        'eidx_loss': float(sums['eidx_loss_sum'] / cv),
        
        # Event accuracy (top-1)
        'acc_tgt': float(sums['acc_tgt_sum'] / cv),
        'mae_eidx': float(sums['mae_eidx_sum'] / cv),
        'mse_epos': float(sums['mse_epos_sum'] / cv),
        
        # Top-3 metrics
        'acc_tgt_top3': float(sums['acc_tgt_top3_sum'] / cv),
        'mae_eidx_top3': float(sums['mae_eidx_top3_sum'] / cv),
        'mse_epos_top3': float(sums['mse_epos_top3_sum'] / cv),
        
        # Top-5 metrics
        'acc_tgt_top5': float(sums['acc_tgt_top5_sum'] / cv),
        'mae_eidx_top5': float(sums['mae_eidx_top5_sum'] / cv),
        'mse_epos_top5': float(sums['mse_epos_top5_sum'] / cv),
        
        # Disentangle losses
        'disentangle_loss': float(sums['disentangle_loss_sum'] / ct),
        'ortho_loss': float(sums['ortho_loss_sum'] / ct),
        'align_loss': float(sums['align_loss_sum'] / ct),
        
        'adv_loss': float(sums['adv_loss_sum'] / ct),
        'disc_acc': float(sums['disc_acc_sum'] / ct),
        'hsic_score': float(sums['hsic_score_sum'] / ct),

        # Overall
        'total_loss': float(sums['total_sum'] / ct),
        'nll': float(sums['nll_sum'] / ct),
        'kld': float(sums['kld_sum'] / ct),
    }
    
    return result
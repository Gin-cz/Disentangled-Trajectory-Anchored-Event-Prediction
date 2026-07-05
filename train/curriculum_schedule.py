# -*- coding: utf-8 -*-
import math
from typing import Dict

class CurriculumScheduler:
    """
    解耦版课程学习调度器 (Decoupled Curriculum Scheduler)
    
    允许独立控制：
    1. 轨迹长度的渐进 (Short -> Full Trajectory)
    2. 事件任务的介入时间 (No Event -> Event)
    """
    
    def __init__(
        self,
        # --- 轨迹相关设置 ---
        traj_short_only_epochs: int = 20,   # 阶段1: 只练短期 (0-20)
        traj_warmup_epochs: int = 10,       # 阶段2: 逐渐引入长期 (20-30)
        initial_short_weight: float = 1.0,  # 初始只看短期
        final_short_weight: float = 0.5,    # 最终长短平衡
        
        # --- 事件相关设置 ---
        event_start_epoch: int = 50,        # 阶段3: 开始引入事件的时间 (50)
        event_warmup_epochs: int = 10,      # 阶段4: 事件权重从0升到1的过程 (50-60)
        initial_event_weight: float = 0.0,
        final_event_weight: float = 1.0,
        
        # --- 这里的 schedule_type 主要控制 warmup 的曲线 ---
        schedule_type: str = 'cosine',      
        
        # --- 权重缩放 ---
        tgt_weight_scale: float = 1.0,
        eidx_weight_scale: float = 1.0,
        epos_weight_scale: float = 1.0,
        
        # --- 解耦权重 ---
        initial_disentangle_weight: float = 0.0,
        final_disentangle_weight: float = 0.1,
        use_hsic: bool = True,
    ):
        # Trajectory Config
        self.traj_short_only_epochs = traj_short_only_epochs
        self.traj_warmup_epochs = traj_warmup_epochs
        self.traj_transition_end = traj_short_only_epochs + traj_warmup_epochs
        self.initial_short_weight = initial_short_weight
        self.final_short_weight = final_short_weight
        
        # Event Config
        self.event_start_epoch = event_start_epoch
        self.event_warmup_epochs = event_warmup_epochs
        self.event_transition_end = event_start_epoch + event_warmup_epochs
        self.initial_event_weight = initial_event_weight
        self.final_event_weight = final_event_weight
        
        # Others
        self.schedule_type = schedule_type
        self.tgt_weight_scale = tgt_weight_scale
        self.eidx_weight_scale = eidx_weight_scale
        self.epos_weight_scale = epos_weight_scale
        
        self.initial_disentangle_weight = initial_disentangle_weight
        self.final_disentangle_weight = final_disentangle_weight
        self.use_hsic = use_hsic

    def _calc_progress(self, current_epoch, start_epoch, warmup_len):
        """通用进度计算器 [0.0, 1.0]"""
        if current_epoch < start_epoch:
            return 0.0
        elif current_epoch >= start_epoch + warmup_len:
            return 1.0
        else:
            return (current_epoch - start_epoch) / max(warmup_len, 1)

    def _schedule_value(self, progress: float, initial: float, final: float) -> float:
        """根据进度计算具体数值"""
        if self.schedule_type == 'cosine':
            cosine_progress = 0.5 * (1 - math.cos(math.pi * progress))
            return initial + (final - initial) * cosine_progress
        elif self.schedule_type == 'linear':
            return initial + (final - initial) * progress
        else: # step
            return final if progress >= 1.0 else initial

    def get_weights(self, epoch):
        # -----------------------------------------------------------
        # 1. 计算轨迹权重的进度 (Trajectory Curriculum)
        # -----------------------------------------------------------
        # 进度: 0.0 (只短) -> 1.0 (平衡)
        traj_progress = self._calc_progress(
            epoch, 
            self.traj_short_only_epochs, 
            self.traj_warmup_epochs
        )
        
        # w_short_traj: 1.0 -> 0.5
        w_short_traj = self._schedule_value(
            traj_progress, 
            self.initial_short_weight, 
            self.final_short_weight
        )

        # -----------------------------------------------------------
        # 2. 计算事件权重的进度 (Event Curriculum)
        # -----------------------------------------------------------
        # 进度: 0.0 (无事件) -> 1.0 (全权重)
        event_progress = self._calc_progress(
            epoch,
            self.event_start_epoch,
            self.event_warmup_epochs
        )
        
        # event_base_weight: 0.0 -> 1.0
        event_base_weight = self._schedule_value(
            event_progress,
            self.initial_event_weight,
            self.final_event_weight
        )
        
        # Disentangle 也可以跟随 event 进度，或者单独设置
        # 这里假设 disentangle 在事件开始训练时同步开始
        disentangle_weight = self._schedule_value(
            event_progress,
            self.initial_disentangle_weight,
            self.final_disentangle_weight
        )

        # -----------------------------------------------------------
        # 3. 确定当前阶段名称 (用于日志)
        # -----------------------------------------------------------
        if epoch < self.traj_short_only_epochs:
            phase = "Short_Traj"
        elif epoch < self.traj_transition_end:
            phase = "Traj_Warmup"
        elif epoch < self.event_start_epoch:
            phase = "Full_Traj" # 长短都练，但还没事件
        elif epoch < self.event_transition_end:
            phase = "Event_Warmup"
        else:
            phase = "Fully_Trained"

        return {
            'traj': 1.0,
            'tgt': event_base_weight * self.tgt_weight_scale,
            'eidx': event_base_weight * self.eidx_weight_scale,
            'epos': event_base_weight * self.epos_weight_scale,
            'disentangle': disentangle_weight,
            'w_short_traj': w_short_traj,
            'phase': phase,
            'traj_progress': traj_progress,
            'event_progress': event_progress,
            'use_hsic': self.use_hsic
        }

    def __str__(self):
        hsic_status = "ON" if self.use_hsic else "OFF"
        return (
            f"CurriculumScheduler (Decoupled):\n"
            f"  HSIC: {hsic_status}\n"
            f"  [Traj] Short-Only: 0-{self.traj_short_only_epochs}\n"
            f"  [Traj] Transition: {self.traj_short_only_epochs}-{self.traj_transition_end} "
            f"(w_short: {self.initial_short_weight}->{self.final_short_weight})\n"
            f"  [Event] Start Epoch: {self.event_start_epoch}\n"
            f"  [Event] Warmup: {self.event_start_epoch}-{self.event_transition_end}\n"
        )

# ============================================================================
# 验证代码：符合你的时间线要求
# ============================================================================
if __name__ == "__main__":
    scheduler = CurriculumScheduler(
        # 1. 轨迹部分
        traj_short_only_epochs=20,   # 前20个epoch只练short
        traj_warmup_epochs=10,       # 20-30: 慢慢加入long (w_short 1.0 -> 0.5)
        
        # 2. 事件部分
        event_start_epoch=50,        # 50之后才开始练事件
        event_warmup_epochs=10,      # 50-60: 事件权重逐渐增加
        
        tgt_weight_scale=10.0,
        eidx_weight_scale=2.0,
        final_disentangle_weight=0.1
    )

    print(scheduler)
    print(f"{'Epoch':^5} | {'Phase':^14} | {'w_short':^7} | {'w_tgt':^7} | {'w_eidx':^7}")
    print("-" * 60)
    
    # 测试关键时间点
    test_epochs = [0, 19, 20, 25, 30, 40, 49, 50, 55, 60, 65]
    
    # 也可以打印所有epoch
    for epoch in range(70):
        if epoch in test_epochs or True: # 打印所有以便你看清楚变化
            w = scheduler.get_weights(epoch)
            print(f"{epoch:5d} | {w['phase']:14s} | {w['w_short_traj']:7.2f} | "
                  f"{w['tgt']:7.2f} | {w['eidx']:7.2f}")
            
            # 在关键节点打印分隔线
            if epoch == 19: print("-" * 60 + " (End Short Only)")
            if epoch == 29: print("-" * 60 + " (Traj Fully Balanced)")
            if epoch == 49: print("-" * 60 + " (Start Event Training)")
            if epoch == 59: print("-" * 60 + " (Fully Active)")
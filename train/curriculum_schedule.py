"""Define the curriculum schedule for trajectory and event objectives."""

import math

class CurriculumScheduler:

    def __init__(
        self,
        traj_short_only_epochs: int = 20,
        traj_warmup_epochs: int = 10,
        initial_short_weight: float = 1.0,
        final_short_weight: float = 0.5,

        event_start_epoch: int = 50,
        event_warmup_epochs: int = 10,
        initial_event_weight: float = 0.0,
        final_event_weight: float = 1.0,

        schedule_type: str = 'cosine',

        tgt_weight_scale: float = 1.0,
        eidx_weight_scale: float = 1.0,
        epos_weight_scale: float = 1.0,

        initial_disentangle_weight: float = 0.0,
        final_disentangle_weight: float = 0.1,
        use_hsic: bool = True,
    ):
        self.traj_short_only_epochs = traj_short_only_epochs
        self.traj_warmup_epochs = traj_warmup_epochs
        self.traj_transition_end = traj_short_only_epochs + traj_warmup_epochs
        self.initial_short_weight = initial_short_weight
        self.final_short_weight = final_short_weight

        self.event_start_epoch = event_start_epoch
        self.event_warmup_epochs = event_warmup_epochs
        self.event_transition_end = event_start_epoch + event_warmup_epochs
        self.initial_event_weight = initial_event_weight
        self.final_event_weight = final_event_weight

        self.schedule_type = schedule_type
        self.tgt_weight_scale = tgt_weight_scale
        self.eidx_weight_scale = eidx_weight_scale
        self.epos_weight_scale = epos_weight_scale

        self.initial_disentangle_weight = initial_disentangle_weight
        self.final_disentangle_weight = final_disentangle_weight
        self.use_hsic = use_hsic

    def _calc_progress(self, current_epoch, start_epoch, warmup_len):
        if current_epoch < start_epoch:
            return 0.0
        elif current_epoch >= start_epoch + warmup_len:
            return 1.0
        else:
            return (current_epoch - start_epoch) / max(warmup_len, 1)

    def _schedule_value(self, progress: float, initial: float, final: float) -> float:
        if self.schedule_type == 'cosine':
            cosine_progress = 0.5 * (1 - math.cos(math.pi * progress))
            return initial + (final - initial) * cosine_progress
        elif self.schedule_type == 'linear':
            return initial + (final - initial) * progress
        else:
            return final if progress >= 1.0 else initial

    def get_weights(self, epoch):
        traj_progress = self._calc_progress(
            epoch,
            self.traj_short_only_epochs,
            self.traj_warmup_epochs
        )

        w_short_traj = self._schedule_value(
            traj_progress,
            self.initial_short_weight,
            self.final_short_weight
        )

        event_progress = self._calc_progress(
            epoch,
            self.event_start_epoch,
            self.event_warmup_epochs
        )

        event_base_weight = self._schedule_value(
            event_progress,
            self.initial_event_weight,
            self.final_event_weight
        )

        disentangle_weight = self._schedule_value(
            event_progress,
            self.initial_disentangle_weight,
            self.final_disentangle_weight
        )

        if epoch < self.traj_short_only_epochs:
            phase = "Short_Traj"
        elif epoch < self.traj_transition_end:
            phase = "Traj_Warmup"
        elif epoch < self.event_start_epoch:
            phase = "Full_Traj"
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

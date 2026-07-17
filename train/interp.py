"""Interpolate trajectory and event-time predictions."""

import torch

def torch_interp_1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    inds = torch.bucketize(x, xp) - 1
    inds = inds.clamp(0, len(xp) - 2)
    x0, x1 = xp[inds], xp[inds + 1]
    y0, y1 = fp[inds], fp[inds + 1]
    slope = (y1 - y0) / (x1 - x0 + 1e-12)
    return y0 + slope * (x - x0)

def interpolate_traj(seq: torch.Tensor) -> torch.Tensor:
    B, C, T, A, D = seq.shape
    seq_out = seq.clone()
    time_idx = torch.arange(T, device=seq.device, dtype=seq.dtype)
    for b in range(B):
        for c in range(C):
            for a in range(A):
                for d in range(D):
                    y = seq[b, c, :, a, d]
                    mask = ~torch.isnan(y)
                    if mask.sum() == 0:
                        seq_out[b, c, :, a, d] = 0.0
                    elif mask.sum() == 1:
                        seq_out[b, c, :, a, d] = y[mask].item()
                    else:
                        seq_out[b, c, :, a, d] = torch_interp_1d(time_idx, time_idx[mask], y[mask])
    return seq_out

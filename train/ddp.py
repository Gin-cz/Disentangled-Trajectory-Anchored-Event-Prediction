"""Provide distributed-training helper functions."""

import torch.distributed as dist
import torch
from typing import Dict


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def reduce_avg_dict(d: Dict[str, float], device: torch.device = None) -> Dict[str, float]:
    if not is_dist_avail_and_initialized():
        return d
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device()) \
                 if torch.cuda.is_available() else torch.device("cpu")

    keys = sorted(d.keys())
    d_tensor = torch.tensor([d[k] for k in keys], device=device, dtype=torch.float32)
    dist.all_reduce(d_tensor, op=dist.ReduceOp.SUM)

    out = {}
    world_size = get_world_size()
    for i, k in enumerate(keys):
        if k == "_batches":
            out[k] = float(d_tensor[i].item())
        else:
            out[k] = float(d_tensor[i].item() / world_size)
    return out

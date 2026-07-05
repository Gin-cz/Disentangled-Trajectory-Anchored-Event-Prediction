import torch
from typing import List, Tuple, Dict


@torch.no_grad()
def reindex_by_labels(x_TBAD: torch.Tensor,
                      src_labels_BA: List[List[str]],
                      dst_labels_BA: List[List[str]]) -> torch.Tensor:
    """
    将 x_TBAD（其 A 维顺序对应 src_labels_BA）重排为 A 维顺序对应 dst_labels_BA。
    x_TBAD: [T,B,A,D]
    src_labels_BA: len B，每个是长度 A 的字符串列表（当前 A 维标签顺序）
    dst_labels_BA: len B，每个是长度 A 的字符串列表（目标 A 维标签顺序）
    """
    T, B, A, D = x_TBAD.shape
    out = torch.zeros_like(x_TBAD)
    for b in range(B):
        src_map = {str(lbl): k for k, lbl in enumerate(src_labels_BA[b])}
        for j, lbl in enumerate(dst_labels_BA[b]):
            k = src_map.get(str(lbl), None)
            if k is not None:
                out[:, b, j, :] = x_TBAD[:, b, k, :]
            else:
                out[:, b, j, :].zero_()  # 目标里有但源里没有时，置零
    return out

@torch.no_grad()
def _compute_ADE_FDE(fut_pred_TBAD: torch.Tensor, fut_gt_TBAD: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    diff = fut_pred_TBAD - fut_gt_TBAD
    l2 = torch.sqrt((diff * diff).sum(dim=3) + 1e-12)  # [T,B,A]
    ADE = l2.mean(dim=(0,2)).mean()
    FDE = l2[-1].mean(dim=1).mean()
    return ADE, FDE

@torch.no_grad()
def align_fut_by_tid(fut_pred: torch.Tensor, ctx_tids: torch.Tensor, fut_tids: torch.Tensor) -> torch.Tensor:
    T,B,A,_ = fut_pred.shape
    aligned = torch.zeros_like(fut_pred)
    for b in range(B):
        mapping = {int(ctx_tids[b,i].item()): i for i in range(A)}
        for k in range(A):
            tid = int(fut_tids[b,k].item())
            j = mapping.get(tid, None)
            if j is not None:
                aligned[:, b, k, :] = fut_pred[:, b, j, :]
            else:
                aligned[:, b, k, :].zero_()
    return aligned


@torch.no_grad()
def eval_best_of_n(fut_samples_TBASD: torch.Tensor, fut_gt_BTAD: torch.Tensor) -> Dict[str, float]:
    device = fut_samples_TBASD.device
    T,B,A,S,D = fut_samples_TBASD.shape
    fut_gt_TBAD = fut_gt_BTAD.permute(1,0,2,3).contiguous()  # [T,B,A,D]
    diff = fut_samples_TBASD - fut_gt_TBAD.unsqueeze(3)      # [T,B,A,S,D]
    l2 = torch.sqrt((diff * diff).sum(dim=4) + 1e-12)        # [T,B,A,S]
    ade_per_sample = l2.mean(dim=(0,2))                      # [B,S]
    best_idx = ade_per_sample.argmin(dim=1)                  # [B]
    idx_expand_A = best_idx.view(1,B,1,1).expand(T,B,A,1)
    l2_best = torch.gather(l2, dim=3, index=idx_expand_A).squeeze(3)
    ADE_top   = l2_best.mean(dim=(0,2)).mean().item()
    FDE_top   = l2_best[-1].mean(dim=1).mean().item()
    l2_sq     = (diff * diff).sum(dim=4)
    variety   = l2_sq.min(dim=3).values.mean().item()
    return {"ADE_top": ADE_top, "FDE_top": FDE_top, "variety": variety}
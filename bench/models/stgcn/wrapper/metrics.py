from typing import Dict, Iterable

import math
import torch


def init_metric_acc(horizon_indices: Iterable[int]) -> Dict[int, Dict[str, float]]:
    return {
        int(idx): {"sum_abs": 0.0, "sum_sq": 0.0, "sum_mape": 0.0, "count": 0.0}
        for idx in horizon_indices
    }


def update_metric_acc(
    acc: Dict[int, Dict[str, float]],
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    horizon_indices: Iterable[int],
    eps: float = 1e-5,
) -> None:
    for idx in horizon_indices:
        step = int(idx)
        err = y_pred[:, step, :] - y_true[:, step, :]
        abs_err = err.abs()
        acc[step]["sum_abs"] += float(abs_err.sum().item())
        acc[step]["sum_sq"] += float((err ** 2).sum().item())
        # Using the true value as denominator (standard MAPE)
        denom = torch.clamp(y_true[:, step, :].abs(), min=eps)
        acc[step]["sum_mape"] += float((abs_err / denom).sum().item())
        acc[step]["count"] += float(err.numel())


def finalize_metric_acc(acc: Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    results: Dict[int, Dict[str, float]] = {}
    for idx, stats in acc.items():
        count = stats["count"]
        if count == 0:
            results[idx] = {"mae": 0.0, "rmse": 0.0, "mape": 0.0}
            continue
        mae = stats["sum_abs"] / count
        rmse = math.sqrt(stats["sum_sq"] / count)
        mape = (stats["sum_mape"] / count) * 100.0
        results[idx] = {"mae": mae, "rmse": rmse, "mape": mape}
    return results

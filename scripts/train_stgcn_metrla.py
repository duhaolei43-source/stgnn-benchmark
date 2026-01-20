import argparse
import csv
import datetime as dt
import os
import random
import subprocess
import time
from typing import Dict, Optional, Tuple

import math
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from bench.data.splits import load_split_json
from bench.models.stgcn.vendor.stgcn_pytorch.stgcn import STGCN
from bench.models.stgcn.wrapper.adj import load_adjacency
from bench.models.stgcn.wrapper.data import (
    build_sequences,
    load_metrla_h5,
    normalize_series,
)
from bench.models.stgcn.wrapper.metrics import (
    finalize_metric_acc,
    init_metric_acc,
    update_metric_acc,
)

try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    class _DummyTqdm:
        def __init__(self, iterable, **kwargs):  # type: ignore
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            return None

    def tqdm(iterable, **kwargs):  # type: ignore
        return _DummyTqdm(iterable)


HORIZON_INDICES = {2: "15", 5: "30", 11: "60"}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _prepare_datasets(
    data_path: str,
    split_path: str,
    t_in: int,
    horizon: int,
) -> Tuple[Dict[str, TensorDataset], Dict[str, object]]:
    data = load_metrla_h5(data_path)
    num_steps, num_nodes, _ = data.shape

    split = load_split_json(split_path, num_steps=num_steps)
    train_range = split["train_range"]
    val_range = split["val_range"]
    test_range = split["test_range"]

    normalized, mean, std = normalize_series(data, train_range)

    train_x, train_y = build_sequences(
        normalized, train_range[0], train_range[1], t_in, horizon
    )
    datasets: Dict[str, TensorDataset] = {
        "train": TensorDataset(
            torch.from_numpy(train_x), torch.from_numpy(train_y)
        )
    }

    if val_range is not None:
        val_x, val_y = build_sequences(
            normalized, val_range[0], val_range[1], t_in, horizon
        )
        datasets["val"] = TensorDataset(
            torch.from_numpy(val_x), torch.from_numpy(val_y)
        )
    if test_range is not None:
        test_x, test_y = build_sequences(
            normalized, test_range[0], test_range[1], t_in, horizon
        )
        datasets["test"] = TensorDataset(
            torch.from_numpy(test_x), torch.from_numpy(test_y)
        )

    meta = {
        "num_steps": num_steps,
        "num_nodes": num_nodes,
        "train_range": train_range,
        "val_range": val_range,
        "test_range": test_range,
        "dataset": split.get("dataset", ""),
        "mean": mean,
        "std": std,
    }
    return datasets, meta


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    adj: torch.Tensor,
    mean: float,
    std: float,
    mape_mode: str,
    mape_min: float,
    debug_metrics: bool,
    epoch: int,
    split_name: str,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    horizon_indices = list(HORIZON_INDICES.keys())
    acc = init_metric_acc(horizon_indices)
    steps = None
    y_true_stats = None
    y_pred_stats = None
    err_stats = None
    if debug_metrics and epoch == 0:
        steps = torch.tensor(horizon_indices, device=device, dtype=torch.long)
        y_true_stats = _init_diag_stats(include_thresholds=True)
        y_pred_stats = _init_diag_stats(include_thresholds=False)
        err_stats = _init_diag_stats(include_thresholds=False)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            pred = model(adj, x)
            pred = pred.permute(0, 2, 1)
            pred = pred * std + mean
            y = y * std + mean
            update_metric_acc(
                acc,
                pred,
                y,
                horizon_indices,
                mape_mode=mape_mode,
                mape_min=mape_min,
            )
            if (
                steps is not None
                and y_true_stats is not None
                and y_pred_stats is not None
                and err_stats is not None
            ):
                y_true_sel = y.index_select(1, steps)
                y_pred_sel = pred.index_select(1, steps)
                _update_diag_stats(y_true_stats, y_true_sel, mape_min=mape_min)
                _update_diag_stats(y_pred_stats, y_pred_sel, mape_min=None)
                _update_diag_stats(err_stats, (y_pred_sel - y_true_sel).abs(), mape_min=None)
    finalized = finalize_metric_acc(acc)
    if mape_mode == "masked":
        total_count = sum(stats["count"] for stats in acc.values())
        mape_count = sum(stats["mape_count"] for stats in acc.values())
        if total_count > 0:
            masked_ratio = 1.0 - (mape_count / total_count)
            if masked_ratio > 0.5:
                print(
                    f"Warning: Epoch {epoch + 1} {split_name} MAPE masked ratio "
                    f"{masked_ratio * 100:.2f}% (>50%), MAPE may be unreliable."
                )
    if y_true_stats is not None and y_pred_stats is not None and err_stats is not None:
        _print_diag_stats(
            epoch=epoch,
            split_name=split_name,
            y_true_stats=y_true_stats,
            y_pred_stats=y_pred_stats,
            err_stats=err_stats,
            mape_min=mape_min,
        )
    results: Dict[str, Dict[str, float]] = {}
    for idx, metrics in finalized.items():
        label = HORIZON_INDICES.get(idx, str(idx))
        results[label] = metrics
    return results


def _format_metrics(prefix: str, metrics: Dict[str, Dict[str, float]]) -> str:
    lines = [f"{prefix}:"]
    for label in ["15", "30", "60"]:
        m = metrics.get(label, {"mae": 0.0, "mape": 0.0, "rmse": 0.0})
        lines.append(
            f"  {label}m MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} MAPE={m['mape']:.4f}"
        )
    return "\n".join(lines)


def _idx_to_hhmm(idx: int, minutes_per_step: int = 5, steps_per_day: int = 288) -> str:
    tod_step = idx % steps_per_day
    minutes = tod_step * minutes_per_step
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"


def _range_to_str(range_value: Optional[list]) -> str:
    if range_value is None:
        return ""
    return f"[{range_value[0]},{range_value[1]}]"


def _metric_value(metrics: Optional[Dict[str, Dict[str, float]]], label: str, key: str) -> str:
    if not metrics:
        return ""
    return f"{metrics.get(label, {}).get(key, 0.0):.4f}"


def _infer_graph_info(graph_path: str) -> Dict[str, str]:
    basename = os.path.basename(graph_path)
    parent = os.path.basename(os.path.dirname(graph_path))
    slot_map = {
        "A_0.npz": ("slot0", "night", "00:00-05:55"),
        "A_1.npz": ("slot1", "AM", "06:00-09:55"),
        "A_2.npz": ("slot2", "midday", "10:00-15:55"),
        "A_3.npz": ("slot3", "PM", "16:00-23:55"),
    }
    if parent == "dds_k4" and basename in slot_map:
        slot, name, span = slot_map[basename]
        return {
            "graph_type": "dds_k4_slot",
            "graph_slot": slot,
            "graph_slot_name": name,
            "graph_slot_span_hhmm": span,
        }
    return {
        "graph_type": "static",
        "graph_slot": "",
        "graph_slot_name": "",
        "graph_slot_span_hhmm": "",
    }


def _get_repo_commit(repo_root: str) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return ""


def _init_diag_stats(include_thresholds: bool) -> Dict[str, float]:
    stats = {
        "count": 0.0,
        "finite_count": 0.0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "nan": 0.0,
        "inf": 0.0,
    }
    if include_thresholds:
        stats["lt_eps"] = 0.0
        stats["lt_min"] = 0.0
    return stats


def _update_diag_stats(
    stats: Dict[str, float], values: torch.Tensor, mape_min: Optional[float]
) -> None:
    stats["count"] += float(values.numel())
    stats["nan"] += float(torch.isnan(values).sum().item())
    stats["inf"] += float(torch.isinf(values).sum().item())
    if mape_min is not None:
        stats["lt_eps"] += float((values.abs() < 1e-6).sum().item())
        stats["lt_min"] += float((values.abs() < mape_min).sum().item())
    finite = torch.isfinite(values)
    if finite.any():
        finite_vals = values[finite]
        stats["finite_count"] += float(finite_vals.numel())
        stats["sum"] += float(finite_vals.sum().item())
        stats["sum_sq"] += float((finite_vals ** 2).sum().item())
        stats["min"] = float(min(stats["min"], float(finite_vals.min().item())))
        stats["max"] = float(max(stats["max"], float(finite_vals.max().item())))


def _finalize_diag_stats(stats: Dict[str, float]) -> Dict[str, float]:
    finite_count = stats["finite_count"]
    if finite_count > 0:
        mean = stats["sum"] / finite_count
        var = stats["sum_sq"] / finite_count - mean ** 2
        if var < 0.0:
            var = 0.0
        std = math.sqrt(var)
        min_val = stats["min"]
        max_val = stats["max"]
    else:
        mean = 0.0
        std = 0.0
        min_val = 0.0
        max_val = 0.0
    return {
        "min": min_val,
        "max": max_val,
        "mean": mean,
        "std": std,
        "count": stats["count"],
        "finite_count": finite_count,
        "nan": stats["nan"],
        "inf": stats["inf"],
        "lt_eps": stats.get("lt_eps", 0.0),
        "lt_min": stats.get("lt_min", 0.0),
    }


def _print_diag_stats(
    epoch: int,
    split_name: str,
    y_true_stats: Dict[str, float],
    y_pred_stats: Dict[str, float],
    err_stats: Dict[str, float],
    mape_min: float,
) -> None:
    y_true = _finalize_diag_stats(y_true_stats)
    y_pred = _finalize_diag_stats(y_pred_stats)
    err = _finalize_diag_stats(err_stats)
    total = y_true["count"] if y_true["count"] > 0 else 1.0
    pct_eps = (y_true["lt_eps"] / total) * 100.0
    pct_min = (y_true["lt_min"] / total) * 100.0
    print(
        f"Diag Epoch {epoch + 1} {split_name} y_true: "
        f"min={y_true['min']:.4f} max={y_true['max']:.4f} "
        f"mean={y_true['mean']:.4f} std={y_true['std']:.4f} "
        f"|abs|<1e-6={pct_eps:.2f}% |abs|<{mape_min}={pct_min:.2f}% "
        f"nan={int(y_true['nan'])} inf={int(y_true['inf'])}"
    )
    print(
        f"Diag Epoch {epoch + 1} {split_name} y_pred: "
        f"min={y_pred['min']:.4f} max={y_pred['max']:.4f} "
        f"mean={y_pred['mean']:.4f} std={y_pred['std']:.4f} "
        f"nan={int(y_pred['nan'])} inf={int(y_pred['inf'])}"
    )
    print(
        f"Diag Epoch {epoch + 1} {split_name} abs_err: "
        f"min={err['min']:.4f} max={err['max']:.4f} mean={err['mean']:.4f}"
    )


def _aggregate_mae(metrics: Dict[str, Dict[str, float]]) -> float:
    values = [metrics[label]["mae"] for label in ["15", "30", "60"] if label in metrics]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _checkpoint_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_mae: float,
    config: Dict[str, object],
) -> Dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_mae": best_val_mae,
        "config": config,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="One-touch STGCN training on METR-LA.")
    parser.add_argument("--data", required=True, help="Path to metr-la.h5")
    parser.add_argument("--split", required=True, help="Path to split_v1.json")
    parser.add_argument("--adj", required=True, help="Path to adjacency (npz or pkl)")
    parser.add_argument("--t_in", type=int, default=12, help="Input timesteps.")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--mape_mode",
        choices=["masked", "clamp"],
        default="masked",
        help="MAPE mode (masked or clamp).",
    )
    parser.add_argument(
        "--mape_min",
        type=float,
        default=1.0,
        help="MAPE minimum denominator or mask threshold.",
    )
    parser.add_argument("--debug_metrics", type=int, default=1, help="Enable metric diagnostics (0/1).")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--min_delta", type=float, default=0.0, help="Early stopping min delta.")
    parser.add_argument("--run_id", default=None, help="Checkpoint run id (defaults to timestamp).")
    parser.add_argument("--wandb", type=int, default=0, help="Enable wandb (0/1).")
    parser.add_argument("--wandb_project", default="stgnn-benchmark", help="Wandb project name.")
    parser.add_argument("--wandb_run_name", default=None, help="Wandb run name.")
    parser.add_argument("--report", type=int, default=0, help="Append CSV report row (0/1).")
    parser.add_argument(
        "--report_path",
        default=os.path.join("results", "reports.csv"),
        help="CSV report path.",
    )
    parser.add_argument("--report_tags", default="", help="Optional report notes/tags.")
    args = parser.parse_args()

    _set_seed(args.seed)
    device = torch.device(args.device)

    datasets, meta = _prepare_datasets(args.data, args.split, args.t_in, args.horizon)
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = None
    if "val" in datasets:
        val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)
    test_loader = None
    if "test" in datasets:
        test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False)

    adj = load_adjacency(args.adj)
    if adj.shape[0] != meta["num_nodes"]:
        raise ValueError(
            f"Adjacency size mismatch: {adj.shape[0]} != num_nodes {meta['num_nodes']}"
        )
    adj_tensor = torch.from_numpy(adj).to(device, dtype=torch.float32)

    model = STGCN(
        num_nodes=meta["num_nodes"],
        num_features=1,
        num_timesteps_input=args.t_in,
        num_timesteps_output=args.horizon,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.L1Loss()

    print(
        "Banner: "
        f"num_steps={meta['num_steps']} num_nodes={meta['num_nodes']} "
        f"train_range={meta['train_range']} val_range={meta['val_range']} "
        f"adj={args.adj} device={device}"
    )

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join("artifacts", "checkpoints", "stgcn", run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    config_snapshot = dict(vars(args))

    use_wandb = args.wandb == 1
    wandb = None
    if use_wandb:
        import wandb as _wandb

        wandb = _wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    best_val_mae = float("inf")
    best_epoch = -1
    patience_counter = 0
    last_epoch = -1
    best_val_metrics: Optional[Dict[str, Dict[str, float]]] = None
    early_stopped = 0
    for epoch in range(args.epochs):
        last_epoch = epoch
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            leave=False,
        )
        for x, y in pbar:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred = model(adj_tensor, x)
            y_target = y.permute(0, 2, 1)
            loss = loss_fn(pred, y_target)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

        epoch_loss /= max(len(train_loader), 1)
        if wandb is not None:
            wandb.log({"train_loss": epoch_loss, "epoch": epoch + 1})

        train_metrics = _evaluate(
            model,
            train_loader,
            device,
            adj_tensor,
            meta["mean"],
            meta["std"],
            args.mape_mode,
            args.mape_min,
            args.debug_metrics == 1,
            epoch,
            "TRAIN",
        )
        print(_format_metrics(f"Epoch {epoch + 1} TRAIN", train_metrics))

        if val_loader is not None:
            val_metrics = _evaluate(
                model,
                val_loader,
                device,
                adj_tensor,
                meta["mean"],
                meta["std"],
                args.mape_mode,
                args.mape_min,
                args.debug_metrics == 1,
                epoch,
                "VAL",
            )
            print(_format_metrics(f"Epoch {epoch + 1} VAL", val_metrics))
        else:
            val_metrics = {}
            print(f"Epoch {epoch + 1} VAL: skipped (no val_range)")

        if wandb is not None:
            for label, metrics in train_metrics.items():
                wandb.log(
                    {
                        f"train/mae_{label}": metrics["mae"],
                        f"train/mape_{label}": metrics["mape"],
                        f"train/rmse_{label}": metrics["rmse"],
                        "epoch": epoch + 1,
                    }
                )
            for label, metrics in val_metrics.items():
                wandb.log(
                    {
                        f"val/mae_{label}": metrics["mae"],
                        f"val/mape_{label}": metrics["mape"],
                        f"val/rmse_{label}": metrics["rmse"],
                        "epoch": epoch + 1,
                    }
                )

        current_val_mae = _aggregate_mae(val_metrics) if val_loader is not None else None
        if current_val_mae is not None:
            if current_val_mae < best_val_mae - args.min_delta:
                best_val_mae = current_val_mae
                best_epoch = epoch + 1
                best_val_metrics = dict(val_metrics)
                patience_counter = 0
                best_state = _checkpoint_state(
                    model, optimizer, epoch + 1, best_val_mae, config_snapshot
                )
                torch.save(best_state, os.path.join(ckpt_dir, "best.pt"))
            else:
                patience_counter += 1

        last_state = _checkpoint_state(
            model, optimizer, epoch + 1, best_val_mae, config_snapshot
        )
        torch.save(last_state, os.path.join(ckpt_dir, "last.pt"))

        if current_val_mae is not None and patience_counter >= args.patience:
            print(
                "Early stopping: "
                f"epoch={epoch + 1} best_val_mae={best_val_mae:.4f} "
                f"best_epoch={best_epoch} patience={args.patience}"
            )
            early_stopped = 1
            break

    test_metrics: Optional[Dict[str, Dict[str, float]]] = None
    if test_loader is not None:
        test_metrics = _evaluate(
            model,
            test_loader,
            device,
            adj_tensor,
            meta["mean"],
            meta["std"],
            args.mape_mode,
            args.mape_min,
            False,
            max(last_epoch, 0),
            "TEST",
        )
        print(_format_metrics("TEST metrics", test_metrics))
        if wandb is not None:
            for label, metrics in test_metrics.items():
                wandb.log(
                    {
                        f"test/mae_{label}": metrics["mae"],
                        f"test/mape_{label}": metrics["mape"],
                        f"test/rmse_{label}": metrics["rmse"],
                        "epoch": max(last_epoch, 0) + 1,
                    }
                )

    if args.report == 1:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        commit = _get_repo_commit(repo_root)
        timestamp = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()
        graph_info = _infer_graph_info(args.adj)
        train_range = meta.get("train_range")
        val_range = meta.get("val_range")
        test_range = meta.get("test_range")
        report_dir = os.path.dirname(args.report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)

        row = {
            "timestamp_utc": timestamp,
            "repo_commit": commit,
            "model": "STGCN",
            "dataset": meta.get("dataset", ""),
            "run_id": run_id,
            "seed": args.seed,
            "device": str(device),
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs_planned": args.epochs,
            "epochs_ran": max(last_epoch + 1, 0),
            "early_stopped": early_stopped,
            "best_epoch": best_epoch,
            "split_json": args.split,
            "num_steps": meta.get("num_steps", ""),
            "train_range": _range_to_str(train_range),
            "val_range": _range_to_str(val_range),
            "test_range": _range_to_str(test_range),
            "train_start_hhmm": _idx_to_hhmm(train_range[0]) if train_range else "",
            "train_end_hhmm": _idx_to_hhmm(train_range[1]) if train_range else "",
            "val_start_hhmm": _idx_to_hhmm(val_range[0]) if val_range else "",
            "val_end_hhmm": _idx_to_hhmm(val_range[1]) if val_range else "",
            "test_start_hhmm": _idx_to_hhmm(test_range[0]) if test_range else "",
            "test_end_hhmm": _idx_to_hhmm(test_range[1]) if test_range else "",
            "graph_path": args.adj,
            "graph_type": graph_info["graph_type"],
            "graph_slot": graph_info["graph_slot"],
            "graph_slot_name": graph_info["graph_slot_name"],
            "graph_slot_span_hhmm": graph_info["graph_slot_span_hhmm"],
            "best_val_mae_15": _metric_value(best_val_metrics, "15", "mae"),
            "best_val_mape_15": _metric_value(best_val_metrics, "15", "mape"),
            "best_val_rmse_15": _metric_value(best_val_metrics, "15", "rmse"),
            "best_val_mae_30": _metric_value(best_val_metrics, "30", "mae"),
            "best_val_mape_30": _metric_value(best_val_metrics, "30", "mape"),
            "best_val_rmse_30": _metric_value(best_val_metrics, "30", "rmse"),
            "best_val_mae_60": _metric_value(best_val_metrics, "60", "mae"),
            "best_val_mape_60": _metric_value(best_val_metrics, "60", "mape"),
            "best_val_rmse_60": _metric_value(best_val_metrics, "60", "rmse"),
            "test_mae_15": _metric_value(test_metrics, "15", "mae"),
            "test_mape_15": _metric_value(test_metrics, "15", "mape"),
            "test_rmse_15": _metric_value(test_metrics, "15", "rmse"),
            "test_mae_30": _metric_value(test_metrics, "30", "mae"),
            "test_mape_30": _metric_value(test_metrics, "30", "mape"),
            "test_rmse_30": _metric_value(test_metrics, "30", "rmse"),
            "test_mae_60": _metric_value(test_metrics, "60", "mae"),
            "test_mape_60": _metric_value(test_metrics, "60", "mape"),
            "test_rmse_60": _metric_value(test_metrics, "60", "rmse"),
            "notes": args.report_tags,
        }

        fieldnames = [
            "timestamp_utc",
            "repo_commit",
            "model",
            "dataset",
            "run_id",
            "seed",
            "device",
            "lr",
            "batch_size",
            "epochs_planned",
            "epochs_ran",
            "early_stopped",
            "best_epoch",
            "split_json",
            "num_steps",
            "train_range",
            "val_range",
            "test_range",
            "train_start_hhmm",
            "train_end_hhmm",
            "val_start_hhmm",
            "val_end_hhmm",
            "test_start_hhmm",
            "test_end_hhmm",
            "graph_path",
            "graph_type",
            "graph_slot",
            "graph_slot_name",
            "graph_slot_span_hhmm",
            "best_val_mae_15",
            "best_val_mape_15",
            "best_val_rmse_15",
            "best_val_mae_30",
            "best_val_mape_30",
            "best_val_rmse_30",
            "best_val_mae_60",
            "best_val_mape_60",
            "best_val_rmse_60",
            "test_mae_15",
            "test_mape_15",
            "test_rmse_15",
            "test_mae_30",
            "test_mape_30",
            "test_rmse_30",
            "test_mae_60",
            "test_mape_60",
            "test_rmse_60",
            "notes",
        ]

        file_exists = os.path.exists(args.report_path)
        with open(args.report_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"Report: appended row to {args.report_path}")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from bench.models.stgcn.vendor.stgcn_pytorch.stgcn import STGCN
from bench.models.stgcn.wrapper.adj import load_adjacency
from bench.models.stgcn.wrapper.data import (
    build_sequences,
    load_metrla_h5,
    load_split_json,
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

    split = load_split_json(split_path, num_steps)
    train_range = split["train_range"]
    val_range = split["val_range"]

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

    meta = {
        "num_steps": num_steps,
        "num_nodes": num_nodes,
        "train_range": train_range,
        "val_range": val_range,
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
) -> Dict[str, Dict[str, float]]:
    model.eval()
    horizon_indices = list(HORIZON_INDICES.keys())
    acc = init_metric_acc(horizon_indices)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            pred = model(adj, x)
            pred = pred.permute(0, 2, 1)
            pred = pred * std + mean
            y = y * std + mean
            update_metric_acc(acc, pred, y, horizon_indices)
    finalized = finalize_metric_acc(acc)
    results: Dict[str, Dict[str, float]] = {}
    for idx, metrics in finalized.items():
        label = HORIZON_INDICES.get(idx, str(idx))
        results[label] = metrics
    return results


def _format_metrics(prefix: str, metrics: Dict[str, Dict[str, float]]) -> str:
    parts = []
    for label in ["15", "30", "60"]:
        m = metrics.get(label, {"mae": 0.0, "mape": 0.0, "rmse": 0.0})
        parts.append(
            f"{label}m MAE={m['mae']:.4f} MAPE={m['mape']:.4f} RMSE={m['rmse']:.4f}"
        )
    return f"{prefix}: " + " | ".join(parts)


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
    parser.add_argument("--wandb", type=int, default=0, help="Enable wandb (0/1).")
    parser.add_argument("--wandb_project", default="stgnn-benchmark", help="Wandb project name.")
    parser.add_argument("--wandb_run_name", default=None, help="Wandb run name.")
    args = parser.parse_args()

    _set_seed(args.seed)
    device = torch.device(args.device)

    datasets, meta = _prepare_datasets(args.data, args.split, args.t_in, args.horizon)
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = None
    if "val" in datasets:
        val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)

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

    use_wandb = args.wandb == 1
    wandb = None
    if use_wandb:
        import wandb as _wandb

        wandb = _wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    for epoch in range(args.epochs):
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
        model, train_loader, device, adj_tensor, meta["mean"], meta["std"]
    )
    print(_format_metrics("TRAIN metrics", train_metrics))

    if val_loader is not None:
        val_metrics = _evaluate(
            model, val_loader, device, adj_tensor, meta["mean"], meta["std"]
        )
        print(_format_metrics("VAL metrics", val_metrics))
    else:
        val_metrics = {}
        print("VAL metrics: skipped (no val_range)")

    if wandb is not None:
        for label, metrics in train_metrics.items():
            wandb.log(
                {
                    f"train/mae_{label}": metrics["mae"],
                    f"train/mape_{label}": metrics["mape"],
                    f"train/rmse_{label}": metrics["rmse"],
                }
            )
        for label, metrics in val_metrics.items():
            wandb.log(
                {
                    f"val/mae_{label}": metrics["mae"],
                    f"val/mape_{label}": metrics["mape"],
                    f"val/rmse_{label}": metrics["rmse"],
                }
            )

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# Usage:
#   python scripts/viz_dds_metrla_map.py --loc data/raw/metr-la/sensor_locations.csv \
#       --graphs_dir artifacts/graphs/metr-la/dds_k4 --k 4 \
#       --split artifacts/splits/metr-la/split_v1.json --h5 data/raw/metr-la/metr-la.h5 \
#       --color_by_speed 1 --edges_per_node 3 --basemap plain \
#       --out_dir artifacts/graphs/metr-la/dds_k4/viz_map

import argparse
import os
import pickle
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from bench.models.stgcn.wrapper.data import load_metrla_h5, load_split_json


SLOT_RANGES = [(0, 71), (72, 119), (120, 191), (192, 287)]
SLOT_NAMES = ["night", "AM", "midday", "PM"]
STEPS_PER_DAY = 288


def _load_locations(path: str, id_col: str, lat_col: str, lon_col: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Missing sensor locations CSV. Provide --loc with columns sensor_id,lat,lon."
        )
    df = pd.read_csv(path)
    for col in [id_col, lat_col, lon_col]:
        if col not in df.columns:
            raise ValueError(
                f"Missing column '{col}' in {path}. "
                f"Available columns: {list(df.columns)}"
            )
    df = df[[id_col, lat_col, lon_col]].copy()
    df[id_col] = df[id_col].astype(str)
    df[lat_col] = pd.to_numeric(df[lat_col], errors="raise")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="raise")
    return df


def _load_pickle(path: str) -> object:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")


def _load_adj(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        with np.load(path) as data:
            if {"data", "indices", "indptr", "shape"}.issubset(set(data.files)):
                try:
                    import scipy.sparse as sp  # type: ignore
                except ImportError as exc:
                    raise ImportError(
                        "Sparse npz adjacency requires scipy; install scipy or provide dense npz."
                    ) from exc
                shape = tuple(data["shape"])
                adj = sp.csr_matrix(
                    (data["data"], data["indices"], data["indptr"]), shape=shape
                )
            elif "A" in data:
                adj = data["A"]
            elif "adj" in data:
                adj = data["adj"]
            elif data.files:
                adj = data[data.files[0]]
            else:
                raise ValueError(f"No arrays found in npz file: {path}")
    elif path.endswith(".pkl"):
        obj = _load_pickle(path)
        if isinstance(obj, dict):
            if "adj_mx" in obj:
                adj = obj["adj_mx"]
            elif "adj" in obj:
                adj = obj["adj"]
            else:
                raise ValueError(f"Unsupported adjacency dict keys: {list(obj.keys())}")
        elif isinstance(obj, (list, tuple)) and len(obj) == 3:
            adj = obj[2]
        else:
            adj = obj
    else:
        raise ValueError(f"Unsupported adjacency format: {path}")

    if hasattr(adj, "toarray"):
        adj = adj.toarray()
    adj = np.asarray(adj)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got shape {adj.shape}")
    return adj.astype(np.float32)


def _load_sensor_mapping(adj_pkl: str) -> Optional[Dict[str, int]]:
    if not adj_pkl:
        return None
    if not os.path.exists(adj_pkl):
        raise FileNotFoundError(f"adj_pkl not found: {adj_pkl}")
    obj = _load_pickle(adj_pkl)
    mapping = None
    if isinstance(obj, dict):
        if "sensor_id_to_ind" in obj:
            mapping = obj["sensor_id_to_ind"]
        elif "id_to_ind" in obj:
            mapping = obj["id_to_ind"]
    elif isinstance(obj, (list, tuple)) and len(obj) == 3:
        mapping = obj[1]
    if mapping is None:
        return None
    return {str(k): int(v) for k, v in mapping.items()}


def _align_locations(
    loc_df: pd.DataFrame,
    mapping: Optional[Dict[str, int]],
    adj_size: int,
    id_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if mapping is None:
        print("Warning: --adj_pkl not provided; assuming CSV order matches adjacency order.")
        print("First 5 location rows:")
        print(loc_df.head().to_string(index=False))
        if len(loc_df) != adj_size:
            raise ValueError(
                f"Location row count {len(loc_df)} does not match adjacency size {adj_size}."
            )
        return loc_df["lon"].to_numpy(), loc_df["lat"].to_numpy()

    ordered = [None] * (max(mapping.values()) + 1)
    missing: List[str] = []
    for _, row in loc_df.iterrows():
        sensor_id = str(row[id_col])
        if sensor_id not in mapping:
            missing.append(sensor_id)
            continue
        ordered[mapping[sensor_id]] = row
    if any(entry is None for entry in ordered):
        raise ValueError(
            "Sensor location alignment failed. "
            f"Missing {sum(entry is None for entry in ordered)} entries from mapping."
        )
    if len(ordered) != adj_size:
        raise ValueError(
            f"Location count {len(ordered)} does not match adjacency size {adj_size}."
        )
    lon = np.array([row["lon"] for row in ordered], dtype=float)
    lat = np.array([row["lat"] for row in ordered], dtype=float)
    if missing:
        print(f"Warning: {len(missing)} sensors in CSV not found in adj mapping.")
    return lon, lat


def _mercator(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    origin_shift = 20037508.34
    x = lon * origin_shift / 180.0
    y = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) / (np.pi / 180.0)
    y = y * origin_shift / 180.0
    return x, y


def _compute_slot_means(
    data: np.ndarray,
    train_range: List[int],
    slot_ranges: List[Tuple[int, int]],
) -> List[np.ndarray]:
    start, end = train_range
    train_idx = np.arange(start, end + 1)
    values = []
    for slot_start, slot_end in slot_ranges:
        slot_mask = (train_idx % STEPS_PER_DAY >= slot_start) & (
            train_idx % STEPS_PER_DAY <= slot_end
        )
        if not np.any(slot_mask):
            values.append(np.full((data.shape[1],), np.nan, dtype=np.float32))
            continue
        slot_data = data[train_idx[slot_mask], :, 0]
        values.append(np.nanmean(slot_data, axis=0))
    return values


def _build_edge_segments(
    adj: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    edges_per_node: int,
    min_weight: float,
) -> List[np.ndarray]:
    segments: List[np.ndarray] = []
    for i in range(adj.shape[0]):
        row = adj[i]
        if edges_per_node > 0:
            order = np.argsort(row)[::-1]
        else:
            order = range(len(row))
        count = 0
        for j in order:
            if i == j:
                continue
            weight = row[j]
            if weight <= min_weight:
                continue
            segments.append(np.array([[xs[i], ys[i]], [xs[j], ys[j]]], dtype=float))
            count += 1
            if edges_per_node > 0 and count >= edges_per_node:
                break
    return segments


def _plot_graph(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    adj: np.ndarray,
    edges_per_node: int,
    min_weight: float,
    node_values: Optional[np.ndarray],
    title: str,
    basemap: str,
    ctx: Optional[object],
    vmin: Optional[float],
    vmax: Optional[float],
) -> None:
    segments = _build_edge_segments(adj, xs, ys, edges_per_node, min_weight)
    if segments:
        lc = LineCollection(segments, colors="black", linewidths=0.5, alpha=0.25, zorder=1)
        ax.add_collection(lc)

    if node_values is None:
        ax.scatter(xs, ys, s=12, c="#1f77b4", edgecolors="white", linewidths=0.2, zorder=2)
    else:
        sc = ax.scatter(
            xs,
            ys,
            s=18,
            c=node_values,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            edgecolors="white",
            linewidths=0.2,
            zorder=2,
        )
        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)

    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    if basemap == "tiles" and ctx is not None:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs="EPSG:3857")
        except Exception as exc:
            print(f"Warning: failed to add basemap tiles ({exc}). Using plain background.")


def _slot_name(k: int, total_k: int) -> str:
    if total_k == 4 and 0 <= k < 4:
        return SLOT_NAMES[k]
    return f"slot-{k}"


def _infer_slot_from_path(path: str) -> Optional[int]:
    match = re.search(r"A_(\d+)\.npz", os.path.basename(path))
    if match:
        return int(match.group(1))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize DDS METR-LA graphs on a map.")
    parser.add_argument("--loc", required=True, help="Sensor locations CSV path.")
    parser.add_argument("--graphs_dir", default="", help="Directory containing A_k.npz files.")
    parser.add_argument("--A", default="", help="Single adjacency file to plot (npz/pkl).")
    parser.add_argument("--k", type=int, default=4, help="Number of slots/graphs in graphs_dir.")
    parser.add_argument("--split", default="", help="Split json path (for speed coloring).")
    parser.add_argument("--h5", default="", help="metr-la.h5 path (for speed coloring).")
    parser.add_argument("--color_by_speed", type=int, default=0, help="Color nodes by speed (0/1).")
    parser.add_argument("--edges_per_node", type=int, default=3, help="Edges per node to draw.")
    parser.add_argument("--min_weight", type=float, default=0.0, help="Minimum edge weight.")
    parser.add_argument("--basemap", choices=["plain", "tiles"], default="plain")
    parser.add_argument("--out_dir", default="", help="Output directory for figures.")
    parser.add_argument("--adj_pkl", default="", help="adj_mx.pkl for sensor alignment.")
    parser.add_argument("--sensor_id_col", default="sensor_id", help="Sensor id column.")
    parser.add_argument("--lat_col", default="lat", help="Latitude column.")
    parser.add_argument("--lon_col", default="lon", help="Longitude column.")
    parser.add_argument("--panel", type=int, default=1, help="Save 2x2 panel when k=4 (0/1).")
    args = parser.parse_args()

    if not args.A and not args.graphs_dir:
        print("Error: provide --A or --graphs_dir.", file=sys.stderr)
        return 1

    loc_df = _load_locations(args.loc, args.sensor_id_col, args.lat_col, args.lon_col)
    loc_df = loc_df.rename(
        columns={args.sensor_id_col: "sensor_id", args.lat_col: "lat", args.lon_col: "lon"}
    )

    adj_paths: List[str] = []
    if args.A:
        adj_paths = [args.A]
    else:
        for idx in range(args.k):
            adj_paths.append(os.path.join(args.graphs_dir, f"A_{idx}.npz"))

    for path in adj_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Adjacency file not found: {path}")

    adj0 = _load_adj(adj_paths[0])
    mapping = _load_sensor_mapping(args.adj_pkl) if args.adj_pkl else None
    lon, lat = _align_locations(loc_df, mapping, adj0.shape[0], "sensor_id")

    if adj0.shape[0] != len(lon):
        raise ValueError(
            f"Adjacency size {adj0.shape[0]} does not match location count {len(lon)}."
        )

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(os.path.dirname(adj_paths[0]), "viz_map")
    os.makedirs(out_dir, exist_ok=True)

    node_values_list: Optional[List[np.ndarray]] = None
    if args.color_by_speed == 1:
        if not args.h5 or not args.split:
            raise ValueError("--h5 and --split are required when --color_by_speed=1.")
        data = load_metrla_h5(args.h5)
        split = load_split_json(args.split, num_steps=data.shape[0])
        node_values_list = _compute_slot_means(data, split["train_range"], SLOT_RANGES)

    ctx = None
    xs = lon
    ys = lat
    basemap = args.basemap
    if basemap == "tiles":
        try:
            import contextily as _ctx  # type: ignore
            ctx = _ctx
            xs, ys = _mercator(lon, lat)
        except Exception as exc:
            print(f"Warning: basemap tiles unavailable ({exc}). Falling back to plain.")
            basemap = "plain"
            xs, ys = lon, lat

    vmin = vmax = None
    if node_values_list is not None:
        all_vals = np.concatenate([vals for vals in node_values_list if vals is not None])
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))

    outputs: List[str] = []
    for path in adj_paths:
        adj = _load_adj(path)
        if adj.shape[0] != adj0.shape[0]:
            raise ValueError(f"Adjacency size mismatch: {path} has {adj.shape}")
        slot_idx = _infer_slot_from_path(path)
        if slot_idx is None:
            slot_idx = 0
        slot_label = _slot_name(slot_idx, args.k)
        node_vals = None
        if node_values_list is not None and 0 <= slot_idx < len(node_values_list):
            node_vals = node_values_list[slot_idx]
        color_label = "speed" if node_vals is not None else "fixed"
        title = (
            f"METR-LA DDS Slot {slot_idx} ({slot_label}) "
            f"| edges_per_node={args.edges_per_node} | color={color_label}"
        )

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        _plot_graph(
            ax,
            xs,
            ys,
            adj,
            args.edges_per_node,
            args.min_weight,
            node_vals,
            title,
            basemap,
            ctx,
            vmin,
            vmax,
        )
        fig_path = os.path.join(out_dir, f"map_A_{slot_idx}.png")
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(fig_path)

    if args.panel == 1 and len(adj_paths) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for idx, path in enumerate(adj_paths):
            adj = _load_adj(path)
            slot_idx = _infer_slot_from_path(path)
            if slot_idx is None:
                slot_idx = idx
            slot_label = _slot_name(slot_idx, args.k)
            node_vals = None
            if node_values_list is not None and 0 <= slot_idx < len(node_values_list):
                node_vals = node_values_list[slot_idx]
            color_label = "speed" if node_vals is not None else "fixed"
            title = (
                f"METR-LA DDS Slot {slot_idx} ({slot_label}) "
                f"| edges_per_node={args.edges_per_node} | color={color_label}"
            )
            ax = axes[idx // 2, idx % 2]
            _plot_graph(
                ax,
                xs,
                ys,
                adj,
                args.edges_per_node,
                args.min_weight,
                node_vals,
                title,
                basemap,
                ctx,
                vmin,
                vmax,
            )
        panel_path = os.path.join(out_dir, "map_A_k4_panel.png")
        fig.savefig(panel_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(panel_path)

    print("Viz output:")
    for path in outputs:
        print(f"- {path}")
    print(f"Saved {len(outputs)} figure(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

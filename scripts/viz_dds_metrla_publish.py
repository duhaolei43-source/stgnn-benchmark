#!/usr/bin/env python3
# Publish-ready DDS visualization + metrics for METR-LA.
# - Sensor CSV must include sensor_id, latitude, longitude (override with CLI flags).
# - Speed coloring is optional: use --speed_mode train_mean and provide --train_split_json + --data_npz/--data_h5.
# - If ordering is unclear, provide --sensor_ids_txt or --adj_pkl to align CSV with adjacency indices.

import argparse
import json
import math
import os
import pickle
import random
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


SLOT_RANGES = [(0, 71), (72, 119), (120, 191), (192, 287)]
STEPS_PER_DAY = 288


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _read_pickle(path: str) -> object:
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
        obj = _read_pickle(path)
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


def _load_sensor_ids_txt(path: str) -> Optional[List[str]]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"sensor_ids_txt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids or None


def _load_adj_sensor_ids(adj_pkl: str) -> Tuple[Optional[List[str]], Optional[Dict[str, int]]]:
    if not adj_pkl:
        return None, None
    if not os.path.exists(adj_pkl):
        raise FileNotFoundError(f"adj_pkl not found: {adj_pkl}")
    obj = _read_pickle(adj_pkl)
    sensor_ids = None
    id_to_ind = None
    if isinstance(obj, dict):
        if "sensor_ids" in obj:
            sensor_ids = [str(s) for s in obj["sensor_ids"]]
        if "sensor_id_to_ind" in obj:
            id_to_ind = {str(k): int(v) for k, v in obj["sensor_id_to_ind"].items()}
    elif isinstance(obj, (list, tuple)) and len(obj) == 3:
        first, second, third = obj
        if isinstance(first, (list, tuple, np.ndarray)):
            sensor_ids = [str(s) for s in first]
        elif isinstance(second, (list, tuple, np.ndarray)):
            sensor_ids = [str(s) for s in second]
        if isinstance(second, dict):
            id_to_ind = {str(k): int(v) for k, v in second.items()}
        elif isinstance(third, dict):
            id_to_ind = {str(k): int(v) for k, v in third.items()}
    return sensor_ids, id_to_ind


def _load_locations(
    path: str, sensor_id_col: str, lat_col: str, lon_col: str
) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Missing sensor locations CSV. Provide --sensor_csv with columns sensor_id,lat,lon."
        )
    df = pd.read_csv(path)
    for col in [sensor_id_col, lat_col, lon_col]:
        if col not in df.columns:
            raise ValueError(
                f"Missing column '{col}' in {path}. "
                f"Available columns: {list(df.columns)}"
            )
    df = df[[sensor_id_col, lat_col, lon_col]].copy()
    df[sensor_id_col] = df[sensor_id_col].astype(str)
    df[lat_col] = pd.to_numeric(df[lat_col], errors="raise")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="raise")
    return df.rename(
        columns={sensor_id_col: "sensor_id", lat_col: "lat", lon_col: "lon"}
    )


def _align_locations(
    loc_df: pd.DataFrame,
    adj_size: int,
    sensor_ids: Optional[List[str]],
    id_to_ind: Optional[Dict[str, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    if sensor_ids:
        if len(sensor_ids) != adj_size:
            raise ValueError(
                f"sensor_ids length {len(sensor_ids)} != adjacency size {adj_size}."
            )
        order = {sid: idx for idx, sid in enumerate(sensor_ids)}
    elif id_to_ind:
        order = id_to_ind
    else:
        print("Warning: no sensor ordering provided; assuming CSV order matches adjacency.")
        print("First 5 rows of locations:")
        print(loc_df.head().to_string(index=False))
        if len(loc_df) != adj_size:
            raise ValueError(
                f"Location row count {len(loc_df)} != adjacency size {adj_size}."
            )
        return loc_df["lon"].to_numpy(), loc_df["lat"].to_numpy()

    ordered = [None] * (max(order.values()) + 1)
    for _, row in loc_df.iterrows():
        sensor_id = str(row["sensor_id"])
        if sensor_id in order:
            ordered[order[sensor_id]] = row
    if any(row is None for row in ordered):
        missing = sum(row is None for row in ordered)
        raise ValueError(f"Location alignment failed; {missing} ids missing from CSV.")
    if len(ordered) != adj_size:
        raise ValueError(
            f"Location count {len(ordered)} != adjacency size {adj_size}."
        )
    lon = np.array([row["lon"] for row in ordered], dtype=float)
    lat = np.array([row["lat"] for row in ordered], dtype=float)
    return lon, lat


def _load_speed_data(data_npz: str, data_h5: str) -> Optional[np.ndarray]:
    if data_npz:
        if not os.path.exists(data_npz):
            print(f"Warning: data_npz not found: {data_npz}")
        else:
            data = np.load(data_npz)
            if "data" in data:
                arr = data["data"]
            elif data.files:
                arr = data[data.files[0]]
            else:
                raise ValueError(f"No arrays found in {data_npz}")
            arr = np.asarray(arr)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            return arr.astype(np.float32)
    if data_h5:
        if not os.path.exists(data_h5):
            print(f"Warning: data_h5 not found: {data_h5}")
            return None
        from bench.models.stgcn.wrapper.data import load_metrla_h5

        return load_metrla_h5(data_h5)
    return None


def _load_split(path: str, num_steps: int) -> Optional[Dict[str, List[int]]]:
    if not path:
        return None
    if not os.path.exists(path):
        print(f"Warning: train_split_json not found: {path}")
        return None
    from bench.models.stgcn.wrapper.data import load_split_json

    return load_split_json(path, num_steps=num_steps)


def _slot_means(
    data: np.ndarray, train_range: List[int], slot_ranges: Sequence[Tuple[int, int]]
) -> List[np.ndarray]:
    start, end = train_range
    train_idx = np.arange(start, end + 1)
    means: List[np.ndarray] = []
    for s_start, s_end in slot_ranges:
        mask = (train_idx % STEPS_PER_DAY >= s_start) & (train_idx % STEPS_PER_DAY <= s_end)
        if not np.any(mask):
            means.append(np.full((data.shape[1],), np.nan, dtype=np.float32))
            continue
        slot_vals = data[train_idx[mask], :, 0]
        means.append(np.nanmean(slot_vals, axis=0))
    return means


def _mercator(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    origin_shift = 20037508.34
    x = lon * origin_shift / 180.0
    y = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) / (np.pi / 180.0)
    y = y * origin_shift / 180.0
    return x, y


def _edges_from_adj(adj: np.ndarray, undirected: bool) -> Dict[Tuple[int, int], float]:
    edges: Dict[Tuple[int, int], float] = {}
    for i in range(adj.shape[0]):
        row = adj[i]
        for j, w in enumerate(row):
            if i == j or w <= 0:
                continue
            if undirected:
                a, b = (i, j) if i < j else (j, i)
                edges[(a, b)] = max(edges.get((a, b), 0.0), float(w))
            else:
                edges[(i, j)] = float(w)
    return edges


def _cap_edges_per_node(
    edges: Dict[Tuple[int, int], float],
    n_nodes: int,
    edges_per_node: int,
    undirected: bool,
) -> Dict[Tuple[int, int], float]:
    if edges_per_node <= 0 or undirected:
        return edges
    by_src: Dict[int, List[Tuple[int, float]]] = {}
    for (i, j), w in edges.items():
        by_src.setdefault(i, []).append((j, w))
    capped: Dict[Tuple[int, int], float] = {}
    for i, nbrs in by_src.items():
        nbrs_sorted = sorted(nbrs, key=lambda x: x[1], reverse=True)[:edges_per_node]
        for j, w in nbrs_sorted:
            capped[(i, j)] = w
    return capped


def _haversine_km(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    r = 6371.0
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def _edge_lengths(
    edges: Dict[Tuple[int, int], float], lon: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    if not edges:
        return np.array([], dtype=float)
    src = np.array([i for i, _ in edges.keys()], dtype=int)
    dst = np.array([j for _, j in edges.keys()], dtype=int)
    return _haversine_km(lon[src], lat[src], lon[dst], lat[dst])


def _neighbors_for_node(
    edges: Iterable[Tuple[int, int]], node: int, undirected: bool
) -> set:
    nbrs = set()
    for i, j in edges:
        if i == node:
            nbrs.add(j)
        if undirected and j == node:
            nbrs.add(i)
    return nbrs


def _plot_edges(
    ax: plt.Axes,
    edges: Iterable[Tuple[int, int]],
    xs: np.ndarray,
    ys: np.ndarray,
    color: str,
    alpha: float,
    lw: float,
) -> None:
    segments = [np.array([[xs[i], ys[i]], [xs[j], ys[j]]], dtype=float) for i, j in edges]
    if not segments:
        return
    lc = LineCollection(segments, colors=color, linewidths=lw, alpha=alpha, zorder=1)
    ax.add_collection(lc)


def _plot_nodes(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    values: Optional[np.ndarray],
    vmin: Optional[float],
    vmax: Optional[float],
    size: float,
) -> Optional[plt.cm.ScalarMappable]:
    if values is None:
        ax.scatter(xs, ys, s=size, c="#6c6c6c", edgecolors="white", linewidths=0.2, zorder=2)
        return None
    sc = ax.scatter(
        xs,
        ys,
        s=size,
        c=values,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        edgecolors="white",
        linewidths=0.2,
        zorder=2,
    )
    return sc


def _format_slot_labels(slot_names: List[str], slot_spans: List[str]) -> List[str]:
    labels = []
    for name, span in zip(slot_names, slot_spans):
        label = f"{name} ({span})" if span else name
        labels.append(label)
    return labels


def _set_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def _maybe_add_basemap(ax: plt.Axes, basemap: str, ctx: Optional[object]) -> None:
    if basemap != "contextily" or ctx is None:
        return
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs="EPSG:3857")
    except Exception as exc:
        print(f"Warning: failed to add basemap tiles ({exc}). Using plain background.")


def _disable_extra_axes(axes: np.ndarray, used: int) -> None:
    total = axes.size
    if used >= total:
        return
    for idx in range(used, total):
        ax = axes.flat[idx]
        ax.set_axis_off()


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish DDS METR-LA figures + metrics.")
    parser.add_argument("--graphs_dir", default="", help="Directory containing A_k.npz.")
    parser.add_argument("--out_dir", default="", help="Output directory.")
    parser.add_argument("--sensor_csv", required=True, help="Sensor locations CSV.")
    parser.add_argument("--k", type=int, default=4, help="Number of slots/graphs.")
    parser.add_argument("--edges_per_node", type=int, default=3, help="Edges per node to draw.")
    parser.add_argument("--slot_names", default="night,AM,midday,PM", help="Comma list.")
    parser.add_argument("--slot_spans", default="00:00-05:55,06:00-09:55,10:00-15:55,16:00-23:55")
    parser.add_argument("--speed_mode", choices=["none", "train_mean"], default="none")
    parser.add_argument("--train_split_json", default="", help="Split json for train_range.")
    parser.add_argument("--data_npz", default="", help="METR-LA data npz (optional).")
    parser.add_argument("--data_h5", default="", help="METR-LA data h5 (optional).")
    parser.add_argument("--sensor_ids_txt", default="", help="graph_sensor_ids.txt path.")
    parser.add_argument("--adj_pkl", default="", help="adj_mx.pkl path (optional).")
    parser.add_argument("--sensor_id_col", default="sensor_id")
    parser.add_argument("--lat_col", default="latitude")
    parser.add_argument("--lon_col", default="longitude")
    parser.add_argument("--undirected", type=int, default=0)
    parser.add_argument("--basemap", choices=["plain", "contextily"], default="plain")
    parser.add_argument("--rare_edges", type=int, default=1, help="Plot rare edges panel (0/1).")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dry_run", type=int, default=0)
    args = parser.parse_args()

    if not args.graphs_dir:
        print("Error: --graphs_dir is required.", file=sys.stderr)
        return 1

    _seed_all(args.seed)

    slot_names = [s.strip() for s in args.slot_names.split(",") if s.strip()]
    slot_spans = [s.strip() for s in args.slot_spans.split(",") if s.strip()]
    while len(slot_spans) < len(slot_names):
        slot_spans.append("")
    slot_labels = _format_slot_labels(slot_names, slot_spans)

    adj_paths = [os.path.join(args.graphs_dir, f"A_{k}.npz") for k in range(args.k)]
    for path in adj_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing adjacency file: {path}")

    out_dir = args.out_dir or os.path.join(args.graphs_dir, "viz_publish")
    os.makedirs(out_dir, exist_ok=True)

    loc_df = _load_locations(args.sensor_csv, args.sensor_id_col, args.lat_col, args.lon_col)
    sensor_ids = _load_sensor_ids_txt(args.sensor_ids_txt)
    adj_ids, adj_id_map = _load_adj_sensor_ids(args.adj_pkl)
    if sensor_ids is None and adj_ids is not None:
        sensor_ids = adj_ids
    lon, lat = _align_locations(loc_df, _load_adj(adj_paths[0]).shape[0], sensor_ids, adj_id_map)

    if args.dry_run == 1:
        print("Dry run OK.")
        print(f"Graphs: {adj_paths}")
        print(f"Output: {out_dir}")
        return 0

    ctx = None
    xs = lon
    ys = lat
    if args.basemap == "contextily":
        try:
            import contextily as _ctx  # type: ignore

            ctx = _ctx
            xs, ys = _mercator(lon, lat)
        except Exception as exc:
            print(f"Warning: basemap unavailable ({exc}). Falling back to plain.")
            args.basemap = "plain"
            xs, ys = lon, lat

    adj_list = [_load_adj(path) for path in adj_paths]
    n_nodes = adj_list[0].shape[0]
    for adj in adj_list[1:]:
        if adj.shape[0] != n_nodes:
            raise ValueError("Adjacency size mismatch across slots.")

    edge_sets: List[Dict[Tuple[int, int], float]] = []
    for adj in adj_list:
        edge_sets.append(_edges_from_adj(adj, args.undirected == 1))

    capped_sets = [
        _cap_edges_per_node(edges, n_nodes, args.edges_per_node, args.undirected == 1)
        for edges in edge_sets
    ]

    print(
        "Loaded graphs: "
        f"N={n_nodes} K={args.k} undirected={args.undirected} "
        f"edges_per_node={args.edges_per_node}"
    )
    print(f"Edge counts per slot: {[len(edges) for edges in edge_sets]}")

    speed_means = None
    if args.speed_mode == "train_mean":
        data = _load_speed_data(args.data_npz, args.data_h5)
        if data is None:
            print("Warning: speed data missing; skipping speed coloring.")
        else:
            split = _load_split(args.train_split_json, num_steps=data.shape[0])
            if split is None:
                print("Warning: train split missing; skipping speed coloring.")
            else:
                if args.k != len(SLOT_RANGES):
                    print(
                        f"Warning: k={args.k} does not match default slot ranges "
                        f"({len(SLOT_RANGES)}). Using available ranges."
                    )
                speed_means = _slot_means(
                    data, split["train_range"], SLOT_RANGES[: min(args.k, len(SLOT_RANGES))]
                )
                if len(speed_means) < args.k:
                    pad = [np.full((n_nodes,), np.nan, dtype=np.float32)] * (
                        args.k - len(speed_means)
                    )
                    speed_means = speed_means + pad

    vmin = vmax = None
    if speed_means is not None:
        vals = np.concatenate(speed_means)
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))

    fig_paths: List[str] = []

    plt.rcParams.update({"font.size": 10})

    def _save_fig(fig: plt.Figure, name: str) -> None:
        png = os.path.join(out_dir, f"{name}.png")
        pdf = os.path.join(out_dir, f"{name}.pdf")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, dpi=300, bbox_inches="tight")
        fig_paths.extend([png, pdf])

    # A) structure-only panel
    cols = 2
    rows = int(math.ceil(args.k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 9))
    axes = np.atleast_2d(axes)
    for k in range(args.k):
        ax = axes[k // cols, k % cols]
        _plot_edges(
            ax,
            capped_sets[k].keys(),
            xs,
            ys,
            color="#2f2f2f",
            alpha=0.25,
            lw=0.5,
        )
        _plot_nodes(ax, xs, ys, None, None, None, size=12)
        title = f"METR-LA DDS Slot {k} ({slot_labels[k]}) | m={args.edges_per_node}"
        ax.set_title(title, fontsize=10)
        _set_axes(ax)
        _maybe_add_basemap(ax, args.basemap, ctx)
    _disable_extra_axes(axes, args.k)
    _save_fig(fig, f"map_structure_panel_k{args.k}_m{args.edges_per_node}")
    plt.close(fig)

    # B) speed-colored panel
    if speed_means is not None:
        fig, axes = plt.subplots(rows, cols, figsize=(12, 9))
        axes = np.atleast_2d(axes)
        sm = None
        for k in range(args.k):
            ax = axes[k // cols, k % cols]
            _plot_edges(
                ax,
                capped_sets[k].keys(),
                xs,
                ys,
                color="#9e9e9e",
                alpha=0.2,
                lw=0.5,
            )
            sm = _plot_nodes(ax, xs, ys, speed_means[k], vmin, vmax, size=16)
            title = f"METR-LA DDS Slot {k} ({slot_labels[k]}) | m={args.edges_per_node} | color=speed"
            ax.set_title(title, fontsize=10)
            _set_axes(ax)
            _maybe_add_basemap(ax, args.basemap, ctx)
        _disable_extra_axes(axes, args.k)
        if sm is not None:
            fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
        _save_fig(fig, f"map_speed_panel_k{args.k}_m{args.edges_per_node}")
        plt.close(fig)
    else:
        print("Warning: speed panel skipped (no speed data).")

    # C) edge diff panel
    edge_sets_keys = [set(edges.keys()) for edges in edge_sets]
    intersection = set.intersection(*edge_sets_keys) if edge_sets_keys else set()
    fig, axes = plt.subplots(rows, cols, figsize=(12, 9))
    axes = np.atleast_2d(axes)
    for k in range(args.k):
        ax = axes[k // cols, k % cols]
        unique = edge_sets_keys[k] - intersection
        _plot_edges(ax, intersection, xs, ys, color="#bdbdbd", alpha=0.2, lw=0.5)
        _plot_edges(ax, unique, xs, ys, color="#d62728", alpha=0.5, lw=0.7)
        _plot_nodes(ax, xs, ys, None, None, None, size=10)
        title = f"METR-LA DDS Slot {k} ({slot_labels[k]}) | common vs unique"
        ax.set_title(title, fontsize=10)
        _set_axes(ax)
        _maybe_add_basemap(ax, args.basemap, ctx)
    _disable_extra_axes(axes, args.k)
    _save_fig(fig, f"map_edge_diff_panel_k{args.k}_m{args.edges_per_node}")
    plt.close(fig)

    # D) rare edges panel
    if args.rare_edges == 1:
        freq: Dict[Tuple[int, int], int] = {}
        for edges in edge_sets_keys:
            for e in edges:
                freq[e] = freq.get(e, 0) + 1
        rare_edges = {e for e, c in freq.items() if c <= 2}
        fig, axes = plt.subplots(rows, cols, figsize=(12, 9))
        axes = np.atleast_2d(axes)
        for k in range(args.k):
            ax = axes[k // cols, k % cols]
            slot_rare = edge_sets_keys[k].intersection(rare_edges)
            _plot_edges(ax, slot_rare, xs, ys, color="#111111", alpha=0.5, lw=0.8)
            _plot_nodes(ax, xs, ys, None, None, None, size=10)
            title = f"METR-LA DDS Slot {k} ({slot_labels[k]}) | rare edges"
            ax.set_title(title, fontsize=10)
            _set_axes(ax)
            _maybe_add_basemap(ax, args.basemap, ctx)
        _disable_extra_axes(axes, args.k)
        _save_fig(fig, f"map_rare_edges_panel_k{args.k}_m{args.edges_per_node}")
        plt.close(fig)

    # E1) edge length distributions
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, edges in enumerate(edge_sets):
        lengths = _edge_lengths(edges, lon, lat)
        if len(lengths) == 0:
            continue
        ax.hist(lengths, bins=30, density=True, alpha=0.4, label=f"slot {k}")
    ax.set_xlabel("Edge length (km)")
    ax.set_ylabel("Density")
    ax.set_title("Edge length distribution by slot")
    ax.legend(frameon=False)
    _save_fig(fig, "edge_length_dist_by_slot")
    plt.close(fig)

    # E2) Jaccard heatmap
    jaccard = np.zeros((args.k, args.k), dtype=float)
    for i in range(args.k):
        for j in range(args.k):
            a = edge_sets_keys[i]
            b = edge_sets_keys[j]
            union = len(a | b)
            jaccard[i, j] = len(a & b) / union if union > 0 else 0.0
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(jaccard, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(args.k))
    ax.set_yticks(range(args.k))
    ax.set_xticklabels([f"{k}" for k in range(args.k)])
    ax.set_yticklabels([f"{k}" for k in range(args.k)])
    ax.set_title("Edge Jaccard similarity")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_fig(fig, "edge_jaccard_heatmap")
    plt.close(fig)

    # E3) neighbor stability
    stability = []
    for i in range(n_nodes):
        nbr_sets = [
            _neighbors_for_node(edges, i, args.undirected == 1)
            for edges in edge_sets_keys
        ]
        sims = []
        for a in range(args.k):
            for b in range(a + 1, args.k):
                u = len(nbr_sets[a] | nbr_sets[b])
                sims.append(len(nbr_sets[a] & nbr_sets[b]) / u if u > 0 else 0.0)
        stability.append(float(np.mean(sims)) if sims else 0.0)
    stability = np.array(stability, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(stability, bins=30, color="#4c72b0", alpha=0.7)
    ax.set_xlabel("Neighbor stability (avg Jaccard)")
    ax.set_ylabel("Count")
    ax.set_title("Neighbor stability across slots")
    _save_fig(fig, "neighbor_stability_hist")
    plt.close(fig)

    # Metrics
    metrics: Dict[str, object] = {
        "N": n_nodes,
        "K": args.k,
        "edges_per_node": args.edges_per_node,
        "total_edges_per_slot": [len(edges) for edges in edge_sets],
        "intersection_size": len(intersection),
        "union_size": len(set.union(*edge_sets_keys)) if edge_sets_keys else 0,
        "jaccard_matrix": jaccard.tolist(),
        "neighbor_stability": {
            "mean": float(np.mean(stability)),
            "median": float(np.median(stability)),
            "std": float(np.std(stability)),
        },
    }

    freq: Dict[Tuple[int, int], int] = {}
    for edges in edge_sets_keys:
        for e in edges:
            freq[e] = freq.get(e, 0) + 1
    freq_counts = {str(k): 0 for k in range(1, args.k + 1)}
    for count in freq.values():
        freq_counts[str(count)] = freq_counts.get(str(count), 0) + 1
    metrics["edge_frequency_counts"] = freq_counts

    slot_metrics = []
    for k, edges in enumerate(edge_sets):
        lengths = _edge_lengths(edges, lon, lat)
        if lengths.size == 0:
            length_stats = {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
        else:
            length_stats = {
                "mean": float(np.mean(lengths)),
                "median": float(np.median(lengths)),
                "p90": float(np.percentile(lengths, 90)),
                "max": float(np.max(lengths)),
            }
        out_deg = np.zeros(n_nodes, dtype=int)
        in_deg = np.zeros(n_nodes, dtype=int)
        if args.undirected == 1:
            for i, j in edges.keys():
                out_deg[i] += 1
                out_deg[j] += 1
            in_deg = out_deg.copy()
        else:
            for i, j in edges.keys():
                out_deg[i] += 1
                in_deg[j] += 1
        slot_metrics.append(
            {
                "slot": k,
                "edge_length": length_stats,
                "out_degree": {
                    "min": int(out_deg.min()),
                    "mean": float(out_deg.mean()),
                    "max": int(out_deg.max()),
                },
                "in_degree": {
                    "min": int(in_deg.min()),
                    "mean": float(in_deg.mean()),
                    "max": int(in_deg.max()),
                },
            }
        )
    metrics["slots"] = slot_metrics

    corr_stats = []
    for k, edges in enumerate(edge_sets):
        if not edges:
            corr_stats.append({"slot": k, "corr_weight_distance": None})
            continue
        weights = np.array(list(edges.values()), dtype=float)
        lengths = _edge_lengths(edges, lon, lat)
        if weights.size < 2 or np.std(weights) == 0 or np.std(lengths) == 0:
            corr = None
        else:
            corr = float(np.corrcoef(weights, lengths)[0, 1])
        corr_stats.append({"slot": k, "corr_weight_distance": corr})
    metrics["weight_length_correlation"] = corr_stats

    metrics_json = os.path.join(out_dir, f"metrics_dds_k{args.k}_m{args.edges_per_node}.json")
    metrics_csv = os.path.join(out_dir, f"metrics_dds_k{args.k}_m{args.edges_per_node}.csv")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    rows = []
    for slot in slot_metrics:
        rows.append(
            {
                "slot": slot["slot"],
                "edge_len_mean": slot["edge_length"]["mean"],
                "edge_len_median": slot["edge_length"]["median"],
                "edge_len_p90": slot["edge_length"]["p90"],
                "edge_len_max": slot["edge_length"]["max"],
                "out_deg_mean": slot["out_degree"]["mean"],
                "out_deg_min": slot["out_degree"]["min"],
                "out_deg_max": slot["out_degree"]["max"],
                "in_deg_mean": slot["in_degree"]["mean"],
                "in_deg_min": slot["in_degree"]["min"],
                "in_deg_max": slot["in_degree"]["max"],
            }
        )
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)

    # Self-checks
    if args.undirected == 0:
        for k, edges in enumerate(edge_sets):
            out_deg = np.zeros(n_nodes, dtype=int)
            for i, _ in edges.keys():
                out_deg[i] += 1
            if abs(out_deg.mean() - args.edges_per_node) > 0.5:
                print(
                    f"Warning: slot {k} avg out-degree {out_deg.mean():.2f} "
                    f"differs from m={args.edges_per_node}."
                )
    if all(edge_sets_keys[0] == s for s in edge_sets_keys[1:]):
        print("Warning: DDS effect not detected (edge sets identical across slots).")

    print("Generated outputs:")
    for path in fig_paths:
        print(f"- {path}")
    print(f"- {metrics_json}")
    print(f"- {metrics_csv}")
    print(f"out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

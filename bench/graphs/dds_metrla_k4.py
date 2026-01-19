import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SLOT_BOUNDARIES = [
    {"slot_id": 0, "name": "night", "start_step": 0, "end_step": 71},
    {"slot_id": 1, "name": "AM", "start_step": 72, "end_step": 119},
    {"slot_id": 2, "name": "midday", "start_step": 120, "end_step": 191},
    {"slot_id": 3, "name": "PM", "start_step": 192, "end_step": 287},
]
SLOT_NAMES = [slot["name"] for slot in SLOT_BOUNDARIES]


def get_slot_id_from_step(step_of_day: int, K: int = 4) -> int:
    if K != 4:
        raise ValueError("Only K=4 is supported for METR-LA slots.")
    if step_of_day < 0 or step_of_day > 287:
        raise ValueError("step_of_day must be in [0, 287].")
    if step_of_day <= 71:
        return 0
    if step_of_day <= 119:
        return 1
    if step_of_day <= 191:
        return 2
    return 3


def _describe_h5_obj(key: str, obj: object) -> str:
    if isinstance(obj, pd.DataFrame):
        return f"{key}: DataFrame shape={obj.shape}"
    if isinstance(obj, np.ndarray):
        return f"{key}: ndarray shape={obj.shape}"
    return f"{key}: {type(obj).__name__}"


def _summarize_h5_candidates(data_h5_path: str, keys: List[str]) -> List[str]:
    summaries: List[str] = []
    for key in keys:
        try:
            obj = pd.read_hdf(data_h5_path, key=key)
        except Exception as exc:
            summaries.append(f"{key}: unreadable ({exc.__class__.__name__})")
            continue
        summaries.append(_describe_h5_obj(key, obj))
    return summaries


def _get_h5_keys(data_h5_path: str) -> List[str]:
    try:
        with pd.HDFStore(data_h5_path, mode="r") as store:
            return store.keys()
    except Exception:
        return []


def _sanitize_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings_list: List[str] = []
    drop_cols: List[str] = []
    index_like = {"index", "time", "timestamp", "date", "datetime"}
    for col in df.columns:
        name = str(col).strip().lower()
        if name in index_like:
            drop_cols.append(col)
    if drop_cols:
        warnings_list.append(f"Dropping index-like columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] != df.shape[1]:
        dropped = [col for col in df.columns if col not in numeric_df.columns]
        warnings_list.append(f"Dropping non-numeric columns: {dropped}")
        df = numeric_df

    return df, warnings_list


def _load_metrla_h5(data_h5_path: str) -> np.ndarray:
    keys: List[str] = []
    try:
        data = pd.read_hdf(data_h5_path)
    except Exception as exc:
        try:
            keys = _get_h5_keys(data_h5_path)
        except Exception as store_exc:
            raise RuntimeError(
                f"Failed to open HDF5 file: {data_h5_path}"
            ) from store_exc
        last_exc = exc
        data = None
        for key in keys:
            try:
                candidate = pd.read_hdf(data_h5_path, key=key)
            except Exception as retry_exc:
                last_exc = retry_exc
                continue
            if isinstance(candidate, pd.DataFrame):
                data = candidate
                break
            last_exc = ValueError(
                f"HDF5 key {key} is not a pandas DataFrame (got {type(candidate).__name__})."
            )
        if data is None:
            candidate_shapes = _summarize_h5_candidates(data_h5_path, keys)
            message = (
                f"Unable to read HDF5 DataFrame from {data_h5_path}. "
                f"Available keys: {keys}. Candidates: {candidate_shapes}"
            )
            print(message)
            raise ValueError(message) from last_exc

    if isinstance(data, pd.DataFrame):
        data, warnings_list = _sanitize_dataframe(data)
        for warning in warnings_list:
            warnings.warn(warning)
        if data.shape[1] == 0:
            if not keys:
                keys = _get_h5_keys(data_h5_path)
            candidate_shapes = _summarize_h5_candidates(data_h5_path, keys)
            message = (
                f"No numeric sensor columns found in {data_h5_path}. "
                f"Available keys: {keys}. Candidates: {candidate_shapes}"
            )
            print(message)
            raise ValueError(message)
        values = data.values
        if values.ndim != 2:
            if not keys:
                keys = _get_h5_keys(data_h5_path)
            candidate_shapes = _summarize_h5_candidates(data_h5_path, keys)
            message = (
                f"Expected 2D (T, N) array from HDF5 DataFrame, got shape {values.shape}. "
                f"Available keys: {keys}. Candidates: {candidate_shapes}"
            )
            print(message)
            raise ValueError(message)
        return values

    if not keys:
        keys = _get_h5_keys(data_h5_path)
    candidate_shapes = _summarize_h5_candidates(data_h5_path, keys)
    message = (
        f"Unsupported HDF5 data format in {data_h5_path}. "
        f"Available keys: {keys}. Candidates: {candidate_shapes}"
    )
    print(message)
    raise ValueError(message)


def _validate_range(name: str, value: object, num_steps: int, required: bool) -> Optional[List[int]]:
    if value is None:
        if required:
            raise ValueError(f"Split json missing required {name}.")
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a list of two integers.")
    start, end = value
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f"{name} entries must be integers.")
    if start < 0 or end < 0 or start > end:
        raise ValueError(f"{name} must satisfy 0 <= start <= end.")
    if end >= num_steps:
        raise ValueError(f"{name} end must be < num_steps ({num_steps}).")
    return [start, end]


def _load_split_json(split_json_path: str, num_steps: int) -> Dict[str, Optional[List[int]]]:
    with open(split_json_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    if "num_steps" in split_data and split_data["num_steps"] != num_steps:
        warnings.warn(
            f"Split json num_steps ({split_data['num_steps']}) does not match H5 length ({num_steps})."
        )

    train_range = _validate_range("train_range", split_data.get("train_range"), num_steps, required=True)
    val_range = _validate_range("val_range", split_data.get("val_range"), num_steps, required=False)
    test_range = _validate_range("test_range", split_data.get("test_range"), num_steps, required=False)

    return {
        "train_range": train_range,
        "val_range": val_range,
        "test_range": test_range,
    }


def _compute_corr_matrix(slot_data: np.ndarray, num_nodes: int) -> np.ndarray:
    if slot_data.shape[0] < 2:
        corr = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    else:
        corr = np.corrcoef(slot_data, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr[corr < 0] = 0.0
    np.fill_diagonal(corr, 0.0)
    return corr


def _top_m_adjacency(corr: np.ndarray, m: int) -> np.ndarray:
    num_nodes = corr.shape[0]
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    if m <= 0 or num_nodes == 0:
        return adj
    indices = np.arange(num_nodes, dtype=np.int64)
    for i in range(num_nodes):
        row = corr[i]
        order = np.lexsort((indices, -row))
        order = order[order != i]
        top = order[:m]
        adj[i, top] = row[top]
    return adj


def _row_normalize(adj: np.ndarray) -> np.ndarray:
    row_sums = adj.sum(axis=1, keepdims=True)
    nonzero = row_sums[:, 0] > 0
    normalized = adj.copy()
    normalized[nonzero] = normalized[nonzero] / row_sums[nonzero]
    return normalized


def _corr_summary(corr: np.ndarray) -> Dict[str, float]:
    num_nodes = corr.shape[0]
    if num_nodes == 0:
        return {"mean_pos": 0.0, "std_pos": 0.0, "zero_fraction": 0.0}
    mask = ~np.eye(num_nodes, dtype=bool)
    values = corr[mask]
    total = values.size
    zeros = int((values == 0).sum())
    pos_vals = values[values > 0]
    if pos_vals.size == 0:
        mean_pos = 0.0
        std_pos = 0.0
    else:
        mean_pos = float(pos_vals.mean())
        std_pos = float(pos_vals.std())
    zero_fraction = float(zeros / total) if total > 0 else 0.0
    return {"mean_pos": mean_pos, "std_pos": std_pos, "zero_fraction": zero_fraction}


def build_dds_graphs_metrla_k4(
    data_h5_path: str,
    out_dir: str,
    m: int = 10,
    seed: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    split_json_path: Optional[str] = None,
) -> Dict[str, object]:
    np.random.seed(seed)

    data = _load_metrla_h5(data_h5_path)
    num_steps, num_nodes = data.shape
    if data.ndim != 2:
        raise ValueError(f"Expected 2D (T, N) data array, got shape {data.shape}.")

    split_source = "ratio"
    split_meta: Dict[str, object] = {}
    if split_json_path:
        split_source = "json"
        split_ranges = _load_split_json(split_json_path, num_steps)
        train_start, train_end = split_ranges["train_range"]
        train_indices = np.arange(train_start, train_end + 1, dtype=np.int64)
        split_meta = {
            "split_json_path": split_json_path,
            "train_range": split_ranges["train_range"],
        }
        if split_ranges["val_range"] is not None:
            split_meta["val_range"] = split_ranges["val_range"]
        if split_ranges["test_range"] is not None:
            split_meta["test_range"] = split_ranges["test_range"]
    else:
        train_end_exclusive = int(num_steps * train_ratio)
        train_indices = np.arange(train_end_exclusive, dtype=np.int64)
        train_start = 0
        train_end = train_end_exclusive - 1

    if split_source == "json":
        print(
            f"Split source: json ({split_json_path}) train_range=[{train_start},{train_end}]"
        )
    else:
        print(f"Split source: ratio train_range=[{train_start},{train_end}]")

    train_steps_of_day = train_indices % 288
    slot_ids = np.array(
        [get_slot_id_from_step(int(step), K=4) for step in train_steps_of_day],
        dtype=np.int64,
    )

    os.makedirs(out_dir, exist_ok=True)
    viz_dir = os.path.join(out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    slot_counts: Dict[int, int] = {}
    slot_stats: Dict[str, Dict[str, object]] = {}
    edge_masks: List[np.ndarray] = []
    paths: Dict[str, str] = {"viz_dir": viz_dir}

    for k in range(4):
        slot_indices = train_indices[slot_ids == k]
        if slot_indices.size > 0:
            if int(slot_indices.min()) < train_start or int(slot_indices.max()) > train_end:
                raise RuntimeError(
                    f"Slot {k} indices leak beyond train_range [{train_start}, {train_end}]."
                )
        slot_counts[k] = int(slot_indices.shape[0])
        slot_data = data[slot_indices]

        corr = _compute_corr_matrix(slot_data, num_nodes)
        if corr.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"Correlation matrix shape mismatch for slot {k}: {corr.shape} != ({num_nodes}, {num_nodes})"
            )
        adj = _top_m_adjacency(corr, m)
        if adj.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"Adjacency matrix shape mismatch for slot {k}: {adj.shape} != ({num_nodes}, {num_nodes})"
            )
        edge_mask = adj > 0
        edge_masks.append(edge_mask)

        out_degree = edge_mask.sum(axis=1)
        if out_degree.size > 0 and int(out_degree.max()) > m:
            warnings.warn(
                f"Slot {k} out-degree exceeds m={m}: max={int(out_degree.max())}"
            )
        out_degree_stats = {
            "min": int(out_degree.min()) if num_nodes > 0 else 0,
            "median": float(np.median(out_degree)) if num_nodes > 0 else 0.0,
            "max": int(out_degree.max()) if num_nodes > 0 else 0,
            "count_lt_m": int((out_degree < m).sum()) if num_nodes > 0 else 0,
        }

        adj = _row_normalize(adj)
        row_sums = adj.sum(axis=1)
        nonzero_rows = row_sums > 0
        if nonzero_rows.any() and not np.allclose(
            row_sums[nonzero_rows], 1.0, atol=1e-6
        ):
            warnings.warn(f"Slot {k} row-stochastic normalization drift detected.")
        row_sum_stats = {
            "min": float(row_sums[nonzero_rows].min()) if nonzero_rows.any() else 0.0,
            "mean": float(row_sums[nonzero_rows].mean()) if nonzero_rows.any() else 0.0,
            "max": float(row_sums[nonzero_rows].max()) if nonzero_rows.any() else 0.0,
        }

        corr_summary = _corr_summary(corr)

        adj_path = os.path.join(out_dir, f"A_{k}.npz")
        np.savez_compressed(adj_path, A=adj)
        paths[f"A_{k}"] = adj_path

        heatmap_path = os.path.join(viz_dir, f"heatmap_k{k}.npz")
        np.savez_compressed(heatmap_path, heatmap=corr)

        slot_stats[f"slot_{k}"] = {
            "outdegree": out_degree_stats,
            "row_sum": row_sum_stats,
            "corr_summary": corr_summary,
            "index_min": int(slot_indices.min()) if slot_indices.size > 0 else -1,
            "index_max": int(slot_indices.max()) if slot_indices.size > 0 else -1,
            "num_steps": int(slot_indices.size),
        }

    overlap: Dict[str, Dict[str, float]] = {}
    for i in range(4):
        for j in range(i + 1, 4):
            inter = int(np.logical_and(edge_masks[i], edge_masks[j]).sum())
            union = int(np.logical_or(edge_masks[i], edge_masks[j]).sum())
            jaccard = float(inter / union) if union > 0 else 0.0
            overlap[f"{i}-{j}"] = {
                "intersection": inter,
                "union": union,
                "jaccard": jaccard,
            }

    overlap_path = os.path.join(viz_dir, "overlap.json")
    with open(overlap_path, "w", encoding="utf-8") as f:
        json.dump(overlap, f, indent=2, sort_keys=True)

    meta = {
        "K": 4,
        "slot_boundaries": SLOT_BOUNDARIES,
        "m": m,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "train_time_index_range": [train_start, train_end],
        "num_nodes": num_nodes,
        "num_train_steps_per_slot": slot_counts,
        "split_source": split_source,
    }
    if split_source == "json":
        meta.update(split_meta)
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    paths["meta"] = meta_path

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for k in range(4):
            heatmap_path = os.path.join(viz_dir, f"heatmap_k{k}.npz")
            heatmap = np.load(heatmap_path)["heatmap"]
            adj = np.load(os.path.join(out_dir, f"A_{k}.npz"))["A"]
            slot_name = SLOT_NAMES[k]
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(heatmap, cmap="viridis", aspect="auto")
            ax.set_title(f"{slot_name} corr")
            fig.tight_layout()
            fig.savefig(os.path.join(viz_dir, f"corr_k{k}.png"))
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(adj, cmap="viridis", aspect="auto")
            ax.set_title(f"{slot_name} adj")
            fig.tight_layout()
            fig.savefig(os.path.join(viz_dir, f"adj_k{k}.png"))
            plt.close(fig)
    except Exception:
        pass

    diagnostics = {"slots": slot_stats, "overlap": overlap}

    print("Per-slot out-degree stats:")
    for slot_key, stats in slot_stats.items():
        outdegree = stats["outdegree"]
        print(
            f"  {slot_key}: min={outdegree['min']} median={outdegree['median']} "
            f"max={outdegree['max']}"
        )
    print("Pairwise edge overlap (Jaccard):")
    for pair_key, stats in overlap.items():
        print(f"  {pair_key}: jaccard={stats['jaccard']}")

    info = {
        "K": 4,
        "m": m,
        "seed": seed,
        "slot_boundaries": SLOT_BOUNDARIES,
        "num_nodes": num_nodes,
        "num_train_steps": int(train_indices.shape[0]),
        "paths": paths,
        "diagnostics": diagnostics,
    }
    return info

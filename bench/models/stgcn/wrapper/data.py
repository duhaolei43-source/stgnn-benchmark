import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _get_h5_keys(data_h5_path: str) -> List[str]:
    try:
        with pd.HDFStore(data_h5_path, mode="r") as store:
            return store.keys()
    except Exception:
        return []


def _read_h5_any(data_h5_path: str):
    try:
        return pd.read_hdf(data_h5_path)
    except Exception:
        keys = _get_h5_keys(data_h5_path)
        last_exc: Optional[Exception] = None
        for key in keys:
            try:
                return pd.read_hdf(data_h5_path, key=key)
            except Exception as exc:
                last_exc = exc
                continue
        raise ValueError(f"Unable to read HDF5 data from {data_h5_path}. Keys: {keys}") from last_exc


def load_metrla_h5(data_h5_path: str) -> np.ndarray:
    data = _read_h5_any(data_h5_path)
    if isinstance(data, pd.DataFrame):
        values = data.values
    elif isinstance(data, np.ndarray):
        values = data
    else:
        raise ValueError(f"Unsupported HDF5 object type: {type(data).__name__}")

    if values.ndim == 2:
        values = values[:, :, None]
    elif values.ndim == 3:
        if values.shape[2] != 1:
            raise ValueError(f"Expected feature dimension size 1, got shape {values.shape}")
    else:
        raise ValueError(f"Expected data of shape (T, N) or (T, N, 1), got {values.shape}")

    return values.astype(np.float32)


def load_split_json(split_json_path: str, num_steps: int) -> Dict[str, Optional[List[int]]]:
    with open(split_json_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    def _validate(name: str) -> Optional[List[int]]:
        value = split_data.get(name)
        if value is None:
            return None
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{name} must be a list of two integers.")
        start, end = value
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"{name} entries must be integers.")
        if start < 0 or end < start or end >= num_steps:
            raise ValueError(f"{name} must satisfy 0 <= start <= end < num_steps ({num_steps}).")
        return [start, end]

    if "num_steps" in split_data and split_data["num_steps"] != num_steps:
        print(
            f"Warning: split num_steps ({split_data['num_steps']}) != H5 length ({num_steps}), using H5."
        )

    train_range = _validate("train_range")
    if train_range is None:
        raise ValueError("split json missing train_range.")

    return {
        "train_range": train_range,
        "val_range": _validate("val_range"),
        "test_range": _validate("test_range"),
    }


def normalize_series(data: np.ndarray, train_range: List[int]) -> Tuple[np.ndarray, float, float]:
    start, end = train_range
    train_data = data[start : end + 1]
    mean = float(train_data.mean())
    std = float(train_data.std())
    if std == 0.0:
        std = 1.0
    normalized = (data - mean) / std
    return normalized, mean, std


def build_sequences(
    data: np.ndarray,
    start: int,
    end: int,
    t_in: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if t_in <= 0 or horizon <= 0:
        raise ValueError("t_in and horizon must be positive.")
    if end < start:
        raise ValueError("Invalid range: end must be >= start.")

    last_start = end - (t_in + horizon) + 1
    if last_start < start:
        raise ValueError("Not enough data to build sequences for the given range.")

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for t in range(start, last_start + 1):
        x_slice = data[t : t + t_in]
        y_slice = data[t + t_in : t + t_in + horizon]
        x = np.transpose(x_slice, (1, 0, 2))
        y = y_slice[:, :, 0]
        xs.append(x)
        ys.append(y)

    X = np.stack(xs, axis=0).astype(np.float32)
    Y = np.stack(ys, axis=0).astype(np.float32)
    return X, Y

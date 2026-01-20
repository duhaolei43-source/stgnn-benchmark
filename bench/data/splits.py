import json
from typing import Dict, List, Optional

import numpy as np


def compute_time_split(num_steps: int, train_ratio: float, val_ratio: float) -> Dict[str, List[int]]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if train_ratio <= 0 or val_ratio < 0:
        raise ValueError("train_ratio must be > 0 and val_ratio must be >= 0.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0.")

    train_end = int(num_steps * train_ratio)
    val_end = int(num_steps * (train_ratio + val_ratio))

    if not (0 < train_end < val_end < num_steps):
        raise ValueError(
            f"Invalid split points: train_end={train_end}, val_end={val_end}, num_steps={num_steps}."
        )

    train_range = [0, train_end - 1]
    val_range = [train_end, val_end - 1]
    test_range = [val_end, num_steps - 1]

    return {
        "train_range": train_range,
        "val_range": val_range,
        "test_range": test_range,
    }


def _validate_range(name: str, value: object, num_steps: int) -> List[int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a list of two integers.")
    start, end = value
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f"{name} entries must be integers.")
    if start < 0 or end < start or end >= num_steps:
        raise ValueError(f"{name} must satisfy 0 <= start <= end < num_steps ({num_steps}).")
    return [start, end]


def validate_split_ranges(
    num_steps: int,
    train_range: object,
    val_range: object,
    test_range: object,
) -> Dict[str, List[int]]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    train_range = _validate_range("train_range", train_range, num_steps)
    val_range = _validate_range("val_range", val_range, num_steps)
    test_range = _validate_range("test_range", test_range, num_steps)

    train_start, train_end = train_range
    val_start, val_end = val_range
    test_start, test_end = test_range

    if train_start != 0:
        raise ValueError("train_range must start at index 0.")
    if train_end >= val_start:
        raise ValueError("train_range must end before val_range starts.")
    if val_end >= test_start:
        raise ValueError("val_range must end before test_range starts.")
    if val_start != train_end + 1:
        raise ValueError("val_range must start immediately after train_range.")
    if test_start != val_end + 1:
        raise ValueError("test_range must start immediately after val_range.")
    if test_end != num_steps - 1:
        raise ValueError("test_range must end at num_steps - 1.")

    return {
        "train_range": train_range,
        "val_range": val_range,
        "test_range": test_range,
    }


def ranges_to_indices(range_value: List[int]) -> np.ndarray:
    if range_value is None:
        raise ValueError("range_value must be provided.")
    start, end = range_value
    return np.arange(start, end + 1, dtype=np.int64)


def load_split_json(split_json_path: str, num_steps: Optional[int] = None) -> Dict[str, object]:
    with open(split_json_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    required_keys = ["dataset", "version", "num_steps", "train_range", "val_range", "test_range"]
    missing = [key for key in required_keys if key not in split_data]
    if missing:
        raise ValueError(f"split json missing keys: {', '.join(missing)}")

    split_num_steps = split_data.get("num_steps")
    if num_steps is None:
        num_steps = split_num_steps
    if num_steps is None:
        raise ValueError("num_steps must be provided or present in split json.")
    if split_num_steps is not None and split_num_steps != num_steps:
        print(
            f"Warning: split num_steps ({split_num_steps}) != data num_steps ({num_steps}), using data."
        )

    ranges = validate_split_ranges(
        num_steps,
        split_data["train_range"],
        split_data["val_range"],
        split_data["test_range"],
    )

    return {
        "dataset": split_data["dataset"],
        "version": split_data["version"],
        "num_steps": num_steps,
        **ranges,
    }

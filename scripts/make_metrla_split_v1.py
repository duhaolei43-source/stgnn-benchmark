import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd

from bench.data.splits import compute_time_split, validate_split_ranges


def _describe_h5_obj(key: str, obj: object) -> str:
    if isinstance(obj, pd.DataFrame):
        return f"{key}: DataFrame shape={obj.shape}"
    if isinstance(obj, pd.Series):
        return f"{key}: Series shape={obj.shape}"
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


def _extract_num_steps(obj: object) -> int:
    if isinstance(obj, pd.DataFrame):
        return int(obj.shape[0])
    if isinstance(obj, pd.Series):
        return int(obj.shape[0])
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return 0
        return int(obj.shape[0])
    raise ValueError(f"Unsupported HDF5 object type: {type(obj).__name__}")


def _read_h5_num_steps(data_h5_path: str) -> int:
    keys: List[str] = []
    try:
        data = pd.read_hdf(data_h5_path)
        return _extract_num_steps(data)
    except Exception as exc:
        keys = _get_h5_keys(data_h5_path)
        last_exc = exc
        for key in keys:
            try:
                candidate = pd.read_hdf(data_h5_path, key=key)
                return _extract_num_steps(candidate)
            except Exception as retry_exc:
                last_exc = retry_exc
                continue
        candidate_shapes = _summarize_h5_candidates(data_h5_path, keys)
        message = (
            f"Unable to read HDF5 data from {data_h5_path}. "
            f"Available keys: {keys}. Candidates: {candidate_shapes}"
        )
        print(message)
        raise ValueError(message) from last_exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Create METR-LA split_v1 artifact.")
    parser.add_argument("--data", required=True, help="Path to metr-la.h5")
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for split_v1.json (under artifacts/splits/...)",
    )
    args = parser.parse_args()

    config_path = os.path.join("configs", "protocol", "metrla_split_v1.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    num_steps = _read_h5_num_steps(args.data)
    split = compute_time_split(
        num_steps=num_steps,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
    )
    validate_split_ranges(
        num_steps,
        split["train_range"],
        split["val_range"],
        split["test_range"],
    )

    output = {
        "dataset": config["dataset"],
        "version": config["version"],
        "num_steps": num_steps,
        "train_ratio": config["train_ratio"],
        "val_ratio": config["val_ratio"],
        "steps_per_day": config["steps_per_day"],
        "train_range": split["train_range"],
        "val_range": split["val_range"],
        "test_range": split["test_range"],
        "split_type": config["split_type"],
        "notes": "",
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)

    print(f"train_range: {output['train_range']}")
    print(f"val_range: {output['val_range']}")
    print(f"test_range: {output['test_range']}")
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

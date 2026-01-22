import os
import pickle
from typing import Any

import numpy as np


def _to_dense(adj: Any) -> np.ndarray:
    if hasattr(adj, "toarray"):
        return adj.toarray()
    return np.asarray(adj)


def _sanitize_adj(adj: np.ndarray) -> np.ndarray:
    assert adj.ndim == 2 and adj.shape[0] == adj.shape[1], (
        f"Adjacency matrix must be square, got shape {adj.shape}"
    )
    return np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)


def load_adjacency(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Adjacency file not found: {path}")

    if path.endswith(".npz"):
        with np.load(path) as data:
            if {"data", "indices", "indptr", "shape"}.issubset(set(data.files)):
                import scipy.sparse as sp

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
        adj = _to_dense(adj)
        adj_type = "npz"
    elif path.endswith(".pkl"):
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except UnicodeDecodeError:
            with open(path, "rb") as f:
                obj = pickle.load(f, encoding="latin1")
        if isinstance(obj, dict):
            if "adj_mx" in obj:
                adj = obj["adj_mx"]
            elif "adj" in obj:
                adj = obj["adj"]
            else:
                raise ValueError(f"Unsupported adjacency dict keys: {list(obj.keys())}")
        elif isinstance(obj, (list, tuple)) and len(obj) == 3:
            adj = obj[2]
        elif isinstance(obj, np.ndarray):
            adj = obj
        else:
            adj = obj
        adj = _to_dense(adj)
        adj_type = "pkl"
    else:
        raise ValueError(f"Unsupported adjacency format: {path}")

    adj = _sanitize_adj(adj)
    print(f"Loaded adjacency: type={adj_type} shape={adj.shape} path={path}")

    return adj.astype(np.float32)


def _sanity_check_adjacency() -> None:
    adj = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    _sanitize_adj(adj)

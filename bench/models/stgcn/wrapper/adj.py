import os
import pickle
from typing import Any

import numpy as np


def _to_dense(adj: Any) -> np.ndarray:
    if hasattr(adj, "toarray"):
        return adj.toarray()
    return np.asarray(adj)


def load_adjacency(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Adjacency file not found: {path}")

    if path.endswith(".npz"):
        data = np.load(path)
        if "A" in data:
            adj = data["A"]
        elif data.files:
            adj = data[data.files[0]]
        else:
            raise ValueError(f"No arrays found in npz file: {path}")
        adj = _to_dense(adj)
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            if "adj_mx" in obj:
                adj = obj["adj_mx"]
            elif "adj" in obj:
                adj = obj["adj"]
            else:
                raise ValueError(f"Unsupported adjacency dict keys: {list(obj.keys())}")
        elif isinstance(obj, (list, tuple)) and len(obj) > 0:
            adj = obj[0]
        else:
            adj = obj
        adj = _to_dense(adj)
    else:
        raise ValueError(f"Unsupported adjacency format: {path}")

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got shape {adj.shape}")

    return adj.astype(np.float32)

from typing import Dict, List


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

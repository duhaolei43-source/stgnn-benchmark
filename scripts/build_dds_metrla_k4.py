import argparse
import os
import sys

from bench.graphs.dds_metrla_k4 import build_dds_graphs_metrla_k4


def _expected_outputs(out_dir: str) -> list:
    expected = [
        os.path.join(out_dir, "A_0.npz"),
        os.path.join(out_dir, "A_1.npz"),
        os.path.join(out_dir, "A_2.npz"),
        os.path.join(out_dir, "A_3.npz"),
        os.path.join(out_dir, "meta.json"),
        os.path.join(out_dir, "viz"),
        os.path.join(out_dir, "viz", "heatmap_k0.npz"),
        os.path.join(out_dir, "viz", "heatmap_k1.npz"),
        os.path.join(out_dir, "viz", "heatmap_k2.npz"),
        os.path.join(out_dir, "viz", "heatmap_k3.npz"),
        os.path.join(out_dir, "viz", "overlap.json"),
    ]
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DDS K=4 graphs for METR-LA.")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to data/raw/metr-la/metr-la.h5",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("artifacts", "graphs", "metr-la", "dds_k4"),
        help="Output directory for graphs and metadata.",
    )
    parser.add_argument("--m", type=int, default=10, help="Top-m neighbors per node.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Val split ratio.")
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split json path (artifacts/splits/metr-la/split_v1.json).",
    )
    args = parser.parse_args()

    build_dds_graphs_metrla_k4(
        data_h5_path=args.data,
        out_dir=args.out,
        m=args.m,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_json_path=args.split,
    )

    expected = _expected_outputs(args.out)
    missing = []
    for path in expected:
        if path.endswith("viz"):
            if not os.path.isdir(path):
                missing.append(path)
        else:
            if not os.path.exists(path):
                missing.append(path)

    if missing:
        print("Missing expected outputs:")
        for path in missing:
            print(f"  {path}")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

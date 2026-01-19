import json
import os
import tempfile
import unittest

from bench.graphs.dds_metrla_k4 import build_dds_graphs_metrla_k4, _load_metrla_h5


class TestDDSMETRLAK4Smoke(unittest.TestCase):
    def test_build_dds_graphs(self) -> None:
        data_path = os.path.join("data", "raw", "metr-la", "metr-la.h5")
        if not os.path.exists(data_path):
            self.skipTest("metr-la.h5 not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            data = _load_metrla_h5(data_path)
            num_steps = int(data.shape[0])
            train_end = min(499, max(num_steps - 1, 0))
            split_path = os.path.join(tmpdir, "split.json")
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "train_range": [0, train_end],
                        "num_steps": num_steps,
                    },
                    f,
                    indent=2,
                )

            info = build_dds_graphs_metrla_k4(
                data_h5_path=data_path,
                out_dir=tmpdir,
                m=5,
                split_json_path=split_path,
            )

            for k in range(4):
                path = os.path.join(tmpdir, f"A_{k}.npz")
                self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "meta.json")))
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "viz")))

            with open(os.path.join(tmpdir, "meta.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["split_source"], "json")
            self.assertEqual(meta["train_range"], [0, train_end])

            slot_stats = info["diagnostics"]["slots"]
            for k in range(4):
                slot_info = slot_stats[f"slot_{k}"]
                index_max = slot_info["index_max"]
                if index_max >= 0:
                    self.assertLessEqual(index_max, train_end)


if __name__ == "__main__":
    unittest.main()

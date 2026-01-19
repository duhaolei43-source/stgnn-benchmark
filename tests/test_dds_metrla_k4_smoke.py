import os
import tempfile
import unittest

from bench.graphs.dds_metrla_k4 import build_dds_graphs_metrla_k4


class TestDDSMETRLAK4Smoke(unittest.TestCase):
    def test_build_dds_graphs(self) -> None:
        data_path = os.path.join("data", "raw", "metr-la", "metr-la.h5")
        if not os.path.exists(data_path):
            self.skipTest("metr-la.h5 not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            info = build_dds_graphs_metrla_k4(
                data_h5_path=data_path,
                out_dir=tmpdir,
                m=5,
            )

            for k in range(4):
                path = os.path.join(tmpdir, f"A_{k}.npz")
                self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "meta.json")))
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "viz")))

            train_end = info["num_train_steps"]
            slot_stats = info["diagnostics"]["slots"]
            for k in range(4):
                slot_info = slot_stats[f"slot_{k}"]
                index_max = slot_info["index_max"]
                if index_max >= 0:
                    self.assertLess(index_max, train_end)


if __name__ == "__main__":
    unittest.main()

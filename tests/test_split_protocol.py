import json
import os
import tempfile
import unittest

import numpy as np

from bench.data.splits import load_split_json, ranges_to_indices, validate_split_ranges


class TestSplitProtocol(unittest.TestCase):
    def test_valid_contiguous_split(self) -> None:
        ranges = validate_split_ranges(10, [0, 5], [6, 7], [8, 9])
        self.assertEqual(ranges["train_range"], [0, 5])
        self.assertEqual(ranges["val_range"], [6, 7])
        self.assertEqual(ranges["test_range"], [8, 9])

        indices = ranges_to_indices(ranges["val_range"])
        self.assertTrue(np.array_equal(indices, np.arange(6, 8, dtype=np.int64)))

    def test_overlap_split_fails(self) -> None:
        with self.assertRaises(ValueError):
            validate_split_ranges(10, [0, 6], [6, 7], [8, 9])

    def test_missing_keys_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "split.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset": "metr-la",
                        "version": "v1",
                        "num_steps": 10,
                        "train_range": [0, 5],
                        "val_range": [6, 7],
                    },
                    f,
                    indent=2,
                )
            with self.assertRaises(ValueError) as ctx:
                load_split_json(path)
            self.assertIn("test_range", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

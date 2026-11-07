import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "stream_benchmarks", Path(__file__).resolve().parents[1] / "benchmark" / "stream_benchmarks.py")
B = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(B)


class TestStreamBenchmarks(unittest.TestCase):
    def test_reference_implementations_agree(self):
        X = np.random.default_rng(0).normal(size=(1000, 3))
        m1, v1 = B.welford_loop(X)
        m2, v2 = B.chunked_stats(X, chunk=70)
        np.testing.assert_allclose(m1, m2)
        np.testing.assert_allclose(v1, v2)

    def test_time_call_reports_requested_repeats(self):
        stats, result = B.time_call(lambda: 42, repeat=4)
        self.assertEqual(len(stats["times"]), 4)
        self.assertEqual(result, 42)
        self.assertLessEqual(stats["min"], stats["max"])

    def test_comparison_detects_mismatch(self):
        with self.assertRaises(AssertionError):
            B._comparison("x", 1, lambda: 1, lambda: 2, 1, lambda a, b: a == b)

    def test_quick_run_and_output_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                report = B.main(["--quick", "--repeat", "1", "--output-dir", tmp])
            data = json.loads((Path(tmp) / "stream_results.json").read_text(encoding="utf-8"))
            md = (Path(tmp) / "stream_results.md").read_text(encoding="utf-8")
        self.assertEqual(len(report["buffer_sizes"]["rows"]), 3)
        self.assertTrue(all(r["speedup"] > 1 for r in data["loop_vs_vectorised"][:2]))
        self.assertIn("Effect of buffer_size", md)

    def test_run_stream_counts_retrains(self):
        from numcompute_stream.datasets import make_drifting_data
        X, y = make_drifting_data(1000, seed=1)
        out = B.run_stream(X, y, buffer_size=250, chunk_size=50, max_depth=4, drift_index=500)
        self.assertEqual(out["n_retrains"], 4)
        self.assertGreater(out["accuracy"], 0.3)


if __name__ == "__main__":
    unittest.main()

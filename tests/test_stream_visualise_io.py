import os
import tempfile
import unittest
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from numcompute_stream import visualise as V  # noqa: E402
from numcompute_stream.datasets import make_drifting_data, write_csv  # noqa: E402
from numcompute_stream.io import read_csv, read_csv_chunks  # noqa: E402


class TestVisualise(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_plot_metric_over_time_returns_figure_and_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "acc.png")
            fig = V.plot_metric_over_time([0.5, np.nan, 0.7, 0.9], "Accuracy", "acc",
                                          retrain_points=[2, 4], save_path=path, show=False)
            self.assertTrue(os.path.getsize(path) > 0)
        self.assertEqual(fig.axes[0].get_title(), "Accuracy")
        self.assertEqual(fig.axes[0].get_ylabel(), "acc")

    def test_compare_models_two_lines_and_different_lengths(self):
        fig = V.compare_models([0.1, 0.2, 0.3], [0.3, 0.4], labels=("A", "B"), show=False)
        lines = fig.axes[0].get_lines()
        self.assertEqual(len(lines), 2)
        self.assertEqual([t.get_text() for t in fig.axes[0].get_legend().get_texts()], ["A", "B"])

    def test_predictions_vs_ground_truth_numeric_and_string(self):
        fig = V.plot_predictions_vs_ground_truth([0, 1, 2, 1], [0, 2, 2, 1], show=False)
        self.assertEqual(len(fig.axes), 3)  # two panels + colorbar
        self.assertIn("1 errors", fig.axes[0].get_title())
        V.plot_predictions_vs_ground_truth(["a", "b"], ["a", "a"], show=False)

    def test_save_pdf_and_show_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.pdf")
            V.compare_models([1, 2], [2, 3], save_path=path, show=False)
            self.assertTrue(os.path.exists(path))

    def test_show_flag_controls_plt_show(self):
        # plt.show is mocked: the Agg test backend is non-interactive and would warn.
        funcs = (lambda **kw: V.plot_metric_over_time([1, 2], **kw),
                 lambda **kw: V.compare_models([1, 2], [2, 3], **kw),
                 lambda **kw: V.plot_predictions_vs_ground_truth([0, 1], [0, 1], **kw))
        for call in funcs:
            with mock.patch.object(V.plt, "show") as shown:
                fig = call(show=True)
                shown.assert_called_once_with()
                self.assertIsNotNone(fig)
            with mock.patch.object(V.plt, "show") as shown:
                call(show=False)
                shown.assert_not_called()
            plt.close("all")

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            V.plot_metric_over_time([], show=False)
        with self.assertRaises(ValueError):
            V.plot_metric_over_time([[1, 2], [3, 4]], show=False)
        with self.assertRaises(ValueError):
            V.compare_models([1], [1], labels=("only",), show=False)
        with self.assertRaises(ValueError):
            V.plot_predictions_vs_ground_truth([1, 2], [1], show=False)
        with self.assertRaises(ValueError):
            V.plot_predictions_vs_ground_truth([], [], show=False)


class TestCsvStreamingIO(unittest.TestCase):
    def test_write_read_roundtrip_with_missing_values(self):
        X, y = make_drifting_data(50, seed=3, missing_rate=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.csv")
            write_csv(path, X, y)
            data = read_csv(path, dtype=float, skip_header=1)
        self.assertEqual(data.shape, (50, 7))
        np.testing.assert_allclose(data[:, :-1], X, equal_nan=True)
        np.testing.assert_array_equal(data[:, -1].astype(int), y)

    def test_read_csv_chunks_matches_full_read(self):
        X, y = make_drifting_data(25, seed=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.csv")
            write_csv(path, X, y)
            chunks = list(read_csv_chunks(path, dtype=float, skip_header=1, chunk_size=10))
            full = read_csv(path, dtype=float, skip_header=1)
        self.assertEqual([c.shape[0] for c in chunks], [10, 10, 5])
        np.testing.assert_allclose(np.vstack(chunks), full, equal_nan=True)

    def test_write_csv_validation(self):
        with self.assertRaises(ValueError):
            write_csv("unused.csv", np.ones((3, 2)), np.ones(2))


if __name__ == "__main__":
    unittest.main()

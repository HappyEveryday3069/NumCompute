import unittest

import numpy as np

from numcompute_stream.stats import (RunningStats, StreamingHistogram,
                                     StreamingQuantiles, histogram)


class TestRunningStats(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(0).normal(5, 3, size=(500, 4))

    def test_chunked_equals_batch(self):
        rs = RunningStats()
        for chunk in np.array_split(self.X, 7):
            rs.update_stats(chunk)
        np.testing.assert_allclose(rs.mean, self.X.mean(0), rtol=1e-12)
        np.testing.assert_allclose(rs.variance(0), self.X.var(0), rtol=1e-10)
        np.testing.assert_allclose(rs.variance(1), self.X.var(0, ddof=1), rtol=1e-10)
        np.testing.assert_allclose(rs.min, self.X.min(0))
        np.testing.assert_allclose(rs.max, self.X.max(0))
        self.assertEqual(rs.n_samples_seen_, 500)

    def test_nan_ignored_per_feature(self):
        X = self.X.copy()
        X[::5, 1] = np.nan
        rs = RunningStats()
        for chunk in np.array_split(X, 4):
            rs.update_stats(chunk)
        np.testing.assert_allclose(rs.mean, np.nanmean(X, 0), rtol=1e-12)
        np.testing.assert_allclose(rs.std, np.nanstd(X, 0), rtol=1e-10)
        self.assertEqual(rs.count_[1], 400)

    def test_all_nan_feature_reports_nan(self):
        X = self.X.copy()
        X[:, 2] = np.nan
        rs = RunningStats().update_stats(X)
        self.assertTrue(np.isnan(rs.mean[2]) and np.isnan(rs.min[2]) and np.isnan(rs.std[2]))
        self.assertFalse(np.isnan(rs.mean[0]))

    def test_constant_feature_zero_variance(self):
        rs = RunningStats().update_stats(np.full((10, 1), 3.0))
        rs.update_stats(np.full((5, 1), 3.0))
        self.assertAlmostEqual(rs.variance()[0], 0.0, places=20)

    def test_empty_chunk_is_noop_and_1d_ok(self):
        rs = RunningStats().update_stats([1.0, 2.0, 3.0])
        rs.update_stats(np.empty((0, 1)))
        self.assertEqual(rs.n_samples_seen_, 3)
        self.assertAlmostEqual(rs.mean[0], 2.0)

    def test_errors(self):
        rs = RunningStats()
        with self.assertRaises(ValueError):
            rs.mean
        rs.update_stats(np.ones((3, 2)))
        with self.assertRaises(ValueError):
            rs.update_stats(np.ones((3, 3)))
        with self.assertRaises(ValueError):
            rs.update_stats(np.array([[np.inf, 1.0]]))
        with self.assertRaises(ValueError):
            rs.update_stats(np.ones((2, 2, 2)))

    def test_merge_and_reset(self):
        a = RunningStats().update_stats(self.X[:200])
        b = RunningStats().update_stats(self.X[200:])
        a.merge(b)
        np.testing.assert_allclose(a.variance(), self.X.var(0), rtol=1e-10)
        with self.assertRaises(TypeError):
            a.merge(3)
        a.reset()
        self.assertIsNone(a.n_features_)

    def test_values_that_would_overflow_the_variance_are_rejected_not_turned_into_nan(self):
        with self.assertRaisesRegex(ValueError, "max_abs"):
            RunningStats().update_stats([[1e150], [-1e150]])
        ok = RunningStats().update_stats([[1e99], [-1e99], [5e98]])
        self.assertTrue(np.isfinite(ok.variance()).all())
        with self.assertRaises(ValueError):
            RunningStats(max_abs=-1)
        self.assertEqual(RunningStats(max_abs=None).update_stats([[1e200]]).max[0], 1e200)

    def test_numerical_stability_large_offset(self):
        x = 1e9 + np.random.default_rng(1).normal(size=(1000, 1))
        rs = RunningStats()
        for chunk in np.array_split(x, 10):
            rs.update_stats(chunk)
        np.testing.assert_allclose(rs.variance(), x.var(0), rtol=1e-6)


class TestStreamingHistogram(unittest.TestCase):
    def test_matches_batch_histogram(self):
        x = np.random.default_rng(2).normal(size=1000)
        h = StreamingHistogram(bins=8, range_bounds=(-3, 3))
        for chunk in np.array_split(x, 9):
            h.update_stats(chunk)
        counts, edges = histogram(x, bins=8, range_bounds=(-3, 3))
        np.testing.assert_array_equal(h.counts, counts)
        np.testing.assert_allclose(h.result()[1], edges)

    def test_window_evicts_old_chunks(self):
        h = StreamingHistogram(bins=2, range_bounds=(0, 2), window=2)
        h.update_stats([0.5, 0.5])
        h.update_stats([1.5])
        h.update_stats([1.5, 1.5])
        np.testing.assert_array_equal(h.counts, [0, 3])
        self.assertEqual(h.n_samples, 3)

    def test_outliers_clipped_and_nan_ignored(self):
        h = StreamingHistogram(bins=2, range_bounds=(0, 2))
        h.update_stats([-5, 5, np.nan, np.inf])
        np.testing.assert_array_equal(h.counts, [1, 2])

    def test_range_inferred_from_first_chunk(self):
        h = StreamingHistogram(bins=4)
        self.assertIsNone(h.result()[1])
        h.update_stats([np.nan])
        self.assertIsNone(h.bin_edges_)
        h.update_stats([0.0, 4.0])
        np.testing.assert_allclose(h.bin_edges_, [0, 1, 2, 3, 4])
        self.assertAlmostEqual(h.density().sum(), 1.0)

    def test_constant_first_chunk_and_reset(self):
        h = StreamingHistogram(bins=3).update_stats([2.0, 2.0])
        self.assertEqual(h.n_samples, 2)
        h.reset()
        self.assertEqual(h.n_samples, 0)

    def test_invalid_arguments(self):
        for kwargs in ({"bins": 0}, {"window": 0}, {"range_bounds": (1, 1)}):
            with self.assertRaises(ValueError):
                StreamingHistogram(**kwargs)


class TestStreamingQuantiles(unittest.TestCase):
    def test_exact_below_capacity(self):
        X = np.random.default_rng(3).normal(size=(300, 2))
        sq = StreamingQuantiles(max_samples=1000)
        for chunk in np.array_split(X, 5):
            sq.update_stats(chunk)
        np.testing.assert_allclose(sq.quantile([0.1, 0.5, 0.9]),
                                   np.quantile(X, [0.1, 0.5, 0.9], axis=0))

    def test_window_keeps_latest_rows(self):
        sq = StreamingQuantiles(window=3)
        sq.update_stats(np.arange(10.0))
        self.assertEqual(sq.n_stored, 3)
        np.testing.assert_allclose(sq.quantile(0.5), [8.0])

    def test_reservoir_bounded_and_approximately_correct(self):
        x = np.random.default_rng(4).uniform(size=20000)
        sq = StreamingQuantiles(max_samples=2000, random_state=0)
        for chunk in np.array_split(x, 40):
            sq.update_stats(chunk)
        self.assertEqual(sq.n_stored, 2000)
        self.assertEqual(sq.n_samples_seen_, 20000)
        self.assertAlmostEqual(float(sq.quantile(0.5)[0]), 0.5, delta=0.05)

    def test_nan_and_errors(self):
        sq = StreamingQuantiles()
        with self.assertRaises(ValueError):
            sq.quantile(0.5)
        sq.update_stats([[1.0, np.nan], [3.0, np.nan]])
        out = sq.quantile(0.5)
        self.assertEqual(out[0], 2.0)
        self.assertTrue(np.isnan(out[1]))
        with self.assertRaises(ValueError):
            sq.quantile(1.5)
        with self.assertRaises(ValueError):
            StreamingQuantiles(max_samples=0)


if __name__ == "__main__":
    unittest.main()

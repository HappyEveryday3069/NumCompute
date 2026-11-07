import unittest

import numpy as np

from numcompute_stream import (DecisionTreeClassifier, Imputer, Pipeline, StandardScaler,
                               StreamTrainer, iter_chunks)
from numcompute_stream.datasets import make_drifting_data
from numcompute_stream.metrics import StreamingAccuracy, StreamingPrecision
from numcompute_stream.stream import estimate_memory
from numcompute_stream.utils import NotFittedError


def _pipe(buffer_size=300, adaptive=False, **tree_kw):
    """Default: fixed preprocessing (constant imputer) in front of the buffered tree.

    ``adaptive=True`` adds the adaptive mean imputer and StandardScaler (used to test the
    pipeline plumbing and to demonstrate why they are not recommended before a buffered tree).
    """
    tree = DecisionTreeClassifier(max_depth=5, buffer_size=buffer_size, **tree_kw)
    if adaptive:
        return Pipeline([("imp", Imputer(strategy="mean")), ("scale", StandardScaler()), ("tree", tree)])
    return Pipeline([("imp", Imputer(strategy="constant", fill_value=0.0)), ("tree", tree)])


class TestStreamingPipeline(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_drifting_data(1200, seed=1)

    def test_partial_fit_chain_updates_every_step(self):
        pipe = _pipe(buffer_size=300, adaptive=True)
        for Xc, yc in iter_chunks(self.X, self.y, 100):
            pipe.partial_fit(Xc, yc)
        scaler, tree = pipe.steps[1][1], pipe.steps[2][1]
        self.assertEqual(int(scaler.n_samples_seen_.max()), 1200)
        self.assertEqual(tree.n_retrains_, 4)
        self.assertEqual(pipe.predict(self.X[:50]).shape, (50,))

    def test_scaler_state_matches_batch_after_stream(self):
        pipe = _pipe(adaptive=True)
        for Xc, yc in iter_chunks(self.X, self.y, 150):
            pipe.partial_fit(Xc, yc)
        batch_imp = Imputer(strategy="mean").fit(self.X)
        batch_scaler = StandardScaler().fit(batch_imp.transform(self.X))
        # streaming scaler saw imputed chunks, so compare means approximately
        np.testing.assert_allclose(pipe.steps[1][1].mean_, batch_scaler.mean_, atol=0.1)

    def test_predict_before_any_fit_raises_not_fitted(self):
        with self.assertRaises(NotFittedError):
            _pipe().predict(self.X[:5])

    def test_predict_before_first_retrain_raises_not_fitted(self):
        pipe = _pipe(buffer_size=500).partial_fit(self.X[:100], self.y[:100])
        with self.assertRaises(NotFittedError):
            pipe.predict(self.X[:5])

    def test_step_without_partial_fit_is_rejected(self):
        class OnlyFit:
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X

        pipe = Pipeline([("bad", OnlyFit()), ("tree", DecisionTreeClassifier())])
        with self.assertRaises(ValueError):
            pipe.partial_fit(self.X[:10], self.y[:10])

    def test_score_predict_proba_and_existing_fit_api(self):
        pipe = _pipe()
        pipe.fit(self.X, self.y)
        self.assertGreater(pipe.score(self.X, self.y), 0.7)
        proba = pipe.predict_proba(self.X[:10])
        np.testing.assert_allclose(proba.sum(1), 1.0)
        with self.assertRaises(ValueError):
            pipe.score(self.X[:5], self.y[:4])

    def test_bad_chunk_shape_propagates_clear_error(self):
        pipe = _pipe().partial_fit(self.X[:50], self.y[:50])
        with self.assertRaises(ValueError):
            pipe.partial_fit(self.X[:10, :3], self.y[:10])


class TestAdaptiveVsFixedPreprocessing(unittest.TestCase):
    """Why the buffered tree is combined with a fixed preprocessing step."""

    def setUp(self):
        self.X, self.y = make_drifting_data(1000, seed=3)

    def test_adaptive_scaler_maps_the_same_value_differently_over_time(self):
        pipe = _pipe(adaptive=True)
        probe = np.ones((1, 6))
        pipe.partial_fit(self.X[:100], self.y[:100])
        scaler = pipe.steps[1][1]
        first = scaler.transform(probe)
        for Xc, yc in iter_chunks(self.X[100:], self.y[100:], 100):
            pipe.partial_fit(Xc, yc)
        self.assertFalse(np.allclose(first, scaler.transform(probe), atol=1e-6))

    def test_fixed_preprocessing_never_changes_its_transformation(self):
        pipe = _pipe()
        chunk = self.X[:100]
        pipe.partial_fit(chunk, self.y[:100])
        imputer = pipe.steps[0][1]
        first = imputer.transform(chunk)
        for Xc, yc in iter_chunks(self.X[100:], self.y[100:], 100):
            pipe.partial_fit(Xc, yc)
        np.testing.assert_array_equal(imputer.transform(chunk), first)

    def test_predictions_are_stable_between_retrains_with_fixed_preprocessing(self):
        pipe = _pipe(buffer_size=300)
        for Xc, yc in iter_chunks(self.X[:300], self.y[:300], 100):
            pipe.partial_fit(Xc, yc)
        tree = pipe.steps[-1][1]
        self.assertEqual(tree.n_retrains_, 1)
        before = pipe.predict(self.X[600:700])
        for Xc, yc in iter_chunks(self.X[300:500], self.y[300:500], 100):
            pipe.partial_fit(Xc, yc)           # buffered only, no retrain yet
        self.assertEqual(tree.n_retrains_, 1)
        np.testing.assert_array_equal(pipe.predict(self.X[600:700]), before)


class TestStreamTrainer(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_drifting_data(1500, seed=2)

    def test_run_logs_one_record_per_chunk(self):
        trainer = StreamTrainer(_pipe(buffer_size=300)).run(iter_chunks(self.X, self.y, 100))
        self.assertEqual(len(trainer.history), 15)
        self.assertEqual(trainer.n_samples_, 1500)
        first = trainer.history[0]
        self.assertFalse(first["scored"])
        self.assertTrue(np.isnan(first["accuracy"]))
        self.assertTrue(all(r["memory_bytes"] > 0 for r in trainer.history))
        self.assertEqual(trainer.retrain_chunks(), [3, 6, 9, 12, 15])

    def test_prequential_accuracy_matches_manual_computation(self):
        pipe = _pipe(buffer_size=200)
        trainer = StreamTrainer(pipe)
        manual_pipe, hits, total = _pipe(buffer_size=200), 0, 0
        for Xc, yc in iter_chunks(self.X, self.y, 100):
            trainer.process_chunk(Xc, yc)
            try:
                hits += int((manual_pipe.predict(Xc) == yc).sum())
                total += len(yc)
            except NotFittedError:
                pass
            manual_pipe.partial_fit(Xc, yc)
        self.assertAlmostEqual(trainer.metrics["accuracy"].result(), hits / total)
        self.assertAlmostEqual(trainer.history[-1]["accuracy"], hits / total)

    def test_summary_series_and_arrays(self):
        trainer = StreamTrainer(_pipe(), rolling_window=200).run(iter_chunks(self.X, self.y, 250))
        s = trainer.summary()
        self.assertEqual((s["chunks"], s["samples"]), (6, 1500))
        self.assertGreater(s["total_fit_time_s"], 0)
        self.assertEqual(trainer.metric_series("accuracy").shape, (6,))
        self.assertEqual(trainer.history_arrays()["chunk"].tolist(), [1, 2, 3, 4, 5, 6])
        with self.assertRaises(KeyError):
            trainer.metric_series("nope")
        self.assertEqual(StreamTrainer(_pipe()).summary()["chunks"], 0)

    def test_score_and_fit_chunk_separately_and_last_predictions(self):
        trainer = StreamTrainer(_pipe(buffer_size=100))
        rec = trainer.fit_chunk(self.X[:100], self.y[:100])
        self.assertTrue(rec["retrained"])
        out = trainer.score_chunk(self.X[100:200], self.y[100:200])
        self.assertTrue(out["scored"])
        self.assertEqual(len(trainer.last_y_pred), 100)
        np.testing.assert_array_equal(trainer.last_y_true, self.y[100:200])

    def test_custom_metrics_reset_and_validation(self):
        trainer = StreamTrainer(_pipe(buffer_size=100),
                                metrics={"acc": StreamingAccuracy(), "prec": StreamingPrecision()})
        trainer.run(iter_chunks(self.X, self.y, 100))
        self.assertIn("prec", trainer.history[-1])
        trainer.reset()
        self.assertEqual((trainer.history, trainer.n_chunks_), ([], 0))
        self.assertTrue(np.isnan(trainer.metrics["acc"].result()))
        with self.assertRaises(ValueError):
            StreamTrainer(_pipe(), metrics={"x": object()})
        with self.assertRaises(ValueError):
            StreamTrainer(object())

    def test_bad_chunk_errors_and_logging(self):
        trainer = StreamTrainer(_pipe())
        with self.assertRaises(ValueError):
            trainer.process_chunk(self.X[:10], self.y[:5])
        with self.assertLogs("numcompute_stream.stream", level="INFO") as logs:
            trainer.process_chunk(self.X[:10], self.y[:10])
        self.assertIn("chunk 1", logs.output[0])

    def test_empty_chunk_is_logged_without_scoring(self):
        trainer = StreamTrainer(_pipe())
        rec = trainer.process_chunk(np.empty((0, 6)), np.empty(0, dtype=int))
        self.assertFalse(rec["scored"])
        self.assertEqual(rec["n_samples"], 0)


class TestHelpers(unittest.TestCase):
    def test_iter_chunks(self):
        chunks = list(iter_chunks(np.arange(10).reshape(5, 2), np.arange(5), 2))
        self.assertEqual([len(c[0]) for c in chunks], [2, 2, 1])
        self.assertEqual(len(list(iter_chunks(np.empty((0, 2)), chunk_size=3))), 0)
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                list(iter_chunks(np.ones((3, 1)), chunk_size=bad))
        with self.assertRaises(TypeError):
            list(iter_chunks(np.ones((3, 1)), chunk_size=True))
        with self.assertRaises(ValueError):
            list(iter_chunks(np.ones((3, 1)), np.ones(2), 1))

    def test_estimate_memory_matches_known_array_sizes(self):
        # An array that owns its buffer costs nbytes + a small header: the data is counted once.
        for arr in (np.zeros(1000), np.zeros(100_000, dtype=np.float32), np.zeros((50, 40))):
            with self.subTest(dtype=arr.dtype, shape=arr.shape):
                estimate = estimate_memory(arr)
                self.assertGreaterEqual(estimate, arr.nbytes)
                self.assertLess(estimate, arr.nbytes + 300)

    def test_estimate_memory_follows_views_to_their_base_and_counts_it_once(self):
        base = np.zeros(10_000)
        view, other_view = base[:10], base[5000:5100]
        owner = estimate_memory(base)
        # a tiny view still keeps the whole 80 kB buffer alive ...
        self.assertGreater(estimate_memory(view), base.nbytes)
        self.assertLess(estimate_memory(view), owner + 300)
        # ... and the buffer is counted once however many views / the owner are reachable
        self.assertLess(estimate_memory([base, view, other_view]), owner + 1000)

    def test_estimate_memory_sums_independent_parts_and_survives_cycles(self):
        a, b = np.zeros(1000), np.zeros(2000)
        self.assertGreaterEqual(estimate_memory({"a": a, "b": b}), a.nbytes + b.nbytes)
        self.assertGreater(estimate_memory([a, b.copy()]), estimate_memory([a, a]))
        loop = []
        loop.append(loop)
        self.assertGreater(estimate_memory(loop), 0)

    def test_datasets_deterministic_and_valid(self):
        a, ya = make_drifting_data(100, seed=5)
        b, yb = make_drifting_data(100, seed=5)
        np.testing.assert_array_equal(np.nan_to_num(a), np.nan_to_num(b))
        np.testing.assert_array_equal(ya, yb)
        for kwargs in ({"n_samples": 0}, {"n_features": 3}, {"drift_at": 2}, {"noise": -1}):
            with self.assertRaises(ValueError):
                make_drifting_data(**kwargs)


if __name__ == "__main__":
    unittest.main()

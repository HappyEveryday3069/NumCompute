import unittest

import numpy as np

from numcompute_stream.datasets import make_drifting_data
from numcompute_stream.tree import (DecisionTreeClassifier, best_split_loop,
                                    best_split_vectorised, impurity_from_counts)
from numcompute_stream.utils import NotFittedError


def _separable(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    return X, (X[:, 1] > 0.2).astype(int)


class TestImpurityAndSplit(unittest.TestCase):
    def test_impurity_values(self):
        self.assertAlmostEqual(float(impurity_from_counts([5, 5], "gini")), 0.5)
        self.assertAlmostEqual(float(impurity_from_counts([5, 5], "entropy")), 1.0)
        self.assertEqual(float(impurity_from_counts([7, 0], "gini")), 0.0)
        self.assertEqual(float(impurity_from_counts([0, 0], "entropy")), 0.0)
        with self.assertRaises(ValueError):
            impurity_from_counts([1, 1], "bad")

    def test_vectorised_equals_loop_with_ties(self):
        rng = np.random.default_rng(3)
        Xn = rng.integers(0, 5, size=(120, 4)).astype(float)  # many tied values
        Yn = np.eye(3)[rng.integers(0, 3, 120)]
        for crit in ("gini", "entropy"):
            for leaf in (1, 5):
                self.assertEqual(best_split_vectorised(Xn, Yn, range(4), crit, leaf),
                                 best_split_loop(Xn, Yn, range(4), crit, leaf))

    def test_no_split_when_constant_or_too_small(self):
        Xn, Yn = np.ones((10, 2)), np.eye(2)[np.arange(10) % 2]
        self.assertIsNone(best_split_vectorised(Xn, Yn, range(2)))
        Xn2 = np.arange(4.0).reshape(-1, 1)
        self.assertIsNone(best_split_vectorised(Xn2, np.eye(2)[[0, 1, 0, 1]], [0], "gini", 3))

    def test_threshold_is_midpoint(self):
        f, thr, gain = best_split_vectorised(np.array([[0.0], [1.0], [2.0], [3.0]]),
                                             np.eye(2)[[0, 0, 1, 1]], [0])
        self.assertEqual((f, thr), (0, 1.5))
        self.assertAlmostEqual(gain, 0.5)


class TestFitPredict(unittest.TestCase):
    def test_learns_separable_rule(self):
        X, y = _separable()
        tree = DecisionTreeClassifier(max_depth=1).fit(X, y)
        self.assertEqual(tree.score(X, y), 1.0)
        self.assertEqual(tree.feature_[0], 1)
        self.assertAlmostEqual(tree.threshold_[0], 0.2, delta=0.3)

    def test_xor_requires_depth_two(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 10, dtype=float)
        y = np.array([0, 1, 1, 0] * 10)
        self.assertEqual(DecisionTreeClassifier(max_depth=2).fit(X, y).score(X, y), 1.0)
        self.assertLess(DecisionTreeClassifier(max_depth=1).fit(X, y).score(X, y), 1.0)

    def test_max_depth_zero_and_depth_limit(self):
        X, y = make_drifting_data(400, missing_rate=0, seed=1)
        self.assertEqual(DecisionTreeClassifier(max_depth=0).fit(X, y).get_n_leaves(), 1)
        for d in (1, 3, 5):
            self.assertLessEqual(DecisionTreeClassifier(max_depth=d).fit(X, y).get_depth(), d)

    def test_min_samples_split_and_leaf(self):
        X, y = make_drifting_data(300, missing_rate=0, seed=2)
        tree = DecisionTreeClassifier(min_samples_split=50, min_samples_leaf=20).fit(X, y)
        leaves = tree.feature_ < 0
        self.assertTrue((tree.n_node_samples_[leaves] >= 20).all())
        internal = ~leaves
        self.assertTrue((tree.n_node_samples_[internal] >= 50).all())

    def test_pure_data_gives_single_leaf(self):
        tree = DecisionTreeClassifier().fit(np.random.rand(10, 2), np.zeros(10, int))
        self.assertEqual(tree.get_n_leaves(), 1)
        np.testing.assert_array_equal(tree.predict(np.random.rand(3, 2)), [0, 0, 0])

    def test_identical_features_conflicting_labels_majority_and_tie(self):
        X = np.ones((4, 1))
        tree = DecisionTreeClassifier().fit(X, np.array([1, 1, 2, 2]))
        self.assertEqual(tree.predict([[1.0]])[0], 1)  # tie -> smallest label
        tree = DecisionTreeClassifier().fit(X, np.array([2, 2, 2, 1]))
        self.assertEqual(tree.predict([[1.0]])[0], 2)

    def test_string_labels_and_proba(self):
        X, y = _separable()
        labels = np.where(y == 1, "yes", "no")
        tree = DecisionTreeClassifier(max_depth=3).fit(X, labels)
        self.assertEqual(set(tree.predict(X)), {"yes", "no"})
        proba = tree.predict_proba(X)
        np.testing.assert_allclose(proba.sum(1), 1.0)
        np.testing.assert_array_equal(tree.classes_, ["no", "yes"])

    def test_entropy_and_gini_both_work(self):
        X, y = make_drifting_data(500, missing_rate=0, noise=0, drift_at=1.0, seed=3)
        for crit in ("gini", "entropy"):
            self.assertGreater(DecisionTreeClassifier(criterion=crit, max_depth=6).fit(X, y).score(X, y), 0.9)

    def test_predict_loop_matches_vectorised(self):
        X, y = make_drifting_data(600, seed=4)
        tree = DecisionTreeClassifier(max_depth=8).fit(X, y)
        np.testing.assert_array_equal(tree.predict(X), tree.predict_loop(X))

    def test_deterministic_and_max_features(self):
        X, y = make_drifting_data(500, missing_rate=0, seed=5)
        kw = dict(max_depth=5, max_features="sqrt", random_state=7)
        a, b = DecisionTreeClassifier(**kw).fit(X, y), DecisionTreeClassifier(**kw).fit(X, y)
        np.testing.assert_array_equal(a.feature_, b.feature_)
        np.testing.assert_array_equal(a.threshold_, b.threshold_)
        for mf in (1, 0.5, "log2"):
            self.assertGreater(DecisionTreeClassifier(max_depth=4, max_features=mf, random_state=0)
                               .fit(X, y).score(X, y), 0.5)

    def test_nan_in_training_and_prediction(self):
        X, y = _separable()
        X[::7, 0] = np.nan
        tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
        Xt = X.copy()
        Xt[:, 1] = np.nan  # the informative feature is missing at predict time
        self.assertEqual(tree.predict(Xt).shape, (200,))
        allnan = np.full((6, 2), np.nan)
        DecisionTreeClassifier().fit(allnan, np.array([0, 1] * 3))  # must not crash

    def test_feature_importances_and_export(self):
        X, y = _separable()
        tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
        imp = tree.feature_importances_
        self.assertAlmostEqual(imp.sum(), 1.0)
        self.assertEqual(int(np.argmax(imp)), 1)
        text = tree.export_text(["a", "b", "c"])
        self.assertIn("b <=", text)
        self.assertIn("class:", text)
        self.assertTrue(np.all(DecisionTreeClassifier(max_depth=0).fit(X, y).feature_importances_ == 0))
        with self.assertRaises(ValueError):
            tree.export_text(["only_one"])


class TestValidation(unittest.TestCase):
    def test_bad_hyperparameters(self):
        for kwargs in ({"criterion": "mse"}, {"max_depth": -1}, {"max_depth": 1.5},
                       {"min_samples_split": 1}, {"min_samples_leaf": 0},
                       {"buffer_size": 0}, {"max_features": 0}, {"max_features": "all"},
                       {"max_features": 1.5}, {"min_impurity_decrease": -0.1},
                       {"buffer_size": True}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                DecisionTreeClassifier(**kwargs)

    def test_bad_data(self):
        tree = DecisionTreeClassifier()
        with self.assertRaises(ValueError):
            tree.fit(np.ones(5), np.ones(5))               # 1D X
        with self.assertRaises(ValueError):
            tree.fit(np.ones((5, 2)), np.ones(4))          # length mismatch
        with self.assertRaises(ValueError):
            tree.fit(np.ones((5, 2)), np.full(5, np.nan))  # NaN labels
        with self.assertRaises(ValueError):
            tree.fit(np.full((5, 2), np.inf), np.ones(5))  # inf
        with self.assertRaises(ValueError):
            tree.fit(np.empty((0, 2)), np.empty(0))        # no samples
        with self.assertRaises(ValueError):
            tree.fit([["a", "b"]], [1])                    # non-numeric

    def test_predict_before_fit_and_wrong_width(self):
        with self.assertRaises(NotFittedError):
            DecisionTreeClassifier().predict(np.ones((2, 2)))
        X, y = _separable()
        tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
        with self.assertRaises(ValueError):
            tree.predict(X[:, :2])
        self.assertEqual(tree.predict(np.empty((0, 3))).shape, (0,))


class TestBufferedStreaming(unittest.TestCase):
    def test_retrains_only_when_buffer_full_and_clears_buffer(self):
        X, y = make_drifting_data(1000, missing_rate=0, seed=6)
        tree = DecisionTreeClassifier(max_depth=4, buffer_size=300)
        tree.partial_fit(X[:100], y[:100])
        tree.partial_fit(X[100:200], y[100:200])
        self.assertFalse(tree.is_fitted)
        self.assertEqual((tree.n_buffered_samples, tree.n_retrains_), (200, 0))
        tree.partial_fit(X[200:300], y[200:300])
        self.assertEqual((tree.n_buffered_samples, tree.n_retrains_), (0, 1))
        self.assertEqual(tree.n_samples_trained_, 300)

    def test_predict_before_first_retrain_raises(self):
        tree = DecisionTreeClassifier(buffer_size=100).partial_fit(np.ones((10, 2)), np.ones(10))
        with self.assertRaises(NotFittedError):
            tree.predict(np.ones((1, 2)))

    def test_predictions_use_latest_tree_between_retrains(self):
        X, y = make_drifting_data(600, missing_rate=0, noise=0, seed=7)
        tree = DecisionTreeClassifier(max_depth=5, buffer_size=200)
        tree.partial_fit(X[:200], y[:200])
        snapshot = tree.predict(X[400:])
        tree.partial_fit(X[200:300], y[200:300])  # buffered, no retrain
        np.testing.assert_array_equal(tree.predict(X[400:]), snapshot)
        tree.partial_fit(X[300:400], y[300:400])  # second retrain
        self.assertEqual(tree.n_retrains_, 2)

    def test_retrain_uses_only_buffered_data(self):
        X, y = _separable(100)
        tree = DecisionTreeClassifier(buffer_size=50)
        tree.partial_fit(X[:50], y[:50])
        tree.partial_fit(X[50:], y[50:])
        reference = DecisionTreeClassifier().fit(X[50:], y[50:])
        np.testing.assert_array_equal(tree.feature_, reference.feature_)
        np.testing.assert_array_equal(tree.threshold_, reference.threshold_)

    def test_chunk_larger_than_buffer_and_many_small_chunks(self):
        X, y = make_drifting_data(500, missing_rate=0, seed=8)
        big = DecisionTreeClassifier(buffer_size=50).partial_fit(X[:400], y[:400])
        self.assertEqual((big.n_retrains_, big.n_samples_trained_), (1, 400))
        small = DecisionTreeClassifier(buffer_size=100)
        for i in range(0, 300, 10):
            small.partial_fit(X[i:i + 10], y[i:i + 10])
        self.assertEqual((small.n_retrains_, small.n_buffered_samples), (3, 0))

    def test_flush_and_empty_chunk(self):
        X, y = _separable(60)
        tree = DecisionTreeClassifier(buffer_size=1000)
        tree.partial_fit(X, y).partial_fit(np.empty((0, 3)), np.empty(0))
        self.assertEqual(tree.n_buffered_samples, 60)
        tree.flush()
        self.assertTrue(tree.is_fitted)
        self.assertEqual(tree.n_buffered_samples, 0)
        self.assertEqual(tree.flush().n_retrains_, 1)  # empty buffer: no-op

    def test_fit_clears_buffer(self):
        X, y = _separable(40)
        tree = DecisionTreeClassifier(buffer_size=1000).partial_fit(X, y)
        tree.fit(X, y)
        self.assertEqual((tree.n_buffered_samples, tree.n_retrains_), (0, 1))

    def test_feature_count_change_rejected_by_default(self):
        tree = DecisionTreeClassifier(buffer_size=10).partial_fit(np.ones((3, 2)), [0, 1, 0])
        with self.assertRaises(ValueError):
            tree.partial_fit(np.ones((3, 4)), [0, 1, 0])

    def test_feature_growth_mode_pads(self):
        tree = DecisionTreeClassifier(buffer_size=6, allow_feature_growth=True, max_depth=3)
        tree.partial_fit(np.array([[0.0], [1.0], [0.0]]), [0, 1, 0])
        tree.partial_fit(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), [1, 0, 1])
        self.assertEqual(tree.n_features_, 2)
        self.assertEqual(tree.predict(np.array([[1.0]])).shape, (1,))       # narrower: padded
        self.assertEqual(tree.predict(np.ones((1, 5))).shape, (1,))         # wider: truncated

    def test_streaming_accuracy_on_drift_with_small_buffer_beats_large(self):
        X, y = make_drifting_data(4000, missing_rate=0, noise=0, drift_at=0.5, seed=9)

        def prequential(buffer_size):
            tree, correct, total = DecisionTreeClassifier(max_depth=6, buffer_size=buffer_size), 0, 0
            for i in range(0, 4000, 100):
                xb, yb = X[i:i + 100], y[i:i + 100]
                if tree.is_fitted:
                    correct += int((tree.predict(xb) == yb).sum())
                    total += 100
                tree.partial_fit(xb, yb)
            return correct / total

        self.assertGreater(prequential(200), prequential(2000))


if __name__ == "__main__":
    unittest.main()

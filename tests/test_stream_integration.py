"""Cross-module integration tests: several streaming components used together."""

import unittest
import warnings

import numpy as np

from numcompute_stream import (DecisionTreeClassifier, Imputer, OneHotEncoder, Pipeline,
                               StandardScaler, StreamTrainer, iter_chunks)
from numcompute_stream.datasets import make_drifting_data
from numcompute_stream.metrics import (StreamingAccuracy, StreamingAUC,
                                       StreamingConfusionMatrix, auc, roc_curve)
from numcompute_stream.tree import _midpoint, best_split_loop, best_split_vectorised
from numcompute_stream.utils import NotFittedError


class TestScalerBeforeBufferedTree(unittest.TestCase):
    """An adaptive scaler in front of a buffered tree invalidates the tree between retrains."""

    @staticmethod
    def _stream():
        rng = np.random.default_rng(0)

        def make(lo, hi):
            x = rng.uniform(lo, hi, size=(100, 1))
            return x, (x[:, 0] > 0.5).astype(int)

        return [make(0, 1) for _ in range(3)], [make(100, 101) for _ in range(2)]

    def _predictions_before_and_after_more_chunks(self, pipe):
        first, later = self._stream()
        probes = np.array([[0.1], [0.3], [0.7], [0.9]])
        for X, y in first:
            pipe.partial_fit(X, y)
        tree = pipe.steps[-1][1]
        self.assertEqual(tree.n_retrains_, 1)
        before = pipe.predict(probes)
        for X, y in later:
            pipe.partial_fit(X, y)  # 200 buffered samples < buffer_size: the tree is not retrained
        self.assertEqual(tree.n_retrains_, 1)
        return before, pipe.predict(probes)

    def test_fixed_preprocessing_keeps_predictions_between_retrains(self):
        pipe = Pipeline([("impute", Imputer(strategy="constant")),
                         ("tree", DecisionTreeClassifier(max_depth=3, buffer_size=300))])
        before, after = self._predictions_before_and_after_more_chunks(pipe)
        np.testing.assert_array_equal(before, [0, 0, 1, 1])
        np.testing.assert_array_equal(after, before)

    def test_adaptive_scaler_changes_predictions_between_retrains(self):
        pipe = Pipeline([("scale", StandardScaler()),
                         ("tree", DecisionTreeClassifier(max_depth=3, buffer_size=300))])
        before, after = self._predictions_before_and_after_more_chunks(pipe)
        np.testing.assert_array_equal(before, [0, 0, 1, 1])
        self.assertFalse(np.array_equal(before, after),
                         "the shifted scaler should move the probes across the old thresholds")


class TestOneHotEncoderWithTree(unittest.TestCase):
    """New categories widen the encoded input; the tree must follow (allow_feature_growth)."""

    @staticmethod
    def _chunks():
        # column 1 ("kind") decides the label; column 0 ("site") is noise that gains the
        # new category "c" in chunk 3. "kind" is the *later* feature, so a layout that
        # inserted the new column in the middle would shift its columns.
        rng = np.random.default_rng(1)

        def chunk(sites):
            site = rng.choice(sites, size=20)
            kind = rng.choice(["x", "y"], size=20)
            site[0], kind[0], kind[1] = sites[-1], "x", "y"
            site[1] = sites[0]
            return np.column_stack([site, kind]).astype(object), (kind == "y").astype(int)

        return [chunk(["a", "b"]), chunk(["a", "b"]), chunk(["a", "b", "c"]), chunk(["a", "b", "c"])]

    @staticmethod
    def _pipe(grow=True):
        return Pipeline([("ohe", OneHotEncoder()),
                         ("tree", DecisionTreeClassifier(max_depth=4, buffer_size=40,
                                                         allow_feature_growth=grow))])

    def test_new_category_widens_input_and_tree_follows(self):
        pipe, chunks = self._pipe(), self._chunks()
        ohe, tree = pipe.steps[0][1], pipe.steps[1][1]
        for X, y in chunks[:2]:
            pipe.partial_fit(X, y)
        self.assertEqual((ohe.n_output_features_, tree.n_features_, tree.n_retrains_), (4, 4, 1))

        pipe.partial_fit(*chunks[2])           # "c" appears: encoder widens, tree still on 4 columns
        self.assertEqual((ohe.n_output_features_, tree.n_features_), (5, 4))
        X3, y3 = chunks[3]
        # between retrains the old tree keeps working: the "kind" columns did not move
        np.testing.assert_array_equal(pipe.predict(X3), y3)

        pipe.partial_fit(X3, y3)               # buffer full again: retrain on the wider input
        self.assertEqual((tree.n_features_, tree.n_retrains_), (5, 2))
        self.assertEqual(pipe.score(X3, y3), 1.0)

    def test_existing_columns_do_not_move_when_a_category_is_added(self):
        X1, _ = self._chunks()[0]
        enc = OneHotEncoder().fit(X1)
        before = enc.transform(X1)
        enc.partial_fit(self._chunks()[2][0])
        after = enc.transform(X1)
        self.assertEqual(after.shape[1], before.shape[1] + 1)
        np.testing.assert_array_equal(after[:, :-1], before)
        self.assertTrue((after[:, -1] == 0).all())  # nothing in X1 has the new category "c"

    def test_tree_without_growth_rejects_the_wider_input(self):
        pipe, chunks = self._pipe(grow=False), self._chunks()
        for X, y in chunks[:2]:
            pipe.partial_fit(X, y)
        with self.assertRaisesRegex(ValueError, "features"):
            pipe.partial_fit(*chunks[2])

    def test_stream_trainer_runs_end_to_end_with_string_categories(self):
        trainer = StreamTrainer(self._pipe(), rolling_window=40).run(self._chunks())
        self.assertEqual(len(trainer.history), 4)
        self.assertEqual(trainer.retrain_chunks(), [2, 4])
        self.assertEqual([r["scored"] for r in trainer.history], [False, False, True, True])
        self.assertEqual(trainer.metrics["accuracy"].result(), 1.0)


class TestTrainerWithAUC(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_drifting_data(2000, seed=4)

    @staticmethod
    def _pipe():
        return Pipeline([("impute", Imputer(strategy="constant")),
                         ("tree", DecisionTreeClassifier(max_depth=5, buffer_size=300, random_state=0))])

    def _manual_prequential(self):
        """Reference: score each chunk before training and collect labels/scores of class 0."""
        pipe, labels, scores, hard = self._pipe(), [], [], []
        for Xc, yc in iter_chunks(self.X, self.y, 100):
            try:
                proba = pipe.predict_proba(Xc)
                pred = pipe.predict(Xc)
            except NotFittedError:
                pass
            else:
                labels.append((yc == 0).astype(int))
                scores.append(proba[:, list(pipe.classes_).index(0)])
                hard.append((pred == 0).astype(float))
            pipe.partial_fit(Xc, yc)
        return np.concatenate(labels), np.concatenate(scores), np.concatenate(hard)

    def _trainer(self):
        metrics = {"accuracy": StreamingAccuracy(),
                   "auc0": StreamingAUC(pos_label=0),
                   "auc0_binned": StreamingAUC(pos_label=0, n_bins=500),
                   "confusion": StreamingConfusionMatrix()}
        return StreamTrainer(self._pipe(), metrics=metrics).run(iter_chunks(self.X, self.y, 100))

    def test_probabilities_reach_the_auc_metric(self):
        trainer = self._trainer()
        labels, scores, hard = self._manual_prequential()
        fpr, tpr, _ = roc_curve(labels, scores)
        expected = auc(fpr, tpr)
        self.assertAlmostEqual(trainer.metrics["auc0"].result(), expected, places=10)
        self.assertAlmostEqual(trainer.history[-1]["auc0"], expected, places=10)
        self.assertAlmostEqual(trainer.metrics["auc0_binned"].result(), expected, delta=0.01)
        # feeding hard 0/1 predictions instead of probabilities would give a different AUC
        hard_auc = auc(*roc_curve(labels, hard)[:2])
        self.assertGreater(abs(expected - hard_auc), 1e-3)

    def test_auc_series_is_nan_until_the_model_can_predict(self):
        trainer = self._trainer()
        series = trainer.metric_series("auc0")
        self.assertTrue(np.isnan(series[:3]).all())      # buffer_size 300 -> first scored chunk is #4
        self.assertTrue(np.isfinite(series[3:]).all())

    def test_confusion_matrix_is_updated_but_not_logged(self):
        trainer = self._trainer()
        self.assertNotIn("confusion", trainer.history[0])
        self.assertNotIn("confusion", trainer.summary())
        scored = sum(r["n_samples"] for r in trainer.history if r["scored"])
        self.assertEqual(int(trainer.metrics["confusion"].result().sum()), scored)

    def test_probability_metric_requires_predict_proba(self):
        class NoProba:
            def partial_fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(len(X))

        with self.assertRaisesRegex(ValueError, "predict_proba"):
            StreamTrainer(NoProba(), metrics={"auc": StreamingAUC()})

    def test_update_proba_validation_and_unseen_positive_class(self):
        proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
        with self.assertRaises(ValueError):
            StreamingAUC().update_proba([0, 1, 0], proba[:, :1], classes=[0, 1])
        with self.assertRaises(ValueError):
            StreamingAUC().update_proba([0, 1], proba, classes=[0, 1])
        m = StreamingAUC(pos_label=7).update_proba([0, 1, 0], proba, classes=[0, 1])
        self.assertTrue(np.isnan(m.result()))              # class 7 never present: single-class stream


class TestExtremeValues(unittest.TestCase):
    PAIRS = [(-1.7e308, 1.7e308), (1e308, 1.7e308), (-1.7e308, -1e308), (-1e308, 1e308),
             (5e-324, 1e-323), (0.0, 5e-324), (-5e-324, 5e-324), (1.0, np.nextafter(1.0, 2.0)),
             (-np.finfo(float).max, np.finfo(float).max), (-1e-300, 1e-300)]

    def test_midpoint_is_finite_and_between_the_two_values(self):
        for a, b in self.PAIRS:
            with self.subTest(a=a, b=b), warnings.catch_warnings():
                warnings.simplefilter("error")
                thr = _midpoint(a, b)
                self.assertTrue(np.isfinite(thr))
                self.assertTrue(a <= thr < b)

    def test_the_naive_formula_overflows_for_the_same_pairs(self):
        with np.errstate(over="ignore"):
            self.assertTrue(np.isinf(-1.7e308 + (1.7e308 - -1.7e308) / 2.0))

    def test_tree_separates_opposite_extremes_without_warnings(self):
        X = np.array([[-1.7e308], [-1.0e308], [1.0e308], [1.7e308]] * 5)
        y = np.array([0, 0, 1, 1] * 5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
            np.testing.assert_array_equal(tree.predict(X), y)
        self.assertTrue(np.isfinite(tree.threshold_[0]))
        self.assertTrue(-1e308 <= tree.threshold_[0] < 1e308)

    def test_vectorised_and_loop_split_search_agree_on_extremes(self):
        rng = np.random.default_rng(0)
        values = np.array([-1.7e308, -1e300, -1e-300, 0.0, 1e-300, 1e300, 1.7e308, 5e-324])
        Xn = rng.choice(values, size=(80, 3))
        Yn = np.eye(3)[rng.integers(0, 3, 80)]
        for criterion in ("gini", "entropy"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                self.assertEqual(best_split_vectorised(Xn, Yn, range(3), criterion, 2),
                                 best_split_loop(Xn, Yn, range(3), criterion, 2))

    def test_nan_fill_median_does_not_overflow_near_float_max(self):
        X = np.full((10, 1), 1.7e308)
        X[0, 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tree = DecisionTreeClassifier().fit(X, [0, 1] * 5)
        self.assertEqual(tree.nan_fill_[0], 1.7e308)

    def test_buffered_pipeline_streams_mixed_magnitudes_without_warnings(self):
        rng = np.random.default_rng(3)
        scale = np.array([1e-300, 1.0, 1e300])
        X = rng.normal(size=(600, 3)) * scale
        y = (X[:, 1] > 0).astype(int)
        pipe = Pipeline([("impute", Imputer(strategy="constant")),
                         ("tree", DecisionTreeClassifier(max_depth=4, buffer_size=200))])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            trainer = StreamTrainer(pipe).run(iter_chunks(X, y, 100))
        self.assertGreater(trainer.metrics["accuracy"].result(), 0.9)


if __name__ == "__main__":
    unittest.main()

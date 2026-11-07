import unittest
import warnings

import numpy as np

from numcompute_stream import metrics as M


class TestStreamingClassificationMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.yt = rng.integers(0, 3, 400)
        self.yp = np.where(rng.random(400) < 0.7, self.yt, rng.integers(0, 3, 400))

    def _stream(self, metric):
        for a, b in zip(np.array_split(self.yt, 9), np.array_split(self.yp, 9)):
            metric.update(a, b)
        return metric

    def test_accuracy_matches_batch(self):
        self.assertAlmostEqual(self._stream(M.StreamingAccuracy()).result(),
                               M.accuracy(self.yt, self.yp))

    def test_confusion_matrix_matches_batch(self):
        cm = self._stream(M.StreamingConfusionMatrix()).result()
        np.testing.assert_array_equal(cm, M.confusion_matrix(self.yt, self.yp))

    def test_confusion_matrix_grows_with_new_classes(self):
        m = M.StreamingConfusionMatrix()
        m.update([0, 1], [0, 1])
        m.update([2, 0], [1, 2])
        np.testing.assert_array_equal(m.classes_, [0, 1, 2])
        np.testing.assert_array_equal(m.result(), [[1, 0, 1], [0, 1, 0], [0, 1, 0]])

    def test_string_labels(self):
        m = M.StreamingAccuracy().update(["a", "b"], ["a", "a"])
        self.assertAlmostEqual(m.result(), 0.5)

    def test_prf_match_batch_macro_and_micro(self):
        for avg in ("macro", "micro"):
            p, r, f = M.precision_recall_f1(self.yt, self.yp, average=avg)
            self.assertAlmostEqual(self._stream(M.StreamingPrecision(average=avg)).result(), p)
            self.assertAlmostEqual(self._stream(M.StreamingRecall(average=avg)).result(), r)
            self.assertAlmostEqual(self._stream(M.StreamingF1(average=avg)).result(), f)

    def test_weighted_and_binary_average(self):
        yt, yp = [0, 0, 0, 1], [0, 0, 1, 1]
        # class0: P=1 R=2/3 ; class1: P=1/2 R=1
        self.assertAlmostEqual(M.StreamingRecall("weighted").update(yt, yp).result(), 3 / 4 * 2 / 3 + 1 / 4)
        self.assertAlmostEqual(M.StreamingPrecision("binary", pos_label=1).update(yt, yp).result(), 0.5)
        self.assertEqual(M.StreamingPrecision("binary", pos_label=7).update(yt, yp).result(), 0.0)

    def test_zero_division_value(self):
        m = M.StreamingPrecision(average="binary", pos_label=1, zero_division=0.25)
        m.update([0, 0], [0, 0])  # class 1 never present -> not in classes -> fallback
        self.assertEqual(m.result(), 0.25)
        m2 = M.StreamingPrecision(average="macro", zero_division=1.0).update([0, 1], [0, 0])
        self.assertAlmostEqual(m2.result(), (0.5 + 1.0) / 2)

    def test_rolling_window_only_uses_recent_samples(self):
        m = M.StreamingAccuracy(window=4)
        m.update([1, 1, 1, 1], [0, 0, 0, 0])
        m.update([1, 1], [1, 1])
        self.assertAlmostEqual(m.result(), 2 / 4)
        m.update([1, 1, 1, 1], [1, 1, 1, 1])
        self.assertEqual(m.result(), 1.0)
        self.assertEqual(m.n_samples_seen_, 10)

    def test_rolling_f1_and_confusion(self):
        cm = M.StreamingConfusionMatrix(window=3)
        cm.update([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(cm.result().sum(), 3)
        np.testing.assert_array_equal(cm.result(), [[0, 1], [0, 2]])

    def test_reset_and_empty_state(self):
        m = M.StreamingAccuracy()
        self.assertTrue(np.isnan(m.result()))
        m.update([1], [1]).reset()
        self.assertTrue(np.isnan(m.result()))
        self.assertEqual(m.n_samples_seen_, 0)

    def test_empty_update_noop(self):
        m = M.StreamingAccuracy().update([], [])
        self.assertEqual(m.n_samples_seen_, 0)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            M.StreamingAccuracy().update([1, 2], [1])
        with self.assertRaises(ValueError):
            M.StreamingAccuracy().update([np.nan], [1.0])
        with self.assertRaises(ValueError):
            M.StreamingAccuracy(window=0)
        with self.assertRaises(ValueError):
            M.StreamingF1(average="bogus")


class TestBatchMetricValidation(unittest.TestCase):
    def test_length_mismatch_is_reported_by_every_batch_metric(self):
        for func in (M.accuracy, M.confusion_matrix, M.precision_recall_f1, M.mse, M.roc_curve):
            with self.subTest(func=func.__name__), self.assertRaises(ValueError):
                func([0, 1, 1], [0, 1])

    def test_label_vectors_must_be_one_dimensional(self):
        with self.assertRaises(ValueError):
            M.accuracy(np.zeros((3, 2)), np.zeros((3, 2)))
        self.assertEqual(M.accuracy([[0], [1], [1]], [0, 1, 0]), 2 / 3)  # (n, 1) is accepted

    def test_mse_rejects_broadcasting_instead_of_returning_an_n_by_n_result(self):
        with self.assertRaises(ValueError):
            M.mse(np.arange(3.0), np.arange(3.0).reshape(3, 1))
        self.assertEqual(M.mse(np.zeros((2, 2)), np.ones((2, 2))), 1.0)  # multi-output is fine

    def test_empty_inputs_give_nan_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertTrue(np.isnan(M.accuracy([], [])))
            self.assertTrue(np.isnan(M.mse([], [])))
            self.assertTrue(all(np.isnan(v) for v in M.precision_recall_f1([], [])))
        with self.assertRaises(ValueError):
            M.roc_curve([], [])

    def test_precision_recall_f1_zero_denominators_are_silent_and_correct(self):
        y_true, y_pred = [0, 0, 1, 2], [0, 0, 0, 0]      # classes 1 and 2 are never predicted
        with warnings.catch_warnings():
            warnings.simplefilter("error")                  # np.where(..., a / b, 0) would warn here
            macro = M.precision_recall_f1(y_true, y_pred, "macro")
            micro = M.precision_recall_f1(y_true, y_pred, "micro")
            weighted = M.precision_recall_f1(y_true, y_pred, "weighted")
        np.testing.assert_allclose(macro, (0.5 / 3, 1 / 3, (2 / 3) / 3))
        np.testing.assert_allclose(micro, (0.5, 0.5, 0.5))
        np.testing.assert_allclose(weighted, (0.25, 0.5, 1 / 3))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            M.precision_recall_f1([0, 1], [0, 1], average="binary")
        for bad in ([-1, 0], [0.5, 1.0], [np.nan, 1.0], ["a", "b"]):
            with self.subTest(labels=bad), self.assertRaises(ValueError):
                M.confusion_matrix(bad, [0, 1])
        for bad in (2, 0, 1.5, True):
            with self.subTest(num_classes=bad), self.assertRaises(ValueError):
                M.confusion_matrix([0, 2], [0, 1], num_classes=bad)
        np.testing.assert_array_equal(M.confusion_matrix([0, 1], [0, 1], num_classes=3),
                                      np.diag([1, 1, 0]))

    def test_roc_curve_and_auc_validation(self):
        for labels, scores in (([0, 2], [0.1, 0.2]), ([0, 1], [0.1, np.nan]), ([0, 1], [0.1, np.inf])):
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                M.roc_curve(labels, scores)
        fpr, tpr, thr = M.roc_curve([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.3])
        self.assertEqual(thr[0], np.inf)
        self.assertEqual(M.auc(fpr, tpr), 1.0)
        for x, y in (([0.0], [0.0]), ([0.0, 1.0, 0.5], [0, 1, 1]), ([0.0, 1.0], [0.0])):
            with self.subTest(x=x), self.assertRaises(ValueError):
                M.auc(x, y)

    def test_roc_curve_with_a_single_class_has_all_zero_rates_for_the_missing_class(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fpr, tpr, _ = M.roc_curve([1, 1], [0.2, 0.9])
        np.testing.assert_array_equal(fpr, np.zeros(3))
        self.assertEqual(tpr[-1], 1.0)


class TestStreamingAUC(unittest.TestCase):
    def test_perfect_and_inverted(self):
        self.assertEqual(M.StreamingAUC().update([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]).result(), 1.0)
        self.assertEqual(M.StreamingAUC().update([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]).result(), 0.0)

    def test_known_value_with_ties_across_chunks(self):
        m = M.StreamingAUC()
        m.update([0, 1], [0.5, 0.5])
        m.update([1, 0], [0.9, 0.1])
        # pairs (pos,neg): (0.5,0.5)=0.5 (0.5,0.1)=1 (0.9,0.5)=1 (0.9,0.1)=1 -> 3.5/4
        self.assertAlmostEqual(m.result(), 0.875)

    def test_matches_trapezoid_auc(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 300)
        s = rng.random(300) + 0.3 * y
        fpr, tpr, _ = M.roc_curve(y, s)
        m = M.StreamingAUC()
        for a, b in zip(np.array_split(y, 6), np.array_split(s, 6)):
            m.update(a, b)
        self.assertAlmostEqual(m.result(), M.auc(fpr, tpr), places=10)

    def test_binned_approximation_close_to_exact(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 2000)
        s = np.clip(rng.random(2000) * 0.7 + 0.3 * y, 0, 1)
        exact, approx = M.StreamingAUC().update(y, s), M.StreamingAUC(n_bins=500).update(y, s)
        self.assertAlmostEqual(exact.result(), approx.result(), delta=0.01)

    def test_single_class_gives_nan_and_window(self):
        self.assertTrue(np.isnan(M.StreamingAUC().update([1, 1], [0.2, 0.3]).result()))
        m = M.StreamingAUC(window=2)
        m.update([0, 1, 0, 1], [0.9, 0.1, 0.1, 0.9])
        self.assertEqual(m.result(), 1.0)  # only the last two samples remain

    def test_validation(self):
        with self.assertRaises(ValueError):
            M.StreamingAUC().update([0, 2], [0.1, 0.2])
        with self.assertRaises(ValueError):
            M.StreamingAUC().update([0, 1], [0.1, np.nan])
        with self.assertRaises(ValueError):
            M.StreamingAUC(n_bins=10).update([0, 1], [0.1, 1.5])
        with self.assertRaises(ValueError):
            M.StreamingAUC(window=5, n_bins=10)
        with self.assertRaises(ValueError):
            M.StreamingAUC().update([0, 1], [0.1])


class TestStreamingMSE(unittest.TestCase):
    def test_matches_batch_and_window(self):
        y, p = np.arange(10.0), np.arange(10.0) + 1
        m = M.StreamingMSE()
        m.update(y[:4], p[:4]).update(y[4:], p[4:])
        self.assertAlmostEqual(m.result(), M.mse(y, p))
        w = M.StreamingMSE(window=2).update([0, 0, 0], [1, 1, 3])
        self.assertAlmostEqual(w.result(), (1 + 9) / 2)
        self.assertTrue(np.isnan(M.StreamingMSE().result()))
        with self.assertRaises(ValueError):
            m.update([1, 2], [1])


if __name__ == "__main__":
    unittest.main()

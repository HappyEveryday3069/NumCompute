"""Tests for the fourth review round: zero-gain splits, object-label NaN checks,
min_impurity_decrease validation and update_proba validation."""

import unittest

import numpy as np

from numcompute_stream import DecisionTreeClassifier
from numcompute_stream.metrics import (StreamingAUC, accuracy, confusion_matrix,
                                       precision_recall_f1)
from numcompute_stream.utils import check_y, has_missing

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_Y = np.array([0, 1, 1, 0])


class TestZeroGainSplits(unittest.TestCase):
    def test_default_learns_xor(self):
        tree = DecisionTreeClassifier().fit(XOR_X, XOR_Y)
        self.assertEqual(tree.score(XOR_X, XOR_Y), 1.0)
        self.assertGreater(tree.get_depth(), 1)

    def test_strict_course_rule_stops_at_root(self):
        tree = DecisionTreeClassifier(allow_zero_gain_splits=False).fit(XOR_X, XOR_Y)
        self.assertEqual(tree.get_n_leaves(), 1)
        self.assertEqual(tree.get_depth(), 0)

    def test_positive_min_impurity_decrease_also_blocks_zero_gain(self):
        tree = DecisionTreeClassifier(min_impurity_decrease=1e-6).fit(XOR_X, XOR_Y)
        self.assertEqual(tree.get_n_leaves(), 1)

    def test_rules_agree_when_gain_is_positive(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        a = DecisionTreeClassifier().fit(X, y)
        b = DecisionTreeClassifier(allow_zero_gain_splits=False).fit(X, y)
        np.testing.assert_array_equal(a.feature_, b.feature_)
        np.testing.assert_array_equal(a.threshold_, b.threshold_)

    def test_all_features_useless_in_identical_data(self):
        # identical rows with different labels: no valid threshold, a leaf either way
        X = np.ones((6, 2))
        y = np.array([0, 1, 0, 1, 0, 1])
        for flag in (True, False):
            tree = DecisionTreeClassifier(allow_zero_gain_splits=flag).fit(X, y)
            self.assertEqual(tree.get_n_leaves(), 1)

    def test_flag_in_params_and_streaming(self):
        tree = DecisionTreeClassifier(allow_zero_gain_splits=False, buffer_size=4)
        self.assertFalse(tree.get_params()["allow_zero_gain_splits"])
        tree.partial_fit(XOR_X, XOR_Y)
        self.assertEqual(tree.get_n_leaves(), 1)


class TestObjectLabelMissing(unittest.TestCase):
    BAD = [np.array([1, np.nan], dtype=object),
           np.array(["passed", np.nan], dtype=object),
           np.array(["passed", None], dtype=object),
           np.array([1, None], dtype=object)]

    def test_check_y_rejects(self):
        for y in self.BAD:
            with self.assertRaises(ValueError):
                check_y(y)

    def test_check_y_accepts_clean_object_labels(self):
        y = np.array(["a", "b", 3], dtype=object)
        self.assertEqual(check_y(y).shape, (3,))
        self.assertFalse(has_missing(np.array(["a", "b"])))
        self.assertFalse(has_missing(np.array([1, 2])))
        self.assertTrue(has_missing(np.array([1.0, np.nan])))

    def test_tree_rejects_before_training(self):
        X = np.zeros((2, 1))
        for y in self.BAD:
            with self.assertRaises(ValueError):
                DecisionTreeClassifier().fit(X, y)
            with self.assertRaises(ValueError):
                DecisionTreeClassifier().partial_fit(X, y)

    def test_batch_metrics_reject(self):
        ok = np.array([1, 0], dtype=object)
        for bad in self.BAD:
            for fn in (accuracy, confusion_matrix, precision_recall_f1):
                with self.assertRaises(ValueError):
                    fn(bad, ok)
                with self.assertRaises(ValueError):
                    fn(ok, bad)


class TestMinImpurityDecreaseValidation(unittest.TestCase):
    def test_invalid_values(self):
        for bad in (np.nan, np.inf, -np.inf, True, False, -1.0, -1e-9, "0.1", None, [0.1]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                DecisionTreeClassifier(min_impurity_decrease=bad)

    def test_valid_values(self):
        for ok in (0, 0.0, 0.25, 1, np.float32(0.1), np.int64(0)):
            tree = DecisionTreeClassifier(min_impurity_decrease=ok)
            self.assertIsInstance(tree.min_impurity_decrease, float)


class TestUpdateProbaValidation(unittest.TestCase):
    def _call(self, proba, classes=(0, 1), y=(0, 1)):
        return StreamingAUC().update_proba(list(y), proba, classes=list(classes))

    def test_rejects_non_finite(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                self._call(np.array([[0.2, 0.8], [bad, 0.1]]))
        # NaN in the non-positive column must be rejected too
        with self.assertRaises(ValueError):
            self._call(np.array([[np.nan, 0.8], [0.2, 0.8]]))

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            self._call(np.array([[0.2, 2.0], [0.8, 0.2]]))
        with self.assertRaises(ValueError):
            self._call(np.array([[-0.2, 1.2], [0.8, 0.2]]))

    def test_rejects_rows_not_summing_to_one(self):
        with self.assertRaises(ValueError):
            self._call(np.array([[0.2, 0.2], [0.8, 0.1]]))

    def test_invalid_call_leaves_state_unchanged(self):
        m = StreamingAUC().update_proba([0, 1], np.array([[0.9, 0.1], [0.2, 0.8]]), [0, 1])
        before = m.result()
        with self.assertRaises(ValueError):
            m.update_proba([0, 1], np.array([[0.9, 0.9], [0.2, 0.8]]), [0, 1])
        self.assertEqual(m.result(), before)

    def test_valid_matrices_accepted(self):
        proba = np.array([[0.9, 0.1], [0.3, 0.7], [0.5, 0.5]])
        m = self._call(proba, y=(0, 1, 1))
        self.assertAlmostEqual(m.result(), 1.0 - 0.0)  # ranks: pos scores .7,.5 > neg .1
        self._call(proba.astype(np.float32), y=(0, 1, 1))  # float32 rounding is tolerated
        self._call(np.zeros((0, 2)), y=())                   # empty chunk

    def test_plain_update_still_takes_any_finite_score(self):
        m = StreamingAUC().update([0, 1, 1], [-5.0, 12.5, 100.0])
        self.assertEqual(m.result(), 1.0)


if __name__ == "__main__":
    unittest.main()

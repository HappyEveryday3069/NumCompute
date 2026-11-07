import unittest

import numpy as np

from numcompute_stream.preprocessing import (Imputer, MinMaxScaler, OneHotEncoder,
                                             SimpleImputer, StandardScaler)
from numcompute_stream.utils import NotFittedError


class TestStandardScalerStreaming(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(0).normal(10, 4, size=(300, 3))

    def test_partial_fit_matches_fit(self):
        a = StandardScaler().fit(self.X)
        b = StandardScaler()
        for chunk in np.array_split(self.X, 6):
            b.partial_fit(chunk)
        np.testing.assert_allclose(a.mean_, b.mean_, rtol=1e-12)
        np.testing.assert_allclose(a.scale_, b.scale_, rtol=1e-10)
        np.testing.assert_allclose(a.transform(self.X), b.transform(self.X), rtol=1e-9)

    def test_transform_is_standardised(self):
        Z = StandardScaler().fit_transform(self.X)
        np.testing.assert_allclose(Z.mean(0), 0, atol=1e-12)
        np.testing.assert_allclose(Z.std(0), 1, atol=1e-12)

    def test_zero_variance_chunk_does_not_divide_by_zero(self):
        s = StandardScaler().partial_fit(np.full((5, 2), 7.0))
        out = s.transform(np.full((2, 2), 7.0))
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_array_equal(s.scale_, [1.0, 1.0])

    def test_nan_ignored_in_fit_and_propagated(self):
        X = self.X.copy()
        X[::4, 0] = np.nan
        s = StandardScaler()
        for chunk in np.array_split(X, 5):
            s.partial_fit(chunk)
        np.testing.assert_allclose(s.mean_[0], np.nanmean(X[:, 0]), rtol=1e-12)
        self.assertTrue(np.isnan(s.transform(X)[0, 0]))

    def test_inverse_and_flags(self):
        s = StandardScaler().fit(self.X)
        np.testing.assert_allclose(s.inverse_transform(s.transform(self.X)), self.X)
        only_mean = StandardScaler(with_std=False).fit(self.X).transform(self.X)
        np.testing.assert_allclose(only_mean, self.X - self.X.mean(0))

    def test_errors(self):
        with self.assertRaises(NotFittedError):
            StandardScaler().transform(self.X)
        s = StandardScaler().partial_fit(self.X)
        with self.assertRaises(ValueError):
            s.partial_fit(self.X[:, :2])
        with self.assertRaises(ValueError):
            s.transform(self.X[:, :2])
        with self.assertRaises(ValueError):
            s.transform(self.X[:, 0])

    def test_fit_resets_previous_state(self):
        s = StandardScaler().partial_fit(self.X + 100)
        s.fit(self.X)
        np.testing.assert_allclose(s.mean_, self.X.mean(0))


class TestMinMaxScaler(unittest.TestCase):
    def test_streaming_range(self):
        s = MinMaxScaler()
        s.partial_fit([[0.0], [5.0]])
        s.partial_fit([[10.0]])
        np.testing.assert_allclose(s.transform([[0.0], [5.0], [10.0]]).ravel(), [0, 0.5, 1])

    def test_huge_magnitudes_are_fine_for_minmax_but_rejected_by_the_standard_scaler(self):
        X = np.array([[-1e200], [0.0], [1e200]])
        np.testing.assert_allclose(MinMaxScaler().fit_transform(X).ravel(), [0, 0.5, 1])
        with self.assertRaises(ValueError):
            StandardScaler().fit(X)

    def test_constant_feature_and_invalid_range(self):
        out = MinMaxScaler().fit_transform(np.full((3, 1), 2.0))
        np.testing.assert_array_equal(out, np.zeros((3, 1)))
        with self.assertRaises(ValueError):
            MinMaxScaler(feature_range=(1, 1))

    def test_custom_range_and_not_fitted(self):
        out = MinMaxScaler((-1, 1)).fit_transform([[0.0], [1.0]])
        np.testing.assert_allclose(out.ravel(), [-1, 1])
        with self.assertRaises(NotFittedError):
            MinMaxScaler().transform([[1.0]])


class TestSimpleImputer(unittest.TestCase):
    def test_streaming_mean(self):
        imp = SimpleImputer(strategy="mean")
        imp.partial_fit([[1.0, np.nan], [3.0, np.nan]])
        imp.partial_fit([[5.0, 4.0]])
        np.testing.assert_allclose(imp.statistics_, [3.0, 4.0])
        out = imp.transform([[np.nan, np.nan]])
        np.testing.assert_allclose(out, [[3.0, 4.0]])

    def test_unseen_feature_uses_fill_value(self):
        imp = SimpleImputer(strategy="mean", fill_value=-1.0).partial_fit([[np.nan, 2.0]])
        np.testing.assert_allclose(imp.statistics_, [-1.0, 2.0])

    def test_most_frequent_streaming_and_tie_rule(self):
        imp = SimpleImputer(strategy="most_frequent")
        imp.partial_fit([[1.0], [2.0], [2.0]])
        imp.partial_fit([[1.0], [np.nan]])
        self.assertEqual(imp.statistics_[0], 1.0)  # tie 1 vs 2 -> smallest

    def test_constant_does_not_mutate_input(self):
        X = np.array([[np.nan, 1.0]])
        out = SimpleImputer(fill_value=9.0).fit_transform(X)
        self.assertTrue(np.isnan(X[0, 0]))
        self.assertEqual(out[0, 0], 9.0)

    def test_errors_and_alias(self):
        with self.assertRaises(ValueError):
            SimpleImputer(strategy="median")
        with self.assertRaises(NotFittedError):
            SimpleImputer().transform([[1.0]])
        self.assertIs(Imputer, SimpleImputer)


class TestOneHotEncoder(unittest.TestCase):
    def test_incremental_expansion_keeps_old_columns(self):
        enc = OneHotEncoder()
        enc.partial_fit([["a"], ["b"]])
        before = enc.transform([["a"], ["b"]])
        enc.partial_fit([["c"], ["a"]])
        after = enc.transform([["a"], ["b"], ["c"]])
        self.assertEqual(enc.n_output_features_, 3)
        np.testing.assert_array_equal(after[:2, :2], before)
        np.testing.assert_array_equal(after[2], [0, 0, 1])

    def test_new_category_of_an_early_feature_does_not_shift_later_features(self):
        enc = OneHotEncoder().fit([["a", "x"], ["b", "y"]])          # columns: a b | x y
        before = enc.transform([["a", "y"]])
        enc.partial_fit([["c", "x"]])                                  # new category in feature 0
        after = enc.transform([["a", "y"]])
        np.testing.assert_array_equal(before, [[1, 0, 0, 1]])
        np.testing.assert_array_equal(after, [[1, 0, 0, 1, 0]])       # "c" became the last column

    def test_unknown_handling(self):
        enc = OneHotEncoder().fit([["a"], ["b"]])
        np.testing.assert_array_equal(enc.transform([["z"]]), [[0, 0]])
        with self.assertRaises(ValueError):
            OneHotEncoder(handle_unknown="error").fit([["a"]]).transform([["z"]])
        with self.assertRaises(ValueError):
            OneHotEncoder(handle_unknown="bad")

    def test_multiple_columns_and_numbers(self):
        enc = OneHotEncoder().fit([[1, "x"], [2, "y"]])
        out = enc.transform([[2, "x"]])
        np.testing.assert_array_equal(out, [[0, 1, 1, 0]])

    def test_missing_and_shape_errors(self):
        with self.assertRaises(ValueError):
            OneHotEncoder().fit([["a"], [None]])
        with self.assertRaises(ValueError):
            OneHotEncoder().fit([["a"], [np.nan]])
        with self.assertRaises(ValueError):
            OneHotEncoder().fit(["a", "b"])
        enc = OneHotEncoder().fit([["a", "b"]])
        with self.assertRaises(ValueError):
            enc.partial_fit([["a"]])
        with self.assertRaises(NotFittedError):
            OneHotEncoder().transform([["a"]])


if __name__ == "__main__":
    unittest.main()

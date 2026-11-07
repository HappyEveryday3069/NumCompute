import unittest
from pathlib import Path

import numpy as np

from numcompute.io import read_csv
from numcompute.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    SimpleImputer,
    OneHotEncoder
)


class TestStandardScaler(unittest.TestCase):

    def test_fit(self):
        X = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ], dtype=float)

        scaler = StandardScaler()
        result = scaler.fit(X)

        self.assertIs(result, scaler)
        np.testing.assert_array_almost_equal(
            scaler.mean_,
            [3, 4]
        )

        np.testing.assert_array_almost_equal(
            scaler.scale_,
            np.std(X, axis=0)
        )

    def test_transform(self):
        X = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ], dtype=float)

        scaler = StandardScaler()
        result = scaler.fit_transform(X)

        expected = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        np.testing.assert_array_almost_equal(
            result,
            expected
        )

    def test_fit_transform(self):
        X = np.array([
            [10, 20],
            [20, 40],
            [30, 60]
        ], dtype=float)

        scaler = StandardScaler()

        result1 = scaler.fit_transform(X)

        scaler2 = StandardScaler()
        result2 = scaler2.fit(X).transform(X)

        np.testing.assert_array_almost_equal(
            result1,
            result2
        )

    def test_output_mean_is_zero(self):
        X = np.array([
            [1, 10],
            [2, 20],
            [3, 30]
        ], dtype=float)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        np.testing.assert_array_almost_equal(
            np.mean(X_scaled, axis=0),
            [0, 0]
        )

    def test_constant_feature(self):
        X = np.array([
            [1, 10],
            [1, 20],
            [1, 30]
        ], dtype=float)

        scaler = StandardScaler()
        result = scaler.fit_transform(X)

        # Constant feature should not produce NaN or inf
        self.assertFalse(np.isnan(result).any())
        self.assertFalse(np.isinf(result).any())

        np.testing.assert_array_equal(
            result[:, 0],
            [0, 0, 0]
        )

    def test_transform_before_fit(self):
        X = np.array([
            [1, 2],
            [3, 4]
        ])

        scaler = StandardScaler()

        with self.assertRaises(ValueError):
            scaler.transform(X)

    def test_single_column(self):
        result = StandardScaler().fit_transform([[1], [3]])
        np.testing.assert_allclose(result, [[-1], [1]])


class TestMinMaxScaler(unittest.TestCase):

    def test_default_range(self):
        X = np.array([
            [1, 10],
            [2, 20],
            [3, 30]
        ], dtype=float)

        scaler = MinMaxScaler()

        result = scaler.fit_transform(X)

        expected = np.array([
            [0, 0],
            [0.5, 0.5],
            [1, 1]
        ])

        np.testing.assert_array_almost_equal(
            result,
            expected
        )

    def test_custom_range(self):
        X = np.array([
            [1],
            [2],
            [3]
        ], dtype=float)

        scaler = MinMaxScaler(feature_range=(-1, 1))

        result = scaler.fit_transform(X)

        expected = np.array([
            [-1],
            [0],
            [1]
        ])

        np.testing.assert_array_almost_equal(
            result,
            expected
        )

    def test_fit_returns_self(self):
        X = np.array([
            [1, 2],
            [3, 4]
        ])

        scaler = MinMaxScaler()

        result = scaler.fit(X)

        self.assertIs(result, scaler)

    def test_min_and_max(self):
        X = np.array([
            [10, 100],
            [20, 200],
            [30, 300]
        ], dtype=float)

        scaler = MinMaxScaler()
        scaler.fit(X)

        np.testing.assert_array_equal(
            scaler.data_min_,
            [10, 100]
        )

        np.testing.assert_array_equal(
            scaler.data_max_,
            [30, 300]
        )

    def test_constant_feature(self):
        X = np.array([
            [5, 10],
            [5, 20],
            [5, 30]
        ], dtype=float)

        scaler = MinMaxScaler()

        result = scaler.fit_transform(X)

        self.assertFalse(np.isnan(result).any())
        self.assertFalse(np.isinf(result).any())

    def test_invalid_feature_range(self):
        X = np.array([
            [1, 2],
            [3, 4]
        ])

        scaler = MinMaxScaler(feature_range=(1, 1))

        with self.assertRaises(ValueError):
            scaler.fit(X)

    def test_transform_before_fit(self):
        X = np.array([
            [1, 2],
            [3, 4]
        ])

        scaler = MinMaxScaler()

        with self.assertRaises(ValueError):
            scaler.transform(X)

    def test_single_column(self):
        result = MinMaxScaler().fit_transform([[1], [3]])
        np.testing.assert_allclose(result, [[0], [1]])


class TestSimpleImputer(unittest.TestCase):

    def test_replace_nan_with_constant(self):
        X = np.array([
            [1, 2],
            [np.nan, 4],
            [5, np.nan]
        ])

        imputer = SimpleImputer(fill_value=0)

        result = imputer.fit_transform(X)

        expected = np.array([
            [1, 2],
            [0, 4],
            [5, 0]
        ])

        np.testing.assert_array_equal(
            result,
            expected
        )

    def test_custom_fill_value(self):
        X = np.array([
            [1, np.nan],
            [np.nan, 4]
        ])

        imputer = SimpleImputer(fill_value=-1)

        result = imputer.fit_transform(X)

        expected = np.array([
            [1, -1],
            [-1, 4]
        ])

        np.testing.assert_array_equal(
            result,
            expected
        )

    def test_no_nan(self):
        X = np.array([
            [1, 2],
            [3, 4]
        ])

        imputer = SimpleImputer(fill_value=0)

        result = imputer.fit_transform(X)

        np.testing.assert_array_equal(
            result,
            X
        )

    def test_all_nan(self):
        X = np.array([
            [np.nan, np.nan],
            [np.nan, np.nan]
        ])

        imputer = SimpleImputer(fill_value=99)

        result = imputer.fit_transform(X)

        expected = np.full((2, 2), 99)

        np.testing.assert_array_equal(
            result,
            expected
        )

    def test_single_column(self):
        result = SimpleImputer(fill_value=-1).fit_transform([[1], [np.nan]])
        np.testing.assert_allclose(result, [[1], [-1]])

    def test_mean_uses_training_statistics(self):
        imputer = SimpleImputer(strategy="mean")
        self.assertIs(imputer.fit([[1, np.nan], [3, 6], [np.nan, 10]]), imputer)
        np.testing.assert_allclose(imputer.statistics_, [2, 8])
        X = np.array([[np.nan, 20], [10, np.nan]])
        original = X.copy()
        np.testing.assert_allclose(imputer.transform(X), [[2, 20], [10, 8]])
        np.testing.assert_array_equal(X, original)

    def test_invalid_strategy(self):
        with self.assertRaisesRegex(ValueError, "Strategy"):
            SimpleImputer(strategy="median").fit([[1, 2]])

    def test_mean_all_missing_column(self):
        with self.assertWarns(RuntimeWarning):
            result = SimpleImputer(strategy="mean").fit_transform(
                [[np.nan, 2], [np.nan, 4]]
            )
        np.testing.assert_allclose(result, [[np.nan, 2], [np.nan, 4]], equal_nan=True)

    def test_empty_transform(self):
        imputer = SimpleImputer().fit([[1, 2]])
        self.assertEqual(imputer.transform(np.empty((0, 2))).shape, (0, 2))


class TestScalerBoundaries(unittest.TestCase):

    def test_single_row(self):
        for scaler in (StandardScaler(), MinMaxScaler()):
            with self.subTest(scaler=type(scaler).__name__):
                np.testing.assert_array_equal(scaler.fit_transform([[4, 9]]), [[0, 0]])

    def test_transform_uses_training_statistics_without_mutation(self):
        cases = [
            (StandardScaler(), [[3, 3], [5, 5]]),
            (MinMaxScaler(), [[2, 2], [3, 3]])
        ]
        for scaler, expected in cases:
            with self.subTest(scaler=type(scaler).__name__):
                training = np.array([[1, 2], [3, 4]])
                X = np.array([[5, 6], [7, 8]])
                self.assertIs(scaler.fit(training), scaler)
                np.testing.assert_allclose(scaler.transform(X), expected)
                np.testing.assert_array_equal(training, [[1, 2], [3, 4]])
                np.testing.assert_array_equal(X, [[5, 6], [7, 8]])
                self.assertEqual(scaler.transform(np.empty((0, 2))).shape, (0, 2))

    def test_refit_replaces_statistics(self):
        for scaler in (StandardScaler(), MinMaxScaler()):
            with self.subTest(scaler=type(scaler).__name__):
                scaler.fit([[1, 2], [3, 4]])
                scaler.fit([[10, 20]])
                np.testing.assert_array_equal(scaler.transform([[10, 20]]), [[0, 0]])

    def test_reversed_range(self):
        with self.assertRaisesRegex(ValueError, "min_value"):
            MinMaxScaler((2, -1)).fit([[1], [2]])

    def test_constant_feature_maps_to_lower_bound(self):
        result = MinMaxScaler((-2, 2)).fit_transform([[5, 10], [5, 20]])
        np.testing.assert_array_equal(result, [[-2, -2], [-2, 2]])


class TestOneHotEncoder(unittest.TestCase):

    def test_fit_and_multiple_columns(self):
        X = [["red", "small"], ["blue", "large"], ["red", "large"]]
        encoder = OneHotEncoder()
        self.assertIs(encoder.fit(X), encoder)
        np.testing.assert_array_equal(encoder.categories_[0], ["blue", "red"])
        np.testing.assert_array_equal(encoder.categories_[1], ["large", "small"])
        result = encoder.transform(X)
        np.testing.assert_array_equal(result, [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]])
        self.assertEqual(result.dtype, np.dtype(float))
        np.testing.assert_array_equal(OneHotEncoder().fit_transform(X), result)

    def test_unknown_category(self):
        encoder = OneHotEncoder().fit([["red", "small"], ["blue", "large"]])
        np.testing.assert_array_equal(encoder.transform([["green", "small"]]), [[0, 0, 0, 1]])

    def test_single_category_and_repeated_rows(self):
        np.testing.assert_array_equal(
            OneHotEncoder().fit_transform([[7], [7]]), [[1], [1]]
        )

    def test_single_row(self):
        np.testing.assert_array_equal(
            OneHotEncoder().fit_transform([["red", "small"]]), [[1, 1]]
        )

    def test_empty_transform(self):
        encoder = OneHotEncoder().fit([["red"], ["blue"]])
        self.assertEqual(encoder.transform(np.empty((0, 1), dtype=object)).shape, (0, 2))

    def test_refit_and_input_unchanged(self):
        encoder = OneHotEncoder().fit([["red"], ["blue"]])
        X = np.array([["green"], ["green"]], dtype=object)
        original = X.copy()
        np.testing.assert_array_equal(encoder.fit_transform(X), [[1], [1]])
        np.testing.assert_array_equal(encoder.categories_[0], ["green"])
        np.testing.assert_array_equal(X, original)


class TestPreprocessingCSV(unittest.TestCase):
    data_dir = Path(__file__).parent / "data"

    def test_scalers_from_csv(self):
        X = read_csv(self.data_dir / "preprocessing.csv", skip_header=1, dtype=float)
        original = X.copy()
        scaler = StandardScaler()
        result = scaler.fit_transform(X)
        np.testing.assert_allclose(scaler.mean_, [3, 10])
        np.testing.assert_allclose(scaler.scale_, [np.sqrt(8 / 3), 1])
        np.testing.assert_allclose(
            result, [[-np.sqrt(1.5), 0], [0, 0], [np.sqrt(1.5), 0]]
        )
        np.testing.assert_allclose(
            MinMaxScaler((-1, 1)).fit_transform(X), [[-1, -1], [0, -1], [1, -1]]
        )
        np.testing.assert_array_equal(X, original)

    def test_one_hot_from_csv(self):
        X = read_csv(self.data_dir / "categories.csv", skip_header=1, dtype=str)
        original = X.copy()
        np.testing.assert_array_equal(
            OneHotEncoder().fit_transform(X),
            [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]]
        )
        np.testing.assert_array_equal(X, original)

    def test_imputation_from_csv(self):
        X = read_csv(self.data_dir / "missing.csv", skip_header=1, dtype=float)
        original = X.copy()
        np.testing.assert_array_equal(
            SimpleImputer(fill_value=-1).fit_transform(X),
            [[170, 65], [180, -1], [-1, 70]]
        )
        np.testing.assert_allclose(
            SimpleImputer(strategy="mean").fit_transform(X),
            [[170, 65], [180, 67.5], [175, 70]]
        )
        np.testing.assert_array_equal(X, original)

    def test_single_row_from_csv(self):
        X = read_csv(self.data_dir / "single_row.csv", skip_header=1, dtype=float)
        self.assertEqual(X.shape, (1, 2))
        for scaler in (StandardScaler(), MinMaxScaler()):
            with self.subTest(scaler=type(scaler).__name__):
                np.testing.assert_array_equal(scaler.fit_transform(X), [[0, 0]])
        np.testing.assert_array_equal(OneHotEncoder().fit_transform(X), [[1, 1]])
        np.testing.assert_array_equal(SimpleImputer().fit_transform(X), [[5, 10]])


if __name__ == "__main__":
    unittest.main()

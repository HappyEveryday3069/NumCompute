import unittest
from pathlib import Path

import numpy as np

from numcompute.io import read_csv
from numcompute.sort_search import (
    stable_sort, multi_key_sort, topk, quickselect, binary_search
)


class TestStableSort(unittest.TestCase):

    def test_values_and_input_unchanged(self):
        values = np.array([3, -1, 3, 0])
        np.testing.assert_array_equal(stable_sort(values), [-1, 0, 3, 3])
        np.testing.assert_array_equal(values, [3, -1, 3, 0])

    def test_axes(self):
        X = [[3, 1], [2, 4]]
        np.testing.assert_array_equal(stable_sort(X), [[1, 3], [2, 4]])
        np.testing.assert_array_equal(stable_sort(X, axis=0), [[2, 1], [3, 4]])
        np.testing.assert_array_equal(stable_sort(X, axis=None), [1, 2, 3, 4])

    def test_stability(self):
        class Item:
            def __init__(self, key, label):
                self.key = key
                self.label = label

            def __lt__(self, other):
                return self.key < other.key

        values = [Item(2, "a"), Item(1, "b"), Item(2, "c"), Item(1, "d")]
        self.assertEqual([item.label for item in stable_sort(values)], ["b", "d", "a", "c"])

    def test_empty_singleton_and_nan(self):
        np.testing.assert_array_equal(stable_sort([]), [])
        np.testing.assert_array_equal(stable_sort([5]), [5])
        np.testing.assert_array_equal(stable_sort([np.nan, 1, -np.inf]), [-np.inf, 1, np.nan])


class TestMultiKeySort(unittest.TestCase):

    def test_priority_ties_and_input_unchanged(self):
        X = np.array([[2, 1, 10], [1, 2, 20], [1, 1, 30], [1, 1, 40]])
        original = X.copy()
        np.testing.assert_array_equal(multi_key_sort(X, [0, 1]), X[[2, 3, 1, 0]])
        np.testing.assert_array_equal(multi_key_sort(X, [1, 0]), X[[2, 3, 0, 1]])
        np.testing.assert_array_equal(X, original)

    def test_empty_keys_returns_copy(self):
        X = np.array([[2, 1], [1, 2]])
        result = multi_key_sort(X, [])
        np.testing.assert_array_equal(result, X)
        self.assertFalse(np.shares_memory(result, X))

    def test_empty_singleton_and_negative_key(self):
        self.assertEqual(multi_key_sort(np.empty((0, 2)), [0]).shape, (0, 2))
        np.testing.assert_array_equal(multi_key_sort([[1, 2]], [0]), [[1, 2]])
        np.testing.assert_array_equal(multi_key_sort([[1, 2], [2, 1]], [-1]), [[2, 1], [1, 2]])

    def test_invalid_dimension_and_key(self):
        for X in ([1, 2], np.zeros((1, 1, 1))):
            with self.subTest(shape=np.shape(X)), self.assertRaisesRegex(ValueError, "2D"):
                multi_key_sort(X, [0])
        with self.assertRaises(IndexError):
            multi_key_sort([[1, 2]], [2])


class TestTopK(unittest.TestCase):

    def test_values_indices_and_boundaries(self):
        values = np.array([4, -2, 7, 1, 7])
        original = values.copy()
        for largest in (True, False):
            for k in (1, 3, len(values)):
                with self.subTest(largest=largest, k=k):
                    expected = [7, 7, 4, 1, -2][:k] if largest else [-2, 1, 4, 7, 7][:k]
                    result, indices = topk(values, k, largest=largest)
                    np.testing.assert_array_equal(result, expected)
                    np.testing.assert_array_equal(values[indices], result)
                    self.assertEqual(len(np.unique(indices)), k)
                    np.testing.assert_array_equal(
                        topk(values, k, largest=largest, return_indices=False), expected
                    )
        np.testing.assert_array_equal(values, original)

    def test_singleton(self):
        result, indices = topk([9], 1)
        np.testing.assert_array_equal(result, [9])
        np.testing.assert_array_equal(indices, [0])

    def test_invalid_k(self):
        for values, k in (([], 1), ([1, 2], 0), ([1, 2], -1), ([1, 2], 3)):
            with self.subTest(values=values, k=k), self.assertRaises(ValueError):
                topk(values, k)
        for k in (1.5, "1", None):
            with self.subTest(k=k), self.assertRaises(TypeError):
                topk([1, 2], k)


class TestQuickselect(unittest.TestCase):

    def test_all_ranks_duplicates_and_no_mutation(self):
        values = np.array([4, -2, 7, 1, 7])
        for k, expected in enumerate([-2, 1, 4, 7, 7], start=1):
            with self.subTest(k=k):
                self.assertEqual(quickselect(values, k), expected)
        np.testing.assert_array_equal(values, [4, -2, 7, 1, 7])
        self.assertEqual(quickselect([9], 1), 9)

    def test_invalid_k(self):
        for values, k in (([], 1), ([1, 2], 0), ([1, 2], -1), ([1, 2], 3)):
            with self.subTest(values=values, k=k), self.assertRaises(ValueError):
                quickselect(values, k)


class TestBinarySearch(unittest.TestCase):

    def test_found_duplicates_and_insertion_positions(self):
        values = np.array([1, 3, 3, 7])
        for x, expected in ((0, (0, False)), (1, (0, True)), (3, (1, True)),
                            (4, (3, False)), (7, (3, True)), (8, (4, False))):
            with self.subTest(x=x):
                result = binary_search(values, x)
                self.assertEqual(result, expected)
                self.assertIsInstance(result[1], bool)
        np.testing.assert_array_equal(values, [1, 3, 3, 7])

    def test_empty_and_singleton(self):
        self.assertEqual(binary_search([], 1), (0, False))
        self.assertEqual(binary_search([2], 2), (0, True))
        self.assertEqual(binary_search([2], 3), (1, False))

    def test_invalid_dimension(self):
        for values in (1, [[1, 2]], np.zeros((1, 1, 1))):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "1D"):
                binary_search(values, 1)


class TestSortSearchCSV(unittest.TestCase):

    def setUp(self):
        self.data_dir = Path(__file__).parent / "data"
        self.X = read_csv(self.data_dir / "sort_search.csv", skip_header=1, dtype=int)
        self.original = self.X.copy()
        self.values = self.X[:, 0]

    def tearDown(self):
        np.testing.assert_array_equal(self.X, self.original)

    def test_stable_sort_from_csv(self):
        np.testing.assert_array_equal(stable_sort(self.values), [-2, 1, 4, 7, 7])

    def test_multi_key_sort_from_csv(self):
        np.testing.assert_array_equal(
            multi_key_sort(self.X, [1, 0]),
            [[-2, 1, 20], [7, 1, 30], [7, 1, 50], [1, 2, 40], [4, 2, 10]]
        )

    def test_topk_from_csv(self):
        for largest, expected in ((True, [7, 7, 4, 1, -2]), (False, [-2, 1, 4, 7, 7])):
            for k in (1, 3, 5):
                with self.subTest(largest=largest, k=k):
                    result, indices = topk(self.values, k, largest=largest)
                    np.testing.assert_array_equal(result, expected[:k])
                    np.testing.assert_array_equal(self.values[indices], result)
                    self.assertEqual(len(np.unique(indices)), k)

    def test_quickselect_from_csv(self):
        for k, expected in enumerate([-2, 1, 4, 7, 7], start=1):
            with self.subTest(k=k):
                self.assertEqual(quickselect(self.values, k), expected)

    def test_binary_search_from_csv(self):
        sorted_values = stable_sort(self.values)
        for target, expected in ((-3, (0, False)), (-2, (0, True)), (2, (2, False)),
                                 (7, (3, True)), (8, (5, False))):
            with self.subTest(target=target):
                self.assertEqual(binary_search(sorted_values, target), expected)

    def test_single_value_from_csv(self):
        X = read_csv(self.data_dir / "single_row.csv", skip_header=1, dtype=int)
        values = X[:, 0]
        np.testing.assert_array_equal(stable_sort(values), [5])
        np.testing.assert_array_equal(multi_key_sort(X, [0]), [[5, 10]])
        result, indices = topk(values, 1)
        np.testing.assert_array_equal(result, [5])
        np.testing.assert_array_equal(indices, [0])
        self.assertEqual(quickselect(values, 1), 5)
        self.assertEqual(binary_search(values, 5), (0, True))
        self.assertEqual(binary_search(values, 6), (1, False))


if __name__ == "__main__":
    unittest.main()

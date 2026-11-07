import unittest

import numpy as np

from numcompute.utils import (
    batched,
    euclidean_distance,
    logsumexp,
    relu,
    sigmoid,
    softmax,
    topk_indices,
)


class TestEuclideanDistance(unittest.TestCase):
    def test_known_distances_and_broadcasting(self):
        np.testing.assert_allclose(
            euclidean_distance([[3, 4], [0, 0]], [0, 0]), [5, 0]
        )

    def test_axes(self):
        x = np.array([[1, 2], [3, 4]])
        np.testing.assert_allclose(euclidean_distance(x, 0, axis=0), np.sqrt([10, 20]))
        self.assertAlmostEqual(euclidean_distance(x, 0, axis=(0, 1)), np.sqrt(30))

    def test_large_values(self):
        result = euclidean_distance([1e200, 1e200], [0, 0])
        self.assertTrue(np.isfinite(result))
        np.testing.assert_allclose(result, np.sqrt(2) * 1e200)

    def test_empty_and_nonfinite(self):
        self.assertEqual(euclidean_distance([], []), 0)
        self.assertEqual(euclidean_distance([np.inf, 1], [0, 0]), np.inf)
        self.assertTrue(np.isnan(euclidean_distance([np.nan, 1], [0, 0])))

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            euclidean_distance([1, 2], [1, 2, 3])

    def test_noncontiguous_input_not_modified(self):
        source = np.arange(12.0).reshape(3, 4)
        original = source.copy()
        x = source[:, ::2]
        np.testing.assert_allclose(euclidean_distance(x, 0), np.sqrt([4, 52, 164]))
        np.testing.assert_array_equal(source, original)


class TestActivations(unittest.TestCase):
    def test_sigmoid_extremes(self):
        with np.errstate(over='raise', invalid='raise'):
            actual = sigmoid([-np.inf, -1000, 0, 1000, np.inf])
        np.testing.assert_allclose(actual, [0, 0, 0.5, 1, 1])

    def test_sigmoid_scalar_empty_and_nan(self):
        self.assertEqual(float(sigmoid(0)), 0.5)
        self.assertEqual(sigmoid([]).shape, (0,))
        self.assertTrue(np.isnan(sigmoid(np.nan)))

    def test_relu_preserves_shape_and_input(self):
        x = np.array([[-2, 0], [3, np.nan]])
        original = x.copy()
        np.testing.assert_allclose(relu(x), [[0, 0], [3, np.nan]], equal_nan=True)
        np.testing.assert_array_equal(x, original)

    def test_softmax_known_and_shift_invariant(self):
        expected = np.array([1, np.e]) / (1 + np.e)
        np.testing.assert_allclose(softmax([1000, 1001]), expected)
        np.testing.assert_allclose(softmax([0, 1]), expected)

    def test_softmax_axes_and_shape(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        for axis in (0, 1):
            with self.subTest(axis=axis):
                result = softmax(x, axis=axis)
                self.assertEqual(result.shape, x.shape)
                np.testing.assert_allclose(result.sum(axis=axis), [1, 1])
                self.assertTrue((result >= 0).all())
        self.assertAlmostEqual(softmax(x, axis=None).sum(), 1)

    def test_softmax_extreme_range(self):
        np.testing.assert_array_equal(softmax([-1e308, 1e308]), [0, 1])

    def test_softmax_rejects_nonfinite(self):
        for x in ([np.nan, 1], [np.inf, 1], [-np.inf, 1]):
            with self.subTest(x=x), self.assertRaises(ValueError):
                softmax(x)

    def test_softmax_empty_axes(self):
        for x, axis in (([], -1), (np.empty((2, 0)), 1), ([], None)):
            with self.subTest(shape=np.shape(x), axis=axis), self.assertRaises(ValueError):
                softmax(x, axis=axis)
        self.assertEqual(softmax(np.empty((0, 3)), axis=1).shape, (0, 3))


class TestLogsumexp(unittest.TestCase):
    def test_large_values(self):
        self.assertAlmostEqual(logsumexp([1000, 1000]), 1000 + np.log(2))

    def test_axes_and_keepdims(self):
        x = np.array([[0.0, np.log(2)], [np.log(3), np.log(4)]])
        np.testing.assert_allclose(logsumexp(x, axis=0), np.log([4, 6]))
        np.testing.assert_allclose(logsumexp(x, axis=1, keepdims=True), np.log([[3], [7]]))
        result = logsumexp(x, axis=None, keepdims=True)
        self.assertEqual(result.shape, (1, 1))
        np.testing.assert_allclose(result, [[np.log(10)]])

    def test_tuple_axis(self):
        self.assertAlmostEqual(logsumexp(np.zeros((2, 3)), axis=(0, 1)), np.log(6))

    def test_infinities_and_nan(self):
        self.assertEqual(logsumexp([-np.inf, -np.inf]), -np.inf)
        self.assertEqual(logsumexp([np.inf, 1]), np.inf)
        self.assertEqual(logsumexp([np.inf, -np.inf]), np.inf)
        self.assertTrue(np.isnan(logsumexp([np.nan, 1])))

    def test_empty_reductions(self):
        self.assertEqual(logsumexp([]), -np.inf)
        np.testing.assert_array_equal(logsumexp(np.empty((2, 0)), axis=1), [-np.inf, -np.inf])


class TestTopKIndices(unittest.TestCase):
    def test_largest_smallest_and_k_boundaries(self):
        x = np.array([4, 9, 1, 7])
        for k in (1, 2, 4):
            with self.subTest(k=k):
                np.testing.assert_array_equal(topk_indices(x, k), [1, 3, 0, 2][:k])
                np.testing.assert_array_equal(topk_indices(x, k, largest=False), [2, 0, 3, 1][:k])

    def test_ties_favour_original_index(self):
        x = [5, 5, 1, 5]
        np.testing.assert_array_equal(topk_indices(x, 2), [0, 1])
        np.testing.assert_array_equal(topk_indices(x, 3, largest=False), [2, 0, 1])
        np.testing.assert_array_equal(topk_indices([3, 3, 3], 2), [0, 1])

    def test_unsigned_noncontiguous_and_no_mutation(self):
        original = np.array([0, 9, 4, 9, 2, 9], dtype=np.uint64)
        copy = original.copy()
        np.testing.assert_array_equal(topk_indices(original[::2], np.int64(2)), [1, 2])
        np.testing.assert_array_equal(original, copy)

    def test_infinite_values(self):
        np.testing.assert_array_equal(topk_indices([np.inf, 2, -np.inf, np.inf], 3), [0, 3, 1])

    def test_invalid_k(self):
        for k in (0, -1, 4):
            with self.subTest(k=k), self.assertRaises(ValueError):
                topk_indices([1, 2, 3], k)
        for k in (True, np.bool_(False), 1.5, '2', None):
            with self.subTest(k=k), self.assertRaises(TypeError):
                topk_indices([1, 2, 3], k)

    def test_invalid_values(self):
        for x in ([], [[1, 2]], [np.nan, 1], ['a', 'b']):
            with self.subTest(x=x), self.assertRaises(ValueError):
                topk_indices(x, 1)


class TestBatched(unittest.TestCase):
    def test_final_short_batch(self):
        self.assertEqual(list(batched([1, 2, 3, 4, 5], 2)), [[1, 2], [3, 4], [5]])

    def test_empty_singleton_and_large_batch(self):
        self.assertEqual(list(batched([], 2)), [])
        self.assertEqual(list(batched([1], 1)), [[1]])
        self.assertEqual(list(batched([1, 2], 10)), [[1, 2]])

    def test_numpy_integer_and_strided_input(self):
        x = np.arange(10)[::2]
        original = x.copy()
        result = list(batched(x, np.int64(2)))
        self.assertEqual([len(batch) for batch in result], [2, 2, 1])
        np.testing.assert_array_equal(np.concatenate(result), x)
        np.testing.assert_array_equal(x, original)

    def test_invalid_size(self):
        for size in (0, -1):
            with self.subTest(size=size), self.assertRaises(ValueError):
                list(batched([1], size))
        for size in (True, 1.5, '2', None):
            with self.subTest(size=size), self.assertRaises(TypeError):
                list(batched([1], size))


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from numcompute.optim import grad, jacobian
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


class Transformer:
    def fit(self, X):
        self.offset = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return np.asarray(X) - self.offset


class Estimator:
    def fit(self, X, y=None):
        self.X = np.array(X)
        self.y = y
        return self

    def predict(self, X):
        return np.sum(X, axis=1)


class Dual(Transformer):
    def predict(self, X):
        return np.sum(X, axis=1)


class TestOptim(unittest.TestCase):
    def test_gradient_analytic(self):
        x = np.array([1., -2., 0.])
        for method in ("forward", "central"):
            np.testing.assert_allclose(grad(lambda z: np.sum(z*z), x, method=method),
                                       2*x, atol=2e-5)
        np.testing.assert_array_equal(x, [1, -2, 0])

    def test_jacobian_analytic(self):
        f = lambda z: np.array([z[0]**2+z[1], z[0]*z[1]])
        for method in ("forward", "central"):
            np.testing.assert_allclose(jacobian(f, [1, 2], method=method),
                                       [[2, 1], [2, 1]], atol=2e-5)

    def test_bad_step(self):
        for h in (0, -1, np.nan, np.inf, True, [1], "bad"):
            for operation in (lambda: grad(np.sum, [1], h=h),
                              lambda: jacobian(lambda x: x, [1], h=h)):
                with self.subTest(h=h), self.assertRaises(ValueError):
                    operation()

    def test_bad_inputs(self):
        for x in (1, [[1]], [np.nan], [np.inf]):
            with self.subTest(x=x), self.assertRaises(ValueError):
                grad(np.sum, x)
        with self.assertRaises(TypeError):
            grad(None, [1])
        with self.assertRaises(ValueError):
            grad(np.sum, [1], method="bad")

    def test_non_scalar_objective(self):
        with self.assertRaises(ValueError):
            grad(lambda x: x, [1])

    def test_changing_vector_shape(self):
        def f(x):
            return np.array([1, 2]) if x[0] == 1 else np.array([x.sum()])
        with self.assertRaises(ValueError):
            jacobian(f, [1, 2])

    def test_bad_vector_outputs(self):
        for f in (lambda x: 1., lambda x: np.array([np.nan])):
            with self.assertRaises(ValueError):
                jacobian(f, [1])

    def test_empty_points(self):
        self.assertEqual(grad(np.sum, []).shape, (0,))
        self.assertEqual(jacobian(lambda x: np.zeros(2), []).shape, (2, 0))

    def test_noncontiguous_point(self):
        x = np.arange(6.)[::2]
        np.testing.assert_allclose(grad(lambda z: z@z, x), 2*x, atol=1e-8)

    def test_unrepresentable_step(self):
        with self.assertRaises(ValueError):
            grad(np.sum, [1e30])

    def test_forward_reuses_baseline(self):
        calls = []
        def f(x):
            calls.append(x.copy())
            return x@x
        grad(f, [1, 2, 3], method="forward")
        self.assertEqual(len(calls), 4)


class TestPipeline(unittest.TestCase):
    def test_scaler_integration(self):
        X = np.array([[1., 10.], [3., 20.], [5., 30.]])
        original = X.copy()
        pipe = Pipeline([("standard", StandardScaler()), ("range", MinMaxScaler())])
        expected = MinMaxScaler().fit_transform(StandardScaler().fit_transform(X))
        np.testing.assert_allclose(pipe.fit_transform(X), expected)
        np.testing.assert_allclose(pipe.transform(X), expected)
        np.testing.assert_array_equal(X, original)
        self.assertIs(pipe.fit(X), pipe)

    def test_imputer_integration(self):
        # Supports the GitHub SimpleImputer and the older local Imputer.
        from numcompute import preprocessing
        cls = getattr(preprocessing, "SimpleImputer", None)
        if cls is None:
            cls = preprocessing.Imputer
        X = np.array([[1., np.nan], [3., 4.], [5., 8.]])
        pipe = Pipeline([("impute", cls()), ("scale", StandardScaler())])
        np.testing.assert_allclose(np.mean(pipe.fit_transform(X), axis=0), 0, atol=1e-12)

    def test_encoder_integration(self):
        pipe = Pipeline([("encode", OneHotEncoder())])
        result = pipe.fit_transform([["red"], ["blue"], ["red"]])
        self.assertEqual(result.shape, (3, 2))
        np.testing.assert_array_equal(pipe.transform([["red"]]), result[:1])

    def test_fallback_and_training_statistics(self):
        pipe = Pipeline([("center", Transformer())])
        pipe.fit([[1.], [3.]])
        np.testing.assert_allclose(pipe.transform([[5.]]), [[3.]])

    def test_unfitted(self):
        pipe = Pipeline([("center", Transformer())])
        for method in (pipe.transform, pipe.predict):
            with self.assertRaises(ValueError):
                method([[1.]])

    def test_invalid_steps(self):
        for steps in ([], None, [("x", object())], [("", Transformer())],
                      [("x", Transformer()), ("x", Transformer())],
                      [("model", Estimator()), ("x", Transformer())], ["bad"]):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                Pipeline(steps)

    def test_estimator(self):
        model = Estimator()
        pipe = Pipeline([("center", Transformer()), ("model", model)])
        labels = np.array([0, 1])
        self.assertIs(pipe.fit([[1.], [3.]], labels), pipe)
        self.assertIs(model.y, labels)
        np.testing.assert_allclose(pipe.predict([[5.]]), [3.])
        with self.assertRaises(ValueError):
            pipe.transform([[5.]])

    def test_estimator_cannot_fit_transform(self):
        with self.assertRaises(ValueError):
            Pipeline([("model", Estimator())]).fit_transform([[1.]])

    def test_dual_method_consistency(self):
        X = [[1.], [3.]]
        a = Pipeline([("dual", Dual())])
        b = Pipeline([("dual", Dual())])
        np.testing.assert_allclose(a.fit_transform(X), b.fit(X).transform(X))
        np.testing.assert_allclose(a.predict([[5.]]), b.predict([[5.]]))

    def test_no_predict(self):
        pipe = Pipeline([("center", Transformer())]).fit([[1.]])
        with self.assertRaises(ValueError):
            pipe.predict([[1.]])


if __name__ == "__main__":
    unittest.main()

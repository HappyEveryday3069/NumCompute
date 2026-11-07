"""
Sequential transformation and optional estimator pipeline.
"""

class Pipeline:
    """Chain named transformers and an optional final estimator.

    Args: steps is a non-empty list of unique (name, object) pairs.
        Transformers need fit/transform; only the last step may use fit/predict.
    Objects are fitted in place. Shapes are defined by the individual steps.
    Raises: ValueError for invalid names, steps or required methods.
    Cost: O(s) setup time/space for s steps; operation costs sum step costs.
    """
    def __init__(self, steps):
        """Store and validate steps; see the class contract for parameters and errors."""
        if not isinstance(steps, list) or not steps:
            raise ValueError("steps must be a non-empty list.")
        self.steps = list(steps)
        self._is_fitted = False
        self._validate_steps()

    def _validate_steps(self):
        """Validate unique names and callable methods; raise ValueError. O(s) time/space."""
        if not self.steps:
            raise ValueError("steps must be non-empty.")
        names = set()
        for i, item in enumerate(self.steps):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("Each step must be a (name, object) pair.")
            name, step = item
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError("Step names must be non-empty, unique strings.")
            names.add(name)
            if not callable(getattr(step, "fit", None)):
                raise ValueError(f"Step '{name}' must implement fit.")
            if not callable(getattr(step, "transform", None)):
                if i != len(self.steps) - 1 or not callable(getattr(step, "predict", None)):
                    raise ValueError(f"Step '{name}' must implement transform.")

    def _check_fitted(self):
        """Reject unfitted/invalid pipelines with ValueError. O(s) time/space."""
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before transform or predict.")
        self._validate_steps()

    def _require_transform(self):
        """Require a final transform method; raise ValueError otherwise. O(1) time/space."""
        if not callable(getattr(self.steps[-1][1], "transform", None)):
            raise ValueError("The final estimator cannot transform; use fit and predict.")

    @staticmethod
    def _fit_transform_step(step, X):
        """Fit/transform X via step; return its output. Errors and cost follow step."""
        if callable(getattr(step, "fit_transform", None)):
            return step.fit_transform(X)
        step.fit(X)
        return step.transform(X)

    def _fit(self, X, y, return_transformed):
        """Fit steps in sequence; return self or transformed X. Errors/cost follow steps."""
        self._is_fitted = False
        self._validate_steps()
        if return_transformed:
            self._require_transform()
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = self._fit_transform_step(step, Xt)
        final = self.steps[-1][1]
        if callable(getattr(final, "transform", None)):
            if return_transformed:
                Xt = self._fit_transform_step(final, Xt)
            else:
                final.fit(Xt)
        elif y is None:
            final.fit(Xt)
        else:
            final.fit(Xt, y)
        self._is_fitted = True
        return Xt if return_transformed else self

    def fit(self, X, y=None):
        """Fit each step sequentially without changing the public API.

        Args: X has the first step's input shape; optional y is passed
            only to a final estimator without transform.
        Returns: self. Transformers receive fit(X), not y.
        Raises: ValueError for invalid steps; step exceptions propagate.
        Cost: sum of intermediate fit/transform and final fit costs;
            O(s) dispatch, with workspace and fitted state defined by steps.
        """
        return self._fit(X, y, return_transformed=False)

    def transform(self, X):
        """Apply every fitted transformer in order.

        Args: X matches the fitted first step's input shape.
        Returns: the final step's transformed array; shape depends on steps.
        Raises: ValueError before fit or if the final step cannot transform;
            step exceptions propagate.
        Cost: sum of transform costs plus O(s) dispatch;
            workspace depends on intermediate arrays.
        """
        self._check_fitted()
        self._require_transform()
        Xt = X
        for _, step in self.steps:
            Xt = step.transform(Xt)
        return Xt

    def fit_transform(self, X, y=None):
        """Fit the pipeline and return its transformed output.

        Args: X matches the first step; y is accepted but unused by transformers.
        Returns: the final transformed array, with shape defined by steps.
        Raises: ValueError if steps are invalid or the final step cannot transform;
            step exceptions propagate.
        Cost: sum of fit/transform costs plus O(s) dispatch;
            workspace and fitted state depend on steps.
        """
        return self._fit(X, y, return_transformed=True)

    def predict(self, X):
        """Transform through intermediate steps, then predict with the last.

        Args: X matches the fitted first step's input shape.
        Returns: the final estimator's prediction array and output shape.
        Raises: ValueError before fit or without final predict;
            step exceptions propagate.
        Cost: intermediate transform costs plus prediction and O(s) dispatch;
            workspace depends on intermediate arrays and the estimator.
        """
        self._check_fitted()
        final = self.steps[-1][1]
        if not callable(getattr(final, "predict", None)):
            raise ValueError("The final step must implement predict.")
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        return final.predict(Xt)

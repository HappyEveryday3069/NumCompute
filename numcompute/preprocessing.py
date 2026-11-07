import numpy as np


class StandardScaler:
    """Standardize features using z-score standardization.

    z = (x - mean) / std
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        # Avoid division by zero
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)

        return self

    def transform(self, X):
        if self.mean_ is None:
            raise ValueError(
                "The mean is missing. StandardScaler should be fitted first."
            )

        X = np.asarray(X, dtype=float)

        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Scale to a given range.

    Default range is [0, 1].
    """

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        min_value, max_value = self.feature_range

        if min_value >= max_value:
            raise ValueError("min_value must be less than max_value")

        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)

        return self

    def transform(self, X):
        if self.data_min_ is None or self.data_max_ is None:
            raise ValueError(
                "The data_min_/data_max_ is missing. MinMaxScaler should be fitted first."
            )

        X = np.asarray(X, dtype=float)

        min_value, max_value = self.feature_range

        data_range = self.data_max_ - self.data_min_

        # Avoid division by zero
        data_range = np.where(data_range == 0, 1.0, data_range)

        X_std = (
                (X - self.data_min_) / data_range
        )

        return (
            X_std * (max_value - min_value)
            + min_value
        )

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class OneHotEncoder:
    """Convert categorical variables to one-hot encoding."""

    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=object)
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object)

        encoded = [
            (X[:, i:i + 1] == cats).astype(float)
            for i, cats in enumerate(self.categories_)
        ]

        return np.hstack(encoded)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class SimpleImputer:
    """Replace Nan values."""

    def __init__(self, strategy="constant", fill_value=0):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        if self.strategy == "constant":
            self.statistics_ = np.full(X.shape[1], self.fill_value)
        elif self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        else:
            raise ValueError("Strategy must be 'constant' or 'mean'.")
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_out = X.copy()

        mask = np.isnan(X_out)
        X_out[mask] = self.statistics_[np.where(mask)[1]]

        return X_out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

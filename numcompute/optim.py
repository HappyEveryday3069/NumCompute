"""
Finite-difference gradients and Jacobians for real-valued functions.
"""

import numpy as np


def _validate_inputs(function, x, h, method):
    """Validate callable, finite x (n,), positive h and difference method.
    Returns (x copy, h); raises TypeError/ValueError. O(n) time/space.
    """
    if not callable(function):
        raise TypeError("function must be callable.")
    x = np.array(x, dtype=float, copy=True)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values.")
    if isinstance(h, (bool, np.bool_)) or np.ndim(h) != 0:
        raise ValueError("h must be a finite positive scalar.")
    try:
        h = float(h)
    except (TypeError, ValueError) as exc:
        raise ValueError("h must be a finite positive scalar.") from exc
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be a finite positive scalar.")
    if method not in ("central", "forward"):
        raise ValueError("method must be 'central' or 'forward'.")
    return x, h


def _evaluate(function, x, shape):
    """Evaluate a copied point and enforce a finite output of shape.
    Returns a float array; raises ValueError on mismatch. O(C_f+n+m) time, O(n+m) space.
    """
    value = np.asarray(function(x.copy()), dtype=float)
    if value.shape != shape:
        raise ValueError(f"Function output must have shape {shape}; got {value.shape}.")
    if not np.all(np.isfinite(value)):
        raise ValueError("Function output must contain only finite values.")
    return value.copy()


def _perturb(x, i, h, direction):
    """Copy x (n,) and shift coordinate i by direction*h.
    Returns (n,); raises ValueError if the step is unrepresentable. O(n) time/space.
    """
    point = x.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        point[i] += direction * h
    if not np.isfinite(point[i]) or point[i] == x[i]:
        raise ValueError("h cannot produce a finite, distinct point; rescale x or h.")
    return point


def grad(f, x, h=1e-5, method="central"):
    """Estimate the gradient using forward or central differences.

    Args: f maps (n,) to a scalar; x is finite (n,); h > 0;
        method is 'central' or 'forward'. Input x is not modified.
    Returns: float array (n,). Central error O(h^2); forward O(h).
    Raises: TypeError/ValueError for invalid inputs or non-finite outputs.
    Cost: O(n*(C_f+n)) time, O(n) space, excluding f workspace.
        Evaluations: n+1 forward or 2n+1 central, including validation.
    """
    x, h = _validate_inputs(f, x, h, method)
    baseline = _evaluate(f, x, ())
    result = np.empty_like(x)
    for i in range(x.size):
        plus = _evaluate(f, _perturb(x, i, h, 1), ())
        if method == "central":
            minus = _evaluate(f, _perturb(x, i, h, -1), ())
            result[i] = (plus - minus) / (2 * h)
        else:
            result[i] = (plus - baseline) / h
    if not np.all(np.isfinite(result)):
        raise ValueError("Derivative is non-finite; rescale the function or step.")
    return result


def jacobian(F, x, h=1e-5, method="central"):
    """Estimate partial derivatives of a vector-valued function.

    Args: F maps finite x (n,) to fixed (m,); h > 0;
        method is 'central' or 'forward'. Input x is not modified.
    Returns: float array (m,n); entry [j,i] is dF_j/dx_i.
    Raises: TypeError/ValueError for invalid inputs or output shape/values.
    Cost: O(n*(C_F+n+m)) time, O(m*n+n+m) space, excluding F workspace.
        Evaluations: n+1 forward or 2n+1 central.
    """
    x, h = _validate_inputs(F, x, h, method)
    baseline = np.asarray(F(x.copy()), dtype=float).copy()
    if baseline.ndim != 1 or not np.all(np.isfinite(baseline)):
        raise ValueError("F(x) must return a finite 1D array.")
    result = np.empty((baseline.size, x.size), dtype=float)
    for i in range(x.size):
        plus = _evaluate(F, _perturb(x, i, h, 1), baseline.shape)
        if method == "central":
            minus = _evaluate(F, _perturb(x, i, h, -1), baseline.shape)
            result[:, i] = (plus - minus) / (2 * h)
        else:
            result[:, i] = (plus - baseline) / h
    if not np.all(np.isfinite(result)):
        raise ValueError("Derivative is non-finite; rescale the function or step.")
    return result

import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    Raises ValueError if probabilities are invalid.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)

    # Check lengths
    if len(x) != len(p):
        raise ValueError("x and p must have same length")

    # Check probabilities non-negative
    if np.any(p < 0):
        raise ValueError("Probabilities must be non-negative")

    # Check probabilities sum to 1 (within tolerance)
    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("Probabilities must sum to 1")

    return float(np.dot(x, p))

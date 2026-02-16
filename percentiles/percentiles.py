import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    Supports scalar or list/array q.
    """
    x = np.sort(np.asarray(x, dtype=float))
    q = np.asarray(q, dtype=float)  # convert list -> array
    n = len(x)

    if n == 0:
        raise ValueError("Empty array")

    pos = (n - 1) * (q / 100.0)

    lower = np.floor(pos).astype(int)
    upper = np.ceil(pos).astype(int)

    fraction = pos - lower

    result = x[lower] + (x[upper] - x[lower]) * fraction

    return result

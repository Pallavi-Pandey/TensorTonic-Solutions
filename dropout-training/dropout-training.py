import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x, dtype=float)

    if rng is None:
        rng = np.random

    keep_prob = 1 - p

    # Binary mask
    mask = (rng.random(x.shape) < keep_prob).astype(float)

    # Scaled mask (inverted dropout)
    pattern = mask / keep_prob

    # Apply
    output = x * pattern

    return output, pattern

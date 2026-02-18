import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    Return (w_new, s_new).
    """
    # Convert to numpy arrays (important)
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)

    # Update moving average of squared gradients
    s_new = beta * s + (1 - beta) * (g ** 2)

    # Update parameters
    w_new = w - lr * g / (np.sqrt(s_new) + eps)

    return w_new, s_new

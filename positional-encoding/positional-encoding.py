import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pos = np.arange(seq_len)[:, np.newaxis]        # (seq_len, 1)
    i = np.arange(d_model)[np.newaxis, :]          # (1, d_model)

    # Compute angle rates
    angle_rates = 1 / (base ** (2 * (i // 2) / d_model))
    angles = pos * angle_rates                     # (seq_len, d_model)

    PE = np.zeros((seq_len, d_model))

    # Even indices -> sin
    PE[:, 0::2] = np.sin(angles[:, 0::2])

    # Odd indices -> cos
    PE[:, 1::2] = np.cos(angles[:, 1::2])

    return PE

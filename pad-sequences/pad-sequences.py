import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L)
    """
    N = len(seqs)

    # Determine max length
    if max_len is None:
        L = max((len(s) for s in seqs), default=0)
    else:
        L = max_len

    # Create padded array
    out = np.full((N, L), pad_value, dtype=float)

    # Fill rows
    for i, seq in enumerate(seqs):
        seq = np.asarray(seq)
        length = min(len(seq), L)  # truncate if needed
        out[i, :length] = seq[:length]

    return out

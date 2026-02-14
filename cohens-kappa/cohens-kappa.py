import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    rater1, rater2: arrays of categorical labels (same length)
    """
    rater1 = np.array(rater1)
    rater2 = np.array(rater2)
    
    assert len(rater1) == len(rater2)

    # Observed agreement
    po = np.mean(rater1 == rater2)

    # Get unique categories
    categories = np.unique(np.concatenate([rater1, rater2]))
    
    # Expected agreement
    pe = 0
    for c in categories:
        p1 = np.mean(rater1 == c)
        p2 = np.mean(rater2 == c)
        pe += p1 * p2

    # Handle edge case (avoid division by zero)
    if pe == 1:
        return 1.0

    kappa = (po - pe) / (1 - pe)
    return kappa

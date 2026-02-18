import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    y: 1D array of class labels
    """
    y = np.asarray(y)
    
    # Get class counts
    _, counts = np.unique(y, return_counts=True)
    
    # Convert to probabilities
    p = counts / counts.sum()
    
    # Keep only non-zero probabilities (stable log)
    p = p[p > 0]
    
    entropy = -np.sum(p * np.log2(p))
    
    return entropy

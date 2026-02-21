import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Map vocab word -> index
    index = {word: i for i, word in enumerate(vocab)}

    vec = np.zeros(len(vocab), dtype=int)

    for token in tokens:
        if token in index:
            vec[index[token]] += 1

    return vec
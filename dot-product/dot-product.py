def dot_product(x, y):
    if len(x) != len(y):
        raise ValueError("Vectors must have the same length")
    return sum(a * b for a, b in zip(x, y))


def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    vocab = set(tokens)
    V = len(vocab)

    counts = {}
    first_counts = {}

    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        counts[(w1, w2)] = counts.get((w1, w2), 0) + 1
        first_counts[w1] = first_counts.get(w1, 0) + 1

    probs = {}

    for w1 in vocab:
        for w2 in vocab:
            c = counts.get((w1, w2), 0)
            denom = first_counts.get(w1, 0) + V
            probs[(w1, w2)] = (c + 1) / denom if denom > 0 else 0.0

    return counts, probs
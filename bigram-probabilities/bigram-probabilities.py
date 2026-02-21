def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    # Vocabulary
    vocab = set(tokens)
    V = len(vocab)

    # Bigram counts
    counts = {}
    first_counts = {}

    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]

        counts[(w1, w2)] = counts.get((w1, w2), 0) + 1
        first_counts[w1] = first_counts.get(w1, 0) + 1

    # Probabilities with add-1 smoothing
    probs = {}

    for w1 in vocab:
        for w2 in vocab:
            c = counts.get((w1, w2), 0)
            denom = first_counts.get(w1, 0) + V
            probs[(w1, w2)] = (c + 1) / denom if denom > 0 else 0.0

    return counts, probs
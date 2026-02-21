import math

def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    N = len(actual_tokens)
    if N == 0:
        return float('inf')

    log_sum = 0.0

    for i in range(N):
        p = prob_distributions[i][actual_tokens[i]]

        # Avoid log(0)
        if p <= 0:
            return float('inf')

        log_sum += math.log(p)

    return math.exp(-log_sum / N)
def popularity_ranking(items, min_votes, global_mean):
    """
    Compute Bayesian weighted rating for each item.
    items: list of [R, v]
    Returns list of scores (same order)
    """
    scores = []

    for R, v in items:
        score = (v / (v + min_votes)) * R + (min_votes / (v + min_votes)) * global_mean
        scores.append(score)

    return scores

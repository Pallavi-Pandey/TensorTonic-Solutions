import numpy as np

def decision_tree_split(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    n_samples, n_features = X.shape

    def gini(labels):
        if len(labels) == 0:
            return 0
        _, counts = np.unique(labels, return_counts=True)
        p = counts / len(labels)
        return 1 - np.sum(p ** 2)

    best_feature = None
    best_threshold = None
    best_score = float('inf')

    for f in range(n_features):
        values = np.unique(X[:, f])

        # Midpoints between consecutive values
        thresholds = (values[:-1] + values[1:]) / 2

        for t in thresholds:
            left = y[X[:, f] <= t]
            right = y[X[:, f] > t]

            if len(left) == 0 or len(right) == 0:
                continue

            score = (
                len(left)/n_samples * gini(left) +
                len(right)/n_samples * gini(right)
            )

            if score < best_score:
                best_score = score
                best_feature = f
                best_threshold = t

    return [best_feature, best_threshold]
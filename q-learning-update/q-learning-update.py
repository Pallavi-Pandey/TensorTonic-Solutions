import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    Q = np.array(Q, dtype=float)  # copy to avoid modifying original

    # Best future value
    best_next = np.max(Q[s_next])

    # TD target
    target = r + gamma * best_next

    # TD update
    Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])

    return Q
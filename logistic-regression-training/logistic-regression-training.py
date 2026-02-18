import numpy as np

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    X: shape (n_samples, n_features)
    y: shape (n_samples,)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n, d = X.shape

    # Initialize parameters
    w = np.zeros(d)
    b = 0.0

    for _ in range(steps):
        # Linear output
        z = X @ w + b

        # Predictions
        pred = _sigmoid(z)

        # Error
        error = pred - y

        # Gradients
        grad_w = (X.T @ error) / n
        grad_b = np.sum(error) / n

        # Update
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b

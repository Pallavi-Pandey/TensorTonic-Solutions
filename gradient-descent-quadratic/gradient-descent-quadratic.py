def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    f(x) = a*x^2 + b*x + c
    """
    x = x0
    
    for _ in range(steps):
        grad = 2 * a * x + b   # derivative of ax^2 + bx + c
        x = x - lr * grad      # gradient descent update
    
    return x

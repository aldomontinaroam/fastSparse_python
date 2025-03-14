import numpy as np

def gen_synthetic(n, p):
    """
    Generates synthetic dataset.

    Parameters:
        n (int): Number of samples
        p (int): Number of features
    
    Returns:
        tuple: (X, y)
    """
    X = np.random.randn(n, p)
    y = np.random.choice([-1, 1], size=n)
    return X, y

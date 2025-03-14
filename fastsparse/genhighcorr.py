import numpy as np
from scipy.stats import multivariate_normal

def gen_synthetic_highcorr(n, p, k, seed, base_cor=0.9):
    """
    Generates a synthetic dataset with exponentially correlated features.

    Parameters:
        n (int): Number of samples
        p (int): Number of features
        k (int): Number of non-zeros in true coefficients
        seed (int): Random seed
        base_cor (float): Base correlation factor

    Returns:
        dict: {'X': Feature matrix, 'y': target vector, 'B': true coefficients}
    """
    np.random.seed(seed)
    
    # Generate correlation matrix
    corr_matrix = np.array([[base_cor ** abs(i - j) for j in range(p)] for i in range(p)])
    
    # Generate features from multivariate normal
    X = multivariate_normal.rvs(mean=np.zeros(p), cov=corr_matrix, size=n)
    
    # Generate true coefficients
    B = np.zeros(p)
    B[:k] = 1  # First k coefficients set to 1
    
    # Generate target values
    noise = np.random.normal(0, 1, n)
    y = X @ B + noise

    return {'X': X, 'y': y, 'B': B}

import numpy as np

def normalize(X):
    """
    Normalizes feature matrix X.
    
    Parameters:
        X (numpy.ndarray): Feature matrix
    
    Returns:
        numpy.ndarray: Normalized matrix
    """
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

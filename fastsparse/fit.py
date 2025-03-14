import numpy as np
import cvxpy as cp

def fastsparse_fit(X, y, lambda_):
    """
    Fits a sparse logistic regression model using L0-regularization.
    
    Parameters:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target vector
        lambda_ (float): Regularization parameter
    
    Returns:
        np.ndarray: Coefficients of the model
    """
    n, p = X.shape
    beta = cp.Variable(p)
    
    # Logistic Loss
    log_loss = cp.sum(cp.logistic(-cp.multiply(y, X @ beta)))
    
    # L0 Regularization using relaxed surrogate (L1 proxy)
    reg = lambda_ * cp.norm(beta, 1)

    # Objective
    objective = cp.Minimize(log_loss + reg)
    problem = cp.Problem(objective)
    problem.solve()
    
    return beta.value

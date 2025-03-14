
import numpy as np
import cvxpy as cp
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from utils import normalize

def fastsparse_coef(model, lambda_=None):
    """
    Extracts coefficients from the model.

    Parameters:
        model (dict): Trained models with coefficients
        lambda_ (float, optional): Specific lambda value to extract coefficients
    
    Returns:
        np.ndarray: Coefficients corresponding to lambda
    """
    if lambda_ is None:
        return model
    return model[lambda_]

def fastsparse_cvfit(X, y, lambda_seq):
    """
    Performs cross-validation over a sequence of lambda values.

    Parameters:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target vector
        lambda_seq (list): List of lambda values
    
    Returns:
        dict: Fitted models for each lambda
    """
    models = {l: fastsparse_fit(X, y, l) for l in lambda_seq}
    return models


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

def plot_fastsparse(model, lambda_seq):
    """
    Plots the regularization path.

    Parameters:
        model (dict): Trained models
        lambda_seq (list): List of lambda values
    """
    coefs = [fastsparse_coef(model, l) for l in lambda_seq]
    plt.plot(lambda_seq, coefs)
    plt.xlabel("Lambda")
    plt.ylabel("Coefficients")
    plt.title("Regularization Path")
    plt.show()

def fastsparse_predict(model, X, lambda_):
    """
    Makes predictions using the fitted sparse model.

    Parameters:
        model (dict): Trained model
        X (numpy.ndarray): Feature matrix
        lambda_ (float): Regularization parameter
    
    Returns:
        numpy.ndarray: Predictions
    """
    beta = fastsparse_coef(model, lambda_)
    return X @ beta

def print_fastsparse(model):
    """
    Prints summary of the fitted sparse model.

    Parameters:
        model (dict): Fitted model object.
    """
    print("FastSparse Model Summary")
    print("----------------------------")
    for lambda_, beta in model.items():
        print(f"Lambda: {lambda_:.6f}, Non-Zero Coefficients: {sum(beta != 0)}")
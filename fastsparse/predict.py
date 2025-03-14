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

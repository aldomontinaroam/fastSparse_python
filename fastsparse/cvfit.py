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

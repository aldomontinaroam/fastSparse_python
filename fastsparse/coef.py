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

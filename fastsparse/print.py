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
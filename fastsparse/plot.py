import matplotlib.pyplot as plt

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

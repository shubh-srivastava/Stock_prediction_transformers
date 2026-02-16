import numpy as np


def calculate_metrics(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    # MAE
    mae = np.mean(np.abs(actual - predicted))

    # MSE
    mse = np.mean((actual - predicted) ** 2)

    # RMSE
    rmse = np.sqrt(mse)

    # RSE (Relative Squared Error)
    numerator = np.sum((actual - predicted) ** 2)
    denominator = np.sum((actual - np.mean(actual)) ** 2)
    rse = numerator / denominator

    return mae, mse, rmse, rse

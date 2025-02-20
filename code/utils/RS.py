import numpy as np


class ComputeRS:

    def __init__(self):
        pass

    @staticmethod
    def rs_statistic(series):
        t = len(series)
        mean = np.mean(series)
        y = series - mean
        r = np.max(np.cumsum(y)) - np.min(np.cumsum(y))
        s = np.std(series)
        return r / s

    @staticmethod
    def compute_S_modified(series):
        t = len(series)  # Number of observations
        mean_y = np.mean(series)  # Mean of the series
        rho_1 = np.abs(np.corrcoef(series[:-1], series[1:])[0, 1])  # First-order autocorrelation

        # Calculate q according to Andrews (1991)
        q = ((3 * t) / 2) ** (1 / 3) * ((2 * rho_1) / (1 - rho_1)) ** (2 / 3)
        q = int(np.floor(q))

        # First term: classical variance
        var_term = np.sum((series - mean_y) ** 2) / t

        # Second term: weighted sum of autocovariances
        auto_cov_term = 0
        for j in range(1, q + 1):  # j ranges from 1 to q
            w_j = 1 - (j / (q + 1))  # Newey-West weights
            sum_cov = np.sum((series[:-j] - mean_y) * (series[j:] - mean_y))  # Lagged autocovariance
            auto_cov_term += w_j * sum_cov

        auto_cov_term = (2 / t) * auto_cov_term

        s_quared = var_term + auto_cov_term
        return s_quared

    @staticmethod
    def rs_modified_statistic(series):
        t = len(series)
        mean = np.mean(series)
        y = series - mean
        cum_sum = np.cumsum(y)
        r = np.max(cum_sum) - np.min(cum_sum)
        sigma = np.sqrt(ComputeRS.compute_S_modified(series))

        return r / sigma

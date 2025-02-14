# rs_statistic.py
import numpy as np


class RSStatisticCalculator:
    @staticmethod
    def rs_statistic(series):
        T = len(series)
        mean = np.mean(series)
        Y = series - mean
        R = np.max(np.cumsum(Y)) - np.min(np.cumsum(Y))
        S = np.std(series)
        return R / S

    @staticmethod
    def compute_S_modified(r):
        T = len(r)
        mean_Y = np.mean(r)
        rho_1 = np.abs(np.corrcoef(r[:-1], r[1:])[0, 1])

        q = int(np.floor(((3 * T) / 2) ** (1 / 3) * ((2 * rho_1) / (1 - rho_1)) ** (2 / 3)))

        var_term = np.sum((r - mean_Y) ** 2) / T
        auto_cov_term = 0

        for j in range(1, q + 1):
            w_j = 1 - (j / (q + 1))
            sum_cov = np.sum((r[:-j] - mean_Y) * (r[j:] - mean_Y))
            auto_cov_term += w_j * sum_cov

        auto_cov_term = (2 / T) * auto_cov_term
        return var_term + auto_cov_term

    @staticmethod
    def rs_modified_statistic(series):
        T = len(series)
        mean = np.mean(series)
        Y = series - mean
        cum_sum = np.cumsum(Y)
        R = np.max(cum_sum) - np.min(cum_sum)

        sigma = np.sqrt(RSStatisticCalculator.compute_S_modified(series))

        return R / sigma

import numpy as np


class ComputeRS:

    def __init__(self):
        pass

    @staticmethod
    def rs_statistic(series, window_size=0):
        if window_size < len(series):
            window_size = len(series)
        s = series.iloc[len(series) - window_size: len(series)]
        mean = np.mean(s)
        y = s - mean
        r = np.max(np.cumsum(y)) - np.min(np.cumsum(y))
        sigma = np.std(s)
        return r / sigma

    @staticmethod
    def compute_S_modified(series, chin=False):
        s = series
        t = len(s)
        mean_y = np.mean(s)
        s = s.squeeze()

        if not chin:
            rho_1 = np.corrcoef(s[:-1], s[1:])[0, 1]

            if rho_1 < 0:
                return np.sum((s - mean_y) ** 2) / t

            q = ((3 * t) / 2) ** (1 / 3) * ((2 * rho_1) / (1 - (rho_1**2))) ** (2 / 3)
        else:
            q = 4*(t/100)**(2/9)

        q = int(np.floor(q))

        var_term = np.sum((s - mean_y) ** 2) / t

        auto_cov_term = 0
        for j in range(1, q + 1):
            w_j = 1 - (j / (q + 1))
            sum_cov = np.sum((s[:-j] - mean_y) * (s[j:] - mean_y))
            auto_cov_term += w_j * sum_cov

        auto_cov_term = (2 / t) * auto_cov_term

        s_quared = var_term + auto_cov_term
        return s_quared

    @staticmethod
    def rs_modified_statistic(series, window_size=0, chin=False):
        if window_size > len(series):
            window_size = len(series)

        s = series.iloc[len(series) - window_size: len(series)]
        y = s - np.mean(s)
        r = np.max(np.cumsum(y)) - np.min(np.cumsum(y))
        sigma = np.sqrt(ComputeRS.compute_S_modified(s, chin))

        return r / sigma

# market_evaluator.py
from rs_statistic import RSStatisticCalculator


class MarketEvaluator:
    @staticmethod
    def evaluate_market_persistence(price_series, window_size=25):
        hurst_exponents = []
        rs_values = []
        rs_modified_values = []
        t_statistics = []

        for start in range(len(price_series) - window_size):
            window = price_series[start:start + window_size]

            rs_value = RSStatisticCalculator.rs_statistic(window)
            rs_modified_value = RSStatisticCalculator.rs_modified_statistic(window)
            t_stat = rs_modified_value / (window.shape[0] ** 0.5)

            rs_values.append(rs_value)
            rs_modified_values.append(rs_modified_value)
            t_statistics.append(t_stat)

        return rs_values, rs_modified_values, t_statistics

# market_evaluator.py
from rs_statistic import RSStatisticCalculator


class MarketEvaluator:
    """
    A class to evaluate market persistence using statistical methods.
    """

    @staticmethod
    def evaluate_market_persistence(price_series: list[float], window_size: int = 25) -> tuple[
        list[float], list[float], list[float]]:
        """
        Evaluates market persistence by computing R/S statistics and t-statistics over rolling windows.

        Args:
            price_series (list[float]): The series of price data.
            window_size (int): The rolling window size for evaluation.

        Returns:
            tuple: Three lists containing R/S values, modified R/S values, and t-statistics.
        """
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

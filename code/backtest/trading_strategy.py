class TradingStrategy:
    @staticmethod
    def select_trading_strategy(rs_t_stat: float, momentum_threshold: float = 1.620,
                                mean_reversion_threshold: float = 0.2) -> str:
        """
        Selects a trading strategy based on the modified R/S statistic.

        Args:
            rs_t_stat (float): The modified R/S t-statistic value.
            momentum_threshold (float): Threshold above which the strategy is classified as momentum.
            mean_reversion_threshold (float): Threshold below which the strategy is classified as mean reversion.

        Returns:
            str: "Momentum" if the value is above the momentum threshold,
                 "Mean Reversion" if below the mean reversion threshold,
                 otherwise "Neutral".
        """
        if rs_t_stat > momentum_threshold:
            return "Momentum"
        elif rs_t_stat < mean_reversion_threshold:
            return "Mean Reversion"
        else:
            return "Neutral"
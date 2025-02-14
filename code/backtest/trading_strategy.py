# trading_strategy.py

class TradingStrategy:
    @staticmethod
    def select_trading_strategy(rs_t_stat, momentum_threshold=1.620, mean_reversion_threshold=0.2):
        """
        Sélectionne une stratégie de trading en fonction de la statistique R/S modifiée.
        """
        if rs_t_stat > momentum_threshold:
            return "Momentum"
        elif rs_t_stat < mean_reversion_threshold:
            return "Mean Reversion"
        else:
            return "Neutral"

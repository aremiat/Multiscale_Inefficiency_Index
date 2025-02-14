# backtester.py
import numpy as np
import plotly.graph_objects as go
from loader import Loader
from market_evaluator import MarketEvaluator
from trading_strategy import TradingStrategy


class Backtester:
    """
    A class to backtest a trading strategy based on market persistence evaluation.
    """

    def __init__(self, ticker: str, start_date: str, end_date: str, display: bool = False, window_size: int = 800,
                 initial_equity: float = 100000):
        """
        Initializes the backtester with necessary parameters.

        Args:
            ticker (str): The stock ticker symbol.
            start_date (str): The start date for historical data.
            end_date (str): The end date for historical data.
            display (bool): Whether to display the performance plot.
            window_size (int): The size of the rolling window for evaluation.
            initial_equity (float): The starting capital for backtesting.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.display = display
        self.window_size = window_size
        self.equity = initial_equity
        self.balance = initial_equity
        self.positions = 0
        self.performance = []

    def run_backtest(self) -> None:
        """
        Executes the backtesting process using historical price data.
        """
        price_data = Loader(self.ticker, self.start_date, self.end_date).fetch_data()
        log_p = np.log(price_data.values)
        r = np.diff(log_p.ravel())

        for i in range(self.window_size, len(r)):
            window_data = r[i - self.window_size:i]
            _, _, t_statistics = MarketEvaluator.evaluate_market_persistence(window_data)
            strategy = TradingStrategy.select_trading_strategy(t_statistics[-1])

            if strategy == "Momentum" and self.positions == 0:
                self.positions = self.balance / price_data.iloc[i].item()
                self.balance -= self.positions * price_data.iloc[i]
            elif strategy == "Mean Reversion" and self.positions != 0:
                self.balance += self.positions * price_data.iloc[i].item()
                self.positions = 0

            equity_value = self.balance + (self.positions * price_data.iloc[i])
            self.performance.append(equity_value)

        if self.display:
            self.plot_performance()

    def plot_performance(self) -> None:
        """
        Plots the equity performance over time.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(self.performance))),
            y=self.performance,
            mode='lines',
            name='Performance'
        ))
        fig.update_layout(
            title=f"Backtest Performance for {self.ticker}",
            xaxis_title="Time",
            yaxis_title="Equity Value",
            template="plotly_dark"
        )
        fig.show()

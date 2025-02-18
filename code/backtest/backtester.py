import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loader import Loader
from market_evaluator import MarketEvaluator
from trading_strategy import TradingStrategy


class Backtester:
    """
    A class to backtest a trading strategy based on market persistence evaluation.
    """

    def __init__(self, ticker: str, start_date: str, end_date: str, data_source: str = "yahoo",
                 csv_path: str = None, display: bool = False, window_size: int = 800,
                 initial_equity: float = 100000):
        """
        Initializes the backtester with necessary parameters.

        Args:
            ticker (str): The stock ticker symbol.
            start_date (str): The start date for historical data.
            end_date (str): The end date for historical data.
            data_source (str): The source of data ('yahoo' for Yahoo Finance or 'csv' for local file).
            csv_path (str, optional): Path to CSV file if using local data.
            display (bool): Whether to display the performance plot.
            window_size (int): The size of the rolling window for evaluation.
            initial_equity (float): The starting capital for backtesting.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data_source = data_source
        self.csv_path = csv_path
        self.display = display
        self.window_size = window_size
        self.equity = initial_equity
        self.balance = initial_equity
        self.positions = 0
        self.performance = []

    def load_data(self) -> pd.Series:
        """
        Loads historical price data from either Yahoo Finance or a CSV file.

        Returns:
            pd.Series: A series containing the adjusted closing prices.
        """
        if self.data_source == "csv":
            if not self.csv_path:
                raise ValueError("CSV file path must be provided when using 'csv' as data source.")

            # Load CSV and filter by ticker and date range
            df = pd.read_csv(self.csv_path, parse_dates=["Date"], index_col="Date")

            if not self.start_date or not self.end_date:
                self.start_date = df.index[0]
                self.end_date = df.index[-1]

            # Filter data within the given range
            filtered_data = df.loc[self.start_date:self.end_date].dropna()
        else:
            filtered_data = Loader(self.ticker, self.start_date, self.end_date).fetch_data()

        return filtered_data

    def run_backtest(self) -> None:
        """
        Executes the backtesting process using historical price data.
        """
        price_data = self.load_data()
        log_p = np.log(price_data.values)
        r = np.diff(log_p.ravel())

        for i in range(self.window_size, len(r)):
            window_data = r[i - self.window_size:i]
            _, _, t_statistics = MarketEvaluator.evaluate_market_persistence(window_data)
            strategy = TradingStrategy.select_trading_strategy(t_statistics[-1])

            if strategy == "Momentum" and self.positions == 0:
                self.positions = self.balance / price_data.iloc[i]
                self.balance -= self.positions * price_data.iloc[i]
            elif strategy == "Mean Reversion" and self.positions != 0:
                self.balance += self.positions * price_data.iloc[i]
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

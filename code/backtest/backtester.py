# backtester.py
import numpy as np
import plotly.graph_objects as go
from data_fetcher import DataFetcher
from market_evaluator import MarketEvaluator
from trading_strategy import TradingStrategy


class Backtester:
    def __init__(self, ticker, start_date, end_date, window_size=800, initial_equity=100000):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.equity = initial_equity
        self.balance = initial_equity
        self.positions = 0
        self.performance = []

    def run_backtest(self):
        price_data = DataFetcher(self.ticker, self.start_date, self.end_date).fetch_data()
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

        self.plot_performance()

    def plot_performance(self):
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

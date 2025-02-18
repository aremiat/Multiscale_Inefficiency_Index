
from backtester import Backtester

if __name__ == "__main__":

    backtester = Backtester(
        ticker="",
        start_date="1995-01-02",
        end_date="2024-12-31",
        data_source="csv",
        csv_path="../Loader/sp500_prices_1995_2024_final.csv",
        display=True,
        window_size=800,
        initial_equity=100000
    )

    # Exécution du backtest
    backtester.run_backtest()

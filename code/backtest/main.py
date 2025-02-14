# main.py
from backtester import Backtester

if __name__ == "__main__":
    ticker = "^GSPC"  # Exemple : Indice S&P 500
    start_date = "1995-01-02"
    end_date = "2024-01-01"

    backtester = Backtester(ticker, start_date, end_date)
    backtester.run_backtest()

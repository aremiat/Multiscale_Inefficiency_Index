# data_fetcher.py
import yfinance as yf
import pandas as pd

class Loader:
    def __init__(self, ticker: str, start_date: str, end_date: str):
        """
        Initializes the Loader object with the stock ticker and date range.

        Args:
            ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple).
            start_date (str): The start date for fetching data (format: 'YYYY-MM-DD').
            end_date (str): The end date for fetching data (format: 'YYYY-MM-DD').
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self) -> pd.Series:
        """
        Fetches the closing stock price data for the specified ticker and date range.

        Uses the Yahoo Finance API to download historical stock data.

        Returns:
            pd.Series: A pandas Series containing the closing stock prices.
        """
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return data['Close']

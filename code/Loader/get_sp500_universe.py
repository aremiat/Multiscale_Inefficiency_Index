import pandas as pd
import yfinance as yf
import numpy as np
import time
import os
import requests


# Function to get the S&P 500 tickers for a specific year
def get_sp500_tickers(year: int) -> list:
    url = f"https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)

    if response.status_code != 200:
        print("Failed to retrieve data from Wikipedia")
        return []

    dfs = pd.read_html(response.text)
    sp500_df = dfs[0]
    tickers = sp500_df['Symbol'].tolist()

    return tickers


# Function to fetch historical closing prices for each ticker
def fetch_prices_for_tickers(tickers, start_date, end_date):
    price_data = {}

    # Create a date range from the start to the end date
    date_range = pd.date_range(start=start_date, end=end_date,
                               freq='B')  # Business days only (excludes weekends and holidays)

    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date)

            if data.empty:
                print(f"No data for {ticker}")
                continue

            # Filter only the 'Close' price
            data = data[['Close']]
            # Reindex the data with the full date range
            data = data.reindex(date_range)
            # Store the data under the ticker symbol
            price_data[ticker] = data['Close']

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

        # Add delay to avoid hitting Yahoo Finance request limits
        time.sleep(1)

    # Convert the dictionary to a DataFrame
    price_df = pd.concat(price_data, axis=1)

    return price_df


# Fetch the tickers for 1995 to 2024
sp500_tickers_by_year = {}
for year in range(1995, 2025):
    tickers = get_sp500_tickers(year)
    sp500_tickers_by_year[year] = tickers


all_tickers = set([ticker for tickers in sp500_tickers_by_year.values() for ticker in tickers])

# Fetch historical prices for all tickers from 1995 to 2024
price_df = fetch_prices_for_tickers(list(all_tickers), start_date='1995-01-02', end_date='2024-12-31')

# Save the data to a CSV file
output_path = 'sp500_prices_1995_2024_final.csv'
price_df.to_csv(output_path)

print(f"Data has been saved to {output_path}")

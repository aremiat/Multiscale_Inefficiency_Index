import pandas as pd
import yfinance as yf
import time
import requests


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


def fetch_prices_for_tickers(tickers, start_date, end_date):
    price_data = {}

    date_range = pd.date_range(start=start_date, end=end_date,
                               freq='B')

    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date)

            if data.empty:
                print(f"No data for {ticker}")
                continue

            data = data[['Close']]
            data = data.reindex(date_range)
            price_data[ticker] = data['Close']

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

        time.sleep(1)

    price_df = pd.concat(price_data, axis=1)

    return price_df

if __name__ == "__main__":

    sp500_tickers_by_year = {}
    for year in range(1995, 2025):
        tickers = get_sp500_tickers(year)
        sp500_tickers_by_year[year] = tickers

    all_tickers = set([ticker for tickers in sp500_tickers_by_year.values() for ticker in tickers])
    price_df = fetch_prices_for_tickers(list(all_tickers), start_date='1995-01-02', end_date='2024-12-31')
    output_path = 'sp500_prices_1995_2024_final.csv'
    price_df.to_csv(output_path)

    print(f"Data has been saved to {output_path}")

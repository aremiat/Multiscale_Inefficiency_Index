import pandas as pd
import requests

def get_sp500_tickers(year: int) -> list:
    """
    Retrieve the list of S&P 500 tickers for a given year from Wikipedia.

    Args:
        year (int): The year to fetch the S&P 500 tickers for.

    Returns:
        list: A list of tickers for the specified year.
    """
    url = f"https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)

    if response.status_code != 200:
        print("Failed to retrieve data from Wikipedia")
        return []

    # Load the page into a pandas DataFrame directly
    dfs = pd.read_html(response.text)

    # The first dataframe should contain the S&P 500 list, we'll check the right one
    sp500_df = dfs[0]

    # Extract tickers for the specified year
    tickers = sp500_df['Symbol'].tolist()

    return tickers


# Loop over the years 1995 to 2024 and collect tickers
sp500_tickers_by_year = {}
for year in range(1995, 2025):
    print(f"Fetching tickers for {year}...")
    tickers = get_sp500_tickers(year)
    sp500_tickers_by_year[year] = tickers
    print(f"Found {len(tickers)} tickers for {year}")

# Convert the dictionary into a DataFrame
data = []
for year, tickers in sp500_tickers_by_year.items():
    for ticker in tickers:
        data.append([year, ticker])

# Create a DataFrame
df = pd.DataFrame(data, columns=['Year', 'Ticker'])

grouped = df.groupby('Year')['Ticker'].apply(list).reset_index()

grouped.to_csv('sp500_tickers_by_year.csv', index=False)

print("Data has been saved to 'sp500_tickers_by_year.csv'.")

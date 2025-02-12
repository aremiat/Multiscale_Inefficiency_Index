import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import os


DATA_PATH = os.path.dirname(__file__) + "/../data"

# Dickey-Fuller Test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    if result[1] < 0.05:
        print("The series is stationary")
    else:
        print("The series is not stationary, differencing recommended")


# Function to calculate the R/S statistic
def rs_statistic(series):
    T = len(series)
    mean = np.mean(series)
    Y = series - mean
    R = np.max(np.cumsum(Y)) - np.min(np.cumsum(Y))
    S = np.std(series)
    return R / S


def compute_S_modified(r):
    T = len(r)  # Number of observations
    mean_Y = np.mean(r)  # Mean of the series
    rho_1 = np.abs(np.corrcoef(r[:-1], r[1:])[0, 1])  # First-order autocorrelation

    # Calculate q according to Andrews (1991)
    q = ((3 * T) / 2) ** (1 / 3) * ((2 * rho_1) / (1 - rho_1)) ** (2 / 3)
    q = int(np.floor(q))

    # First term: classical variance
    var_term = np.sum((r - mean_Y) ** 2) / T

    # Second term: weighted sum of autocovariances
    auto_cov_term = 0
    for j in range(1, q + 1):  # j ranges from 1 to q
        w_j = 1 - (j / (q + 1))  # Newey-West weights
        sum_cov = np.sum((r[:-j] - mean_Y) * (r[j:] - mean_Y))  # Lagged autocovariance
        auto_cov_term += w_j * sum_cov

    auto_cov_term = (2 / T) * auto_cov_term

    S_squared = var_term + auto_cov_term
    return S_squared


# Function to calculate the modified R/S statistic
def rs_modified_statistic(series):
    T = len(series)
    mean = np.mean(series)
    Y = series - mean
    cum_sum = np.cumsum(Y)
    R = np.max(cum_sum) - np.min(cum_sum)

    sigma = np.sqrt(compute_S_modified(series))

    return R / sigma

all_results = pd.DataFrame()

if __name__ == "__main__":

    tickers = ["^GSPC", "^FTSE", "^SBF250", "^TOPX", "^GSPTSE"]
    for ticker in tickers:
        # 1968-01-02, 1996-06-10 TOPX True
        # 1995-01-02, 2024-12-31 GSPC True
        p = yf.download(ticker, start="2023-12-31", end="2024-12-31", progress=False)['Close']
        log_p = np.log(p.values)
        r = np.diff(log_p.ravel())

        # Stationarity test of the series (Dickey-Fuller)
        adf_test(r)

        rs_modified = rs_modified_statistic(r)
        S_modified = compute_S_modified(r)

        rs_value = rs_statistic(r)
        rs_modified_value = rs_modified_statistic(r)

        hurst_rs = np.log(rs_value) / np.log(len(r))
        hurst_rs_modified = np.log(rs_modified_value) / np.log(len(r))
        critical_value = rs_modified_value / np.sqrt(len(r))

        h_true = bool(critical_value > 1.620)  # critical values (10, 5, 0.5) are 1.620, 1.747, 2.098

        print(f"Ticker: {ticker}")
        print(f"R/S Statistic: {rs_value}")
        print(f"Modified R/S Statistic: {rs_modified_value}")
        print(f"Hurst Exponent: {hurst_rs}")
        print(f"Modified Hurst Exponent: {hurst_rs_modified}")
        print(f"Critical Value of the Modified Hurst Exponent: {critical_value}")
        print("Long memory: ", h_true)

        ticker = ticker.replace("^", "")

        df_results = pd.DataFrame({
            "Ticker": [ticker],
            "R/S": [np.round(rs_value, 2)],  # Round to 2 decimal places
            "Hurst Exponent": [np.round(hurst_rs, 2)],  # Round to 2 decimal places
            "Modified Hurst Exponent": [np.round(hurst_rs_modified, 2)],  # Round to 2 decimal places
            "Critical Value": [np.round(critical_value, 2)],  # Round to 2 decimal places
            "Long Memory": [h_true]
        })

        all_results = pd.concat([all_results, df_results])


all_results.to_csv(f"{DATA_PATH}/hurst_results.csv", index=False)
print(all_results)

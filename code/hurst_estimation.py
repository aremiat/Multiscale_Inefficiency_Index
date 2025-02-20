import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import os
from utils.RS import ComputeRS

DATA_PATH = os.path.dirname(__file__) + "/../data"

# Dickey-Fuller Test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    if result[1] < 0.05:
        print("The series is stationary")
        return result[1]
    else:
        print("The series is not stationary, differencing recommended")
        return result[1]


all_results = pd.DataFrame()

if __name__ == "__main__":

    tickers = ["^GSPC", "^FTSE", "^SBF250", "^TOPX", "^GSPTSE"]
    adf_data = pd.DataFrame()
    for ticker in tickers:
        # 1968-01-02, 1996-06-10 TOPX True
        # 1995-01-02, 2024-12-31 GSPC True
        p = yf.download(ticker, start="1995-01-02", end="2024-12-31", progress=False)['Close']
        ticker = ticker.replace("^", "")
        p_val_before = adf_test(p)

        log_p = np.log(p.values)
        r = np.diff(log_p.ravel())

        # Stationarity test of the series (Dickey-Fuller)
        p_val_after = adf_test(r)

        adf_data = pd.concat([adf_data, pd.DataFrame({
            "Ticker": [ticker],
            "P-Value on prices": [round(p_val_before, 3)],  # Round for clarity
            "P-Value on log differentiated return": [round(p_val_after, 3)]
        })], ignore_index=True)

        rs_modified = ComputeRS.rs_modified_statistic(r)
        S_modified = ComputeRS.compute_S_modified(r)

        rs_value = ComputeRS.rs_statistic(r)
        rs_modified_value = ComputeRS.rs_modified_statistic(r)

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

        df_results = pd.DataFrame({
            "Ticker": [ticker],
            "R/S": [f"{rs_value:.3f}"],  # Format to 3 decimal places
            "Hurst Exponent": [f"{hurst_rs:.3f}"],  # Format to 3 decimal places
            "Modified Hurst Exponent": [f"{hurst_rs_modified:.3f}"],  # Format to 3 decimal places
            "Critical Value": [f"{critical_value:.3f}"],  # Format to 3 decimal places
            "Long Memory": [h_true]
        })

        all_results = pd.concat([all_results, df_results])

all_results.to_csv(f"{DATA_PATH}/hurst_results.csv", index=False)
adf_data.to_csv(f"{DATA_PATH}/adf_results.csv", index=False)
print(all_results)

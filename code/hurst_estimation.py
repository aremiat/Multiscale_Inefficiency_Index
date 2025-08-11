import numpy as np
import pandas as pd
import os
from utils.RS import ComputeRS
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.ADF import adf_test

DATA_PATH = os.path.dirname(__file__) + "/../data"
IMG_PATH = os.path.dirname(__file__) + "/../img"


all_results = pd.DataFrame()

if __name__ == "__main__":
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 250)

    tickers = ["^GSPC", "^RUT", "^FTSE", "^N225", "^GDAXI"]
    noms_indices = {
        "^GSPC": "S\&P 500",
        "^RUT": "Russell 2000",
        "^FTSE": "FTSE 100",
        "^N225": "Nikkei 225",
        "^GDAXI": "DAX"
    }

    adf_data = pd.DataFrame()

    p = pd.read_csv(f"{DATA_PATH}/index_prices2.csv", index_col=0, parse_dates=True)
    fig = go.Figure()
    for ticker in tickers:
        p_ticker = p[ticker].dropna()
        ticker = ticker.replace("^", "")
        p_ticker = p_ticker.loc["1987-09-10": "2025-02-28"]
        log_p = np.log(p_ticker)
        r = log_p.diff().dropna()
        r = r.resample('M').last().dropna()
        rolling_critical = r.rolling(120).apply(
            lambda window: ComputeRS.rs_modified_statistic(window, window_size=len(window), chin=False) / np.sqrt(
                len(window)),
            raw=False
        ).dropna()

        p_val_before = adf_test(log_p)
        p_val_after = adf_test(r)
        adf_data = pd.concat([adf_data, pd.DataFrame({
            "Ticker": [ticker],
            "P-Value on log prices": [format(p_val_before, ".3f")],
            "P-Value on log differentiated return": [format(p_val_after, ".3f")]
        })], ignore_index=True)


        Q_tild = ComputeRS.rs_modified_statistic(r, window_size=len(r), chin=False)
        rs_value = ComputeRS.rs_statistic(r, window_size=len(r))

        hurst_rs = np.log(rs_value) / np.log(len(r))
        hurst_rs_modified = np.log(Q_tild) / np.log(len(r))
        critical_value = Q_tild / np.sqrt(len(r))

        h_true = bool(np.round(critical_value, 2) >= 1.620)
        print(hurst_rs, hurst_rs_modified)
        print(np.round(critical_value, 2))
        if h_true:
            print(f"Long memory: {h_true} for {ticker}")
            print(f"Ticker: {ticker} \n")
            print(f"R/S Statistic: {rs_value} \n")
            print(f"Modified R/S Statistic: {Q_tild} \n")
            print(f"Hurst Exponent: {hurst_rs} \n")
            print(f"Modified Hurst Exponent: {hurst_rs_modified} \n")
            print(f"Critical Value of the Modified Hurst Exponent: {critical_value} \n")
            print("Long memory: ", h_true)

        df_results = pd.DataFrame({
            "Ticker": [ticker],
            "R/S": [f"{rs_value:.3f}"],
            "Hurst Exponent": [f"{hurst_rs:.3f}"],
            "Modified Hurst Exponent": [f"{hurst_rs_modified:.3f}"],
            "Critical Value": [f"{critical_value:.3f}"],
            "Long Memory": [h_true]
        })
    #
        all_results = pd.concat([all_results, df_results])
    print(all_results)
    print(adf_data)
    # all_results["Ticker"] = all_results["Ticker"].map(noms_indices)
    # all_results.to_csv(f"{DATA_PATH}/hurst_results.csv", index=False)
    # adf_data["Ticker"] = adf_data["Ticker"].map(noms_indices)
    # adf_data.to_csv(f"{DATA_PATH}/adf_results.csv", index=False)

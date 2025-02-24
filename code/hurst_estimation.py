import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import os
from utils.RS import ComputeRS
import plotly.graph_objects as go

DATA_PATH = os.path.dirname(__file__) + "/../data"
IMG_PATH = os.path.dirname(__file__) + "/../img"

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
    # for ticker in tickers:
        # 1968-01-02, 1996-06-10 TOPX True
        # 1995-01-02, 2024-12-31 GSPC True

    # start_years = ["1995-01-02", "2005-01-02", "2015-01-02"]  # de 1995 à 2015
    #
    # for sy in start_years:
    for ticker in tickers:
        p = yf.download(ticker, start="1968-01-02", end="2024-12-31", progress=False)['Close']
        ticker = ticker.replace("^", "")
        p_val_before = adf_test(p)

        log_p = np.log(p)
        r = log_p.diff().dropna()
        #
        # Stationarity test of the series (Dickey-Fuller)
        p_val_after = adf_test(r)

        # adf_data = pd.concat([adf_data, pd.DataFrame({
        #     "Ticker": [ticker],
        #     "P-Value on prices": [round(p_val_before, 3)],  # Round for clarity
        #     "P-Value on log differentiated return": [round(p_val_after, 3)]
        # })], ignore_index=True)

        rs_modified = ComputeRS.rs_modified_statistic(r)
        S_modified = ComputeRS.compute_S_modified(r)

        rs_value = ComputeRS.rs_statistic(r)
        rs_modified_value = ComputeRS.rs_modified_statistic(r)

        hurst_rs = np.log(rs_value) / np.log(len(r))
        hurst_rs_modified = np.log(rs_modified_value) / np.log(len(r))
        critical_value = rs_modified_value / np.sqrt(len(r))

        h_true = bool(critical_value.values > 1.620)  # critical values (10, 5, 0.5) are 1.620, 1.747, 2.098

        print(f"Ticker: {ticker}")
        print(f"R/S Statistic: {rs_value}")
        print(f"Modified R/S Statistic: {rs_modified_value}")
        print(f"Hurst Exponent: {hurst_rs}")
        print(f"Modified Hurst Exponent: {hurst_rs_modified}")
        print(f"Critical Value of the Modified Hurst Exponent: {critical_value}")
        print("Long memory: ", h_true)
    #
    #     df_results = pd.DataFrame({
    #         "Ticker": [ticker],
    #         "R/S": [f"{rs_value:.3f}"],  # Format to 3 decimal places
    #         "Hurst Exponent": [f"{hurst_rs:.3f}"],  # Format to 3 decimal places
    #         "Modified Hurst Exponent": [f"{hurst_rs_modified:.3f}"],  # Format to 3 decimal places
    #         "Critical Value": [f"{critical_value:.3f}"],  # Format to 3 decimal places
    #         "Long Memory": [h_true]
    #     })
    #
    #     all_results = pd.concat([all_results, df_results])


    # Study of timestamp choice on the estimation

    # p = pd.read_csv(f"Loader/sp500_prices_1995_2024_final.csv", index_col=0, parse_dates=True)
    # #
    # p = p['BA'].dropna()
    # p = p.iloc[1:]
    # p = p.astype(float)
    # p.index = pd.to_datetime(p.index, format='%d/%m/%Y')
    #
    # log_p = np.log(p)
    # r = log_p.diff().dropna()

 #    start_years = range(1995, 2025 - 1 + 1)  # de 1995 à 2015
 # # date centrale de chaque fenêtre
 #    timestamp = [10]
 #    for tms in timestamp:
 #        hurst_values = []
 #        window_dates = []
 #        for year in start_years:
 #            start_date = pd.Timestamp(f"{year}-01-01")
 #            end_date = pd.Timestamp(f"{year + tms}-01-01")  # fenêtre de 10 ans
 #            window_data = r[start_date:end_date]
 #
 #            if len(window_data) > 0:
 #                h_val = ComputeRS.rs_modified_statistic(window_data.values)
 #                critical_value = h_val / np.sqrt(len(window_data))
 #                hurst_values.append(critical_value)
 #                central_date = start_date + (end_date - start_date) / 2
 #                window_dates.append(central_date)
 #            else:
 #                print(f"Pas de données pour la fenêtre {year} - {year + tms}")
 #
 #        # Tracé avec Plotly
 #        fig = go.Figure()
 #        fig.add_trace(go.Scatter(
 #            x=window_dates,
 #            y=hurst_values,
 #            mode='lines+markers',
 #            name='Critical Value'
 #        ))
 #        fig.update_layout(
 #            title=f"Critical Value on a {tms}-year rolling window (moving by 1 year) Boeing Stock, from 1995 to 2015",
 #            xaxis_title="Date (central year of the window)",
 #            yaxis_title="Critical Value",
 #            template="plotly_white"
 #        )
 #        fig.show()
 #
 #        output_filename = os.path.join(IMG_PATH, f"timestamp_analysis_critical_value_{tms}ans_BA.png")
 #        fig.write_image(output_filename)
 #        print(f"Graphique sauvegardé sous : {output_filename}")

# all_results.to_csv(f"{DATA_PATH}/hurst_results.csv", index=False)
# # adf_data.to_csv(f"{DATA_PATH}/adf_results.csv", index=False)
# print(all_results)

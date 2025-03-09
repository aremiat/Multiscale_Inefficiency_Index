import pandas as pd
import numpy as np
import os
from utils.RS import ComputeRS
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_PATH = os.path.dirname(__file__) + "/../data"
TICKER = ["^RUT"]

if __name__ == "__main__":

    p = pd.read_csv(f"{DATA_PATH}/index_prices2.csv", index_col=0, parse_dates=True)
    frequencies = {"D" : 2520, "W" : 520, "M" : 120}
    frequencies_results = pd.DataFrame()
    rolling_periods = pd.DataFrame()
    fig = go.Figure()
    start_years = ["1987-09-10", "1990-01-02", "1995-01-02", "2005-01-02", "2010-01-02", "2015-01-02"]
    # for freq, period in frequencies.items():
    #     p_ticker = p[TICKER].dropna()
    #     p_ticker = p_ticker.loc["1987-09-10": "2025-02-28"]
    #     log_p = np.log(p_ticker)
    #     r = log_p.diff().dropna()
    #     r = r.resample(freq).last().dropna()
    #     rolling_critical = r.rolling(period).apply(
    #         lambda window: ComputeRS.rs_modified_statistic(window, window_size=len(window), chin=False) / np.sqrt(
    #             len(window)),
    #         raw=False
    #     ).dropna()
    #     #
    #     # fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
    #     #                     subplot_titles=(f"{ticker} Price Evolution", "Rolling Critical Value"))
    #     # fig.add_trace(go.Scatter(x=p_ticker.index, y=p_ticker["^RUT"], mode='lines', name=f'Russel Price', line=dict(color='red')),
    #     #               row=1, col=1)
    #     # fig.add_trace(go.Scatter(x=rolling_critical.index, y=rolling_critical, mode='lines', name='Rolling Critical Value',
    #     #                          line=dict(color='green')), row=2, col=1)
    #     # fig.add_trace(
    #     #     go.Scatter(x=rolling_critical.index, y=[1.620] * len(rolling_critical), mode='lines', name='Threshold (V=1.620)',
    #     #                line=dict(color='red', dash='dash')), row=2, col=1)
    #     # fig.update_layout(title_text=f"{ticker} Analysis", height=800, width=1000, showlegend=True)
    #     # fig.update_xaxes(title_text="Date")
    #     # fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    #     # fig.update_yaxes(title_text="Critical Value 10%", row=2, col=1)
    #     # fig.show()
    #
    #     Q_tild = ComputeRS.rs_modified_statistic(r, window_size=len(r), chin=False)  # R-s modified
    #     rs_value = ComputeRS.rs_statistic(r, window_size=len(r))  # R/S
    #
    #     hurst_rs = np.log(rs_value) / np.log(len(r))
    #     hurst_rs_modified = np.log(Q_tild) / np.log(len(r))
    #     critical_value = Q_tild / np.sqrt(len(r))
    #
    #     h_true = bool(np.round(critical_value, 2) >= 1.620)  # critical values (10, 5, 0.5) are 1.620, 1.747, 2.098
    #     frequencies_results = pd.concat([frequencies_results, pd.DataFrame({
    #         "Frequency": [freq],
    #         "Hurst Exponent": [f"{hurst_rs.iloc[0]:.3f}"],
    #         "Modified Hurst Exponent": [f"{hurst_rs_modified:.3f}"],
    #         "Critical Value": [f"{critical_value:.3f}"],
    #         "Long Memory": [h_true]
    #     })], ignore_index=False)
    #
    #     fig.add_trace(go.Scatter(
    #         x=rolling_critical.index,
    #         y=rolling_critical["^RUT"],
    #         mode='lines',
    #         name=f'Rolling Critical {freq}'
    #     ))
    #
    #     rolling_periods = pd.concat([rolling_periods, pd.DataFrame({
    #         f"Rolling Critical {freq}": [rolling_critical]
    #     })], ignore_index=False)
    #
    # fig.add_trace(go.Scatter(
    #     x=rolling_critical.index,
    #     y=[1.620] * len(rolling_critical),
    #     mode='lines',
    #     name='Threshold (V=1.620)',
    #     line=dict(color='red', dash='dash')
    # ))
    #
    # fig.update_layout(
    #     title="Rolling Critical Value for Different Frequencies On Russel 2000",
    #     xaxis_title="Date",
    #     yaxis_title="Critical Value",
    #     showlegend=True
    # )
    #
    # fig.show()

    # rolling_periods.to_csv(f"{DATA_PATH}/rolling_periods.csv")
    # frequencies_results.to_csv(f"{DATA_PATH}/frequencies_results.csv", index=False)
    years_results = pd.DataFrame()
    for st in start_years:
        p_ticker = p[TICKER].dropna()
        p_ticker = p_ticker.loc[st: "2025-01-02"]
        log_p = np.log(p_ticker)
        r = log_p.diff().dropna()
        r = r.resample("M").last().dropna()
        Q_tild = ComputeRS.rs_modified_statistic(r, window_size=len(r), chin=False)
        rs_value = ComputeRS.rs_statistic(r, window_size=len(r))
        hurst_rs = np.log(rs_value) / np.log(len(r))
        hurst_rs_modified = np.log(Q_tild) / np.log(len(r))
        critical_value = Q_tild / np.sqrt(len(r))
        h_true = bool(np.round(critical_value, 2) >= 1.620)
        years_results = pd.concat([years_results, pd.DataFrame({
            "Period": [st + " - 2025-01-02"],
            "Hurst Exponent": [f"{hurst_rs.iloc[0]:.3f}"],
            "Modified Hurst Exponent": [f"{hurst_rs_modified:.3f}"],
            "Critical Value": [f"{critical_value:.3f}"],
            "Long Memory": [h_true]
        })], ignore_index=False)

    years_results.to_csv(f"{DATA_PATH}/timestamp_analysis.csv", index=False)
print('done')
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.RS import ComputeRS
import os


DATA_PATH = os.path.dirname(__file__) + "/../data"


if __name__ == "__main__":

    data = pd.read_csv(os.path.join(DATA_PATH, "russel_stocks.csv"), index_col=0, parse_dates=True)['^RUT']
    log_prices = np.log(data).dropna()
    returns = log_prices.diff().dropna()
    returns = returns

    rolling_modified_rs = returns.rolling(window=120).apply(
        lambda window: ComputeRS.rs_modified_statistic(window, len(window), chin=False) / np.sqrt(len(window)),
        raw=False
    ).dropna()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=(f"Russel Price Evolution", "Rolling Hurst Exponent"))
    fig.add_trace(go.Scatter(x=log_prices.index, y=log_prices, mode='lines', name=f'Russel Price', line=dict(color='red')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling_modified_rs.index, y=rolling_modified_rs, mode='lines', name='Rolling Hurst',
                             line=dict(color='green')), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=rolling_modified_rs.index, y=[1.620] * len(rolling_modified_rs), mode='lines', name='Threshold (V=1.620)',
                   line=dict(color='red', dash='dash')), row=2, col=1)
    fig.update_layout(title_text=f"Russel Analysis", height=800, width=1000, showlegend=True)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Rolling modified rs", row=2, col=1)
    fig.show()
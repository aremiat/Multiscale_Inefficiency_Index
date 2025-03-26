import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backtest import ComputeRS
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data")

if __name__ == "__main__":
# SP500 vs Russel 2000
    p = pd.read_csv(f"{DATA_PATH}/index_prices2.csv", index_col=0, parse_dates=True)
    all_prices = pd.concat([p["^GSPC"], p["^RUT"]], axis=1)
    all_prices = all_prices.loc["1987-10-09": "2024-12-31"]
    all_p = all_prices.pct_change().dropna()
    all_p['Diff'] = all_p['^GSPC'] - all_p["^RUT"]
    r = all_p['Diff']
    rebase_p = all_prices.loc["1987-10-09": "2024-12-31"].dropna()
    rebase_p = rebase_p / rebase_p.iloc[0]
    rolling_critical = r.rolling(120).apply(
        lambda window: ComputeRS.rs_modified_statistic(window, window_size=len(window), chin=False) / np.sqrt(
            len(window)),
        raw=False
    ).dropna()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=(f"Sp500 and Russel Price Evolution", "Rolling Critical Value"))
    fig.add_trace(
        go.Scatter(x=rebase_p.index, y=rebase_p["^GSPC"], mode='lines', name=f'Sp500 Price', line=dict(color='black')),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=rebase_p.index, y=rebase_p["^RUT"], mode='lines', name=f'Russel Price', line=dict(color='red')),
        row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling_critical.index, y=rolling_critical, mode='lines', name='Rolling Critical Value',
                             line=dict(color='green')), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=rolling_critical.index, y=[1.620] * len(rolling_critical), mode='lines',
                   name='Threshold (V=1.620)',
                   line=dict(color='red', dash='dash')), row=2, col=1)
    fig.update_layout(title_text=f"Sp500/Russel Analysis", height=800, width=1000, showlegend=True)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Critical Value 10%", row=2, col=1)
    fig.show()

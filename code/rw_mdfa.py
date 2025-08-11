import numpy as np
import plotly.graph_objects as go
import os
import pandas as pd

IMG_PATH = os.path.dirname(__file__) + "/../img/"
DATA_PATH = os.path.dirname(__file__) + "/../data/"
N = 10000
q_list = np.linspace(-5, 5, 21)
scales = np.unique(np.floor(np.logspace(np.log10(10), np.log10(500), 10)).astype(int))
s_example = 880

if __name__ == "__main__":
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 250)

    np.random.seed(43)
    increments = np.random.normal(0, 1, N)
    random_walk = np.cumsum(increments)
    returns = np.diff(random_walk)
    returns_centered = returns - np.mean(returns)
    Y = np.cumsum(returns_centered)
    data = pd.read_csv(os.path.join(DATA_PATH, "index_prices2.csv"), index_col=0, parse_dates=True)['^RUT']
    df_ticker = data.loc["1987-09-10":"2025-02-28"]
    df_ticker = df_ticker.dropna()
    returns = np.log(df_ticker).diff().dropna()
    returns_centered = returns - np.mean(returns)
    Y = np.cumsum(returns_centered)
    Y = Y.values.flatten()
    n_segments = len(returns) // s_example
    dates = pd.date_range(start="1987-09-10", periods=len(returns), freq='B')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=Y,
        mode='lines',
        name='Cumulative Profile'
    ))

    Y = pd.Series(Y, index=dates)
    for v in range(n_segments):
        start = v * s_example
        end = (v + 1) * s_example
        segment = Y.iloc[start:end]
        idx = segment.index
        coeffs = np.polyfit(np.arange(s_example), segment.values, 1)
        fit = np.polyval(coeffs, np.arange(s_example))
        fig.add_trace(go.Scatter(
            x=idx,
            y=fit + np.mean(returns),
            mode='lines',
            line=dict(dash='dash', width=5, color='black'),
            name=f"Segment {v}"
        ))
        fig.add_vline(x=idx[0], line=dict(color='gray', dash='dot'))
        fig.add_vline(x=idx[-1], line=dict(color='gray', dash='dot'))

    fig.update_layout(
        title=f"Cumulative Profile with Segment Partitioning (segment size = {s_example}) and Trend Removal, Random Walk",
        xaxis_title="Index",
        yaxis_title="Cumulative Profile"
    )
    fig.show()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.RS import ComputeRS
import os


DATA_PATH = os.path.dirname(__file__) + "/../data"

def non_overlapping_rolling(series, window, func):
    """
    Applique la fonction 'func' à des fenêtres non chevauchantes de taille 'window' sur la série.
    Retourne une Series avec, pour chaque segment, la valeur calculée, indexée par la date
    du dernier point du segment.
    """
    results = []
    indices = []
    n_segments = len(series) // window
    for i in range(n_segments):
        seg = series.iloc[i * window: (i + 1) * window]
        results.append(func(seg))
        indices.append(seg.index[-1])
    return pd.Series(results, index=indices)


if __name__ == "__main__":
    window_size = 120
    tickers = ['^RUT', '^GSPC']

    for tick in tickers:
        if tick == '^RUT':
            name = "Russel 2000"
        elif tick == '^GSPC':
            name = "S&P 500"

        data = pd.read_csv(os.path.join(DATA_PATH, "russel_stocks.csv"), index_col=0, parse_dates=True)[tick]
        data = data.loc["1987-09-10":"2025-02-28"]
        log_prices = np.log(data).dropna()
        returns = log_prices.diff().dropna()

        rolling_modified_rs = returns.rolling(window=window_size).apply(
            lambda window: np.log(ComputeRS.rs_modified_statistic(window, len(window), chin=False)) / np.log(len(window)),
            raw=False
        ).dropna()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.1,
                            row_heights=[0.7, 0.3],
                            subplot_titles=(f"{name} Log Price Evolution", "Rolling Modified R/S"))
        fig.add_trace(go.Scatter(x=log_prices.index, y=log_prices, mode='lines', name=f'Russel Price', line=dict(color='red')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=rolling_modified_rs.index, y=rolling_modified_rs, mode='lines', name='Rolling Modified R/S',
                                 line=dict(color='green')), row=2, col=1)

        fig.update_layout(title_text=f"{name} Analysis", showlegend=True)
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Log Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Rolling modified rs", row=2, col=1)
        fig.show()
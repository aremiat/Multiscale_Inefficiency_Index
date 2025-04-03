import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.RS import ComputeRS
import os


DATA_PATH = os.path.dirname(__file__) + "/../data"
IMG_PATH = os.path.dirname(__file__) + "/../img"
LOADER_PATH = os.path.dirname(__file__) + "/Loader"

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
    # Chargement des données pour '^RUT'
    data = pd.read_csv(os.path.join(DATA_PATH, "russel_stocks.csv"), index_col=0, parse_dates=True)['^RUT']
    log_prices = np.log(data).dropna()
    returns = log_prices.diff().dropna()

    # Calcul du rolling Hurst classique (overlapping) via R/S statistic sur 120 jours
    rolling_hurst = returns.rolling(window=120).apply(
        lambda window: np.log(ComputeRS.rs_statistic(window, len(window))) / np.log(len(window)),
        raw=False
    ).dropna()

    # Calcul du rolling modified R/S (non chevauchant) sur 120 jours
    rolling_modified_rs_non_overlap = non_overlapping_rolling(
        returns,
        window=120,
        func=lambda window: np.log(ComputeRS.rs_statistic(window, len(window))) / np.log(len(window))
    )

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=(f"Russel Price Evolution", "Rolling Modified R/S"))
    fig.add_trace(go.Scatter(x=log_prices.index, y=log_prices, mode='lines', name=f'Russel Price', line=dict(color='red')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling_hurst.index, y=rolling_hurst, mode='lines', name='Rolling Hurst',
                             line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=rolling_modified_rs_non_overlap.index, y=rolling_modified_rs_non_overlap, mode='lines',
                                name='Rolling Modified R/S (non overlapping)', line=dict(color='blue')), row=2, col=1)

    fig.update_layout(title_text=f"Russel Analysis", showlegend=True)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Rolling modified rs", row=2, col=1)
    fig.show()
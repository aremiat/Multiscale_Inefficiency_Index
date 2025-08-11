import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from statsmodels.tsa.stattools import acf
import plotly.io as pio
import os
import pandas as pd

IMAGE_PATH = os.path.dirname(__file__) + "/../img"

np.random.seed(43)
H_VALUES = [0.2, 0.35, 0.5, 0.65, 0.8]

def fbm(T, H, N=1000):
    """  Generate fractional Brownian motion (fBm) using Cholesky decomposition.

    Args:
        T (float): Total time duration.
        H (float): Hurst exponent (0 < H < 1).
        N (int): Number of time steps.
    Returns:
        times (np.ndarray): Array of time points.
        X (np.ndarray): Generated fBm values at those time points.
    """
    times = np.linspace(0, T, N)
    cov_matrix = np.array(
        [[0.5 * (t1 ** (2 * H) + t2 ** (2 * H) - abs(t1 - t2) ** (2 * H)) for t1 in times] for t2 in times])
    L = np.linalg.cholesky(cov_matrix + 1e-10 * np.eye(N))
    W = np.random.randn(N)
    X = np.dot(L, W)
    return times, X

if __name__ == "__main__":
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 250)

    fig = sp.make_subplots(rows=2, cols=5, subplot_titles=[f"H = {H}" for H in H_VALUES] + ["Autocorrelation"] * 5,
                           vertical_spacing=0.1)

    for i, H in enumerate(H_VALUES):
        times, X = fbm(T=1, H=H, N=1000)
        autocorr = acf(X, nlags=40, fft=True)

        fig.add_trace(
            go.Scatter(
                x=times, y=X, mode='lines', name=f"H={H}",
                line=dict(color='black')
            ),
            row=1, col=i + 1
        )
        fig.add_trace(
            go.Bar(
                x=np.arange(len(autocorr)), y=autocorr, name=f"ACF H={H}",
                marker_color='black'
            ),
            row=2, col=i + 1
        )
    fig.update_layout(
        title="Fractional Brownian Motion & Autocorrelation",
        height=800, width=1200, showlegend=False
    )
    for ann in fig.layout.annotations:
        if ann.text.startswith("H ="):
            ann.font.size = 30
    for i in range(1, 6):
        fig.update_yaxes(range=[-2.2, 3], row=1, col=i),

    pio.write_image(fig, f"{IMAGE_PATH}/fdm_autocorr.png", scale=5, width=1200, height=1000)
    fig.show()

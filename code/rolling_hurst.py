import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def compute_S_modified(r):
    T = len(r)  # Number of observations
    mean_Y = np.mean(r)  # Mean of the series
    rho_1 = np.abs(np.corrcoef(r[:-1], r[1:])[0, 1])  # First-order autocorrelation

    # Calculate q according to Andrews (1991)
    q = ((3 * T) / 2) ** (1 / 3) * ((2 * rho_1) / (1 - rho_1)) ** (2 / 3)
    q = int(np.floor(q))

    # First term: classical variance
    var_term = np.sum((r - mean_Y) ** 2) / T

    # Second term: weighted sum of autocovariances
    auto_cov_term = 0
    for j in range(1, q + 1):  # j ranges from 1 to q
        w_j = 1 - (j / (q + 1))  # Newey-West weights
        sum_cov = np.sum((r[:-j] - mean_Y) * (r[j:] - mean_Y))  # Lagged autocovariance
        auto_cov_term += w_j * sum_cov

    auto_cov_term = (2 / T) * auto_cov_term

    S_squared = var_term + auto_cov_term
    return S_squared

def rs_modified_statistic(series):
    T = len(series)
    mean = np.mean(series)
    Y = series - mean
    cum_sum = np.cumsum(Y)
    R = np.max(cum_sum) - np.min(cum_sum)

    sigma = np.sqrt(compute_S_modified(series))

    return R / sigma




if __name__ == "__main__":

    ticker = "^GSPC"
    adf_data = pd.DataFrame()
    # 1968-01-02, 1996-06-10 TOPX True
    # 1995-01-02, 2024-12-31 GSPC True
    p = yf.download(ticker, start="1995-01-02", end="2024-12-31", progress=False)['Close']
    ticker = ticker.replace("^", "")

    window_size = 252
    log_p = np.log(p.values)
    r = np.diff(log_p.ravel())

    hurst_values = []
    critical_values = []

    for i in range(len(r) - window_size + 1):
        window_data = r[i:i + window_size]
        rs_value = rs_modified_statistic(window_data)

        if rs_value is not np.nan and rs_value > 0:
            hurst_value = np.log(rs_value) / np.log(len(window_data))
            critical_value = rs_value / np.sqrt(len(window_data))
        else:
            hurst_value = np.nan
            critical_value = np.nan

        hurst_values.append(hurst_value)
        critical_values.append(critical_value)

    # Créer une série temporelle
    hurst_series = pd.Series(hurst_values, index=p.index[window_size:])
    critical_series = pd.Series(critical_values, index=p.index[window_size:])

    # 📊 Création des sous-graphiques interactifs
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("S&P 500 Price Evolution", "Rolling Critical Value"))

    # 1️⃣ Graphique du prix du S&P 500
    fig.add_trace(go.Scatter(x=p.index, y=p, mode='lines', name='S&P 500 Price', line=dict(color='black')), row=1,
                  col=1)

    # 2️⃣ Graphique de la valeur critique
    fig.add_trace(go.Scatter(x=critical_series.index, y=critical_series, mode='lines', name='Rolling Critical Value',
                             line=dict(color='green')), row=2, col=1)

    # Ajout de la ligne de seuil H = 0.5
    fig.add_trace(
        go.Scatter(x=critical_series.index, y=[0.5] * len(critical_series), mode='lines', name='Threshold (H=0.5)',
                   line=dict(color='red', dash='dash')), row=2, col=1)

    # 📌 Mise en forme du graphique
    fig.update_layout(title_text=f"S&P 500 Analysis ({ticker})", height=800, width=1000, showlegend=True)

    # 📅 Formatage des axes
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Critical Value", row=2, col=1)

    # 📊 Affichage interactif
    fig.show()

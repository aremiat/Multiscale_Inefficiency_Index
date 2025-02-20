import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.utils import ComputeRS


tickers = [
    "AAPL",  # Technologie (méga-cap)
    "JPM",   # Financier
    "JNJ",   # Santé
    "XOM",   # Énergie
    "WMT",   # Consommation de base
    "BA",    # Industriel (aéronautique)
    "DIS",   # Communication/Divertissement
    "LIN",   # Matériaux (chimie)
    "NEE",   # Utilitaires
    "SPG"    # Immobilier (REIT)
]

if __name__ == "__main__":

    # ticker = "MSFT"
    # adf_data = pd.DataFrame()
    # 1968-01-02, 1996-06-10 TOPX True
    # 1995-01-02, 2024-12-31 GSPC True
    # p = yf.download(ticker, start="1995-01-02", end="2024-12-31", progress=False)['Close']

    p = pd.read_csv("Loader/sp500_prices_1995_2024_final.csv", index_col=0, parse_dates=True)

    for ticker in tickers:
        p_ticker = p[ticker].dropna()
        p_ticker = p_ticker.iloc[1:]
        p_ticker = p_ticker.astype(float)
        p_ticker.index = pd.to_datetime(p_ticker.index, format='%d/%m/%Y')
        # ticker = ticker.replace("^", "")
        # p_ticker = p_ticker.loc['2018-12-31':'2024-12-31']

        window_size = 252
        log_p = np.log(p_ticker) # .values
        r = np.diff(log_p) # .ravel()

        hurst_values = []
        critical_values = []

        for i in range(len(r) - window_size + 1):
            window_data = r[i:i + window_size]
            rs_value = ComputeRS.rs_modified_statistic(window_data)

            if rs_value is not np.nan and rs_value > 0:
                hurst_value = np.log(rs_value) / np.log(len(window_data))
                critical_value = rs_value / np.sqrt(len(window_data))
            else:
                hurst_value = np.nan
                critical_value = np.nan

            hurst_values.append(hurst_value)
            critical_values.append(critical_value)

        # Créer une série temporelle
        hurst_series = pd.Series(hurst_values, index=p_ticker.index[window_size:])
        critical_series = pd.Series(critical_values, index=p_ticker.index[window_size:])

        # 📊 Création des sous-graphiques interactifs
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=(f"{ticker} Price Evolution", "Rolling Critical Value"))

        # 1️⃣ Graphique du prix du S&P 500
        fig.add_trace(go.Scatter(x=p_ticker.index, y=p_ticker, mode='lines', name=f'{ticker} Price', line=dict(color='black')), row=1,
                      col=1)

        # 2️⃣ Graphique de la valeur critique
        fig.add_trace(go.Scatter(x=critical_series.index, y=critical_series, mode='lines', name='Rolling Critical Value',
                                 line=dict(color='green')), row=2, col=1)

        # Ajout de la ligne de seuil H = 0.5
        fig.add_trace(
            go.Scatter(x=critical_series.index, y=[1.620] * len(critical_series), mode='lines', name='Threshold (V=1.620)',
                       line=dict(color='red', dash='dash')), row=2, col=1)

        # 📌 Mise en forme du graphique
        fig.update_layout(title_text=f"{ticker} Analysis", height=800, width=1000, showlegend=True)

        # 📅 Formatage des axes
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Critical Value 10%", row=2, col=1)

        # 📊 Affichage interactif
        fig.show()

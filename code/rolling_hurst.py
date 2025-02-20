import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.RS import ComputeRS
import os

DATA_PATH = os.path.dirname(__file__) + "/../data"

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

    # Chargement des prix
    p = pd.read_csv("Loader/sp500_prices_1995_2024_final.csv", index_col=0, parse_dates=True)
    results = pd.DataFrame()

    for ticker in tickers:
        p_ticker = p[ticker].dropna()
        p_ticker = p_ticker.iloc[1:]
        p_ticker = p_ticker.astype(float)
        p_ticker.index = pd.to_datetime(p_ticker.index, format='%d/%m/%Y')

        window_size = 252  # Taille de la fenêtre (ex. 252 jours)
        log_p = np.log(p_ticker)
        r = np.diff(log_p)  # Rendements log

        hurst_values = []
        critical_values = []

        # Calcul en fenêtre glissante
        for i in range(len(r) - window_size + 1):
            window_data = r[i:i + window_size]
            rs_value = ComputeRS.rs_modified_statistic(window_data)

            if not np.isnan(rs_value) and rs_value > 0:
                hurst_value = np.log(rs_value) / np.log(len(window_data))
                critical_value = rs_value / np.sqrt(len(window_data))
            else:
                hurst_value = np.nan
                critical_value = np.nan

            hurst_values.append(hurst_value)
            critical_values.append(critical_value)

        # Création des séries temporelles
        hurst_series = pd.Series(hurst_values, index=p_ticker.index[window_size:])
        critical_series = pd.Series(critical_values, index=p_ticker.index[window_size:])

        # Calcul des statistiques descriptives pour Hurst et Critical Values
        critical_desc = critical_series.describe()

        # Création d'un DataFrame pour ce ticker avec les stats
        ticker_stats = pd.DataFrame({
            'Ticker': [ticker],
            'Count': [round(critical_desc['count'], 3)],
            'Mean': [round(critical_desc['mean'], 3)],
            'Std': [round(critical_desc['std'], 3)],
            'Min': [round(critical_desc['min'], 3)],
            '25\%': [round(critical_desc['25%'], 3)],
            '50\%': [round(critical_desc['50%'], 3)],
            '75\%': [round(critical_desc['75%'], 3)],
            'Max': [round(critical_desc['max'], 3)]
        })

        # Concaténation des résultats pour chaque ticker
        results = pd.concat([results, ticker_stats], ignore_index=True)

        # (Optionnel) Affichage ou sauvegarde de graphiques pour chaque ticker
        # fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
        #                     subplot_titles=(f"{ticker} Price Evolution", "Rolling Critical Value"))
        # fig.add_trace(go.Scatter(x=p_ticker.index, y=p_ticker, mode='lines', name=f'{ticker} Price', line=dict(color='black')), row=1, col=1)
        # fig.add_trace(go.Scatter(x=critical_series.index, y=critical_series, mode='lines', name='Rolling Critical Value', line=dict(color='green')), row=2, col=1)
        # fig.add_trace(go.Scatter(x=critical_series.index, y=[1.620] * len(critical_series), mode='lines', name='Threshold (V=1.620)', line=dict(color='red', dash='dash')), row=2, col=1)
        # fig.update_layout(title_text=f"{ticker} Analysis", height=800, width=1000, showlegend=True)
        # fig.update_xaxes(title_text="Date")
        # fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        # fig.update_yaxes(title_text="Critical Value 10%", row=2, col=1)
        # fig.show()

    # Sauvegarde des résultats
    results.to_csv(f"{DATA_PATH}/statistical_hurst_results.csv", index=False)
    print(results)

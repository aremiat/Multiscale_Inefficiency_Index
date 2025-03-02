import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.RS import ComputeRS
import os

DATA_PATH = os.path.dirname(__file__) + "/../data"
IMG_PATH = os.path.dirname(__file__) + "/../img"

tickers = [
    "NVDA",  # Technologie (méga-cap)
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
        r = log_p.diff().dropna()  # Rendements log

        hurst_values = []
        critical_values = []

        # Calcul en fenêtre glissante
        for i in range(len(r) - window_size + 1):
            window_data = r[i:i + window_size]
            rs_value = ComputeRS.rs_modified_statistic(window_data, len(window_data))

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
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=(f"{ticker} Price Evolution", "Rolling Critical Value"))
        fig.add_trace(go.Scatter(x=p_ticker.index, y=p_ticker, mode='lines', name=f'{ticker} Price', line=dict(color='black')), row=1, col=1)
        fig.add_trace(go.Scatter(x=critical_series.index, y=critical_series, mode='lines', name='Rolling Critical Value', line=dict(color='green')), row=2, col=1)
        fig.add_trace(go.Scatter(x=critical_series.index, y=[1.620] * len(critical_series), mode='lines', name='Threshold (V=1.620)', line=dict(color='red', dash='dash')), row=2, col=1)
        fig.update_layout(title_text=f"{ticker} Analysis", height=800, width=1000, showlegend=True)
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Critical Value 10%", row=2, col=1)
        fig.show()
        #
        # output_filename = os.path.join(IMG_PATH, f"rolling_critical_value_{ticker}.png")
        # fig.write_image(output_filename)
        # print(f"Graphique sauvegardé sous : {output_filename}")

    # p = pd.read_csv(f"{DATA_PATH}/index_prices.csv", index_col=0, parse_dates=True)
    #
    # years = range(1970, 2025 - 1 + 1)  # de 1995 à 2015
    # length = range(1, 36, 5)
    #
    # all_prices = pd.DataFrame()
    # ticker = "^GSPC"
    # for l in length:
    #     crit_values = []
    #     for y in years:
    #         p_ticker = p[ticker].dropna()
    #         start_date = pd.Timestamp(f"{y}-01-02")
    #         end_date = pd.Timestamp(f"{y + l}-02-01")
    #         print(start_date, end_date)
    #
    #         p_ticker = p_ticker.loc[start_date: end_date]
    #         #
    #         log_p = np.log(p_ticker)
    #         r = log_p.diff().dropna()
    #         #
    #         #
    #         # # #
    #         # # # Stationarity test of the series (Dickey-Fuller)
    #         # # p_val_after = adf_test(r)
    #         #
    #         # # adf_data = pd.concat([adf_data, pd.DataFrame({
    #         # #     "Ticker": [ticker],
    #         # #     "P-Value on prices": [round(p_val_before, 3)],  # Round for clarity
    #         # #     "P-Value on log differentiated return": [round(p_val_after, 3)]
    #         # # })], ignore_index=True)
    #         #
    #         Q_tild = ComputeRS.rs_modified_statistic(r, window_size=l*252) # R-s modified
    #
    #         rs_value = ComputeRS.rs_statistic(r, window_size=l*252) # R/S
    #
    #         hurst_rs = np.log(rs_value) / np.log(len(r))
    #         hurst_rs_modified = np.log(Q_tild) / np.log(len(r))
    #         critical_value = Q_tild / np.sqrt(len(r))
    #
    #         h_true = bool(np.round(critical_value, 2) >= 1.620)  # critical values (10, 5, 0.5) are 1.620, 1.747, 2.098
    #         print(np.round(critical_value, 2))
    #         crit_values.append(critical_value)
    #         # if h_true:
    #         #     print(f"Long memory: {h_true} for {ticker} in {y} - {y + 15}")
    #         #     print(f"Ticker: {ticker} \n")
    #         #     print(f"R/S Statistic: {rs_value} \n")
    #         #     print(f"Modified R/S Statistic: {Q_tild} \n")
    #         #     print(f"Hurst Exponent: {hurst_rs} \n")
    #         #     print(f"Modified Hurst Exponent: {hurst_rs_modified} \n")
    #         #     print(f"Critical Value of the Modified Hurst Exponent: {critical_value} \n")
    #         #     print("Long memory: ", h_true)
    #     crit_values = pd.Series(crit_values, index=years)
    #     crit_values.to_csv(f"{DATA_PATH}/critical_values_{l}ans_{ticker}.csv")
    #
    # for l in length:
    #     crit_values = pd.read_csv(f"{DATA_PATH}/critical_values_{l}ans_{ticker}.csv", index_col=0)
    #     crit_values.index = crit_values.index.astype(int)
    #     y_values = crit_values.iloc[:, 0]
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(
    #         x=crit_values.index,
    #         y=y_values.values,
    #         mode='lines+markers',
    #         name='Critical Value'
    #     ))
    #     fig.update_layout(
    #         title=f"Critical Value on a {l}-year rolling window (moving by 1 year) S&P 500, from 1970 to 2024",
    #         xaxis_title="Year",
    #         yaxis_title="Critical Value",
    #         template="plotly_white"
    #     )
    #     fig.show()
    #
    #     output_filename = os.path.join(IMG_PATH, f"critical_value_{l}ans_{ticker}.png")
    #     fig.write_image(output_filename)
    #     print(f"Graphique sauvegardé sous : {output_filename}")



    # Sauvegarde des résultats
    # results.to_csv(f"{DATA_PATH}/statistical_hurst_results.csv", index=False)
    # print(results)

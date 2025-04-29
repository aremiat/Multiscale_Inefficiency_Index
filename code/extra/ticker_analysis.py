import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.RS import ComputeRS
import os
from aeqlib import lisa
import yfinance as yf

tickers = ['MSDEWIN Index', ' NDWUMAT Index', 'NDWUTEL Index', 'NDWUCDIS Index',
           'NDWUIND Index', 'NDWUFNCL Index', 'NDWURLCL Index', 'NDWUCSTA Index',
           'NDWUUTIL Index', 'NDWUENR Index', 'NDWUIT Index', 'NDWUHC Index']

DATA_PATH = os.path.dirname(__file__) + "/../data"
IMG_PATH = os.path.dirname(__file__) + "/../img"
LOADER_PATH = os.path.dirname(__file__) + "/Loader"


russell2000_tickers = [
    "CALM",   # Cal-Maine Foods, Inc. - Consommation courante
    "NARI",   # Inari Medical, Inc. - Santé
    "FARO",   # FARO Technologies, Inc. - Technologies/Industries
    "ALGT",   # Allegiant Travel Company - Consommation discrétionnaire/Voyages
    "SMTC",   # SMTC Corporation - Technologies/Ingénierie
    "UCBI",   # United Community Banks, Inc. - Financier
    "VRTV",   # Veritiv Corporation - Industries/Distribution
    "MGP",    # MGP Ingredients, Inc. - Consommation courante/Alimentation
    "AGYS",   # Agilysys, Inc. - Technologies/Logiciels pour l’hôtellerie
    "HCCI"    # Heritage-Crystal Clean, Inc. - Services industriels/Environnement
]

if __name__ == "__main__":

    # Chargement des prix
    p = pd.read_csv(f"{LOADER_PATH}/sp500_prices_1995_2024_final.csv", index_col=0, parse_dates=True)
    results = pd.DataFrame()

    for ticker in russell2000_tickers:
        p_ticker = yf.download(ticker, start="1995-01-02", end="2024-12-25")["Close"]

        window_size = 120
        log_p = np.log(p_ticker)
        r = log_p.diff().dropna()  # Rendements log
        r_m = r

        rolling_critical = r_m.rolling(window_size).apply(
            lambda window: ComputeRS.rs_modified_statistic(window, window_size=len(window), chin=False) / np.sqrt(
                len(window)),
            raw=False
        ).dropna()

        des_crit = rolling_critical[ticker].describe()
        # Création d'un DataFrame pour ce ticker avec les stats
        ticker_stats = pd.DataFrame({
            'Ticker': [ticker],
            'Count': [round(des_crit.loc['count'], 3)],
            'Mean': [round(des_crit.loc['mean'], 3)],
            'Std': [round(des_crit.loc['std'], 3)],
            'Min': [round(des_crit.loc['min'], 3)],
            '25\%': [round(des_crit.loc['25%'], 3)],
            '50\%': [round(des_crit.loc['50%'], 3)],
            '75\%': [round(des_crit.loc['75%'], 3)],
            'Max': [round(des_crit.loc['max'], 3)]
        })

        # Concaténation des résultats pour chaque ticker
        results = pd.concat([results, ticker_stats], ignore_index=True)

        # (Optionnel) Affichage ou sauvegarde de graphiques pour chaque ticker
        # fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
        #                     subplot_titles=(f"{ticker} Price Evolution", "Rolling Critical Value"))
        # fig.add_trace(go.Scatter(x=p_ticker.index, y=p_ticker, mode='lines', name=f'{ticker} Price', line=dict(color='black')), row=1, col=1)
        # fig.add_trace(go.Scatter(x=rolling_critical.index, y=rolling_critical, mode='lines', name='Rolling Critical Value', line=dict(color='green')), row=2, col=1)
        # fig.add_trace(go.Scatter(x=rolling_critical.index, y=[1.620] * len(rolling_critical), mode='lines', name='Threshold (V=1.620)', line=dict(color='red', dash='dash')), row=2, col=1)
        # fig.update_layout(title_text=f"{ticker} Analysis", height=800, width=1000, showlegend=True)
        # fig.update_xaxes(title_text="Date")
        # fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        # fig.update_yaxes(title_text="Critical Value 10%", row=2, col=1)
        # fig.show()
        #
        # output_filename = os.path.join(IMG_PATH, f"rolling_critical_value_{ticker}.png")
        # fig.write_image(output_filename)
        # print(f"Graphique sauvegardé sous : {output_filename}")
    results = results.dropna()
    print(results)





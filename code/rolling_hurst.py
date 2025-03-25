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
        # p_ticker = p[ticker].dropna()
        # p_ticker = p_ticker.iloc[1:]
        # p_ticker = p_ticker.astype(float)
        p_ticker = yf.download(ticker, start="1995-01-02", end="2024-12-25")["Close"]
        # p_ticker.index = pd.to_datetime(p_ticker.index, format='%d/%m/%Y')

        window_size = 120  # Taille de la fenêtre (ex. 252 jours)
        log_p = np.log(p_ticker)
        r = log_p.diff().dropna()  # Rendements log
        r_m = r.resample('M').last().dropna()

        rolling_critical = r_m.rolling(window_size).apply(
            lambda window: ComputeRS.rs_modified_statistic(window, window_size=len(window), chin=False) / np.sqrt(
                len(window)),
            raw=False
        ).dropna()

        des_crit = rolling_critical.describe()
        print(des_crit)
        # Création d'un DataFrame pour ce ticker avec les stats
        ticker_stats = pd.DataFrame({
            'Ticker': [ticker],
            'Count': [round(des_crit['count'], 3)],
            'Mean': [round(des_crit['mean'], 3)],
            'Std': [round(des_crit['std'], 3)],
            'Min': [round(des_crit['min'], 3)],
            '25\%': [round(des_crit['25%'], 3)],
            '50\%': [round(des_crit['50%'], 3)],
            '75\%': [round(des_crit['75%'], 3)],
            'Max': [round(des_crit['max'], 3)]
        })

        # Concaténation des résultats pour chaque ticker
        results = pd.concat([results, ticker_stats], ignore_index=True)

        # (Optionnel) Affichage ou sauvegarde de graphiques pour chaque ticker
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=(f"{ticker} Price Evolution", "Rolling Critical Value"))
        fig.add_trace(go.Scatter(x=p_ticker.index, y=p_ticker, mode='lines', name=f'{ticker} Price', line=dict(color='black')), row=1, col=1)
        fig.add_trace(go.Scatter(x=rolling_critical.index, y=rolling_critical, mode='lines', name='Rolling Critical Value', line=dict(color='green')), row=2, col=1)
        fig.add_trace(go.Scatter(x=rolling_critical.index, y=[1.620] * len(rolling_critical), mode='lines', name='Threshold (V=1.620)', line=dict(color='red', dash='dash')), row=2, col=1)
        fig.update_layout(title_text=f"{ticker} Analysis", height=800, width=1000, showlegend=True)
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Critical Value 10%", row=2, col=1)
        fig.show()

        output_filename = os.path.join(IMG_PATH, f"rolling_critical_value_{ticker}.png")
        fig.write_image(output_filename)
        print(f"Graphique sauvegardé sous : {output_filename}")

    p = pd.read_csv(f"{DATA_PATH}/index_prices.csv", index_col=0, parse_dates=True)

    years = range(1970, 2025 - 1 + 1)  # de 1995 à 2015
    length = range(1, 36, 5)

    all_prices = pd.DataFrame()
    ticker = "^GSPC"
    for l in length:
        crit_values = []
        for y in years:
            p_ticker = p[ticker].dropna()
            start_date = pd.Timestamp(f"{y}-01-02")
            end_date = pd.Timestamp(f"{y + l}-02-01")
            print(start_date, end_date)

            p_ticker = p_ticker.loc[start_date: end_date]
            #
            log_p = np.log(p_ticker)
            r = log_p.diff().dropna()

            Q_tild = ComputeRS.rs_modified_statistic(r, window_size=l*252)  # R-s modified

            rs_value = ComputeRS.rs_statistic(r, window_size=l*252)  # R/S

            hurst_rs = np.log(rs_value) / np.log(len(r))
            hurst_rs_modified = np.log(Q_tild) / np.log(len(r))
            critical_value = Q_tild / np.sqrt(len(r))

            h_true = bool(np.round(critical_value, 2) >= 1.620)  # critical values (10, 5, 0.5) are 1.620, 1.747, 2.098
            print(np.round(critical_value, 2))
            crit_values.append(critical_value)
        crit_values = pd.Series(crit_values, index=years)
        crit_values.to_csv(f"{DATA_PATH}/critical_values_{l}ans_{ticker}.csv")


    # SP500 vs Russel 2000
    tickers = ['SPX Index', 'RTY Index']
    p = lisa.get_prices(tickers = tickers)
    p = pd.read_csv(f"{DATA_PATH}/index_prices2.csv", index_col=0, parse_dates=True)
    all_prices = pd.concat([p["SPX Index"], p["RTY Index"]], axis=1)
    all_prices = all_prices.loc["1985-12-29": "2025-02-28"]
    all_p = all_prices.pct_change().dropna()
    all_p['Diff'] = all_p['SPX Index'] - all_p['RTY Index']
    r = all_p['Diff']
    r_m = r.resample('M').last().dropna()
    rebase_p = all_prices.loc["1985-12-29": "2025-02-28"]
    rebase_p = rebase_p / rebase_p.iloc[0]
    rolling_critical = r_m.rolling(120).apply(
        lambda window: ComputeRS.rs_modified_statistic(window, window_size=len(window), chin=False) / np.sqrt(
            len(window)),
        raw=False
    ).dropna()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=(f"Sp500 and Russel Price Evolution", "Rolling Critical Value"))
    fig.add_trace(
        go.Scatter(x=rebase_p.index, y=rebase_p["SPX Index"], mode='lines', name=f'Sp500 Price', line=dict(color='black')),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=rebase_p.index, y=rebase_p["RTY Index"], mode='lines', name=f'Russel Price', line=dict(color='red')),
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

    # Sectorial analysis US
    tickers = ['MSDEWIN Index', ' NDWUMAT Index', 'NDWUTEL Index', 'NDWUCDIS Index',
               'NDWUIND Index', 'NDWUFNCL Index', 'NDWURLCL Index', 'NDWUCSTA Index',
               'NDWUUTIL Index', 'NDWUENR Index', 'NDWUIT Index', 'NDWUHC Index']
    prices_msci = lisa.get_prices(tickers = tickers)
    prices_msci = prices_msci.drop(columns={'NDWURLCL Index'})
    r_prices = prices_msci.pct_change().dropna()
    r_m_prices = r_prices.resample('M').last()
    for col in r_prices.columns[1:]:
        base_col = r_prices['MSDEWIN Index']
        r_prices[col + '_diff'] = r_prices[col] - base_col

    rolling_critical_dict = {}

    # Itération sur les colonnes souhaitées (par exemple, de la 10ème colonne à la fin)
    for col in r_m_prices.columns[10:]:
        rolling_critical_dict[col] = r_m_prices[col].rolling(120).apply(
            lambda window: ComputeRS.rs_modified_statistic(window, window_size=len(window), chin=False) / np.sqrt(len(window)),
            raw=False
        ).dropna()

    # Reconstruction d'un DataFrame à partir du dictionnaire
    rolling_critical_df = pd.DataFrame(rolling_critical_dict)
    rebase_p = prices_msci / prices_msci.iloc[0]
    for col in prices_msci.columns[1:]:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=(f"{col} Price Evolution", "Rolling Critical Value"))
        fig.add_trace(
            go.Scatter(x=rebase_p.index, y=rebase_p["MSDEWIN Index"], mode='lines', name=f'MSCI World Price',
                       line=dict(color='black')),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=rebase_p.index, y=rebase_p[col], mode='lines', name=f'{col} Price', line=dict(color='red')),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=rolling_critical_df.index, y=rolling_critical_df[col + '_diff'], mode='lines',
                                 name='Rolling Critical Value',
                                 line=dict(color='green')), row=2, col=1)
        fig.add_trace(
            go.Scatter(x=rolling_critical_df.index, y=[1.620] * len(rolling_critical_df), mode='lines',
                       name='Threshold (V=1.620)',
                       line=dict(color='red', dash='dash')), row=2, col=1)
        fig.update_layout(title_text=f"{col} Analysis", height=800, width=1000, showlegend=True)
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Critical Value 10%", row=2, col=1)
        fig.show()

    # Sectorial analysis EUROPE
    tickers = ['SXXP Index', 'SX6P Index', 'SXBSCP Index', 'SXFINP Index',
               'SX8P Index', 'SXDP Index', 'SXIDUP Index ', 'SX86P Index']
    p = lisa.get_prices(tickers = tickers)
    p = p.loc['2002-06-28':]
    r_prices = p.pct_change().dropna()
    for col in r_prices.columns[:-1]:
        base_col = r_prices['SXXP Index']
        r_prices[col + '_diff'] = r_prices[col] - base_col
    rolling_critical_dict = {}
    r_m_prices = r_prices.resample('M').last()
    for col in r_m_prices.columns[7:]:
        rolling_critical_dict[col] = r_m_prices[col].rolling(120).apply(
            lambda window: ComputeRS.rs_modified_statistic(window, window_size=len(window), chin=False) / np.sqrt(len(window)),
            raw=False
        ).dropna()
    # Reconstruction d'un DataFrame à partir du dictionnaire
    rolling_critical_df = pd.DataFrame(rolling_critical_dict)

    rebase_p = p / p.iloc[0]
    for col in p.columns[:-1]:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=(f"{col} Price Evolution", "Rolling Critical Value"))
        fig.add_trace(
            go.Scatter(x=rebase_p.index, y=rebase_p["SXXP Index"], mode='lines', name=f'MSCI World Price',
                       line=dict(color='black')),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=rebase_p.index, y=rebase_p[col], mode='lines', name=f'{col} Price', line=dict(color='red')),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=rolling_critical_df.index, y=rolling_critical_df[col + '_diff'], mode='lines',
                                 name='Rolling Critical Value',
                                 line=dict(color='green')), row=2, col=1)
        fig.add_trace(
            go.Scatter(x=rolling_critical_df.index, y=[1.620] * len(rolling_critical_df), mode='lines',
                       name='Threshold (V=1.620)',
                       line=dict(color='red', dash='dash')), row=2, col=1)
        fig.update_layout(title_text=f"{col} Analysis", height=800, width=1000, showlegend=True)
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Critical Value 10%", row=2, col=1)
        fig.show()


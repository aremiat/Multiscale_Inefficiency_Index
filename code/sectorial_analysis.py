import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.RS import ComputeRS
import os
from aeqlib import lisa

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data")

if __name__ == "__main__":

    # Sectorial analysis US
    tickers = ['MSDEWIN Index', ' NDWUMAT Index', 'NDWUTEL Index', 'NDWUCDIS Index',
               'NDWUIND Index', 'NDWUFNCL Index', 'NDWUCSTA Index',
               'NDWUUTIL Index', 'NDWUENR Index', 'NDWUIT Index', 'NDWUHC Index']
    prices_msci = lisa.get_prices(tickers = tickers, end='2024-12-31')
    prices_msci = prices_msci.loc['2000-07-17':]
    r_prices = prices_msci.pct_change().dropna()
    for col in r_prices.columns[1:]:
        base_col = r_prices['MSDEWIN Index']
        r_prices[col + '_diff'] = r_prices[col] - base_col

    rolling_critical_dict = {}

    # Itération sur les colonnes souhaitées (par exemple, de la 10ème colonne à la fin)
    for col in r_prices.columns[9:]:
        rolling_critical_dict[col] = r_prices[col].rolling(120).apply(
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
            go.Scatter(x=prices_msci.index, y=prices_msci["MSDEWIN Index"], mode='lines', name=f'MSCI World Price',
                       line=dict(color='black')),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=prices_msci.index, y=prices_msci[col], mode='lines', name=f'{col} Price', line=dict(color='red')),
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

    for col in r_prices.columns[7:]:
        rolling_critical_dict[col] = r_prices[col].rolling(120).apply(
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
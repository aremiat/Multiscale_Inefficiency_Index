import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.RS import ComputeRS
import os
from utils.MFDFA import ComputeMFDFA
from utils.DFA import rolling_hurst_dfa
from scipy.stats import skew, kurtosis
from scipy.stats import jarque_bera
import yfinance as yf


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

def compute_inefficiency_index_abs_value(delta_alpha_diff, rolling_hurst):
        """
        Combine la différence de largeur de spectre (delta_alpha_diff),
        l'écart absolu (rolling Hurst - 0.5).
        """
        return delta_alpha_diff * abs(rolling_hurst - 0.5)


def compute_inefficiency_index(delta_alpha_diff, rolling_hurst):
    """
    Combine la différence de largeur de spectre (delta_alpha_diff),
    l'écart absolu (rolling Hurst - 0.5).
    """
    return delta_alpha_diff * (rolling_hurst - 0.5)


if __name__ == "__main__":
    # Chargement des données pour '^RUT'
    window_mfdfa = 1008
    q_list = np.linspace(-4, 4, 17)
    scales = np.unique(np.logspace(np.log10(10), np.log10(200), 10, dtype=int))
    tickers = ["^FCHI", "^GSPC", "^RUT", "^FTSE", "^N225"]
    multi_asset_tickers = ['BTC-USD', 'EURUSD=X', 'GBPUSD=X']
    # data =  pd.read_csv(os.path.join(DATA_PATH, "multi_assets.csv"), index_col=0, parse_dates=True)
    data = pd.read_csv(os.path.join(DATA_PATH, "index_prices2.csv"), index_col=0, parse_dates=True)

    # tickers = ["^BVSP","^MXX", "000001.SS"]
    # data = yf.download(tickers, start="2000-01-01")  # prix journaliers
    # data = data.xs("Close", level=data.columns.names.index('Price'), axis=1)

    for tick in tickers:
        # print(tick)
        # if tick == '^RUT':
        #     name = "Russel 2000"
        # elif tick == '^GSPC':
        #     name = "S&P 500"
        # elif tick == '^FTSE':
        #     name = "FTSE 100"
        # elif tick == '^N225':
        #     name = "Nikkei 225"
        # elif tick == '^GDAXI':
        #     name = "DAX"
    # for name in data.columns:
    #     if name in multi_asset_tickers:
            name = tick
            print(name)

            # tick = name
            # Chargement des données
            df = data[tick]
            df = df.loc["1987-09-10":"2025-02-28"]
            log_prices = np.log(df).dropna()
            returns = log_prices.diff().dropna()

            stat, p_value = jarque_bera(returns)
            print(f"Jarque-Bera test for {name}: stat={stat}, p-value={p_value}")

            # Calcul du rolling Hurst classique (overlapping) via R/S modified statistic sur 120 jours
            # rolling_hurst = returns.rolling(window=120).apply(
            #     lambda window: np.log(ComputeRS.rs_modified_statistic(window, len(window))) / np.log(len(window)),
            #     raw=False
            # ).dropna()

            rolling_hurst = rolling_hurst_dfa(returns, window = 120)

            np.random.seed(42)
            surrogate_returns = ComputeMFDFA.surrogate_gaussian_corr(returns.values)
            surrogate_returns = pd.Series(surrogate_returns, index=returns.index)
            surrogate_returns.name = name


            stat, p_value = jarque_bera(surrogate_returns)
            print(f"Jarque-Bera test for {name}: stat={stat}, p-value={p_value}")

            i = 1

            while p_value < 0.05:
                np.random.seed(i + 42)
                surrogate_returns = ComputeMFDFA.surrogate_gaussian_corr(returns.values)
                surrogate_returns = pd.Series(surrogate_returns, index=returns.index)
                surrogate_returns.name = name
                stat, p_value = jarque_bera(surrogate_returns)
                print(f"Jarque-Bera test for {name}: stat={stat}, p-value={p_value}")
                i += 1

            spectrum_width = ComputeMFDFA.mfdfa_rolling(surrogate_returns, window_mfdfa, q_list, scales)
            spectrum_width = spectrum_width.loc[spectrum_width.index.intersection(rolling_hurst.index)]
            prices_aligned = log_prices.loc[log_prices.index.intersection(spectrum_width.index)]


            # fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
            #                     vertical_spacing=0.1,
            #                     row_heights=[0.7, 0.3],
            #                     subplot_titles=(f"{name} Price Evolution", "Spectrum Width"))
            # fig.add_trace(go.Scatter(x=prices_aligned.index, y=prices_aligned, mode='lines', name=f'{name} Price', line=dict(color='red')),
            #               row=1, col=1)
            # fig.add_trace(go.Scatter(x=spectrum_width.index, y=spectrum_width, mode='lines', name='Spectrum Width',
            #                          line=dict(color='green')), row=2, col=1)
            #
            # fig.update_layout(title_text=f"{name} Analysis", showlegend=True)
            # fig.update_xaxes(title_text="Date")
            # fig.update_yaxes(title_text="Log Price ($)", row=1, col=1)
            # fig.update_yaxes(title_text="Spectrum Width", row=2, col=1)
            # fig.show()

            common_date = rolling_hurst.index.intersection(spectrum_width.index)
            spectrum_width_aligned = spectrum_width.loc[common_date]
            rolling_hurst_aligned = rolling_hurst.loc[common_date]
            prices_aligned = log_prices.loc[common_date]
            inef_index = compute_inefficiency_index(spectrum_width_aligned, rolling_hurst_aligned)
            inef_index_abs = compute_inefficiency_index_abs_value(spectrum_width_aligned, rolling_hurst_aligned)


            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.1,
                                row_heights=[0.7, 0.3],
                                subplot_titles=(f"{name} Price Evolution", "Inefficiency Index (abs)"))
            fig.add_trace(go.Scatter(x=prices_aligned.index, y=prices_aligned, mode='lines', name=f'{name} Price', line=dict(color='red')),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=inef_index_abs.index, y=inef_index_abs, mode='lines', name=f'{name} Inefficience Index (abs)',
                                     line=dict(color='green')), row=2, col=1)

            fig.update_layout(title_text=f"{name} Analysis", showlegend=True)
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Log Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Spectrum Width", row=2, col=1)
            fig.show()

            # fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
            #                     vertical_spacing=0.1,
            #                     row_heights=[0.7, 0.3],
            #                     subplot_titles=(f"{name} Price Evolution", "Inefficiency Index (signed)"))
            # fig.add_trace(
            #     go.Scatter(x=log_prices.index, y=log_prices, mode='lines', name=f'{name} Price', line=dict(color='red')),
            #     row=1, col=1)
            # fig.add_trace(
            #     go.Scatter(x=inef_index.index, y=inef_index, mode='lines', name='Inefficience Index',
            #                line=dict(color='green')), row=2, col=1)
            #
            # fig.update_layout(title_text=f"{name} Analysis", showlegend=True)
            # fig.update_xaxes(title_text="Date")
            # fig.update_yaxes(title_text="Log Price ($)", row=1, col=1)
            # fig.update_yaxes(title_text="Spectrum Width", row=2, col=1)
            # fig.show()
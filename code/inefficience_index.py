import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.RS import ComputeRS
import os
from utils.MFDFA import ComputeMFDFA
from scipy.stats import jarque_bera
from utils.scaled_window_variance import ScaledWindowedVariance


DATA_PATH = os.path.dirname(__file__) + "/../data"
IMG_PATH = os.path.dirname(__file__) + "/../img"
LOADER_PATH = os.path.dirname(__file__) + "/Loader"

def non_overlapping_rolling(series, window, func):
    """
    Apply a function to non-overlapping segments of a time series.
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
        Compute the inefficiency index as the product of the difference in spectrum width
        and the absolute deviation of the rolling Hurst exponent from 0.5.
        """
        return delta_alpha_diff * abs(rolling_hurst - 0.5)


def compute_inefficiency_index(delta_alpha_diff, rolling_hurst):
    """
    Compute the inefficiency index as the product of the difference in spectrum width
    and the deviation of the rolling Hurst exponent from 0.5.
    """
    return delta_alpha_diff * (rolling_hurst - 0.5)

def hurst_swv(window: np.ndarray | pd.Series,
              method: str = "SD",
              exclusions: bool = True) -> float:
    """
    Compute the Hurst exponent using the Scaled Windowed Variance method.
    Parameters
    ----------
    window : np.ndarray | pd.Series
        The time series window to analyze.
    method : str, optional
        The method to use for the estimation. Options are "SD" for standard deviation,
        "MAD" for median absolute deviation, or "R/S" for rescaled range.
    exclusions : bool, optional
        Whether to exclude the first and last 10% of the data points in the window.
        Default is True, which excludes these points to avoid edge effects.
    Returns
    -------
    float
        The estimated Hurst exponent for the given window.

    """
    series = pd.Series(window, copy=False)
    estimator = ScaledWindowedVariance(series,
                                       method=method,
                                       exclusions=exclusions)
    return estimator.estimate()


if __name__ == "__main__":

    window_mfdfa = 252
    q_list = np.linspace(-4, 4, 17)
    scales = np.unique(np.logspace(np.log10(10), np.log10(50), 10, dtype=int))
    tickers = ["^FCHI", "^GSPC", "^RUT", "^FTSE", "^N225"]
    data = pd.read_csv(os.path.join(DATA_PATH, "index_prices2.csv"), index_col=0, parse_dates=True)
    # data = pd.read_csv(os.path.join(DATA_PATH, "ssec.csv"), index_col=0, parse_dates=True)


    for tick in tickers:
        if tick == '^RUT':
            name = "Russel 2000"
        elif tick == '^GSPC':
            name = "S&P 500"
        elif tick == '^FTSE':
            name = "FTSE 100"
        elif tick == '^N225':
            name = "Nikkei 225"
        elif tick == '^GDAXI':
            name = "DAX"

        name = tick
        print(name)

        df = data[tick]
        df = df.loc["1987-09-10":"2025-02-28"]
        log_prices = np.log(df).dropna()
        returns = log_prices.diff().dropna()

        stat, p_value = jarque_bera(returns)
        print(f"Jarque-Bera test for {name}: stat={stat}, p-value={p_value}")

        rolling_hurst = returns.rolling(window=120).apply(
            lambda window: np.log(ComputeRS.rs_modified_statistic(window, len(window))) / np.log(len(window)),
            raw=False
        ).dropna()

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

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.1,
                            row_heights=[0.7, 0.3],
                            subplot_titles=(f"{name} Price Evolution", "rolling_hurst"))
        fig.add_trace(go.Scatter(x=prices_aligned.index, y=prices_aligned, mode='lines', name=f'{name} Price',
                                 line=dict(color='red')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=rolling_hurst.index, y=rolling_hurst, mode='lines', name='rolling_hurst',
                                 line=dict(color='green')), row=2, col=1)

        fig.update_layout(title_text=f"{name} Analysis", showlegend=True)
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Log Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="rolling_hurst", row=2, col=1)
        fig.show()


        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.1,
                            row_heights=[0.7, 0.3],
                            subplot_titles=(f"{name} Price Evolution", "Spectrum Width"))
        fig.add_trace(go.Scatter(x=prices_aligned.index, y=prices_aligned, mode='lines', name=f'{name} Price', line=dict(color='red')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=spectrum_width.index, y=spectrum_width, mode='lines', name='Spectrum Width',
                                 line=dict(color='green')), row=2, col=1)

        fig.update_layout(title_text=f"{name} Analysis", showlegend=True)
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Log Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Spectrum Width", row=2, col=1)
        fig.show()

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

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.1,
                            row_heights=[0.7, 0.3],
                            subplot_titles=(f"{name} Price Evolution", "Inefficiency Index (signed)"))
        fig.add_trace(
            go.Scatter(x=log_prices.index, y=log_prices, mode='lines', name=f'{name} Price', line=dict(color='red')),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=inef_index.index, y=inef_index, mode='lines', name='Inefficience Index',
                       line=dict(color='green')), row=2, col=1)

        fig.update_layout(title_text=f"{name} Analysis", showlegend=True)
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Log Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Spectrum Width", row=2, col=1)
        fig.show()
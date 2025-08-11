import os
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.RS import ComputeRS

DATA_PATH = os.path.dirname(__file__) + "/../data"

def mfdfa(signal, scales, q_list, order=1):
    """
    Compute the Multifractal Detrended Fluctuation Analysis (MF-DFA) of a signal.
    Args:
        signal (np.ndarray): Input signal (time series).
        scales (np.ndarray): Scales to use in MF-DFA.
        q_list (np.ndarray): List of q values for MF-DFA.
        order (int): Order of polynomial fit for detrending.
    Returns:
        Fq (np.ndarray): 2D array of shape (len(q_list), len(scales)),
                         where Fq[i, j] is the q-order fluctuation function for scale s_j.
    """
    N = len(signal)
    signal = signal - np.mean(signal)
    Y = np.cumsum(signal)
    Fq = np.zeros((len(q_list), len(scales)))

    for i, s in enumerate(scales):
        s = int(s)
        if s < 2:
            continue
        n_segments = N // s
        F_seg = []
        for v in range(n_segments):
            segment = Y[v * s:(v + 1) * s]
            idx = np.arange(s)
            coeffs = np.polyfit(idx, segment, order)
            fit = np.polyval(coeffs, idx)
            F_seg.append(np.mean((segment - fit) ** 2))
        for v in range(n_segments):
            segment = Y[N - (v + 1) * s:N - v * s]
            idx = np.arange(s)
            coeffs = np.polyfit(idx, segment, order)
            fit = np.polyval(coeffs, idx)
            F_seg.append(np.mean((segment - fit) ** 2))
        F_seg = np.array(F_seg)
        F_seg[F_seg < 1e-10] = 1e-10
        for j, q in enumerate(q_list):
            if np.abs(q) < 1e-6:
                Fq[j, i] = np.exp(0.5 * np.mean(np.log(F_seg)))
            else:
                Fq[j, i] = (np.mean(F_seg ** (q / 2))) ** (1 / q)
    return Fq


def compute_alpha_falpha(q_list, h_q):
    """
    Legendre transform to compute alpha(q) and f(alpha) from h(q).
    Args:
        q_list (np.ndarray): List of q values.
        h_q (np.ndarray): Corresponding h(q) values.
    Returns:
        alpha (np.ndarray): Array of alpha(q) values.
        f_alpha (np.ndarray): Array of f(alpha) values.
    """
    dq = q_list[1] - q_list[0]
    dh_dq = np.gradient(h_q, dq)
    alpha = h_q + q_list * dh_dq
    f_alpha = q_list * (alpha - h_q) + 1
    return alpha, f_alpha


def mfdfa_rolling(series, window_size, q_list, scales, order=1):
    """
    Apply MF-DFA in a rolling window manner to compute the multifractal spectrum width Δα.
    Args:
        series (pd.Series or np.ndarray): Time series data.
        window_size (int): Size of the rolling window.
        q_list (np.ndarray): List of q values for MF-DFA.
        scales (np.ndarray): Scales to use in MF-DFA.
        order (int): Order of polynomial fit for detrending.
    Returns:
        pd.Series: Series of Δα values with the rolling index.
    """
    if isinstance(series, pd.Series):
        data = series.values
        index_data = series.index
    else:
        data = np.array(series)
        index_data = np.arange(len(data))

    alpha_widths = []
    rolling_index = []
    nb_points = len(data)

    for start in range(nb_points - window_size + 1):
        end = start + window_size
        window_data = data[start:end]
        Fq = mfdfa(window_data, scales, q_list, order=order)
        h_q = []
        log_scales = np.log(scales)
        for j, q in enumerate(q_list):
            log_Fq = np.log(Fq[j, :])
            coeffs = np.polyfit(log_scales, log_Fq, 1)
            h_q.append(coeffs[0])
        h_q = np.array(h_q)
        alpha, _ = compute_alpha_falpha(q_list, h_q)
        delta_alpha = alpha.max() - alpha.min()
        alpha_widths.append(delta_alpha)
        rolling_index.append(index_data[end - 1])
    return pd.Series(alpha_widths, index=rolling_index, name="alpha_width")


def simulate_msm_volatility(T, K=3, b=1.5, p_vec=[1.0, 0.5, 0.1], sigma_base=1.0):
    """
    Simulate the Markov Chain Multifracal (MSM) volatility model.

    Args:
        T (int): Number of time steps to simulate.
        K (int): Number of components in the MSM model.
        b (float): Base value for the components.
        p_vec (list): Probability vector for each component.
        sigma_base (float): Base volatility.
    Returns:
        np.ndarray: Simulated MSM volatility series.
    """
    msm_components = np.ones((T, K))
    for k in range(K):
        for t in range(1, T):
            if np.random.rand() < p_vec[k]:
                msm_components[t, k] = b if np.random.rand() < 0.5 else 1.0
            else:
                msm_components[t, k] = msm_components[t - 1, k]
    msm_vol = sigma_base * np.prod(msm_components, axis=1)
    return msm_vol



def forecast_msm_volatility(n_steps, K=3, b=1.5, p_vec=[1.0, 0.5, 0.1],
                            sigma_base=1.0, n_sims=1000):
    """
    Run a simulation of the MSM volatility model for n_steps.
    Args:
        n_steps (int): Number of steps to simulate.
        K (int): Number of components in the MSM model.
        b (float): Base value for the components.
        p_vec (list): Probability vector for each component.
        sigma_base (float): Base volatility.
        n_sims (int): Number of simulations to run.
    Returns:
        mean_forecast (np.ndarray): Mean forecast of volatility.
        median_forecast (np.ndarray): Median forecast of volatility.
        pct_05 (np.ndarray): 5th percentile of the forecast.
        pct_95 (np.ndarray): 95th percentile of the forecast.
    """

    results = np.zeros((n_sims, n_steps))
    for sim in range(n_sims):
        components = np.where(np.random.rand(K) < 0.5, b, 1.0)
        for step in range(n_steps):
            for k in range(K):
                if np.random.rand() < p_vec[k]:
                    components[k] = b if np.random.rand() < 0.5 else 1.0
            vol = sigma_base * np.prod(components)
            results[sim, step] = vol

    mean_forecast = results.mean(axis=0)
    median_forecast = np.median(results, axis=0)
    pct_05 = np.percentile(results, 5, axis=0)
    pct_95 = np.percentile(results, 95, axis=0)

    return mean_forecast, median_forecast, pct_05, pct_95



ticker = "^RUT"
data = pd.read_csv(f"{DATA_PATH}/russell_2000.csv", index_col=0, parse_dates=True)


returns = np.log(data).diff().dropna()
r_m = returns.resample('M').last()


window_size = 120
q_list = np.linspace(-5, 5, 21)
scales = np.unique(np.floor(np.logspace(np.log10(10), np.log10(80), 10)).astype(int))


alpha_width_series = mfdfa_rolling(r_m, window_size, q_list, scales, order=1)


rolling_critical = r_m.rolling(window_size).apply(
    lambda window: ComputeRS.rs_modified_statistic(window, len(window), chin=False) / np.sqrt(len(window)),
    raw=False
).dropna()
alpha_width_series.index = rolling_critical.index

T = len(r_m)
np.random.seed(43)
msm_vol_series = simulate_msm_volatility(T, K=3, b=1.5, p_vec=[1.0, 0.5, 0.1], sigma_base=1.0)
msm_vol = pd.Series(msm_vol_series, index=r_m.index)
msm_vol.index = r_m.index
msm_vol_rolling = msm_vol.rolling(window=12, min_periods=1).mean()


common_index = rolling_critical.index.intersection(alpha_width_series.index).intersection(msm_vol_rolling.index)
price_series_aligned = data[ticker]
price_series_aligned = price_series_aligned / price_series_aligned.iloc[0]
price_series_aligned = price_series_aligned.loc["1997-08-31": "2025-02-28"]
rolling_critical_aligned = rolling_critical.loc[common_index]
rolling_critical_aligned = rolling_critical_aligned[ticker]
alpha_width_aligned = alpha_width_series.loc[common_index]
msm_vol_aligned = msm_vol_rolling.loc[common_index]

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.4, 0.6],
    vertical_spacing=0.06,
    specs=[
        [{"secondary_y": False}],
        [{"secondary_y": True}]
    ],
    subplot_titles=("Price", "Rolling Critical Value, Δα et Volatilité MSM")
)


fig.add_trace(go.Scatter(
    x=price_series_aligned.index,
    y=price_series_aligned.values,
    mode='lines',
    name='Price',
    line=dict(color='blue')
), row=1, col=1)
fig.update_yaxes(title_text="Price ($)", row=1, col=1)


fig.add_trace(go.Scatter(
    x=rolling_critical_aligned.index,
    y=rolling_critical_aligned.values,
    mode='lines',
    name='Rolling Critical Value (R/S mod.)',
    line=dict(color='green')
), row=2, col=1, secondary_y=False)
fig.add_trace(go.Scatter(
    x=msm_vol_aligned.index,
    y=msm_vol_aligned.values,
    mode='lines',
    name='Volatilité MSM',
    line=dict(color='blue')
), row=2, col=1, secondary_y=False)


fig.add_trace(go.Scatter(
    x=rolling_critical.index,
    y=[1.620] * len(rolling_critical),
    mode='lines',
    name=f'Seuil = {1.620}',
    line=dict(color='red', dash='dash')
), row=2, col=1, secondary_y=False)

fig.add_trace(go.Scatter(
    x=alpha_width_aligned.index,
    y=alpha_width_aligned.values,
    mode='lines+markers',
    name='Δα (MF-DFA)',
    marker=dict(color='purple')
), row=2, col=1, secondary_y=True)

fig.update_layout(
    title="Comparaison : Price, Rolling Critical Value, Δα et Volatilité MSM",
    template="plotly_white",
    legend=dict(x=0.02, y=0.98)
)
fig.update_yaxes(title_text="Critical Value & MSM Volatility", row=2, col=1, secondary_y=False)
fig.update_yaxes(title_text="Δα (MF-DFA)", row=2, col=1, secondary_y=True)
fig.update_xaxes(title_text="Date", row=2, col=1)

fig.show()

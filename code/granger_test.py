import os
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

# -----------------------------
# 1. MF-DFA Functions
# -----------------------------
def mfdfa(signal, scales, q_list, order=1):
    N = len(signal)
    signal = signal - np.mean(signal)
    Y = np.cumsum(signal)

    Fq = np.zeros((len(q_list), len(scales)))
    for j, s in enumerate(scales):
        s = int(s)
        if s < 2:
            continue
        n_segments = N // s
        F_seg = []
        # First division: non-overlapping from the start
        for v in range(n_segments):
            segment = Y[v * s:(v + 1) * s]
            idx = np.arange(s)
            coeffs = np.polyfit(idx, segment, order)
            fit = np.polyval(coeffs, idx)
            F_seg.append(np.mean((segment - fit) ** 2))
        # Second division: from the end
        for v in range(n_segments):
            segment = Y[N - (v + 1) * s:N - v * s]
            idx = np.arange(s)
            coeffs = np.polyfit(idx, segment, order)
            fit = np.polyval(coeffs, idx)
            F_seg.append(np.mean((segment - fit) ** 2))
        F_seg = np.array(F_seg)
        F_seg[F_seg < 1e-10] = 1e-10  # avoid log(0)
        # Compute F_q(s) for each q
        for k, q in enumerate(q_list):
            if np.abs(q) < 1e-6:
                Fq[k, j] = np.exp(0.5 * np.mean(np.log(F_seg)))
            else:
                Fq[k, j] = (np.mean(F_seg ** (q / 2))) ** (1 / q)
    return Fq

def compute_alpha_falpha(q_list, h_q):
    q_list = np.array(q_list)
    tau_q = q_list * h_q - 1
    alpha = np.gradient(tau_q, q_list)
    f_alpha = q_list * alpha - tau_q
    return alpha, f_alpha

def mfdfa_rolling(series, window_size, q_list, scales, order=1):
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
        # 1) Compute F_q(s) on the window
        Fq = mfdfa(window_data, scales, q_list, order=order)
        # 2) Compute h(q) by linear regression of log(Fq) vs log(s)
        h_q = []
        log_scales = np.log(scales)
        for j, q in enumerate(q_list):
            log_Fq = np.log(Fq[j, :])
            coeffs = np.polyfit(log_scales, log_Fq, 1)
            h_q.append(coeffs[0])
        h_q = np.array(h_q)
        # 3) Legendre transformation to get alpha and f(alpha)
        alpha, f_alpha = compute_alpha_falpha(q_list, h_q)
        # 4) Spectrum width Δα
        alpha_width = alpha.max() - alpha.min()
        alpha_widths.append(alpha_width)
        rolling_index.append(index_data[end - 1])
    return pd.Series(alpha_widths, index=rolling_index, name="alpha_width")

# -----------------------------
# 2. R/S Functions (already provided)
# -----------------------------
class ComputeRS:
    @staticmethod
    def rs_statistic(series, window_size=0):
        if window_size < len(series):
            window_size = len(series)
        s = series.iloc[len(series) - window_size: len(series)]
        mean = np.mean(s)
        y = s - mean
        r = np.max(np.cumsum(y)) - np.min(np.cumsum(y))
        sigma = np.std(s)
        return r / sigma

    @staticmethod
    def compute_S_modified(series, chin=False):
        s = series
        t = len(s)
        mean_y = np.mean(s)
        s = s.squeeze()
        if not chin:
            rho_1 = np.corrcoef(s[:-1], s[1:])[0, 1]
            if rho_1 < 0:
                return np.sum((s - mean_y) ** 2) / t
            q = ((3 * t) / 2)**(1 / 3) * ((2 * rho_1) / (1 - (rho_1 ** 2)))**(2 / 3)
        else:
            q = 4 * (t / 100)**(2 / 9)
        q = int(np.floor(q))
        var_term = np.sum((s - mean_y) ** 2) / t
        auto_cov_term = 0
        for j in range(1, q + 1):
            w_j = 1 - (j / (q + 1))
            sum_cov = np.sum((s[:-j] - mean_y) * (s[j:] - mean_y))
            auto_cov_term += w_j * sum_cov
        auto_cov_term = (2 / t) * auto_cov_term
        s_quared = var_term + auto_cov_term
        return s_quared

    @staticmethod
    def rs_modified_statistic(series, window_size=0, chin=False):
        if window_size > len(series):
            window_size = len(series)
        s = series.iloc[len(series)-window_size: len(series)]
        y = s - np.mean(s)
        r = np.max(np.cumsum(y)) - np.min(np.cumsum(y))
        sigma = np.sqrt(ComputeRS.compute_S_modified(s, chin))
        return r / sigma

def compute_rolling_metric(series, window_size, method='modified', rolling_type='overlapping', chin=False):
    if rolling_type == 'overlapping':
        if method == 'modified':
            roll = series.rolling(window_size).apply(
                lambda window: np.log(ComputeRS.rs_modified_statistic(window, len(window), chin=chin)) / np.log(len(window)),
                raw=False
            ).dropna()
        elif method == 'traditional':
            roll = series.rolling(window_size).apply(
                lambda window: np.log(ComputeRS.rs_statistic(window, len(window))) / np.log(len(window)),
                raw=False
            ).dropna()
        else:
            raise ValueError("Unknown method")
    else:
        raise ValueError("Only overlapping rolling supported in this example")
    return roll


# -----------------------------
# 4. Inefficiency Index and Positioning Strategy
# -----------------------------
def compute_inefficiency_index(delta_alpha, rolling_hurst):
    """
    Compute the inefficiency index for a single market as:
    ineff_index = Δα × |H_rolling - 0.5|
    """
    return delta_alpha * np.abs(rolling_hurst - 0.5)

# -----------------------------
# VAR and Granger Causality Testing for Inefficiency Indices
# -----------------------------
if __name__ == "__main__":
    # Set paths and tickers
    DATA_PATH = os.path.join(os.path.dirname(__file__), "../data")
    df_prices = pd.read_csv(f"{DATA_PATH}/index_prices2.csv", index_col=0, parse_dates=True)
    ticker1 = "^GSPC"  # S&P 500
    ticker2 = "^RUT"    # Russell 2000

    # Select price data and restrict date range
    df_prices = df_prices[[ticker1, ticker2]].loc["1987-10-09":"2025-02-28"]

    # Compute daily returns and log prices
    df_returns = df_prices.pct_change().dropna()
    log_prices = np.log(df_prices)
    log_returns = log_prices.diff().dropna()

    # Parameters for MF-DFA rolling
    mfdfa_window = 252
    q_list = np.linspace(-3, 3, 13)
    scales = np.unique(np.logspace(np.log10(10), np.log10(50), 10, dtype=int))

    # Compute rolling multifractal spectrum width (Δα) for each ticker
    rolling_delta_SP500 = mfdfa_rolling(log_returns[ticker1], mfdfa_window, q_list, scales, order=1)
    rolling_delta_RUT   = mfdfa_rolling(log_returns[ticker2], mfdfa_window, q_list, scales, order=1)

    # Compute rolling Hurst exponent for each ticker using ComputeRS (on log returns)
    rolling_hurst_SP500 = compute_rolling_metric(log_returns[ticker1], mfdfa_window, method='modified', rolling_type='overlapping', chin=False)
    rolling_hurst_RUT   = compute_rolling_metric(log_returns[ticker2], mfdfa_window, method='modified', rolling_type='overlapping', chin=False)

    # Align the computed rolling series (use common dates)
    common_dates_sp500 = rolling_delta_SP500.index.intersection(rolling_hurst_SP500.index)
    rolling_delta_SP500 = rolling_delta_SP500.loc[common_dates_sp500]
    rolling_hurst_SP500 = rolling_hurst_SP500.loc[common_dates_sp500]

    common_dates_rut = rolling_delta_RUT.index.intersection(rolling_hurst_RUT.index)
    rolling_delta_RUT = rolling_delta_RUT.loc[common_dates_rut]
    rolling_hurst_RUT = rolling_hurst_RUT.loc[common_dates_rut]

    # Compute inefficiency indices for each market
    ineff_index_SP500 = compute_inefficiency_index(rolling_delta_SP500, rolling_hurst_SP500)
    ineff_index_RUT   = compute_inefficiency_index(rolling_delta_RUT, rolling_hurst_RUT)

    # Combine inefficiency indices into one DataFrame for VAR analysis
    ineff_df = pd.DataFrame({
        'SP500_ineff': ineff_index_SP500,
        'RUT_ineff': ineff_index_RUT
    }).dropna()

    # Check stationarity with ADF test and difference if needed
    def adf_test(series):
        result = adfuller(series)
        print(f"ADF p-value for {series.name}: {result[1]:.4f}")
        return result[1]

    p_sp500 = adf_test(ineff_df['SP500_ineff'])
    p_rut = adf_test(ineff_df['RUT_ineff'])

    if p_sp500 > 0.05:
        ineff_df['SP500_ineff_diff'] = ineff_df['SP500_ineff'].diff()
    else:
        ineff_df['SP500_ineff_diff'] = ineff_df['SP500_ineff']

    if p_rut > 0.05:
        ineff_df['RUT_ineff_diff'] = ineff_df['RUT_ineff'].diff()
    else:
        ineff_df['RUT_ineff_diff'] = ineff_df['RUT_ineff']

    ineff_var = ineff_df[['SP500_ineff_diff', 'RUT_ineff_diff']].dropna()

    # Fit a VAR model on the inefficiency indices
    model = VAR(ineff_var)
    lag_results = model.select_order(maxlags=10)
    print(lag_results.summary())
    opt_lag = lag_results.aic
    results_var = model.fit(opt_lag)
    print(results_var.summary())

    # Granger Causality Test: Does SP500 inefficiency predict RUT inefficiency?
    test_sp500_to_rut = results_var.test_causality('RUT_ineff_diff', ['SP500_ineff_diff'], kind='f')
    print("Granger Causality Test: SP500 inefficiency -> RUT inefficiency")
    print(test_sp500_to_rut.summary())

    # Granger Causality Test: Does RUT inefficiency predict SP500 inefficiency?
    test_rut_to_sp500 = results_var.test_causality('SP500_ineff_diff', ['RUT_ineff_diff'], kind='f')
    print("Granger Causality Test: RUT inefficiency -> SP500 inefficiency")
    print(test_rut_to_sp500.summary())

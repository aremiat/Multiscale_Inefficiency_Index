import os
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from itertools import permutations
from utils.RS import ComputeRS
from utils.MFDFA import ComputeMFDFA
from scipy.stats import jarque_bera

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

def adf_test_and_diff(series, name):
    """Test ADF ; retourne la série stationnaire (éventuellement différenciée)."""
    pval = adfuller(series.dropna())[1]
    if pval > 0.05:
        print(f"{name}: non stationnaire (p={pval:.3f}) → différenciation")
        return series.diff().dropna(), True
    else:
        print(f"{name}: stationnaire (p={pval:.3f})")
        return series, False

def granger_pairwise(df, maxlags=10):
    """
    Pour chaque paire (i → j), ajuste un VAR et teste H0 : i ne cause pas j.
    Retourne DataFrame p-values indexées [cause → effet].
    """
    tickers = df.columns
    pvals = pd.DataFrame(index=tickers, columns=tickers, dtype=float)

    model = VAR(df)
    sel = model.select_order(maxlags=maxlags)
    lag = sel.aic
    res = model.fit(lag)

    for cause, effect in permutations(tickers, 2):
        test = res.test_causality(effect, [cause], kind='f')
        pvals.loc[cause, effect] = test.pvalue
    return pvals

if __name__ == "__main__":
    # 1. Paramètres
    DATA_PATH = os.path.join(os.path.dirname(__file__), "../data")
    tickers = ["^GSPC", "^RUT", "^FTSE", "^N225", "^GDAXI"]

    # 2. Chargement des prix et log-retours
    df_prices = pd.read_csv(f"{DATA_PATH}/index_prices2.csv",
                            index_col=0, parse_dates=True)[tickers]
    df_prices = df_prices.loc["1987-09-10":"2025-02-28"]
    log_returns = np.log(df_prices).diff().dropna()

    # 3. Paramètres MF-DFA rolling
    mfdfa_window = 1008
    q_list = np.linspace(-3, 3, 13)
    scales = np.unique(np.logspace(np.log10(10),
                                   np.log10(200), 10, dtype=int))
    np.random.seed(42)
    # 4. Calcul rolling Δα et Hurst
    rolling_delta = {}
    rolling_hurst = {}
    for tic in tickers:
        log_returns_surrogate = ComputeMFDFA.surrogate_gaussian_corr(pd.Series(log_returns[tic].values,
                                                                               index=log_returns.index))
        log_returns_surrogate = pd.Series(log_returns_surrogate, index=log_returns.index)


        log_returns_surrogate.name = tic

        stat, p_value = jarque_bera(log_returns_surrogate)
        print(f"Jarque-Bera test for {tic}: stat={stat}, p-value={p_value}")

        i = 1

        while p_value < 0.05:
            np.random.seed(i + 42)
            surrogate_returns = ComputeMFDFA.surrogate_gaussian_corr(log_returns_surrogate.values)
            surrogate_returns = pd.Series(surrogate_returns, index=log_returns.index)
            surrogate_returns.name = tic
            stat, p_value = jarque_bera(surrogate_returns)
            print(f"Jarque-Bera test for {tic}: stat={stat}, p-value={p_value}")
            i += 1

        rolling_delta[tic] = ComputeMFDFA.mfdfa_rolling(
            log_returns_surrogate, mfdfa_window, q_list, scales, order=1
        )
        rolling_hurst[tic] = compute_rolling_metric(
            log_returns[tic], mfdfa_window,
            method='modified', rolling_type='overlapping', chin=False
        )

    # 5. Intersection des dates communes
    idx = set(rolling_delta[tickers[0]].index)
    for tic in tickers:
        idx &= set(rolling_delta[tic].index) & set(rolling_hurst[tic].index)
    idx = sorted(idx)

    # 6. Construction DataFrames ineff_abs et ineff_signed
    df_abs = pd.DataFrame({tic: compute_inefficiency_index_abs_value(
                                rolling_delta[tic].loc[idx],
                                rolling_hurst[tic].loc[idx])
                           for tic in tickers}, index=idx).dropna()
    df_signed = pd.DataFrame({tic: compute_inefficiency_index(
                                   rolling_delta[tic].loc[idx],
                                   rolling_hurst[tic].loc[idx])
                              for tic in tickers}, index=idx).dropna()

    # 7. Sauvegarde métriques individuelles
    out_dir = os.path.join(DATA_PATH, "inefficiency granger")
    os.makedirs(out_dir, exist_ok=True)
    for tic in tickers:
        pd.DataFrame({
            'delta_alpha': rolling_delta[tic].loc[idx],
            'hurst': rolling_hurst[tic].loc[idx],
            'ineff_abs': df_abs[tic],
            'ineff_signed': df_signed[tic],
        }).dropna().to_csv(os.path.join(out_dir, f"{tic[1:]}_metrics.csv"))

    # 8. Préparation des séries stationnaires pour Granger
    df_abs_stat = pd.DataFrame()
    df_signed_stat = pd.DataFrame()
    # on diff uniquement si nécessaire, et on aligne ensuite
    for tic in tickers:
        s_abs, _ = adf_test_and_diff(df_abs[tic], f"{tic} abs")
        s_sgn, _ = adf_test_and_diff(df_signed[tic], f"{tic} signed")
        df_abs_stat[tic] = s_abs
        df_signed_stat[tic] = s_sgn

    # tronquer au plus court afin d'avoir les mêmes dates pour tous
    df_abs_stat = df_abs_stat.dropna()
    df_signed_stat = df_signed_stat.dropna()

    # 9. Granger causality pairwise
    pvals_abs = granger_pairwise(df_abs_stat)
    pvals_signed = granger_pairwise(df_signed_stat)

    print("✅ Calculs terminés.")
    print(pvals_abs)
    print(pvals_signed)

    # 10. Sauvegarde des p-values
    pvals_abs.to_csv(os.path.join(out_dir, "granger_pvalues_abs.csv"))
    pvals_signed.to_csv(os.path.join(out_dir, "granger_pvalues_signed.csv"))

    print("✅ Calculs terminés.")
    print("→ inefficiency metrics + CSV individuels dans", out_dir)
    print("→ p-values Granger dans granger_pvalues_*.csv")

import os
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera

from utils.MFDFA import ComputeMFDFA

DATA_PATH = os.path.dirname(__file__) + "/../data"
q_list = np.linspace(-3, 3, 13)
scales = np.unique(np.floor(np.logspace(np.log10(10), np.log10(500), 10)).astype(int))
TICKERS = ["^GSPC", "^RUT", "^FTSE", "^N225", "^GDAXI"]
N_RESAMPLES = 500
SEED = 45

def delta_alpha(series: np.ndarray, scales: np.ndarray, q_list: np.ndarray) -> float:
    """Return the multifractal spectrum width Δα for *series*."""
    Fq = ComputeMFDFA.mfdfa(series, scales, q_list, order=1)
    log_scales = np.log(scales)

    h_q = []
    for i, q in enumerate(q_list):
        slope, _ = np.polyfit(log_scales, np.log(Fq[i, :]), 1)
        h_q.append(slope)

    alpha, _ = ComputeMFDFA.compute_alpha_falpha(q_list, np.asarray(h_q))
    return alpha.max() - alpha.min()


def bootstrap_dalpha(returns: pd.Series,
                     scales: np.ndarray,
                     q_list: np.ndarray,
                     n_resamples: int,
                     rng: np.random.Generator):
    """Return two arrays (Δα_boot, Δα_sur_boot) of length *n_resamples*."""
    n = len(returns)
    boot_orig = np.empty(n_resamples)
    boot_surr = np.empty(n_resamples)

    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resample = returns.values[idx]
        boot_orig[b] = delta_alpha(resample, scales, q_list)
        surrogate = ComputeMFDFA.surrogate_gaussian_corr(returns.values)
        boot_surr[b] = delta_alpha(surrogate, scales, q_list)

    return boot_orig, boot_surr


def summary_stats(arr: np.ndarray):
    mean = arr.mean()
    std = arr.std(ddof=1)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return mean, std, lo, hi


if __name__ == "__main__":
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 250)

    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    data_equity = pd.read_csv(os.path.join(DATA_PATH, "index_prices2.csv"), index_col=0, parse_dates=True)
    df = data_equity.loc["1987-09-10":"2025-02-28"]

    NAME_MAP = {
        "^RUT": "Russell2000",
        "^GSPC": "S&P500",
        "^FTSE": "FTSE100",
        "^N225": "Nikkei225",
        "^GDAXI": "DAX"
    }

    results = []

    for ticker in TICKERS:
        prices = df[ticker].dropna()
        returns = np.log(prices).diff().dropna()

        dalpha_orig = delta_alpha(returns.values, scales, q_list)
        dalpha_sur_once = delta_alpha(ComputeMFDFA.surrogate_gaussian_corr(returns.values), scales, q_list)

        boot_orig, boot_sur = bootstrap_dalpha(returns, scales, q_list, N_RESAMPLES, rng)

        mo, so, lo, hi = summary_stats(boot_orig)
        ms, ss, ls, hs = summary_stats(boot_sur)

        name = NAME_MAP.get(ticker, ticker)
        print(f"\n===== {name} =====")
        print(f"Δα original            : {dalpha_orig:.4f}")
        print(f"Δα surrogate (single)   : {dalpha_sur_once:.4f}")
        print(f"Bootstrap Δα original   : mean={mo:.4f}, std={so:.4f}, 95% CI=({lo:.4f}, {hi:.4f})")
        print(f"Bootstrap Δα surrogate  : mean={ms:.4f}, std={ss:.4f}, 95% CI=({ls:.4f}, {hs:.4f})")

        results.append({
            'Index': name,
            'Δα_orig': dalpha_orig,
            'Δα_surrogate': dalpha_sur_once,
            'orig_mean': mo,
            'orig_std': so,
            'orig_CI_low': lo,
            'orig_CI_high': hi,
            'sur_mean': ms,
            'sur_std': ss,
            'sur_CI_low': ls,
            'sur_CI_high': hs
        })

        results_df = pd.DataFrame(results.round(3))
        out_path = os.path.join(DATA_PATH, "inefficiency_results.csv")
        results_df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")

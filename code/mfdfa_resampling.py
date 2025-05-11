# -*- coding: utf-8 -*-
"""mfdfa_resampling.py

Compute the MF‑DFA multifractal spectrum width (Δα = α_max − α_min) for a set of
stock‑index return series and evaluate estimation error via bootstrap. At each
bootstrap iteration we also generate a phase‑randomised surrogate so that we
obtain a reference distribution where heavy‑tail effects are preserved but
long‑range correlations are destroyed.

Outputs (printed to stdout for each ticker):
  • Δα on the original series;
  • Δα on one surrogate realisation (quick sanity check);
  • mean, standard deviation and 95 % percentile CI of Δα over
    n_resamples for both the original and surrogate.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera

from utils.MFDFA import ComputeMFDFA  # adjust if the module lives elsewhere

# ------------------------ PARAMETERS --------------------------------------

DATA_PATH = os.path.dirname(__file__) + "/../data"

q_list = np.linspace(-3, 3, 13)
scales = np.unique(np.floor(np.logspace(np.log10(10), np.log10(500), 10)).astype(int))

TICKERS = ["^GSPC", "^RUT", "^FTSE", "^N225", "^GDAXI"]
N_RESAMPLES = 500
SEED = 45

# -------------------- UTILITY FUNCTIONS -----------------------------------

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
        # --- bootstrap on the *original* data ---
        idx = rng.integers(0, n, size=n)
        resample = returns.values[idx]
        boot_orig[b] = delta_alpha(resample, scales, q_list)

        # --- generate a *new* surrogate each time ---
        surrogate = ComputeMFDFA.surrogate_gaussian_corr(returns.values)
        boot_surr[b] = delta_alpha(surrogate, scales, q_list)

    return boot_orig, boot_surr


def summary_stats(arr: np.ndarray):
    mean = arr.mean()
    std = arr.std(ddof=1)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return mean, std, lo, hi

# --------------------------- MAIN ----------------------------------------

if __name__ == "__main__":
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    # --- read data ---
    data_equity = pd.read_csv(os.path.join(DATA_PATH, "index_prices2.csv"), index_col=0, parse_dates=True)
    df = data_equity.loc["1987-09-10":"2025-02-28"]

    NAME_MAP = {
        "^RUT": "Russell 2000",
        "^GSPC": "S&P 500",
        "^FTSE": "FTSE 100",
        "^N225": "Nikkei 225",
        "^GDAXI": "DAX"
    }

    for ticker in TICKERS[:2]:
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

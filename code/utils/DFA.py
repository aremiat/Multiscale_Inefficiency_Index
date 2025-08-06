from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyfit, polyval
from typing import Sequence, Tuple

__all__ = [
    "dfa_hurst",
    "rolling_hurst_dfa",
    "ComputeDFA",
]


def _profile(series: np.ndarray) -> np.ndarray:
    """Compute the profile of a time series by removing the mean and computing the cumulative sum."""
    x = series[np.isfinite(series)]
    return np.cumsum(x - x.mean())


def _local_fluctuation(y: np.ndarray, scale: int, order: int = 1) -> float:
    """Compute the local fluctuation for a given scale and polynomial order."""
    n = len(y)
    if scale >= n:
        raise ValueError("`scale` doit être < len(profile)`")

    n_seg = n // scale
    rms = []
    idx = np.arange(scale)
    for i in range(n_seg):
        seg = y[i * scale : (i + 1) * scale]
        coef = polyfit(idx, seg, order)
        trend = polyval(idx, coef)
        rms.append(np.mean((seg - trend) ** 2))

    return np.sqrt(np.mean(rms))


def dfa_hurst(
    series: Sequence[float] | np.ndarray | pd.Series,
    scales: Sequence[int] | None = None,
    order: int = 1,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the Hurst exponent using the Detrended Fluctuation Analysis (DFA) method.
    """
    x = np.asarray(series, dtype=float)
    y = _profile(x)
    n = len(y)

    if scales is None:
        min_s, max_s = 10, n // 5
        scales = np.unique(np.logspace(np.log10(min_s), np.log10(max_s), 10).astype(int))
    else:
        scales = np.asarray(scales, int)
        if scales.min() < 2 or scales.max() >= n:
            raise ValueError("Échelles DFA hors‑limites")

    fluct = np.array([_local_fluctuation(y, s, order) for s in scales])

    coef = polyfit(np.log2(scales), np.log2(fluct), 1)
    hurst = coef[1]  # pente
    return float(hurst), scales, fluct


class ComputeDFA:
    """Utils class for computing the Hurst exponent using DFA."""

    def __init__(self):
        pass

    # ---------------------------------------------------------------------
    @staticmethod
    def dfa_statistic(
        series: pd.Series,
        window_size: int = 0,
        order: int = 1,
        scales: Sequence[int] | None = None,
    ) -> float:
        """
        Compute the Hurst exponent using the DFA method on a given series.
        """
        if not isinstance(series, pd.Series):
            raise TypeError("`series` doit être une pandas.Series")

        if window_size <= 0 or window_size > len(series):
            window_size = len(series)

        window = series.iloc[-window_size:].dropna()
        if len(window) < 4:
            return np.nan  # insuffisant

        h, *_ = dfa_hurst(window.values, scales=scales, order=order)
        return h

def rolling_hurst_dfa(
    series: pd.Series,
    window: int = 252,
    step: int = 1,
    order: int = 1,
    scales: Sequence[int] | None = None,
    min_valid: int | None = None,
) -> pd.Series:
    """Compute the Hurst exponent using DFA on rolling windows."""
    if not isinstance(series, pd.Series):
        raise TypeError("`series` doit être une pandas.Series")

    if min_valid is None:
        min_valid = window // 2

    idx, vals = [], []
    for start in range(0, len(series) - window + 1, step):
        end = start + window
        seg = series.iloc[start:end]
        if seg.count() < min_valid:
            vals.append(np.nan)
        else:
            h = ComputeDFA.dfa_statistic(seg, window_size=window, order=order, scales=scales)
            vals.append(h)
        idx.append(series.index[end - 1])

    return pd.Series(vals, index=idx, name="H_DFA")
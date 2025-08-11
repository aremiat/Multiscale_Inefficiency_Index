import sys
import os
from typing import Optional, Sequence, Union, Tuple


import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.MFDFA import ComputeMFDFA
from utils.RS import ComputeRS


class MultifractalFramework:
    """
    Cadre multifractal pour gérer l'analyse multifractale.
    """

    def __init__(
        self,
        data: pd.Series,
        ticker1: str,
        ticker2: str,
        window_hurst: int = 0,
        window_mfdfa: int = 0,
        q_list: Optional[np.ndarray] = None,
        scales: Optional[np.ndarray] = None,
        order: int = 1,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the MultifractalFramework.

        Args:
            data (pd.Series): Series of financial data.
            ticker1 (str): First ticker symbol.
            ticker2 (str): Second ticker symbol.
            window_hurst (int): Window size for Hurst exponent calculation.
            window_mfdfa (int): Window size for MF-DFA calculation.
            q_list (Optional[np.ndarray]): List of q values for MF-DFA.
            scales (Optional[np.ndarray]): Scales for MF-DFA.
            order (int): Polynomial order for detrending in MF-DFA.
            verbose (bool): If True, print verbose output during calculations.
        """

        self.data: pd.Series = data
        self.ticker1: str = ticker1
        self.ticker2: str = ticker2
        self.window_hurst: int = window_hurst
        self.window_mfdfa: int = window_mfdfa
        self.q_list: np.ndarray = q_list if q_list is not None else np.linspace(-5, 5, 21)
        self.scales: np.ndarray = (
            scales
            if scales is not None
            else np.unique(np.floor(np.logspace(np.log10(10), np.log10(500), 10)).astype(int))
        )
        self.order: int = order
        self.verbose: bool = verbose

        # attributs
        self.alpha_widths: list = []
        self.delta_alpha_diff: Optional[pd.Series] = None
        self.delta_alpha_diff_t1: Optional[pd.Series] = None
        self.delta_alpha_diff_t2: Optional[pd.Series] = None
        self.inefficiency_index: Optional[pd.Series] = None


    def compute_multifractal(self, shuffle, surrogate) -> None:
        """
        Compute the multifractal spectrum using MF-DFA.
        Args:
            shuffle (bool): If True, compute the shuffled surrogate.
            surrogate (bool): If True, compute the Gaussian correlated surrogate.
        """
        if self.verbose:
            print("[Verbose] compute_multifractal…")
        self.Fq = ComputeMFDFA.mfdfa(self.data, self.scales, self.q_list, self.order)
        self.h_q = ComputeMFDFA.compute_h_q(self.q_list, self.Fq, self.scales)
        self.alpha, self.f_alpha = ComputeMFDFA.compute_alpha_falpha(self.q_list, self.h_q)

        if shuffle:
            self.shuffle = ComputeMFDFA.shuffle(self.data)
            self.surogate = ComputeMFDFA.surrogate_gaussian_corr(self.data)

            self.Fq_shuffle = ComputeMFDFA.mfdfa(self.shuffle, self.scales, self.q_list, self.order)
            self.Fq_surogate = ComputeMFDFA.mfdfa(self.surogate, self.scales, self.q_list, self.order)

        if surrogate:
            self.h_q_shuffle = ComputeMFDFA.compute_h_q(self.q_list, self.Fq_shuffle, self.scales)
            self.h_q_surogate = ComputeMFDFA.compute_h_q(self.q_list, self.Fq_surogate, self.scales)

            self.alpha_shuffle, self.f_alpha_shuffle = ComputeMFDFA.compute_alpha_falpha(self.q_list, self.h_q_shuffle)
            self.alpha_surogate, self.f_alpha_surogate = ComputeMFDFA.compute_alpha_falpha(self.q_list, self.h_q_surogate)
        if self.verbose:
            print("[Verbose] End compute_multifractal")

    def compute_delta_alpha_diff(self) -> pd.Series:
        """
        Compute the difference in multifractal spectrum widths (Δα) between two tickers.
        Returns:
            pd.Series: Delta alpha difference indexed by window end positions.
        """
        if self.verbose:
            print("[Verbose] compute_delta_alpha_diff…")
        if self.delta_alpha_diff is None:
            self.delta_alpha_diff_t1 = ComputeMFDFA.mfdfa_rolling(
                self.data[self.ticker1], self.window_mfdfa, self.q_list, self.scales, self.order
            )
            self.delta_alpha_diff_t2 = ComputeMFDFA.mfdfa_rolling(
                self.data[self.ticker2], self.window_mfdfa, self.q_list, self.scales, self.order
            )
            self.delta_alpha_diff = self.delta_alpha_diff_t1 - self.delta_alpha_diff_t2
        if self.verbose:
            print("[Verbose] compute_delta_alpha_diff ended")
        return self.delta_alpha_diff

    def compute_inefficiency_index(self) -> pd.Series:
        """
        Compute the inefficiency index based on the Hurst exponent and delta alpha difference.
        Returns:
            pd.Series: Inefficiency index indexed by window end positions.
        """
        self.rolling_hurst = self.data.rolling(window=self.window_hurst).apply(
            lambda window: np.log(ComputeRS.rs_modified_statistic(window, len(window), chin=False)) / np.log(len(window)),
            raw=False
        ).dropna()

        if self.verbose:
            print("[Verbose] compute_inefficiency_index…")
        if self.delta_alpha_diff is None:
            self.compute_delta_alpha_diff()
        self.inefficiency_index = self.delta_alpha_diff * (self.rolling_hurst - 0.5).abs()
        if self.verbose:
            print("[Verbose] compute_inefficiency_index ended")
        return pd.Series(self.inefficiency_index, index=self.rolling_hurst.index, name="inefficiency_index")

    def plot_multifractal_results(self, shuffle: bool = False, surogate: bool = False) -> None:
        """
        Plot the results of the multifractal analysis.
        """
        if self.verbose:
            print(f"[Verbose] plot_results (shuffle={shuffle}, surogate={surogate})…")
        # h(q)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.q_list, y=self.h_q, mode="lines+markers", name=f"h(q) {self.data.name}"))
        fig.update_layout(
            title=f"Spectre h(q) {self.data.name}", xaxis_title="q", yaxis_title="h(q)", template="plotly_white"
        )
        fig.show()
        # multifractal
        if shuffle and not surogate:
            # shuffle only
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=self.alpha, y=self.f_alpha, mode="lines+markers", name="f(α)"))
            fig1.add_trace(go.Scatter(x=self.alpha_shuffle, y=self.f_alpha_shuffle, mode="lines+markers", name="f(α) shuffle"))
            fig1.update_layout(
                title="Spectre multifractal shuffle", xaxis_title="α", yaxis_title="f(α)", template="plotly_white"
            )
            fig1.show()
        elif surogate and not shuffle:
            # surogate only
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=self.alpha, y=self.f_alpha, mode="lines+markers", name="f(α)"))
            fig2.add_trace(
                go.Scatter(x=self.alpha_surogate, y=self.f_alpha_surogate, mode="lines+markers", name="f(α) surogate")
            )
            fig2.update_layout(
                title="Spectre multifractal surogate", xaxis_title="α", yaxis_title="f(α)", template="plotly_white"
            )
            fig2.show()
        elif surogate and shuffle:
            # both
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=self.alpha, y=self.f_alpha, mode="lines+markers", name="f(α)"))
            fig3.add_trace(go.Scatter(x=self.alpha_shuffle, y=self.f_alpha_shuffle, mode="lines+markers", name="f(α) shuffle"))
            fig3.add_trace(
                go.Scatter(x=self.alpha_surogate, y=self.f_alpha_surogate, mode="lines+markers", name="f(α) surogate")
            )
            fig3.update_layout(
                title="Spectre multifractal shuffle & surogate",
                xaxis_title="α",
                yaxis_title="f(α)",
                template="plotly_white",
            )
            fig3.show()

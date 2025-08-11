from typing import List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

class ComputeMFDFA:
    @staticmethod
    def mfdfa(
        signal: Union[np.ndarray, Sequence[float]],
        scales: Sequence[int],
        q_list: Sequence[float],
        order: int = 1
    ) -> np.ndarray:
        """
        Compute MF-DFA for a time series.

        Args:
            signal: 1D array-like of data points (e.g., returns).
            scales: Sequence of window sizes (ints) to use in DFA.
            q_list: Sequence of moment orders q.
            order: Polynomial order for detrending (default: 1, linear).

        Returns:
            Fq: 2D NumPy array of shape (len(q_list), len(scales)),
                where Fq[i, j] is F_q(s) for q=q_list[i] and s=scales[j].
        """
        data = np.asarray(signal, dtype=float)
        N: int = data.size
        Y: np.ndarray = np.cumsum(data - np.mean(data))
        Fq: np.ndarray = np.zeros((len(q_list), len(scales)), dtype=float)

        for j, s in enumerate(scales):
            s_int: int = int(s)
            if s_int < 2 or s_int > N:
                continue
            n_segments: int = N // s_int
            F_seg: List[float] = []

            # Forward segmentation
            for v in range(n_segments):
                segment = Y[v * s_int : (v + 1) * s_int]
                idx = np.arange(s_int)
                coeffs = np.polyfit(idx, segment, order)
                fit = np.polyval(coeffs, idx)
                F_seg.append(float(np.mean((segment - fit) ** 2)))

            # Reverse segmentation
            for v in range(n_segments):
                segment = Y[N - (v + 1) * s_int : N - v * s_int]
                idx = np.arange(s_int)
                coeffs = np.polyfit(idx, segment, order)
                fit = np.polyval(coeffs, idx)
                F_seg.append(float(np.mean((segment - fit) ** 2)))

            F_seg_arr = np.maximum(np.array(F_seg, dtype=float), 1e-10)

            for k, q in enumerate(q_list):
                if abs(q) < 1e-8:
                    Fq[k, j] = np.exp(0.5 * np.mean(np.log(F_seg_arr)))
                else:
                    Fq[k, j] = (np.mean(F_seg_arr ** (q / 2.0))) ** (1.0 / q)
        return Fq

    @staticmethod
    def compute_alpha_falpha(
        q_list: Sequence[float],
        h_q: Sequence[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute alpha(q) and f(alpha) via Legendre transform from h(q).

        Args:
            q_list: Sequence of q values.
            h_q: Sequence of corresponding h(q) values.

        Returns:
            alpha: NumPy array of alpha(q).
            f_alpha: NumPy array of f(alpha).
        """
        q_arr = np.asarray(q_list, dtype=float)
        h_arr = np.asarray(h_q, dtype=float)
        dq = q_arr[1] - q_arr[0]
        dh_dq = np.gradient(h_arr, dq)
        alpha = h_arr + q_arr * dh_dq
        f_alpha = q_arr * (alpha - h_arr) + 1.0
        return alpha, f_alpha

    @staticmethod
    def mfdfa_rolling(
        series: Union[pd.Series, Sequence[float]],
        window_size: int,
        q_list: Sequence[float],
        scales: Sequence[int],
        order: int = 1
    ) -> pd.Series:
        """
        Apply rolling MF-DFA on a series and return the multifractal width (Delta alpha).

        Args:
            series: Pandas Series or array-like of data.
            window_size: Number of points in each rolling window.
            q_list: Sequence of q values.
            scales: Sequence of scales for MF-DFA.
            order: Polynomial order for detrending.

        Returns:
            Pandas Series of Delta alpha indexed by window end positions.
        """
        if isinstance(series, pd.Series):
            data = series.values.astype(float)
            idx = series.index
        else:
            data = np.asarray(series, dtype=float)
            idx = pd.RangeIndex(start=0, stop=len(data))

        alpha_widths: List[float] = []
        timestamps: List = []
        N = len(data)

        for start in range(N - window_size + 1):
            end = start + window_size
            window = data[start:end]
            Fq = ComputeMFDFA.mfdfa(window, scales, q_list, order)

            h_q = []
            log_s = np.log(scales)
            for j in range(len(q_list)):
                log_F = np.log(Fq[j, :])
                h_q.append(np.polyfit(log_s, log_F, 1)[0])
            h_q_arr = np.asarray(h_q, dtype=float)

            alpha, _ = ComputeMFDFA.compute_alpha_falpha(q_list, h_q_arr)
            alpha_widths.append(float(alpha.max() - alpha.min()))
            timestamps.append(idx[end - 1])

        return pd.Series(alpha_widths, index=timestamps, name="alpha_width")

    @staticmethod
    def compute_h_q(q_list: Sequence[float], Fq: np.ndarray, scales: Sequence[int]) -> np.ndarray:
        """
        Compute h(q) from F(q) using the relation h(q) = d(log(F(q))) / d(log(s)).

        Args:
            q_list: Sequence of q values.
            Fq: 2D NumPy array of F(q) values.
            scales: Sequence of scales.

        Returns:
            h_q: NumPy array of h(q) values.
        """
        log_scales = np.log(scales)
        h_q = np.zeros((len(q_list),), dtype=float)
        for i, q in enumerate(q_list):
            log_Fq = np.log(Fq[i, :])
            h_q[i] = np.polyfit(log_scales, log_Fq, 1)[0]
        return h_q

    @staticmethod
    def shuffle(series: Sequence[float]) -> np.ndarray:
        """
        Generate a shuffled surrogate by randomizing the order of the input series.

        Args:
            series: Input 1D sequence.

        Returns:
            Shuffled surrogate series as NumPy array.
        """
        arr = pd.Series(series)
        arr = arr.sample(frac=1, random_state=56).reset_index(drop=True)
        return arr

    @staticmethod
    def surrogate_gaussian_corr(
        series: Sequence[float]
    ) -> np.ndarray:
        """
        Generate a Gaussian-correlated surrogate by phase randomization.

        Args:
            series: Input 1D sequence.

        Returns:
            Surrogate series as NumPy array.
        """
        arr = np.asarray(series, dtype=float)
        N = arr.size
        Xf = np.fft.rfft(arr)
        amplitudes = np.abs(Xf)
        random_phases = np.exp(2j * np.pi * np.random.rand(Xf.size))
        random_phases[0] = 1.0
        if N % 2 == 0:
            random_phases[-1] = 1.0
        Xf_surr = amplitudes * random_phases
        surrogate = np.fft.irfft(Xf_surr, n=N)
        return surrogate

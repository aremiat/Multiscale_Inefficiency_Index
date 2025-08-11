import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .abstract_estimation import AbstractHurstEstimator


class ScaledWindowedVariance(AbstractHurstEstimator):
    """
    This implementation is based on the paper:
    "Evaluating scaled windowed variance methods for estimating
    the Hurst coefficient of time series" from Physica A. 1997 July 15; 241(3-4): 606–626.
    """

    def __init__(self, data: pd.Series,
                 method: str = 'SD',
                 exclusions: bool = False):
        """
        ScaledWindowedVariance calculates the scaled windowed variance of a time series.
        It allows for different methods of calculation and can exclude certain values based on the method.

        Parameters:
        data (pd.Series): The time series data to analyze.
        min_window (int): The minimum window size for the rolling calculation.
        max_window (int): The maximum window size for the rolling calculation.
        custom_window_list (list): A custom list of window sizes to use instead of the default range.
        """

        self.data = data
        self.method = method

        if self.method not in ['SD', 'LD', 'BD']:
            raise ValueError("Method must be one of 'SD', 'LD' or 'BD'.")

        self.exclusions = exclusions

        self.N = len(data)
        if self.N < 2:
            raise ValueError("Data must contain at least two points.")

        self.min_window = int(np.log2(2))
        self.max_window = int(np.floor(np.log2(self.N)))

        self.window_sizes = 2 ** np.arange(self.min_window, self.max_window + 1)

    def _manage_detrending(self, window):
        """Apply the appropriate detrending method."""
        if self.method == 'SD':
            return window
        elif self.method == 'LD':
            return self._detrend_linear(window)
        elif self.method == 'BD':
            return self._detrend_bridge(window)

    def _detrend_linear(self, window):
        """Remove linear trend from window using regression."""
        x = np.arange(len(window))
        slope, intercept = np.polyfit(x, window, 1)
        trend = slope * x + intercept
        return window - trend

    def _detrend_bridge(self, window):
        """Remove bridge trend from window."""
        if len(window) < 2:
            return window
        x = np.arange(len(window))
        first, last = window[0], window[-1]
        trend = np.linspace(first, last, len(window))
        return window - trend

    def _find_exclusions_bounds(self):
        """Determine the exclusions based on the method."""

        if self.exclusions is False:
            return 0, 0

        lower_window_exclusion = {"SD": [0, 0], "LD": [1, 0], "BD": [1, 0]}

        exclusions_dict = {
            6: {"SD": [0, 2], "LD": [2, 0], "BD": [1, 0]},
            7: {"SD": [0, 3], "LD": [2, 1], "BD": [1, 0]},
            8: {"SD": [0, 3], "LD": [2, 2], "BD": [1, 0]},
            9: {"SD": [1, 4], "LD": [2, 2], "BD": [2, 2]},
            10: {"SD": [1, 4], "LD": [2, 2], "BD": [2, 3]},
            11: {"SD": [1, 5], "LD": [3, 4], "BD": [2, 4]},
            12: {"SD": [1, 5], "LD": [3, 5], "BD": [2, 4]},
            13: {"SD": [2, 6], "LD": [3, 5], "BD": [2, 5]},
            14: {"SD": [2, 7], "LD": [4, 5], "BD": [3, 6]},
            15: {"SD": [2, 7], "LD": [5, 5], "BD": [3, 7]},
            16: {"SD": [3, 7], "LD": [6, 5], "BD": [3, 7]},
            17: {"SD": [4, 7], "LD": [7, 5], "BD": [3, 7]},
        }

        if self.max_window in exclusions_dict:
            return exclusions_dict[self.max_window][self.method]
        elif self.max_window > max(exclusions_dict.keys()):
            return exclusions_dict[max(exclusions_dict.keys())][self.method]
        else:
            return lower_window_exclusion[self.method]

    def _create_exclusion_mask(self):
        lower_bound, upper_bound = self._find_exclusions_bounds()

        mask = np.zeros(len(self.window_sizes), dtype=bool)
        mask[lower_bound:len(self.window_sizes) - upper_bound] = True

        return mask

    def estimate(self):
        avg_sds = []
        valid_window_sizes = []

        for n in self.window_sizes:
            num_windows = self.N // n
            if num_windows < 1:
                continue

            sds = [np.std(self._manage_detrending(self.data[i * n:(i + 1) * n]), ddof=0) for i in range(num_windows)]

            avg_sd = np.mean(sds)
            avg_sds.append(avg_sd)
            valid_window_sizes.append(n)

        valid_window_sizes = np.array(valid_window_sizes)
        avg_sds = np.array(avg_sds)

        mask = np.zeros(len(valid_window_sizes), dtype=bool)
        if self.exclusions:
            lower_bound, upper_bound = self._find_exclusions_bounds()
            mask[lower_bound:len(valid_window_sizes) - upper_bound] = True
        else:
            mask[:] = True

        x = np.log2(valid_window_sizes)
        y = np.log2(avg_sds)

        slope, _ = np.polyfit(x[mask], y[mask], 1)

        return slope
from abc import ABC, abstractmethod
import numpy as np


class AbstractHurstEstimator(ABC):
    """
    Abstract base class for Hurst exponent estimators.
    """

    def __init__(self, time_series: np.ndarray):
        """
        Parameters:
            time_series (np.ndarray): The time series data to analyze.
        """
        self.ts = time_series

    @abstractmethod
    def estimate(self) -> float:
        """
        Estimate the Hurst exponent from the provided data.

        Returns:
            float: The estimated Hurst exponent.
        """
        pass

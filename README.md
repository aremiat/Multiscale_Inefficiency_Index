Hurst Estimation and Fractional Brownian Motion
Project Overview

This project is focused on estimating the Hurst exponent (H) for various financial time series data using different methods, such as the traditional R/S method and the modified R/S method. The main goal is to evaluate the long-term memory properties of financial markets and examine how momentum strategies might be influenced by these properties. We also simulate Fractional Brownian Motion (fBm) with different Hurst exponents to study their impact on financial modeling.
Files and Directories

    code/
        hurst_estimation.py: This script contains the code for estimating the Hurst exponent using the R/S and modified R/S methods.
        mbf.py: This script implements the simulation of Fractional Brownian Motion (fBm) and its corresponding autocorrelation functions.

    data/
        hurst_results.csv: A CSV file storing the results of Hurst exponent calculations for different financial time series.

    hurst_estimation.tex: The LaTeX document summarizing the methodology, results, and analysis.

Setup and Installation

    Clone the repository:

git clone <repository_url>

Set up a virtual environment:

python3 -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

Install required dependencies:

    pip install -r requirements.txt

Usage
Hurst Exponent Estimation

To estimate the Hurst exponent for financial time series data:

    Run the hurst_estimation.py script:

    python code/hurst_estimation.py

    The script will calculate the Hurst exponent using both the traditional R/S method and the modified R/S method. Results will be saved in the hurst_results.csv file.

Fractional Brownian Motion Simulation

To simulate Fractional Brownian Motion (fBm) with different Hurst exponents:

    Run the mbf.py script:

    python code/mbf.py

    This will generate simulations of fBm and compute the autocorrelation functions for different Hurst exponents, which can be analyzed in the context of financial data.

Results and Discussion
Hurst Exponent and Momentum Strategies

Based on the results, only the TOPX stock index showed statistical persistence, with the Hurst exponent being greater than 0.5. This implies that for most financial time series, despite the apparent long-term memory, momentum strategies may not be reliable unless validated by more robust statistical methods.
Fractional Brownian Motion

Fractional Brownian Motion models are useful for simulating market behaviors with varying degrees of long-term dependence. The results provide insights into the behavior of financial assets under different Hurst exponent values.
Contributing

Feel free to fork the repository, create branches, and submit pull requests for any improvements or bug fixes.
References

    Lo, A.W., "Long-Term Memory in Stock Market Prices" (PDF Link)
    Mignon, V., "Méthodes d'estimation de l'exposant de Hurst. Application aux rentabilités boursières", Économie & Prévision, 2003.
    Suhonen, A., Lennkh, M., and Perez, F., "Quantifying Backtest Overfitting in Alternative Beta Strategies", The Journal of Portfolio Management, 2017.

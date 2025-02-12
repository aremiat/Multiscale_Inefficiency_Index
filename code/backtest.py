import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy import stats
import plotly.graph_objects as go


# Function to calculate the R/S statistic
def rs_statistic(series):
    T = len(series)
    mean = np.mean(series)
    Y = series - mean
    R = np.max(np.cumsum(Y)) - np.min(np.cumsum(Y))
    S = np.std(series)
    return R / S


# Function to compute S_modified according to Andrews (1991)
def compute_S_modified(r):
    T = len(r)  # Number of observations
    mean_Y = np.mean(r)  # Mean of the series
    rho_1 = np.abs(np.corrcoef(r[:-1], r[1:])[0, 1])  # First-order autocorrelation

    # Calculate q according to Andrews (1991)
    q = ((3 * T) / 2) ** (1 / 3) * ((2 * rho_1) / (1 - rho_1)) ** (2 / 3)
    q = int(np.floor(q))

    # First term: classical variance
    var_term = np.sum((r - mean_Y) ** 2) / T

    # Second term: weighted sum of autocovariances
    auto_cov_term = 0
    for j in range(1, q + 1):  # j ranges from 1 to q
        w_j = 1 - (j / (q + 1))  # Newey-West weights
        sum_cov = np.sum((r[:-j] - mean_Y) * (r[j:] - mean_Y))  # Lagged autocovariance
        auto_cov_term += w_j * sum_cov

    auto_cov_term = (2 / T) * auto_cov_term

    S_squared = var_term + auto_cov_term
    return S_squared


# Function to calculate the modified R/S statistic
def rs_modified_statistic(series):
    T = len(series)
    mean = np.mean(series)
    Y = series - mean
    cum_sum = np.cumsum(Y)
    R = np.max(cum_sum) - np.min(cum_sum)

    sigma = np.sqrt(compute_S_modified(series))

    return R / sigma


# Function to calculate the t-statistic for the modified R/S statistic
def rs_t_statistic(rs_value, T):
    # The critical value for long memory at 10% significance (from Lo 1991)
    return rs_value / np.sqrt(T)


# Data acquisition: Collect price data
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']


# Test evaluation: Check for market persistence (momentum vs mean-reversion)
def evaluate_market_persistence(price_series, window_size=25):
    # Create rolling windows of data
    hurst_exponents = []
    rs_values = []
    rs_modified_values = []
    t_statistics = []

    for start in range(len(price_series) - window_size):
        window = price_series[start:start + window_size]

        # Calculate R/S statistic
        rs_value = rs_statistic(window)
        rs_modified_value = rs_modified_statistic(window)

        # Calculate the t-statistic for modified R/S
        t_stat = rs_t_statistic(rs_modified_value, window.shape[0])

        rs_values.append(rs_value)
        rs_modified_values.append(rs_modified_value)
        t_statistics.append(t_stat)

    return rs_values, rs_modified_values, t_statistics


# Strategy Selection: Momentum or Mean Reversion
def select_trading_strategy(rs_t_stat, momentum_threshold=1.620, mean_reversion_threshold=0.2):
    # Define strategy based on R/S and t-statistic for persistence
    if rs_t_stat > momentum_threshold:
        return "Momentum"
    elif rs_t_stat < mean_reversion_threshold:
        return "Mean Reversion"
    else:
        return "Neutral"


# Risk Management: Adjust position sizes
def dynamic_risk_management(strategy, equity, risk_factor=0.02):
    # Example of adjusting position size based on strategy
    if strategy == "Momentum":
        position_size = equity * risk_factor  # Higher risk for momentum strategy
    elif strategy == "Mean Reversion":
        position_size = equity * (risk_factor / 2)  # Lower risk for mean-reversion strategy
    else:
        position_size = equity * 0.01  # Default low-risk size for neutral strategy
    return position_size


# Simulate Trading: Backtesting the strategy
def backtest_strategy(ticker, start_date, end_date, window_size=800):
    price_data = fetch_data(ticker, start_date, end_date)
    log_p = np.log(price_data.values)
    r = np.diff(log_p.ravel())

    equity = 100000  # Starting with $100,000
    balance = equity
    positions = 0
    entry_price = 0  # To track entry price for stop-loss logic
    mom_num = 0
    mom_mean = 0

    performance = []

    for i in range(window_size, 1000):
        print(i)
        window_data = r[i - window_size:i]
        rs_values, rs_modified_values, t_statistics = evaluate_market_persistence(window_data)

        strategy = select_trading_strategy(t_statistics[-1])  # Use the last t-statistic
        position_size = dynamic_risk_management(strategy, balance)

        if strategy == "Momentum" and positions == 0:  # Buy in momentum
            positions = position_size / price_data.iloc[i].item()
            balance -= positions * price_data.iloc[i]
            entry_price = price_data.iloc[i]  # Store entry price
            mom_num += 1

        elif strategy == "Mean Reversion" and positions != 0:
            balance += positions * price_data.iloc[i]
            positions = 0
            mom_mean += 1


        equity_value = balance + (positions * r[i])
        performance.append(equity_value)

    print(f"Number of Momentum Trades: {mom_num}")
    print(f"Number of Mean Reversion Trades: {mom_mean}")
    print("Final Equity Value:", equity_value)

    fig = go.Figure()

    # Ajouter la courbe de performance
    fig.add_trace(go.Scatter(
        x=list(range(len(performance))),  # L'axe X correspond aux indices (temps)
        y=performance,  # L'axe Y correspond à la performance de l'équité
        mode='lines',  # Utiliser des lignes pour la courbe
        name='Performance'  # Nom de la trace pour l'inclure dans la légende
    ))

    # Ajouter un titre et des labels d'axes
    fig.update_layout(
        title=f"Backtest Performance for {ticker}",
        xaxis_title="Time",
        yaxis_title="Equity Value",
        template="plotly_dark",  # Style du graphique (vous pouvez changer le thème ici)
    )

    # Afficher le graphique
    fig.show()

    return performance


# Main function to test the strategy
if __name__ == "__main__":
    ticker = "^GSPC"  # Example: S&P 500 index
    start_date = "1995-01-02"
    end_date = "2024-01-01"
    backtest_strategy(ticker, start_date, end_date)

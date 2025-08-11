import os.path

import numpy as np
import pandas as pd
from arch import arch_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from aeqlib.quantlib import volatility



def build_state_space(K, m0):
    """
    Build the state space for the Markov State Model (MSM) with K components.

    Args:
        K (int): Number of components in each state.
        m0 (float): Initial multiplier value.
    Returns:
        states (list of tuples): List of states, each state is a tuple of K components.
    """

    d = 2 ** K
    states = []
    product_cache = np.zeros(d)

    for i in range(d):
        bits = np.binary_repr(i, width=K)
        mults = []
        for bit in bits:
            if bit == '0':
                mults.append(m0)
            else:
                mults.append(2.0 - m0)
        states.append(tuple(mults))
        product_cache[i] = np.prod(mults)

    return states, product_cache


def build_transition_matrix(states, K, gamma_list):
    """
    Build the transition matrix A for the Markov State Model (MSM).

    Args:
        states (list of tuples): List of states, each state is a tuple of K components.
        K (int): Number of components in each state.
        gamma_list (list of float): List of probabilities for switching between states.
    Returns:
        A (np.ndarray): Transition matrix of shape (d, d) where d = 2^K.
    """
    d = len(states)
    A = np.zeros((d, d))

    for i in range(d):
        s_i = states[i]
        for j in range(d):
            s_j = states[j]
            prob = 1.0
            for k in range(K):
                if s_i[k] == s_j[k]:
                    prob *= (1.0 - gamma_list[k])
                else:
                    if abs(s_i[k] + s_j[k] - 2.0) < 1e-10:
                        prob *= gamma_list[k]
                    else:
                        prob = 0.0
                        break
            A[i, j] = prob
        row_sum = A[i, :].sum()
        if row_sum > 1e-14:
            A[i, :] /= row_sum

    return A



def msm_loglik(data, sigma, states, product_cache, A):
    """
    Compute the log-likelihood of the Markov State Model (MSM) given the data.

    Args:
        data (np.ndarray): Time series data (returns).
        sigma (float): Volatility parameter.
        states (list of tuples): List of states, each state is a tuple of K components.
        product_cache (np.ndarray): Precomputed product of multipliers for each state.
        A (np.ndarray): Transition matrix of shape (d, d) where d = 2^K.
    Returns:
        logL (float): Log-likelihood of the MSM given the data.
    """
    T = len(data)
    d = len(states)

    eigvals, eigvecs = np.linalg.eig(A.T)

    idx_stat = np.argmin(np.abs(eigvals - 1.0))
    pi0 = eigvecs[:, idx_stat].real
    pi0 = pi0 / pi0.sum()

    pi_t = pi0

    logL = 0.0

    for t in range(T):
        fx = np.zeros(d)
        for j in range(d):
            vol = sigma * np.sqrt(product_cache[j])
            if vol < 1e-14:
                fx[j] = 0.0
            else:
                z = data[t] / vol
                fx[j] = (1.0 / vol) * (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z ** 2)


        piA = pi_t @ A
        numer = piA * fx
        denom = numer.sum()
        if denom < 1e-14:
            logL += np.log(1e-300)
            pi_t = pi0
        else:
            logL += np.log(denom)
            pi_t = numer / denom

    return logL


def estimate_msm_binomial(
        data, K=3, b=2.0,  # param b pour gamma_k = 1 - (1-gamma_1)*b^(k-1)
        m0_grid=None, gamma1_grid=None, sigma_grid=None
):
    """
    Compute the best parameters for the Markov State Model (MSM) using a grid search.

    Args:
        data (np.ndarray): Time series data (returns).
        K (int): Number of components in the MSM.
        b (float): Base value for the components.
        m0_grid (np.ndarray): Grid of initial multiplier values (m0).
        gamma1_grid (np.ndarray): Grid of probabilities for the first component (gamma1).
        sigma_grid (np.ndarray): Grid of volatility parameters (sigma).
    Returns:
        best_params (tuple): Tuple of the best parameters (m0, gamma1, sigma).
        best_logL (float): Log-likelihood of the best parameters.
    """
    if m0_grid is None:
        m0_grid = np.linspace(0.6, 1.4, 5)
    if gamma1_grid is None:
        gamma1_grid = np.linspace(0.1, 0.9, 5)
    if sigma_grid is None:
        sigma_grid = np.linspace(0.5, 2.0, 5)

    total_iterations = len(m0_grid) * len(gamma1_grid) * len(sigma_grid)
    current_iter = 0

    best_logL = -1e15
    best_params = (None, None, None)

    data_arr = np.array(data)

    for m0 in m0_grid:
        states, product_cache = build_state_space(K, m0)
        for gamma1 in gamma1_grid:
            gamma_list = []
            for k in range(1, K + 1):
                gamma_k = 1.0 - (1.0 - gamma1) ** (b ** (k - 1))
                gamma_list.append(gamma_k)
            A = build_transition_matrix(states, K, gamma_list)
            for sigma in sigma_grid:
                current_iter += 1
                progress = 100.0 * current_iter / total_iterations
                print(f"Progress: {progress:.2f}% completed", end="\r")

                ll = msm_loglik(data_arr, sigma, states, product_cache, A)
                if ll > best_logL:
                    best_logL = ll
                    best_params = (m0, gamma1, sigma)

    print()
    return best_params, best_logL


def forward_filter(data, sigma_opt, states, product_cache, A):
    T = len(data)
    d = len(states)
    eigvals, eigvecs = np.linalg.eig(A.T)
    idx_stat = np.argmin(np.abs(eigvals - 1.0))
    pi0 = eigvecs[:, idx_stat].real
    pi0 = pi0 / pi0.sum()

    pi_t = pi0
    pi_history = np.zeros((T, d))
    loglik = 0.0
    for t in range(T):
        fx = np.zeros(d)
        for j in range(d):
            vol_j = sigma_opt * np.sqrt(product_cache[j])
            z = data[t] / vol_j
            fx[j] = (1.0 / vol_j) * (1.0 / np.sqrt(2*np.pi)) * np.exp(-0.5 * z**2)
        piA = pi_t @ A
        numer = piA * fx
        denom = numer.sum()
        if denom < 1e-14:
            denom = 1e-14
        loglik += np.log(denom)
        pi_t = numer / denom
        pi_history[t, :] = pi_t
    return pi_history, loglik

def fitted_volatility(pi_history, sigma_opt, product_cache):
    T, d = pi_history.shape
    vol_fit = np.zeros(T)
    for t in range(T):
        exp_var_t = sigma_opt**2 * np.sum(pi_history[t, :] * product_cache)
        vol_fit[t] = np.sqrt(exp_var_t)
    return vol_fit


if __name__ == "__main__":
    np.random.seed(43)
    K = 5
    b = 2

    DATA_PATH = os.path.dirname(__file__) + "/../data"
    data_df = pd.read_csv(f"{DATA_PATH}/russell_2000.csv", index_col=0)

    returns = data_df['^RUT'].pct_change().dropna().values

    T = len(returns)
    T_train = int(0.8 * T)
    train_data = returns[:T_train]
    test_data = returns[T_train:]

    m0_grid = np.linspace(0.1, 1.9, 20)
    gamma1_grid = np.linspace(0.01, 0.99, 20)
    sigma_grid = np.linspace(0.001, 0.1, 20)

    best_params, best_logL = estimate_msm_binomial(train_data,
                                                   K=K, b=b,
                                                   m0_grid=m0_grid,
                                                   gamma1_grid=gamma1_grid,
                                                   sigma_grid=sigma_grid)
    m0_opt, gamma1_opt, sigma_opt = best_params
    print("Best param (m0, gamma1, sigma) =", best_params, "LogLik =", best_logL)

    n_train = len(train_data)
    k_params = 3
    AIC = 2 * k_params - 2 * best_logL
    BIC = k_params * np.log(n_train) - 2 * best_logL
    print(f"AIC (train) = {AIC:.2f}")
    print(f"BIC (train) = {BIC:.2f}")

    states, product_cache = build_state_space(K, m0_opt)
    gamma_list_opt = []
    for k_ in range(1, K+1):
        val = 1.0 - (1.0 - gamma1_opt)**(b**(k_-1))
        gamma_list_opt.append(max(0.0, min(1.0, val)))
    A = build_transition_matrix(states, K, gamma_list_opt)

    pi_history_train, loglik_in_sample = forward_filter(train_data, sigma_opt, states, product_cache, A)
    pi_last_train = pi_history_train[-1, :]

    vol_fit_train = fitted_volatility(pi_history_train, sigma_opt, product_cache)

    T_test = len(test_data)
    pi_history_test = np.zeros((T_test, len(states)))
    vol_forecast = np.zeros(T_test)

    pi_t = pi_last_train.copy()
    for t in range(T_test):
        pi_pred  = pi_t @ A

        expected_var = sigma_opt**2 * np.sum(pi_pred * product_cache)
        vol_forecast[t] = np.sqrt(expected_var)

        fx = np.zeros(len(states))
        for j in range(len(states)):
            vol_j = sigma_opt * np.sqrt(product_cache[j])
            if vol_j < 1e-14:
                fx[j] = 1e-300
            else:
                z = test_data[t] / vol_j
                fx[j] = (1.0 / vol_j) * (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z ** 2)

        numer = pi_pred * fx
        denom = numer.sum()
        if denom < 1e-14:
            denom = 1e-14
        pi_t = numer / denom

        pi_history_test[t, :] = pi_t

    realized_vol_test = pd.Series(test_data).rolling(window=21).std().values
    realized_vol_test = np.where(np.isnan(realized_vol_test), 0.0, realized_vol_test)

    valid_idx = np.where(realized_vol_test > 0)[0]
    vol_fore_oos = vol_forecast[valid_idx]
    vol_real_oos = realized_vol_test[valid_idx]

    rmse_oos = np.sqrt(np.mean((vol_fore_oos - vol_real_oos) ** 2))
    corr_oos = np.corrcoef(vol_fore_oos, vol_real_oos)[0, 1]
    print(f"Out-of-sample RMSE = {rmse_oos:.5f}")
    print(f"Out-of-sample Corr  = {corr_oos:.3f}")

    fig = make_subplots(rows=1, cols=1, subplot_titles=["Out-of-sample forecast (MSM) vs. Realized Vol"])
    test_range = np.arange(T_test)

    fig.add_trace(go.Scatter(
        x=test_range,
        y=vol_forecast,
        mode='lines',
        name='Forecasted MSM Vol'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=test_range,
        y=realized_vol_test,
        mode='lines',
        name='Realized Vol (21d rolling)'
    ), row=1, col=1)

    fig.update_layout(
        title=f"Out-of-sample MSM Vol Forecast vs. Realized Vol\n(RMSE={rmse_oos:.4f}, Corr={corr_oos:.3f})",
        xaxis_title="Test sample index",
        yaxis_title="Volatility"
    )
    fig.show()

import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.RS import ComputeRS
from utils.MFDFA import ComputeMFDFA


def non_overlapping_rolling(series, window, func):
    results = []
    indices = []
    n_segments = len(series) // window
    for i in range(n_segments):
        seg = series.iloc[i * window:(i + 1) * window]
        results.append(func(seg))
        indices.append(seg.index[-1])
    return pd.Series(results, index=indices)


def compute_rolling_metric(series, window_size, method='modified', rolling_type='overlapping', chin=False):
    series_shifted = series
    if rolling_type == 'overlapping':
        if method == 'modified':
            roll = series_shifted.rolling(window_size).apply(
                lambda window: np.log(ComputeRS.rs_modified_statistic(window, len(window), chin=chin)) / np.log(
                    len(window)),
                raw=False
            ).dropna()
        elif method == 'traditional':
            roll = series_shifted.rolling(window_size).apply(
                lambda window: np.log(ComputeRS.rs_statistic(window, len(window))) / np.log(len(window)),
                raw=False
            ).dropna()
        else:
            raise ValueError("Méthode inconnue")
    elif rolling_type == 'nonoverlapping':
        if method == 'modified':
            roll = non_overlapping_rolling(series_shifted, window_size,
                                           lambda window: np.log(ComputeRS.rs_modified_statistic(window, len(window),
                                                                                                 chin=chin)) / np.log(
                                               len(window))
                                           )
        elif method == 'traditional':
            roll = non_overlapping_rolling(series_shifted, window_size,
                                           lambda window: np.log(ComputeRS.rs_statistic(window, len(window))) / np.log(
                                               len(window))
                                           )
            full_index = pd.date_range(start=roll.index.min(), end=roll.index.max(), freq='B')
            rolling_full = roll.reindex(full_index)
            roll = rolling_full.ffill()
        else:
            raise ValueError("Méthode inconnue")
    else:
        raise ValueError("Type de rolling inconnu")
    return roll


# =====================================================================
# Fonction pour calculer le momentum
# =====================================================================
def compute_momentum(diff_series, shift_days=20, window_size=220):
    diff_shifted = diff_series.shift(shift_days)
    momentum = diff_shifted.rolling(window_size).mean()
    return momentum


# =====================================================================
# Calcul de l'indice d'inefficience et stratégie de positionnement
# =====================================================================
def compute_inefficiency_index(delta_alpha_diff, rolling_hurst):
    """
    Combine la différence de largeur de spectre (delta_alpha_diff),
    l'écart absolu (rolling Hurst - 0.5).
    """
    return delta_alpha_diff * abs(rolling_hurst - 0.5)


def compute_positions_with_inefficiency(rolling_hurst, momentum, ineff_index, ticker1, ticker2,
                                        threshold_h=0.5, threshold_ineff=1e-6):
    positions = pd.DataFrame(index=rolling_hurst.index, columns=[ticker1, ticker2])
    pos1, pos2 = [], []
    for idx in rolling_hurst.index:
        H_val = rolling_hurst.loc[idx]
        m_val = momentum.loc[idx]
        ineff = ineff_index.loc[idx]
        if H_val > threshold_h:
            if ineff > threshold_ineff and m_val > 0:
                pos1.append(0.8)
                pos2.append(0.2)
            elif ineff < -threshold_ineff and m_val < 0:
                pos1.append(0.2)
                pos2.append(0.8)
            else:
                pos1.append(0.5)
                pos2.append(0.5)
        else:
                pos1.append(0.5)
                pos2.append(0.5)
    positions[ticker1] = pos1
    positions[ticker2] = pos2
    return positions


# =====================================================================
# Backtest et calcul des performances
# =====================================================================
def run_backtest(all_p, positions, ticker1, ticker2, fee_rate=0.0005):
    portfolio_returns = positions[ticker1] * all_p[ticker1] + positions[ticker2] * all_p[ticker2]
    transaction_costs = (positions.diff().abs() * all_p).sum(axis=1) * fee_rate
    transaction_costs = transaction_costs.fillna(0)
    portfolio_returns = portfolio_returns - transaction_costs
    cumulative_returns = (1 + portfolio_returns).cumprod()
    return cumulative_returns, portfolio_returns


def compute_performance_stats(daily_returns: pd.Series, freq=252):
    daily_returns = daily_returns.dropna()
    if len(daily_returns) == 0:
        return np.nan, np.nan, np.nan, np.nan
    total_ret = (1 + daily_returns).prod() - 1
    nb_obs = len(daily_returns)
    annual_return = (1 + total_ret) ** (freq / nb_obs) - 1
    daily_vol = daily_returns.std()
    annual_vol = daily_vol * np.sqrt(freq)
    daily_mean = daily_returns.mean()
    sharpe_ratio = (annual_return / annual_vol)
    cum_curve = (1 + daily_returns).cumprod()
    running_max = cum_curve.cummax()
    drawdown = (cum_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    return annual_return, annual_vol, sharpe_ratio, max_drawdown

# Plot the rolling Hurst signal and position switches
def plot_positions_and_hurst(rolling_hurst, positions, ticker1, ticker2):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rolling_hurst.index,
        y=rolling_hurst,
        mode='lines',
        name='Rolling Hurst Signal',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=positions.index,
        y=positions[ticker1],
        mode='lines',
        name=f'{ticker1} Position',
        line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=positions.index,
        y=positions[ticker2],
        mode='lines',
        name=f'{ticker2} Position',
        line=dict(color='orange')
    ))

    # Add layout details
    fig.update_layout(
        title="Rolling Hurst Signal and Position Switches",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white"
    )

    fig.show()

# Count the number of position switches
def count_position_switches(positions, ticker1, ticker2):
    switches_ticker1 = (positions[ticker1].diff().abs() > 0).sum()
    switches_ticker2 = (positions[ticker2].diff().abs() > 0).sum()
    total_switches = switches_ticker1 + switches_ticker2
    print(f"Number of position switches: {total_switches}")

def compute_positions(rolling_signal, momentum, ticker1, ticker2, default=0.5, threshold=0.5):
    # On suppose que les deux indices ont la même fréquence
    positions = pd.DataFrame(index=rolling_signal.index, columns=[ticker1, ticker2])
    positions[ticker1] = default
    positions[ticker2] = default
    condition = rolling_signal > threshold
    positions.loc[condition & (momentum > 0), ticker1] = 0.8
    positions.loc[condition & (momentum > 0), ticker2] = 0.2
    positions.loc[condition & (momentum < 0), ticker1] = 0.2
    positions.loc[condition & (momentum < 0), ticker2] = 0.8
    return positions


# =====================================================================
# 6. Main : Chargement des données et exécution de la stratégie
# =====================================================================
if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "../data")
    IMG_PATH = os.path.join(os.path.dirname(__file__), "../img")

    # Lecture des prix (indices SP500 et Russell)
    p = pd.read_csv(f"{DATA_PATH}/index_prices2.csv", index_col=0, parse_dates=True)
    ticker1 = "^GSPC"
    ticker2 = "^RUT"
    if ticker1 == "^GSPC":
        name = "S&P 500"
    elif ticker1 == "^RUT":
        name = "Russell 2000"
    all_prices = pd.concat([p[ticker1], p[ticker2]], axis=1)
    all_prices = all_prices.loc["1987-10-09": "2025-02-28"]
    all_p = all_prices.pct_change().dropna()
    all_p['Diff'] = all_p[ticker1] - all_p[ticker2]
    r = all_p['Diff']

    momentum = compute_momentum(r, shift_days=20, window_size=220).dropna()
    mfdfa_window = 1008
    q_list = np.linspace(-3, 3, 13)
    scales = np.unique(np.logspace(np.log10(10), np.log10(200), 10, dtype=int))
    rolling_delta_ticker1 = ComputeMFDFA.mfdfa_rolling(np.log(all_prices[ticker1]).diff().dropna().shift(1),
                                                       mfdfa_window, q_list, scales, order=1).dropna()
    rolling_delta_ticker2 = ComputeMFDFA.mfdfa_rolling(np.log(all_prices[ticker2]).diff().dropna().shift(1),
                                                       mfdfa_window, q_list, scales, order=1).dropna()
    rolling_delta_ticker1.index.name = "Date"
    rolling_delta_ticker2.index.name = "Date"

    common_dates_mfdfa = rolling_delta_ticker1.index.intersection(rolling_delta_ticker2.index)
    delta_alpha_diff = (
                rolling_delta_ticker1.loc[common_dates_mfdfa] - rolling_delta_ticker2.loc[common_dates_mfdfa])

    # do a plot of the difference between the two rolling delta alpha
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=delta_alpha_diff.index, y=delta_alpha_diff,
    #                             mode='lines', name='Delta Alpha Difference',
    #                             line=dict(color='blue')))
    # fig.update_layout(title="Delta Alpha Difference",
    #                     xaxis_title="Date",
    #                     yaxis_title="Delta Alpha Difference",
    #                     template="plotly_white")
    # fig.show()

    # Configurations pour le rolling sur RS
    rolling_configs = {
        "ModifOverlap120": {"method": "modified", "rolling_type": "overlapping", "window_size": 120},
        # "TradOverlap120": {"method": "traditional", "rolling_type": "overlapping", "window_size": 120},
        # "ModifOverlap252": {"method": "modified", "rolling_type": "overlapping", "window_size": 252},
        # "ModifOverlap504": {"method": "modified", "rolling_type": "overlapping", "window_size": 504},
        # "ModifOverlap1260": {"method": "modified", "rolling_type": "overlapping", "window_size": 1260},
        # "ModifOverlap2520": {"method": "modified", "rolling_type": "overlapping", "window_size": 2520},
    }
    start_dates = ["1995-01-02", "2000-01-02", "2005-01-02", "2010-01-02", "2015-01-02", "2020-01-02"]

    # for start_date in start_dates:
    # cumulative_returns_dict = {}
    # performance_results = []
    for config_name, config in rolling_configs.items():
        cumulative_returns_dict = {}
        performance_results = []
        w_s = config["window_size"]
        p_config = all_p.copy()

        rolling_signal = compute_rolling_metric(r.shift(1),
                                                config["window_size"],
                                                method=config["method"],
                                                rolling_type=config["rolling_type"],
                                                chin=False).dropna()

        common_dates = rolling_signal.index.intersection(momentum.index).intersection(delta_alpha_diff.index)
        signal = rolling_signal.loc[common_dates]
        mom = momentum.loc[common_dates]
        delta_alpha_diff_aligned = delta_alpha_diff.loc[common_dates]
        first_valid_index = signal.first_valid_index()

        ineff_index = pd.Series(
            compute_inefficiency_index(delta_alpha_diff_aligned, signal),
            index=common_dates
        ).dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ineff_index.index, y=ineff_index,
                                 mode='lines', name='Inefficiency Index',
                                 line=dict(color='purple')))
        fig.update_layout(title="Inefficiency Index",
                          xaxis_title="Date",
                          yaxis_title="Inefficiency Index",
                          template="plotly_white")
        fig.show()
        ineff_index.index.name = "Date"
        # ineff_index.to_csv(f"{DATA_PATH}/inefficiency_index.csv")

        positions = compute_positions_with_inefficiency(signal, mom, ineff_index, ticker1, ticker2,
                                                        threshold_h=0.5, threshold_ineff=1e-6)

        positions_filtered = positions.loc[first_valid_index:]
        p_config_filtered = p_config.loc[first_valid_index:]
        common_dates_bp = positions_filtered.index.intersection(p_config_filtered.index)
        positions_final = positions_filtered.loc[common_dates_bp]
        all_p_config_final = p_config_filtered.loc[common_dates_bp]
        cum_returns, port_returns = run_backtest(all_p_config_final, positions_final, ticker1, ticker2,
                                                 fee_rate=0.005)
        cumulative_returns_dict[config_name] = cum_returns
        ann_ret, ann_vol, sharpe, max_dd = compute_performance_stats(port_returns)
        performance_results.append({
            "Strategy": config_name,
            "Annualized Return": round(ann_ret * 100, 3),
            "Annualized Volatility": round(ann_vol * 100, 3),
            "Sharpe": round(sharpe, 3),
            "Max Drawdown": round(max_dd * 100, 3)
        })
        # plot_positions_and_hurst(rolling_hurst=rolling_signal, positions=positions, ticker1=ticker1, ticker2=ticker2)
        # count_position_switches(positions, ticker1, ticker2)

        # fig_backtest = make_subplots(
        #     rows=2, cols=1, shared_xaxes=True,
        #     row_heights=[0.5, 0.5],
        #     vertical_spacing=0.05,
        #     subplot_titles=("Log Cumulative Return", "Positions")
        # )
        fig_backtest = go.Figure()

        fig_backtest.add_trace(
            go.Scatter(
                x=cum_returns.index,
                y=np.log(cum_returns),
                mode='lines',
                name=config_name,
                line=dict(color="blue")  # you can choose any color
            )

        )
        fig_backtest.update_layout(
            title=f"Cumulative Returns - Strategy: {config_name}",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template="plotly_white"
        )

        # Modif Overlap 120 No Filter
        p = all_p.copy()
        rolling_signal = compute_rolling_metric(r.shift(1), 120,
                                                method="modified",
                                                rolling_type="overlapping",
                                                chin=False).dropna()
        common_dates = rolling_signal.index.intersection(momentum.index)
        signal = rolling_signal.loc[common_dates]
        mom = momentum.loc[common_dates]
        positions = compute_positions(signal, mom, ticker1, ticker2, default=0.5, threshold=0.5)
        common_dates_bp = positions.index.intersection(p.index)
        positions = positions.loc[common_dates_bp]
        p_config_filtered = p_config.loc[first_valid_index:]
        common_dates_bp = positions_filtered.index.intersection(p_config_filtered.index)
        positions_final = positions_filtered.loc[common_dates_bp]
        all_p_config_final = p_config_filtered.loc[common_dates_bp]
        # all_p_config = p.loc[common_dates_bp]
        cum_returns, port_returns = run_backtest(all_p_config_final, positions, ticker1, ticker2, fee_rate=0.005)
        cumulative_returns_dict["ModifOverlap120NoFilter"] = cum_returns
        ann_ret, ann_vol, sharpe, max_dd = compute_performance_stats(port_returns)
        performance_results.append({
            "Strategy": "ModifOverlap120NoFilter",
            "Annualized Return": round(ann_ret * 100, 3),
            "Annualized Volatility": round(ann_vol * 100, 3),
            "Sharpe": round(sharpe, 3),
            "Max Drawdown": round(max_dd * 100, 3)
        })

        # Ports de comparaison : SP500, Russell et portefeuille 50/50
        new_p = all_p.loc[first_valid_index:]
        sp500_returns = new_p[ticker1]
        sp500_cumulative = (1 + sp500_returns).cumprod()
        russell_returns = new_p[ticker2]
        russell_cumulative = (1 + russell_returns).cumprod()
        portfolio_50_50_returns = 0.5 * new_p[ticker1] + 0.5 * new_p[ticker2]
        portfolio_50_50_cumulative = (1 + portfolio_50_50_returns).cumprod()
        # =====================================================================
        # Visualisation des courbes cumulées des stratégies
        # =====================================================================
        # first_strategy, cum_returns = list(cumulative_returns_dict.items())[0]
        # fig_backtest.add_trace(
        #     go.Scatter(
        #         x=cum_returns.index,
        #         y=np.log(cum_returns),
        #         mode='lines',
        #         name=first_strategy,
        #         line=dict(color="blue")
        #     )
        # )
        # fig_backtest.update_layout(
        #     title=f"Cumulative Returns - Strategy: {first_strategy}",
        #     xaxis_title="Date",
        #     yaxis_title="Cumulative Return",
        #     template="plotly_white"
        # )
        # fig_backtest.show()

        # fig_backtest.add_trace(
        #     go.Scatter(
        #         x=sp500_cumulative.index,
        #         y=np.log(sp500_cumulative),
        #         mode='lines',
        #         name="Long Only S&P500",
        #         line=dict(color='red')
        #     ),
        #     row=1, col=1
        # )
        fig_backtest.add_trace(
            go.Scatter(
                x=portfolio_50_50_cumulative.index,
                y=np.log(portfolio_50_50_cumulative),
                mode='lines',
                name="50/50 Portfolio",
                line=dict(color='black')
            )
        )

        # ticker_cols = positions.columns.tolist()
        #
        # fig_backtest.add_trace(
        #     go.Scatter(
        #         x=positions.index,
        #         y=positions[ticker_cols[0]],
        #         mode='lines',
        #         name=f'Position {ticker_cols[0]}',
        #         line=dict(color='green')
        #     ),
        #     row=2, col=1
        # )
        # fig_backtest.add_trace(
        #     go.Scatter(
        #         x=positions.index,
        #         y=positions[ticker_cols[1]],
        #         mode='lines',
        #         name=f'Position {ticker_cols[1]}',
        #         line=dict(color='orange')
        #     ),
        #     row=2, col=1
        # )

        # Mise en forme
        fig_backtest.update_layout(
            title="Cumulative Return",
            template="plotly_white",
        )

        # Légendes axes
        fig_backtest.update_xaxes(title_text="Date")
        fig_backtest.update_yaxes(title_text="Log Cumulative Return")
        # fig_backtest.update_yaxes(title_text="Positions", row=2, col=1)
        fig_backtest.show()

        # =====================================================================
        # Calcul des performances des portefeuilles de comparaison
        # =====================================================================
        ann_ret_sp500, ann_vol_sp500, sharpe_sp500, max_dd_sp500 = compute_performance_stats(sp500_returns)
        ann_ret_russell, ann_vol_russell, sharpe_russell, max_dd_russell = compute_performance_stats(russell_returns)
        ann_ret_50_50, ann_vol_50_50, sharpe_50_50, max_dd_50_50 = compute_performance_stats(portfolio_50_50_returns)

        performance_results.append({
            "Strategy": "Long Only SP500",
            "Annualized Return": round(ann_ret_sp500 * 100, 3),
            "Annualized Volatility": round(ann_vol_sp500 * 100, 3),
            "Sharpe": round(sharpe_sp500, 3),
            "Max Drawdown": round(max_dd_sp500 * 100, 3)
        })
        performance_results.append({
            "Strategy": "Long Only Russell",
            "Annualized Return": round(ann_ret_russell * 100, 3),
            "Annualized Volatility": round(ann_vol_russell * 100, 3),
            "Sharpe": round(sharpe_russell, 3),
            "Max Drawdown": round(max_dd_russell * 100, 3)
        })
        performance_results.append({
            "Strategy": "50/50 Portfolio",
            "Annualized Return": round(ann_ret_50_50 * 100, 3),
            "Annualized Volatility": round(ann_vol_50_50 * 100, 3),
            "Sharpe": round(sharpe_50_50, 3),
            "Max Drawdown": round(max_dd_50_50 * 100, 3)
        })

        df_results = pd.DataFrame(performance_results)
        print("=== Performance Summary ===")
        print(df_results)
        # df_results.to_csv(f"{DATA_PATH}/backtest_results_start_{start_date}.csv", index=False)
        df_results.to_csv(f"{DATA_PATH}/backtest_long_neutral_results.csv", index=False)
        # fig_backtest.write_image(f"{IMG_PATH}/backtest_long_neutral.png", width=1200, height=800)

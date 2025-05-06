import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from aeqlib.portfolio_construction import Portfolio

from utils.RS import ComputeRS
from utils.MFDFA import ComputeMFDFA
import time


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


def compute_nav_with_inefficiency(
        rolling_hurst, momentum, ineff_index,
        ticker1, ticker2,
        threshold_h=0.5, threshold_ineff=1e-6, portfolio_50_50: bool = False):

    dates = rolling_hurst.index
    w1, w2 = 0.5, 0.5

    pos1, pos2 = [], []

    for t in dates:
        H = rolling_hurst.loc[t]
        m = momentum.loc[t]
        ineff = ineff_index.loc[t]

        # 1) SIGNAL FORT → override et reset du drift
        if H > threshold_h:
            if ineff > threshold_ineff and m > 0:
                w1, w2 = 1.0, 0.0
            elif ineff < -threshold_ineff and m < 0:
                w1, w2 = 0.0, 1.0
            elif ineff < -threshold_ineff and m > 0:
                w1, w2 = w1, w2
            elif ineff > threshold_ineff and m < 0:
                w1, w2 = w1, w2
        # 2) PAS DE SIGNAL (ou signal faible) → drift multiplicatif
        else:
            w1, w2 = 0.5, 0.5


        if portfolio_50_50:
            w1, w2 = 0.5, 0.5
        # valeur de portefeuille après PnL de la veille

        # 3) on stocke
        pos1.append(w1)
        pos2.append(w2)

    return pd.DataFrame({ticker1: pos1, ticker2: pos2}, index=dates)

def compute_nav_with_inefficiency(
    rolling_hurst: pd.Series,
    momentum: pd.Series,
    ineff_index: pd.Series,
    ticker1: str,
    ticker2: str,
    threshold_h: float = 0.5,
    threshold_ineff: float = 1e-6,
    min_hurst_days: int = 5,
    portfolio_50_50: bool = False
) -> pd.DataFrame:
    """
    Compute portfolio weights based on rolling Hurst, momentum, and inefficiency index.

    Parameters:
        rolling_hurst: Series of Hurst exponent values indexed by date.
        momentum: Series of momentum signals indexed by date.
        ineff_index: Series of inefficiency index values indexed by date.
        ticker1: Name of first asset.
        ticker2: Name of second asset.
        threshold_h: Hurst threshold to consider a "strong" long-memory signal.
        threshold_ineff: Inefficiency threshold for signal filtering.
        min_hurst_days: Minimum number of consecutive days Hurst must exceed threshold_h before taking a position.
        portfolio_50_50: If True, always take a 50/50 position.

    Returns:
        DataFrame of weights for ticker1 and ticker2 over time.
    """
    dates = rolling_hurst.index
    w1, w2 = 0.5, 0.5

    pos1, pos2 = [], []
    # Counter for consecutive days H > threshold_h
    hurst_counter = 0

    for t in dates:
        H = rolling_hurst.loc[t]
        m = momentum.loc[t]
        ineff = ineff_index.loc[t]

        # Update hurst counter
        if H > threshold_h:
            hurst_counter += 1
        else:
            hurst_counter = 0

        # Only apply signals if Hurst has been > threshold_h for enough days
        if hurst_counter >= min_hurst_days:
            # Strong Hurst signal: override and reset drift
            if ineff > threshold_ineff and m > 0:
                w1, w2 = 1.0, 0.0
            elif ineff < -threshold_ineff and m < 0:
                w1, w2 = 0.0, 1.0
            elif ineff < -threshold_ineff and m > 0:
                w1, w2 = w1, w2
            elif ineff > threshold_ineff and m < 0:
                w1, w2 = w1, w2
        else:
            # No strong Hurst signal: drift back to 50/50
            w1, w2 = 0.5, 0.5

        # Force 50/50 if requested
        if portfolio_50_50:
            w1, w2 = 0.5, 0.5

        # Store positions
        pos1.append(w1)
        pos2.append(w2)

    return pd.DataFrame({ticker1: pos1, ticker2: pos2}, index=dates)

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
                pos1.append(1)
                pos2.append(0)
            elif ineff < -threshold_ineff and m_val < 0:
                pos1.append(0)
                pos2.append(1)
            else:
                pos1.append(0.5)
                pos2.append(0.5)
        else:
                pos1.append(0.5)
                pos2.append(0.5)
    positions[ticker1] = pos1
    positions[ticker2] = pos2
    return positions



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


def compute_positions(rolling_signal, momentum, ticker1, ticker2, default=0.5, threshold=0.5):
    # On suppose que les deux indices ont la même fréquence
    positions = pd.DataFrame(index=rolling_signal.index, columns=[ticker1, ticker2])
    positions[ticker1] = default
    positions[ticker2] = default
    condition = rolling_signal > threshold
    positions.loc[condition & (momentum > 0), ticker1] = 1
    positions.loc[condition & (momentum > 0), ticker2] = 0
    positions.loc[condition & (momentum < 0), ticker1] = 0
    positions.loc[condition & (momentum < 0), ticker2] = 1
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
    start_time = time.time()

    momentum = compute_momentum(r, shift_days=20, window_size=220).dropna()
    mfdfa_window = 1008
    q_list = np.linspace(-3, 3, 13)
    scales = np.unique(np.logspace(np.log10(10), np.log10(200), 10, dtype=int))
    rolling_delta_ticker1 = ComputeMFDFA.mfdfa_rolling_opti(np.log(all_prices[ticker1]).diff().dropna().shift(1),
                                                       mfdfa_window, q_list, scales, order=1).dropna()


    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=rolling_delta_ticker1.index, y=rolling_delta_ticker1,
    #                             mode='lines', name='Rolling Delta Alpha S&P500',
    #                             line=dict(color='blue')))
    #
    # fig.update_layout(title="Rolling Delta Alpha S&P500",
    #                     xaxis_title="Date",
    #                     yaxis_title="Rolling Delta Alpha S&P500",
    #                     template="plotly_white")
    # fig.show()


    rolling_delta_ticker2 = ComputeMFDFA.mfdfa_rolling_opti(np.log(all_prices[ticker2]).diff().dropna().shift(1),
                                                       mfdfa_window, q_list, scales, order=1).dropna()

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=rolling_delta_ticker2.index, y=rolling_delta_ticker2,
    #                             mode='lines', name='Rolling Delta Alpha Russell 2000',
    #                             line=dict(color='blue')))
    # fig.update_layout(title="Rolling Delta Alpha Russell 2000",
    #                     xaxis_title="Date",
    #                     yaxis_title="Rolling Delta Alpha Russell 2000",
    #                     template="plotly_white")
    # fig.show()

    rolling_delta_ticker1.index.name = "Date"
    rolling_delta_ticker2.index.name = "Date"

    common_dates_mfdfa = rolling_delta_ticker1.index.intersection(rolling_delta_ticker2.index)
    delta_alpha_diff = (
                rolling_delta_ticker1.loc[common_dates_mfdfa] - rolling_delta_ticker2.loc[common_dates_mfdfa])

    # do a plot of the difference between the two rolling delta alpha
    # delta_alpha_diff_m = delta_alpha_diff.resample('M').last()
    # delta_alpha_diff = pd.DataFrame(10, index=all_prices.index, columns=[ticker1])
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=delta_alpha_diff_m.index, y=delta_alpha_diff_m,
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
    # start_dates = ["1995-01-02", "2000-01-02", "2005-01-02", "2010-01-02", "2015-01-02"]

    # for start_date in start_dates:
    #     cumulative_returns_dict = {}
    #     performance_results = []
    for config_name, config in rolling_configs.items():
        cumulative_returns_dict = {}
        performance_results = []
        w_s = config["window_size"]
        p_config = all_p.copy().shift(1)

        rolling_signal = compute_rolling_metric(r.shift(1),
                                                config["window_size"],
                                                method=config["method"],
                                                rolling_type=config["rolling_type"],
                                                chin=False).dropna()
        # rolling_signal = rolling_signal.loc['2015-01-02':]

        common_dates = rolling_signal.index.intersection(momentum.index).intersection(delta_alpha_diff.index)
        signal = rolling_signal.loc[common_dates]
        mom = momentum.loc[common_dates]
        delta_alpha_diff_aligned = delta_alpha_diff.loc[common_dates]
        first_valid_index = signal.first_valid_index()
        ineff_index = pd.Series(
            compute_inefficiency_index(delta_alpha_diff_aligned, signal),
            index=common_dates
        ).dropna()


        all_signal = pd.concat([signal.rename('rolling_hurst'), mom.rename('momentum'), ineff_index.rename('ineff_index')],
                               axis=1)

        cond_long_t1 = (
                (all_signal['momentum'] > 0) &
                (all_signal['ineff_index'] > 0) &
                (all_signal['rolling_hurst'] > 0.5)
        )
        cond_long_t2 = (
                (all_signal['momentum'] < 0) &
                (all_signal['ineff_index'] < 0) &
                (all_signal['rolling_hurst'] > 0.5)
        )
        all_signal[ticker1] = 0.5
        all_signal[ticker2] = 0.5

        all_signal.loc[cond_long_t1, ticker1] = 1.0
        all_signal.loc[cond_long_t1, ticker2] = 0.0
        all_signal.loc[cond_long_t2, ticker1] = 0.0
        all_signal.loc[cond_long_t2, ticker2] = 1.0



        positions = all_signal[[ticker1, ticker2]].copy()
        # positions = compute_nav_with_inefficiency(signal, mom, ineff_index, ticker1, ticker2)

        pos_diff = positions.diff()
        rebalance_mask = (pos_diff != 0).any(axis=1)
        rebalancing_dates = positions.index[rebalance_mask]
        rebalancing_pos = positions.loc[rebalancing_dates]
        portfolio = Portfolio(rebalancing_pos, all_prices.loc['1991-11-15':], keep_currency_effect=True, include_dividends=False,
                              transaction_fees=0.005, management_fees=0)
        nav = portfolio.nav
        cum_returns = nav / nav.iloc[0]
        print(cum_returns)

        ann_ret, ann_vol, sharpe, max_dd = compute_performance_stats(nav.pct_change().dropna())
        performance_results.append({
            "Strategy": config_name,
            "Annualized Return": round(ann_ret * 100, 3),
            "Annualized Volatility": round(ann_vol * 100, 3),
            "Sharpe": round(sharpe, 3),
            "Max Drawdown": round(max_dd * 100, 3)
        })


        inef_sp500 = pd.Series(
            compute_inefficiency_index(rolling_delta_ticker1, signal),
            index=common_dates
        ).dropna()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=inef_sp500.index, y=inef_sp500,
                                 mode='lines', name='Inefficiency Index sp500',
                                 line=dict(color='purple')))
        fig.update_layout(title="Inefficiency Index",
                          xaxis_title="Date",
                          yaxis_title="Inefficiency Index sp500",
                          template="plotly_white")
        fig.show()

        inef_russel = pd.Series(
            compute_inefficiency_index(rolling_delta_ticker2, signal),
            index=common_dates
        ).dropna()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=inef_russel.index, y=inef_russel,
                                 mode='lines', name='Inefficiency Index Russell 2000',
                                 line=dict(color='purple')))
        fig.update_layout(title="Inefficiency Index",
                          xaxis_title="Date",
                          yaxis_title="Inefficiency Index Russell 2000",
                          template="plotly_white")
        fig.show()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ineff_index.index, y=ineff_index,
                                 mode='lines', name='Inefficiency Index',
                                 line=dict(color='purple')))
        fig.update_layout(title="Inefficiency Index",
                          xaxis_title="Date",
                          yaxis_title="Inefficiency Index",
                          template="plotly_white")
        # fig.show()
        ineff_index.index.name = "Date"
        # ineff_index.to_csv(f"{DATA_PATH}/inefficiency_index.csv")

        plot_positions_and_hurst(rolling_hurst=rolling_signal, positions=positions, ticker1=ticker1, ticker2=ticker2)
        # count_position_switches(positions, ticker1, ticker2)

        fig_backtest = go.Figure()

        fig_backtest.add_trace(
            go.Scatter(
                x=cum_returns.index,
                y=cum_returns,
                mode='lines',
                name=config_name,
                line=dict(color="blue")  # you can choose any color
            )

        ),
        fig_backtest.update_layout(
            title=f"Cumulative Returns - Strategy: {config_name}",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template="plotly_white"
        )

        # Modif Overlap 120 No Filter
        cond_long_t1 = (
                (all_signal['momentum'] > 0) &
                (all_signal['rolling_hurst'] > 0.5)
        )
        cond_long_t2 = (
                (all_signal['momentum'] < 0) &
                (all_signal['rolling_hurst'] > 0.5)
        )
        all_signal[ticker1] = 0.5
        all_signal[ticker2] = 0.5

        all_signal.loc[cond_long_t1, ticker1] = 1.0
        all_signal.loc[cond_long_t1, ticker2] = 0.0
        all_signal.loc[cond_long_t2, ticker1] = 0.0
        all_signal.loc[cond_long_t2, ticker2] = 1.0

        positions = compute_nav_with_inefficiency(signal, mom, ineff_index, ticker1, ticker2)
        pos_diff = positions.diff()
        rebalance_mask = (pos_diff != 0).any(axis=1)
        rebalancing_dates = positions.index[rebalance_mask]
        rebalancing_pos = positions.loc[rebalancing_dates]
        portfolio = Portfolio(rebalancing_pos, all_prices.loc['1991-11-15':], keep_currency_effect=True, include_dividends=False,
                              transaction_fees=0.0005, management_fees=0)
        nav = portfolio.nav
        cum_returns_no_filter = nav / nav.iloc[0]
        print(cum_returns_no_filter)
        ann_ret, ann_vol, sharpe, max_dd = compute_performance_stats(nav.pct_change().dropna())
        performance_results.append({
            "Strategy": "ModifOverlap120NoFilter",
            "Annualized Return": round(ann_ret * 100, 3),
            "Annualized Volatility": round(ann_vol * 100, 3),
            "Sharpe": round(sharpe, 3),
            "Max Drawdown": round(max_dd * 100, 3)
        })

        all_signal[ticker1] = 0.5
        all_signal[ticker2] = 0.5
        positions_50 = all_signal[[ticker1, ticker2]].copy()
        pos_diff_50 = positions_50.diff()
        rebalance_mask_50 = (pos_diff_50 != 0).any(axis=1)
        rebalancing_dates_50 = positions_50.index[rebalance_mask_50]
        rebalancing_pos_50 = positions_50.loc[rebalancing_dates_50]
        portfolio_50_50 = Portfolio(rebalancing_pos_50, all_prices.loc['1991-11-15':], keep_currency_effect=True, include_dividends=False,
                              transaction_fees=0.0, management_fees=0)
        nav_50 = portfolio_50_50.nav
        cum_returns_50 = nav_50 / nav_50.iloc[0]
        print(cum_returns_50)

        ann_ret_50, ann_vol_50, sharpe_50, max_dd_50 = compute_performance_stats(nav_50.pct_change().dropna())
        performance_results.append({
            "Strategy": '50/50 Portfolio',
            "Annualized Return": round(ann_ret_50 * 100, 3),
            "Annualized Volatility": round(ann_vol_50 * 100, 3),
            "Sharpe": round(sharpe_50, 3),
            "Max Drawdown": round(max_dd_50 * 100, 3)
        })


        new_p = all_prices.loc[first_valid_index:]
        new_p = new_p.loc[new_p.index.intersection(cum_returns.index)]
        sp500_cumulative = (new_p.loc[:,ticker1] / new_p.loc[:, ticker1].iloc[0]).dropna()
        russell_cumulative = (new_p.loc[:,ticker2] / new_p.loc[:, ticker2].iloc[0]).dropna()

        # =====================================================================
        # Visualisation des courbes cumulées des stratégies
        # =====================================================================
        first_strategy, cum_returns = list(cumulative_returns_dict.items())[0]
        fig_backtest.add_trace(
            go.Scatter(
                x=cum_returns.index,
                y=cum_returns,
                mode='lines',
                name=first_strategy,
                line=dict(color="blue")
            )
        )
        fig_backtest.update_layout(
            title=f"Cumulative Returns - Strategy: {first_strategy}",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template="plotly_white"
        )
        # fig_backtest.show()

        fig_backtest.add_trace(
            go.Scatter(
                x=sp500_cumulative.index,
                y=sp500_cumulative,
                mode='lines',
                name="Long Only S&P500",
                line=dict(color='red')
            )
        )

        fig_backtest.add_trace(
            go.Scatter(
                x=russell_cumulative.index,
                y=russell_cumulative,
                mode='lines',
                name="Long Only Russell",
                line=dict(color='green')
            )
        )

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
        ann_ret_sp500, ann_vol_sp500, sharpe_sp500, max_dd_sp500 = compute_performance_stats(sp500_cumulative.pct_change().dropna())
        ann_ret_russell, ann_vol_russell, sharpe_russell, max_dd_russell = compute_performance_stats(russell_cumulative.pct_change().dropna())

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

        df_results = pd.DataFrame(performance_results)
        print("=== Performance Summary ===")
        print(df_results)
        # df_results.to_csv(f"{DATA_PATH}/backtest_results_start_{start_date}.csv", index=False)
        # df_results.to_csv(f"{DATA_PATH}/backtest_results_window_{w_s}.csv", index=False)
        # df_results.to_csv(f"{DATA_PATH}/backtest_long_neutral_results.csv", index=False)
        # fig_backtest.write_image(f"{IMG_PATH}/backtest_long_neutral.png", width=1200, height=800)

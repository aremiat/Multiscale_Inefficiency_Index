import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go

from utils.RS import ComputeRS
from utils.MFDFA import ComputeMFDFA


def compute_inefficiency_index(delta_alpha_diff, rolling_hurst):
    """
    Compute the inefficiency index as the product of the difference in spectrum width
    and the deviation of the rolling Hurst exponent from 0.5.
    Args:
        delta_alpha_diff (pd.Series): Series of differences in spectrum width.
        rolling_hurst (pd.Series): Series of rolling Hurst exponents.
    Returns:
        pd.Series: Series representing the inefficiency index.
    """
    return delta_alpha_diff * (rolling_hurst - 0.5).abs()

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
            raise ValueError("Unknown rolling method")
    else:
        raise ValueError("Unknown rolling type")
    return roll


def compute_performance_stats(daily_returns: pd.Series, freq=252):
    """Compute performance statistics from daily returns.

    Args:
        daily_returns (pd.Series): Daily returns of the asset.
        freq (int): Frequency of returns, default is 252 for daily returns.
    Returns:
        tuple: Annualized return, annualized volatility, Sharpe ratio, and maximum drawdown.
    """
    daily_returns = daily_returns.dropna()
    if len(daily_returns) == 0:
        return np.nan, np.nan, np.nan, np.nan
    total_ret = (1 + daily_returns).prod() - 1
    nb_obs = len(daily_returns)
    annual_return = (1 + total_ret) ** (freq / nb_obs) - 1
    daily_vol = daily_returns.std()
    annual_vol = daily_vol * np.sqrt(freq)
    sharpe_ratio = (annual_return / annual_vol)
    cum_curve = (1 + daily_returns).cumprod()
    running_max = cum_curve.cummax()
    drawdown = (cum_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    return annual_return, annual_vol, sharpe_ratio, max_drawdown


def run_backtest(positions: pd.DataFrame, prices: pd.Series) -> pd.Series:
    """Run a backtest based on positions and prices.

    Args:
        positions (pd.DataFrame): DataFrame of positions with dates as index.
        prices (pd.Series): Series of asset prices with dates as index.
    Returns:
        pd.Series: Series of net asset value (NAV) over time.
    """
    positions = positions.squeeze()
    aligned_prices = prices.reindex(positions.index).fillna(method='ffill')
    returns = aligned_prices.pct_change().fillna(0)
    net_returns = positions.shift(1).fillna(0) * returns
    nav = (1 + net_returns).cumprod()
    return nav


mfdfa_window = 252
q_list = np.linspace(-4, 4, 17)
scales = np.unique(np.logspace(np.log10(10), np.log10(50), 10, dtype=int))

if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "../data")
    IMG_PATH = os.path.join(os.path.dirname(__file__), "../img")

    np.random.seed(43)

    p = pd.read_csv(f"{DATA_PATH}/ssec.csv", index_col=0, parse_dates=True).dropna()
    r = p.pct_change().dropna()

    surrogate_returns = ComputeMFDFA.surrogate_gaussian_corr(r['ssec'])
    surrogate_returns = pd.Series(surrogate_returns, index=r.index)

    rolling_delta_ssec = ComputeMFDFA.mfdfa_rolling(surrogate_returns.shift(1),
                                                       mfdfa_window, q_list, scales, order=1).dropna()
    dates = pd.date_range(start='2001-09-03', periods=len(rolling_delta_ssec), freq='B')
    rolling_delta_ssec.index = dates
    rolling_delta_ssec.index.name = "Date"


    rolling_configs = {
        "ModifOverlap120": {"method": "modified", "rolling_type": "overlapping", "window_size": 120},
    }

    for config_name, config in rolling_configs.items():
        cumulative_returns_dict = {}
        performance_results = []
        w_s = config["window_size"]
        p_config = p.copy().shift(1)

        rolling_signal = compute_rolling_metric(r.shift(1),
                                                config["window_size"],
                                                method=config["method"],
                                                rolling_type=config["rolling_type"],
                                                chin=False).dropna()


        common_dates = rolling_signal.index.intersection(rolling_delta_ssec.index)
        signal = rolling_signal.loc[common_dates]
        rolling_delta_ssec_aligned = rolling_delta_ssec.loc[common_dates]
        first_valid_index = signal.first_valid_index()
        ineff_index = pd.Series(
            compute_inefficiency_index(rolling_delta_ssec_aligned, signal['ssec']),
            index=common_dates
        ).dropna()
        all_prices_aligned = p.loc[common_dates]

        rolling_std_ineff = (ineff_index.rolling(window=120, min_periods=116).std() * 1.5).dropna()
        ineff_index = ineff_index.reindex(rolling_std_ineff.index)
        common_dates = rolling_signal.index.intersection(rolling_std_ineff.index)
        signal = rolling_signal.loc[common_dates]
        backtest_prices = p.loc[common_dates]

        positions = pd.Series(index=rolling_std_ineff.index, dtype=int)
        positions_inef = pd.Series(index=rolling_std_ineff.index, dtype=int)

        current_pos = 1
        current_pos_without_ineff = 1
        for idx in positions.index:
            hurst = signal['ssec'].loc[idx]
            ineff = ineff_index.loc[idx]
            rolling_std_ineff_value = rolling_std_ineff.loc[idx]
            if current_pos == 1:
                if (hurst < 0.5) and (ineff > rolling_std_ineff_value):
                    current_pos = -1
            else:
                if (hurst > 0.5) or (ineff < rolling_std_ineff_value):
                    current_pos = 1
            positions.loc[idx] = current_pos

        current_pos_without_ineff = 1
        for idx in positions_inef.index:
            hurst = signal['ssec'].loc[idx]
            if current_pos_without_ineff == 1:
                if (hurst < 0.5):
                    current_pos_without_ineff = -1
            else:
                if (hurst > 0.5):
                    current_pos_without_ineff = 1
            positions_inef.loc[idx] = current_pos_without_ineff


        print(np.sum(positions.diff().fillna(0) != 0), "number of positions for inefficiency strategy")
        print(np.sum(positions_inef.diff().fillna(0) != 0), "number of positions for non-ineff strategy")

        long_only_return = (1 + backtest_prices['ssec'].pct_change().fillna(0)).cumprod()

        nav = run_backtest(pd.DataFrame(positions), backtest_prices['ssec'])
        nav_without_ineff = run_backtest(pd.DataFrame(positions_inef), backtest_prices['ssec'])


        ################################################################################################################
        ############################################### Plotting the results ###########################################
        ################################################################################################################

        fig_backtest = go.Figure()

        fig_backtest.add_trace(
            go.Scatter(
                x=nav.index,
                y=nav,
                mode='lines',
                name=config_name,
                line=dict(color="blue")
            )

        ),
        fig_backtest.update_layout(
            title=f"Cumulative Returns - Strategy: {config_name}",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template="plotly_white"
        )

        fig_backtest.add_trace(
            go.Scatter(
                x=long_only_return.index,
                y=long_only_return,
                mode='lines',
                name="Long Only SSEC",
                line=dict(color='red')
            )
        )

        fig_backtest.update_layout(
            title="Cumulative Return",
            template="plotly_white",
        )

        fig_backtest.update_xaxes(title_text="Date")
        fig_backtest.update_yaxes(title_text="Log Cumulative Return")
        fig_backtest.show()

        ################################################################################################################
        ############################################### Performances ###################################################
        ################################################################################################################

        ann_ret, ann_vol, sharpe, max_dd = compute_performance_stats(nav.pct_change().dropna())
        ann_ret_ssec, ann_vol_ssec, sharpe_ssec, max_dd_ssec = compute_performance_stats(long_only_return.pct_change().dropna())
        ann_ret_without_ineff, ann_vol_without_ineff, sharpe_without_ineff, max_dd_without_ineff = compute_performance_stats(nav_without_ineff.pct_change().dropna())

        performance_results.append({
            "Strategy": 'Long/Short SSEC with inefficiency',
            "Annualized Return": round(ann_ret * 100, 3),
            "Annualized Volatility": round(ann_vol * 100, 3),
            "Sharpe": round(sharpe, 3),
            "Max Drawdown": round(max_dd * 100, 3)
        })

        performance_results.append({
            "Strategy": "Long Only SSEC",
            "Annualized Return": round(ann_ret_ssec * 100, 3),
            "Annualized Volatility": round(ann_vol_ssec * 100, 3),
            "Sharpe": round(sharpe_ssec, 3),
            "Max Drawdown": round(max_dd_ssec * 100, 3)
        })

        performance_results.append({
            "Strategy": "Long/short SSEC without inefficiency",
            "Annualized Return": round(ann_ret_without_ineff * 100, 3),
            "Annualized Volatility": round(ann_vol_without_ineff * 100, 3),
            "Sharpe": round(sharpe_without_ineff, 3),
            "Max Drawdown": round(max_dd_without_ineff * 100, 3)
        })

        df_results = pd.DataFrame(performance_results)
        print("=== Performance Summary ===")
        print(df_results)
        df_results.to_csv(os.path.join(DATA_PATH, f"perfs_with_and_without_inef.csv"), index=False)

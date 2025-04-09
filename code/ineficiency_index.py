import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


# =====================================================================
# 1. Fonctions MF-DFA pour le calcul du spectre multifractal
# (vous devez définir ou importer mfdfa() et compute_alpha_falpha())
# =====================================================================
def mfdfa(signal, scales, q_list, order=1):
    """
    Calcule le MF-DFA pour une série temporelle.

    Paramètres:
        signal : tableau numpy, la série (par exemple, rendements) à analyser.
        scales : liste des échelles (tailles de segments) à utiliser.
        q_list : liste des ordres q pour lesquels calculer la fonction de fluctuation.
        order : ordre du polynôme pour le detrending (1 = linéaire par défaut).

    Retourne:
        Fq : matrice de taille (len(q_list), len(scales)) contenant F_q(s) pour chaque q et chaque échelle s.
    """
    N = len(signal)
    signal = signal - np.mean(signal)
    Y = np.cumsum(signal)

    Fq = np.zeros((len(q_list), len(scales)))

    # Pour chaque échelle s
    for j, s in enumerate(scales):
        s = int(s)
        if s < 2:
            continue
        n_segments = N // s
        F_seg = []
        # Division classique : découpage non chevauchant depuis le début
        for v in range(n_segments):
            segment = Y[v * s:(v + 1) * s]
            idx = np.arange(s)
            # Ajustement polynomial de degré 'order'
            coeffs = np.polyfit(idx, segment, order)
            fit = np.polyval(coeffs, idx)
            F_seg.append(np.mean((segment - fit) ** 2))

        # Deuxième division : découpage depuis la fin pour couvrir la totalité de la série
        for v in range(n_segments):
            segment = Y[N - (v + 1) * s:N - v * s]
            idx = np.arange(s)
            coeffs = np.polyfit(idx, segment, order)
            fit = np.polyval(coeffs, idx)
            F_seg.append(np.mean((segment - fit) ** 2))

        F_seg = np.array(F_seg)
        # Éviter les problèmes avec des valeurs nulles
        F_seg[F_seg < 1e-10] = 1e-10

        # Calcul de F_q(s) pour chaque valeur de q
        for k, q in enumerate(q_list):
            if np.abs(q) < 1e-6:
                # q = 0 : utilisation de la moyenne géométrique (limite q->0)
                Fq[k, j] = np.exp(0.5 * np.mean(np.log(F_seg)))
            else:
                Fq[k, j] = (np.mean(F_seg ** (q / 2))) ** (1 / q)

    return Fq


def compute_alpha_falpha(q_list, h_q):
    """
    Calcule alpha et f(alpha) à partir de h(q) via une transformée de Legendre.
    alpha(q) = d[tau(q)]/dq avec tau(q) = q*h(q)-1.
    f(alpha) = q*alpha - tau(q)
    """
    q_list = np.array(q_list)
    tau_q = q_list * h_q - 1
    # Calcul numérique de la dérivée d_tau/dq
    alpha = np.gradient(tau_q, q_list)
    f_alpha = q_list * alpha - tau_q
    return alpha, f_alpha


def mfdfa_rolling(series, window_size, q_list, scales, order=1):
    """
    Applique MF-DFA sur des fenêtres glissantes de taille 'window_size'
    et renvoie la largeur du spectre multifractal (Delta alpha) pour chaque fenêtre.

    Paramètres:
        series      : pd.Series ou array-like, la série de données.
        window_size : int, nombre de points par fenêtre glissante.
        q_list      : liste/array des ordres q pour MF-DFA.
        scales      : liste/array d'échelles (tailles de segments).
        order       : ordre du polynôme pour le detrending local (1 = linéaire par défaut).

    Retourne:
        pd.Series contenant la largeur du spectre Δα
        pour chaque fenêtre, indexé par la date (ou l'index) correspondant
        à la fin de la fenêtre.
    """
    if isinstance(series, pd.Series):
        data = series.values
        index_data = series.index
    else:
        data = np.array(series)
        index_data = np.arange(len(data))

    alpha_widths = []
    rolling_index = []
    nb_points = len(data)

    for start in range(nb_points - window_size + 1):
        end = start + window_size
        window_data = data[start:end]

        # 1) Calcul de Fq(s) sur la fenêtre
        Fq = mfdfa(window_data, scales, q_list, order=order)

        # 2) Calcul de h(q) par régression linéaire log(Fq) vs log(s)
        h_q = []
        log_scales = np.log(scales)
        for j, q in enumerate(q_list):
            log_Fq = np.log(Fq[j, :])
            coeffs = np.polyfit(log_scales, log_Fq, 1)  # la pente correspond à h(q)
            h_q.append(coeffs[0])
        h_q = np.array(h_q)

        # 3) Transformation de Legendre pour obtenir alpha et f(alpha)
        alpha, f_alpha = compute_alpha_falpha(q_list, h_q)

        # 4) Largeur du spectre multifractal
        alpha_width = alpha.max() - alpha.min()
        alpha_widths.append(alpha_width)

        # Associer la valeur à la fin de la fenêtre
        rolling_index.append(index_data[end - 1])

    return pd.Series(alpha_widths, index=rolling_index, name="alpha_width")


# =====================================================================
# 2. Classe et fonctions RS (déjà fournies)
# =====================================================================
class ComputeRS:
    def __init__(self):
        pass

    @staticmethod
    def rs_statistic(series, window_size=0):
        if window_size < len(series):
            window_size = len(series)
        s = series.iloc[len(series) - window_size: len(series)]
        mean = np.mean(s)
        y = s - mean
        r = np.max(np.cumsum(y)) - np.min(np.cumsum(y))
        sigma = np.std(s)
        return r / sigma

    @staticmethod
    def compute_S_modified(series, chin=False):
        s = series
        t = len(s)
        mean_y = np.mean(s)
        s = s.squeeze()
        if not chin:
            rho_1 = np.corrcoef(s[:-1], s[1:])[0, 1]
            if rho_1 < 0:
                return np.sum((s - mean_y) ** 2) / t
            q = ((3 * t) / 2) ** (1 / 3) * ((2 * rho_1) / (1 - (rho_1 ** 2))) ** (2 / 3)
        else:
            q = 4 * (t / 100) ** (2 / 9)
        q = int(np.floor(q))
        var_term = np.sum((s - mean_y) ** 2) / t
        auto_cov_term = 0
        for j in range(1, q + 1):
            w_j = 1 - (j / (q + 1))
            sum_cov = np.sum((s[:-j] - mean_y) * (s[j:] - mean_y))
            auto_cov_term += w_j * sum_cov
        auto_cov_term = (2 / t) * auto_cov_term
        s_quared = var_term + auto_cov_term
        return s_quared

    @staticmethod
    def rs_modified_statistic(series, window_size=0, chin=False):
        if window_size > len(series):
            window_size = len(series)
        s = series.iloc[len(series) - window_size: len(series)]
        y = s - np.mean(s)
        r = np.max(np.cumsum(y)) - np.min(np.cumsum(y))
        sigma = np.sqrt(ComputeRS.compute_S_modified(s, chin))
        return r / sigma


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
# 3. Fonction pour calculer le momentum
# =====================================================================
def compute_momentum(diff_series, shift_days=20, window_size=220):
    diff_shifted = diff_series.shift(shift_days)
    momentum = diff_shifted.rolling(window_size).mean()
    return momentum


# =====================================================================
# 4. Calcul de l'indice d'inefficience et stratégie de positionnement
# =====================================================================
def compute_inefficiency_index(delta_alpha_diff, rolling_hurst, momentum):
    """
    Combine la différence de largeur de spectre (delta_alpha_diff),
    l'écart absolu (rolling Hurst - 0.5) et le signal de momentum.
    """
    return delta_alpha_diff * abs(rolling_hurst - 0.5)


def compute_positions_with_inefficiency(rolling_hurst, momentum, ineff_index, ticker1, ticker2,
                                        threshold_h=0.5, threshold_ineff=0.01):
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
            # if ineff < threshold_ineff and m_val > 0:
            #     pos1.append(0.2)
            #     pos2.append(0.8)
            # elif ineff > -threshold_ineff and m_val < 0:
            #     pos1.append(0.8)
            #     pos2.append(0.2)
            # else:
                pos1.append(0.5)
                pos2.append(0.5)
    positions[ticker1] = pos1
    positions[ticker2] = pos2
    return positions


# =====================================================================
# 5. Backtest et calcul des performances
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
    sharpe_ratio = (daily_mean / daily_vol) * np.sqrt(freq) if daily_vol != 0 else np.nan
    cum_curve = (1 + daily_returns).cumprod()
    running_max = cum_curve.cummax()
    drawdown = (cum_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    return annual_return, annual_vol, sharpe_ratio, max_drawdown

# Plot the rolling Hurst signal and position switches
# Plot the rolling Hurst signal and position switches
def plot_positions_and_hurst(rolling_hurst, positions, ticker1, ticker2):
    fig = go.Figure()

    # Add rolling Hurst signal
    fig.add_trace(go.Scatter(
        x=rolling_hurst.index,
        y=rolling_hurst,
        mode='lines',
        name='Rolling Hurst Signal',
        line=dict(color='blue')
    ))

    # Add position switches for ticker1
    fig.add_trace(go.Scatter(
        x=positions.index,
        y=positions[ticker1],
        mode='lines',
        name=f'{ticker1} Position',
        line=dict(color='green')
    ))

    # Add position switches for ticker2
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
    all_prices = pd.concat([p[ticker1], p[ticker2]], axis=1)
    all_prices = all_prices.loc["1987-10-09": "2025-02-28"]
    all_p = all_prices.pct_change().dropna()
    all_p['Diff'] = all_p[ticker1] - all_p[ticker2]
    r = all_p['Diff']

    # Calcul du momentum sur la série Diff
    momentum = compute_momentum(r, shift_days=20, window_size=220)

    # Paramètres pour MF-DFA rolling
    mfdfa_window = 252
    q_list = np.linspace(-3, 3, 13)
    scales = np.unique(np.logspace(np.log10(10), np.log10(50), 10, dtype=int))

    # Calcul rolling de la largeur du spectre Δα pour chaque indice
    rolling_delta_ticker1 = mfdfa_rolling(np.log(all_prices[ticker1]).diff().dropna(), mfdfa_window, q_list, scales,
                                          order=1)
    rolling_delta_ticker2 = mfdfa_rolling(np.log(all_prices[ticker2]).diff().dropna(), mfdfa_window, q_list, scales,
                                          order=1)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=(f"Rolling Spectrum Width Δα - {ticker1}",
                                        f"Rolling Spectrum Width Δα - {ticker2}"))

    # Plot for Ticker 1
    fig.add_trace(go.Scatter(x=rolling_delta_ticker1.index, y=rolling_delta_ticker1,
                             mode='lines', name=f'Rolling Δα {ticker1}',
                             line=dict(color='blue')), row=1, col=1)

    # Plot for Ticker 2
    fig.add_trace(go.Scatter(x=rolling_delta_ticker2.index, y=rolling_delta_ticker2,
                             mode='lines', name=f'Rolling Δα {ticker2}',
                             line=dict(color='orange')), row=2, col=1)

    fig.update_layout(height=800, width=1000, title_text="Comparison of Rolling Δα for Two Tickers",
                      showlegend=True)

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Δα Ticker1", row=1, col=1)
    fig.update_yaxes(title_text="Δα Ticker2", row=2, col=1)

    fig.show()
    common_dates_mfdfa = rolling_delta_ticker1.index.intersection(rolling_delta_ticker2.index)
    delta_alpha_diff = (
                rolling_delta_ticker1.loc[common_dates_mfdfa] - rolling_delta_ticker2.loc[common_dates_mfdfa])

    # Configurations pour le rolling sur RS (exemple)
    rolling_configs = {
        "ModifOverlap120": {"method": "modified", "rolling_type": "overlapping", "window_size": 120},
        # "TradOverlap120": {"method": "traditional", "rolling_type": "overlapping", "window_size": 120},
        "ModifOverlap252": {"method": "modified", "rolling_type": "overlapping", "window_size": 252},
    }

    cumulative_returns_dict = {}
    performance_results = []

    for config_name, config in rolling_configs.items():
        p_config = all_p.copy()
        rolling_signal = compute_rolling_metric(r.shift(1), config["window_size"],
                                                method=config["method"],
                                                rolling_type=config["rolling_type"],
                                                chin=False).dropna()
        common_dates = rolling_signal.index.intersection(momentum.index).intersection(delta_alpha_diff.index)
        signal = rolling_signal.loc[common_dates]
        mom = momentum.loc[common_dates]
        delta_alpha_diff_aligned = delta_alpha_diff.loc[common_dates]

        # Calcul de l'indice d'inefficience
        ineff_index = pd.Series(
            compute_inefficiency_index(delta_alpha_diff_aligned, signal, mom),
            index=common_dates
        )

        # Calcul des positions
        positions = compute_positions_with_inefficiency(signal, mom, ineff_index, ticker1, ticker2,
                                                        threshold_h=0.5, threshold_ineff=0.01)

        # Alignement avec les rendements journaliers pour backtest
        common_dates_bp = positions.index.intersection(p_config.index)
        positions = positions.loc[common_dates_bp]
        all_p_config = p_config.loc[common_dates_bp]

        cum_returns, port_returns = run_backtest(all_p_config, positions, ticker1, ticker2, fee_rate=0.005)
        cumulative_returns_dict[config_name] = cum_returns

        # Calcul des statistiques de performance
        ann_ret, ann_vol, sharpe, max_dd = compute_performance_stats(port_returns)
        performance_results.append({
            "Strategy": config_name,
            "Annualized Return": round(ann_ret * 100, 2),
            "Annualized Volatility": round(ann_vol * 100, 2),
            "Sharpe": round(sharpe, 2),
            "Max Drawdown": round(max_dd * 100, 2)
        })
        plot_positions_and_hurst(rolling_hurst=rolling_signal, positions=positions, ticker1=ticker1, ticker2=ticker2)
        count_position_switches(positions, ticker1, ticker2)

    p = all_p.copy()
    rolling_signal = compute_rolling_metric(r.shift(1), 120,
                                            method="modified",
                                            rolling_type="overlapping",
                                            chin=False).dropna()
    # On aligne avec la série du momentum (on garde les dates communes)
    common_dates = rolling_signal.index.intersection(momentum.index)
    signal = rolling_signal.loc[common_dates]
    mom = momentum.loc[common_dates]

    # Calcul des positions
    positions = compute_positions(signal, mom, ticker1, ticker2, default=0.5, threshold=0.5)

    # Pour le backtest, on utilise les rendements journaliers (all_p)
    common_dates_bp = positions.index.intersection(p.index)
    positions = positions.loc[common_dates_bp]
    all_p_config = p.loc[common_dates_bp]

    cum_returns, port_returns = run_backtest(all_p_config, positions, ticker1, ticker2, fee_rate=0.005)
    cumulative_returns_dict["ModifOverlap120NoFilter"] = cum_returns

    # Calcul des performances
    ann_ret, ann_vol, sharpe, max_dd = compute_performance_stats(port_returns)
    performance_results.append({
        "Strategy": "ModifOverlap120NoFilter",
        "Annualized Return": round(ann_ret * 100, 2),
        "Annualized Volatility": round(ann_vol * 100, 2),
        "Sharpe": round(sharpe, 2),
        "Max Drawdown": round(max_dd * 100, 2)
    })



    # plot the inefficiency index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ineff_index.index, y=ineff_index,
                             mode='lines', name='Inefficiency Index',
                             line=dict(color='purple')))
    fig.update_layout(title="Inefficiency Index",
                      xaxis_title="Date",
                      yaxis_title="Inefficiency Index",
                      template="plotly_white")
    fig.show()

    # =====================================================================
    # Visualisation des courbes cumulées des stratégies
    # =====================================================================
    # Get the first strategy from the cumulative_returns_dict
    first_strategy, cum_returns = list(cumulative_returns_dict.items())[0]

    # Create a new figure and plot only that strategy
    fig_backtest = go.Figure()
    fig_backtest.add_trace(
        go.Scatter(
            x=cum_returns.index,
            y=np.log(cum_returns),
            mode='lines',
            name=first_strategy,
            line=dict(color="blue")  # you can choose any color
        )
    )
    fig_backtest.update_layout(
        title=f"Cumulative Returns - Strategy: {first_strategy}",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template="plotly_white"
    )
    fig_backtest.show()

    # Ports de comparaison : SP500, Russell et portefeuille 50/50
    new_p = all_p.loc["1988-10-18":"2025-02-28"]
    sp500_returns = new_p[ticker1]
    sp500_cumulative = (1 + sp500_returns).cumprod()
    russell_returns = new_p[ticker2]
    russell_cumulative = (1 + russell_returns).cumprod()
    portfolio_50_50_returns = 0.5 * new_p[ticker1] + 0.5 * new_p[ticker2]
    portfolio_50_50_cumulative = (1 + portfolio_50_50_returns).cumprod()

    fig_backtest.add_trace(
        go.Scatter(
            x=sp500_cumulative.index,
            y=np.log(sp500_cumulative),
            mode='lines',
            name="Long Only SP500",
            line=dict(color='red')
        )
    )
    # fig_backtest.add_trace(
    #     go.Scatter(
    #         x=russell_cumulative.index,
    #         y=russell_cumulative,
    #         mode='lines',
    #         name="Long Only Russell",
    #         line=dict(color='orange')
    #     )
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
    fig_backtest.show()

    # =====================================================================
    # Calcul des performances des portefeuilles de comparaison
    # =====================================================================
    ann_ret_sp500, ann_vol_sp500, sharpe_sp500, max_dd_sp500 = compute_performance_stats(sp500_returns)
    ann_ret_russell, ann_vol_russell, sharpe_russell, max_dd_russell = compute_performance_stats(russell_returns)
    ann_ret_50_50, ann_vol_50_50, sharpe_50_50, max_dd_50_50 = compute_performance_stats(portfolio_50_50_returns)

    performance_results.append({
        "Strategy": "Long Only SP500",
        "Annualized Return": round(ann_ret_sp500 * 100, 2),
        "Annualized Volatility": round(ann_vol_sp500 * 100, 2),
        "Sharpe": round(sharpe_sp500, 2),
        "Max Drawdown": round(max_dd_sp500 * 100, 2)
    })
    performance_results.append({
        "Strategy": "Long Only Russell",
        "Annualized Return": round(ann_ret_russell * 100, 2),
        "Annualized Volatility": round(ann_vol_russell * 100, 2),
        "Sharpe": round(sharpe_russell, 2),
        "Max Drawdown": round(max_dd_russell * 100, 2)
    })
    performance_results.append({
        "Strategy": "50/50 Portfolio",
        "Annualized Return": round(ann_ret_50_50 * 100, 2),
        "Annualized Volatility": round(ann_vol_50_50 * 100, 2),
        "Sharpe": round(sharpe_50_50, 2),
        "Max Drawdown": round(max_dd_50_50 * 100, 2)
    })

    df_results = pd.DataFrame(performance_results)
    print("=== Performance Summary ===")
    print(df_results)
    df_results.to_csv(f"{DATA_PATH}/backtest_long_neutral_results.csv", index=False)
    # fig_backtest.write_image(f"{IMG_PATH}/backtest_long_neutral.png", width=1200, height=800)

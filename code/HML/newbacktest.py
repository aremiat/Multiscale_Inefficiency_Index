import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -----------------------------
# Classe et fonctions de calcul RS
# -----------------------------
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
        t = len(s)  # Number of observations
        mean_y = np.mean(s)  # Mean of the series
        s = s.squeeze()

        if not chin:
            rho_1 = np.corrcoef(s[:-1], s[1:])[0, 1] # First-order autocorrelation

            if rho_1 < 0:
                return np.sum((s - mean_y) ** 2) / t

            # Calculate q according to Andrews (1991)
            q = ((3 * t) / 2) ** (1 / 3) * ((2 * rho_1) / (1 - (rho_1**2))) ** (2 / 3)
        else:
            q = 4*(t/100)**(2/9)

        # lower bound for q
        q = int(np.floor(q))

        # First term: classical variance
        var_term = np.sum((s - mean_y) ** 2) / t

        # Second term: weighted sum of autocovariances
        auto_cov_term = 0
        for j in range(1, q + 1):  # j ranges from 1 to q
            w_j = 1 - (j / (q + 1))  # Newey-West weights
            sum_cov = np.sum((s[:-j] - mean_y) * (s[j:] - mean_y))  # Lagged autocovariance
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


# -----------------------------
# Fonction de rolling non chevauchant
# -----------------------------
def non_overlapping_rolling(series, window, func):
    results = []
    indices = []
    n_segments = len(series) // window
    for i in range(n_segments):
        seg = series.iloc[i * window: (i + 1) * window]
        results.append(func(seg))
        indices.append(seg.index[-1])
    return pd.Series(results, index=indices)


# -----------------------------
# Fonctions modulaires pour le rolling
# -----------------------------
def compute_rolling_metric(series, window_size, method='modified', rolling_type='overlapping', chin=False):
    """
    Calcule la statistique de rolling à partir d'une série.

    - method : 'modified' ou 'traditional'
    - rolling_type : 'overlapping' ou 'nonoverlapping'
    """
    # On décale la série d'un jour pour utiliser le signal du jour précédent
    series_shifted = series.shift(1)
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
            full_index = pd.date_range(start=roll.index.min(),
                                       end=roll.index.max(),
                                       freq='B')
            rolling_modified_rs_full = roll.reindex(full_index)
            roll = rolling_modified_rs_full.ffill()
        else:
            raise ValueError("Méthode inconnue")
    else:
        raise ValueError("Type de rolling inconnu")
    return roll


# -----------------------------
# Fonction pour calculer le momentum
# -----------------------------
def compute_momentum(diff_series, shift_days=20, window_size=220):
    # Décalage puis moyenne mobile
    diff_shifted = diff_series.shift(shift_days)
    momentum = diff_shifted.rolling(window_size).mean()
    return momentum


# -----------------------------
# Fonction pour définir les positions en fonction du signal et du momentum
# -----------------------------
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


# -----------------------------
# Fonction de backtest
# -----------------------------
def run_backtest(all_p, positions, ticker1, ticker2, fee_rate=0.0005):
    portfolio_returns = positions[ticker1] * all_p[ticker1] + positions[ticker2] * all_p[ticker2]
    transaction_costs = (positions.diff().abs() * all_p).sum(axis=1) * fee_rate
    transaction_costs = transaction_costs.fillna(0)
    portfolio_returns = portfolio_returns - transaction_costs
    cumulative_returns = (1 + portfolio_returns).cumprod()
    return cumulative_returns, portfolio_returns


# -----------------------------
# Fonction de calcul des statistiques de performance (déjà définie)
# -----------------------------
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


# -----------------------------
# Main : chargement des données et test de plusieurs configurations de rolling
# -----------------------------
if __name__ == "__main__":
    # Chemin d'accès aux données
    DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data")
    p = pd.read_csv(f"{DATA_PATH}/index_prices2.csv", index_col=0, parse_dates=True)
    ticker1 = "^GSPC"
    ticker2 = "^RUT"
    # Sélection des deux indices
    all_prices = pd.concat([p[ticker1], p[ticker2]], axis=1)
    all_prices = all_prices.loc["1987-10-09": "2025-02-28"]
    all_p = all_prices.pct_change().dropna()
    # Calcul de la différence entre les deux indices
    all_p['Diff'] = all_p[ticker1] - all_p[ticker2]

    # Pour le backtest, on rebase les prix
    rebase_p = all_prices.loc["1987-10-09": "2025-02-28"].dropna()
    rebase_p = rebase_p / rebase_p.iloc[0]

    # Définition de la série sur laquelle appliquer le rolling (ici la Différence)
    r = all_p['Diff']

    # Calcul du momentum
    momentum = compute_momentum(r, shift_days=20, window_size=220)

    # Dictionnaire des configurations de rolling à tester
    # Chaque configuration est définie par :
    # - méthode : 'modified' ou 'traditional'
    # - rolling_type : 'overlapping' ou 'nonoverlapping'
    # - window_size
    rolling_configs = {
        "ModifOverlap_120": {"method": "modified", "rolling_type": "overlapping", "window_size": 120},
        "TradOverlap_120": {"method": "traditional", "rolling_type": "overlapping", "window_size": 120},
        "ModifNonOverlap_120": {"method": "traditional", "rolling_type": "nonoverlapping", "window_size": 120},
        # On peut ajouter d'autres configurations...
    }

    # Dictionnaires pour stocker les résultats
    cumulative_returns_dict = {}
    performance_results = []

    # Pour chaque configuration, calcul du rolling, du signal, des positions et backtest
    for config_name, config in rolling_configs.items():
        # Calcul du rolling metric (signal) sur r
        p = all_p.copy()
        rolling_signal = compute_rolling_metric(r, config["window_size"],
                                                method=config["method"],
                                                rolling_type=config["rolling_type"],
                                                chin=False)
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

        cum_returns, port_returns = run_backtest(all_p_config, positions, ticker1, ticker2, fee_rate=0.0005)
        cumulative_returns_dict[config_name] = cum_returns

        # Calcul des performances
        ann_ret, ann_vol, sharpe, max_dd = compute_performance_stats(port_returns)
        performance_results.append({
            "Strategy": config_name,
            "Annualized Return": round(ann_ret * 100, 2),
            "Annualized Volatility": round(ann_vol * 100, 2),
            "Sharpe": round(sharpe, 2),
            "Max Drawdown": round(max_dd * 100, 2)
        })

    # -----------------------------
    # Visualisation : toutes les courbes cumulées sur un même graphique
    # -----------------------------
    fig_backtest = go.Figure()
    colors = ["blue", "purple", "green", "red", "brown"]
    for i, (strategy, cum_returns) in enumerate(cumulative_returns_dict.items()):
        fig_backtest.add_trace(
            go.Scatter(
                x=cum_returns.index,
                y=cum_returns,
                mode='lines',
                name=strategy,
                line=dict(color=colors[i % len(colors)])
            )
        )
    fig_backtest.update_layout(
        title="Comparison of strategies (Cumulative Returns)",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template="plotly_white"
    )


    new_p = all_p.loc[common_dates_bp] # On fait commencer tous les portefeuilles au même moment

    # Calcul des portefeuilles de comparaison
    sp500_returns = new_p[ticker1]
    sp500_cumulative = (1 + sp500_returns).cumprod()

    russel_returns = new_p[ticker2]
    russel_cumulative = (1 + russel_returns).cumprod()

    portfolio_50_50_returns = 0.5 * new_p[ticker1] + 0.5 * new_p[ticker2]
    portfolio_50_50_cumulative = (1 + portfolio_50_50_returns).cumprod()

    # Ajout des portefeuilles de comparaison
    fig_backtest.add_trace(
        go.Scatter(
            x=sp500_cumulative.index,
            y=sp500_cumulative,
            mode='lines',
            name="SP500",
            line=dict(color='red')
        )
    )
    fig_backtest.add_trace(
        go.Scatter(
            x=russel_cumulative.index,
            y=russel_cumulative,
            mode='lines',
            name="Russel",
            line=dict(color='orange')
        )
    )

    fig_backtest.add_trace(
        go.Scatter(
            x=portfolio_50_50_cumulative.index,
            y=portfolio_50_50_cumulative,
            mode='lines',
            name="Portefeuille 50/50",
            line=dict(color='brown')
        )
    )

    fig_backtest.show()

    # ajout des performances des portefeuilles de comparaison
    ann_ret_sp500, ann_vol_sp500, sharpe_sp500, max_dd_sp500 = compute_performance_stats(sp500_returns)
    ann_ret_russel, ann_vol_russel, sharpe_russel, max_dd_russel = compute_performance_stats(russel_returns)
    ann_ret_50_50, ann_vol_50_50, sharpe_50_50, max_dd_50_50 = compute_performance_stats(portfolio_50_50_returns)

    performance_results.append({
        "Stratégie": "Long Only SP500",
        "Annual Return (%)": round(ann_ret_sp500 * 100, 2),
        "Annual Vol (%)": round(ann_vol_sp500 * 100, 2),
        "Sharpe": round(sharpe_sp500, 2),
        "Max Drawdown (%)": round(max_dd_sp500 * 100, 2)
    })
    performance_results.append({
        "Stratégie": "Long Only Russell",
        "Annual Return (%)": round(ann_ret_russel * 100, 2),
        "Annual Vol (%)": round(ann_vol_russel * 100, 2),
        "Sharpe": round(sharpe_russel, 2),
        "Max Drawdown (%)": round(max_dd_russel * 100, 2)
    })
    performance_results.append({
        "Stratégie": "50/50 Portfolio",
        "Annual Return (%)": round(ann_ret_50_50 * 100, 2),
        "Annual Vol (%)": round(ann_vol_50_50 * 100, 2),
        "Sharpe": round(sharpe_50_50, 2),
        "Max Drawdown (%)": round(max_dd_50_50 * 100, 2)
    })

    # Création d'un DataFrame récapitulatif des performances
    df_results = pd.DataFrame(performance_results)
    print("=== Tableau Récapitulatif des Stratégies ===")
    print(df_results)
    df_results.to_csv(f"{DATA_PATH}/backtest_long_neutral_results.csv", index=False)


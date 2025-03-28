import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


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


def compute_performance_stats(daily_returns: pd.Series, freq=252):
    """
    Calcule les stats de performance à partir d'une série de rendements journaliers.

    :param daily_returns: pd.Series de rendements quotidiens (peut contenir des NaN, qu'on exclura).
    :param freq: Nombre de jours de bourse par an (252 en général).
    :return: (annual_return, annual_vol, sharpe_ratio, max_drawdown)
    """
    # Nettoyage NaN éventuels
    daily_returns = daily_returns.dropna()

    if len(daily_returns) == 0:
        return np.nan, np.nan, np.nan, np.nan

    # 1) Rendement annualisé = [Produit(1+r)]^(freq / nb_observations) - 1
    total_ret = (1 + daily_returns).prod() - 1  # rendement total brut
    nb_obs = len(daily_returns)
    annual_return = (1 + total_ret) ** (freq / nb_obs) - 1

    # 2) Volatilité annualisée = std(daily_returns) * sqrt(freq)
    daily_vol = daily_returns.std()
    annual_vol = daily_vol * np.sqrt(freq)

    # 3) Sharpe ratio (approx) = (moyenne(r) / std(r)) * sqrt(freq)
    #    (Hypothèse : taux risk-free ~ 0)
    daily_mean = daily_returns.mean()
    sharpe_ratio = (daily_mean / daily_vol) * np.sqrt(freq) if daily_vol != 0 else np.nan

    # 4) Max drawdown (DD = (cumul - max_cumul) / max_cumul )
    cum_curve = (1 + daily_returns).cumprod()
    running_max = cum_curve.cummax()
    drawdown = (cum_curve - running_max) / running_max  # Valeurs négatives ou 0
    max_drawdown = drawdown.min()  # Le plus bas drawdown
    # Souvent, on l'affiche comme un pourcentage négatif (ex: -0.3 => -30%)

    return annual_return, annual_vol, sharpe_ratio, max_drawdown

if __name__ == "__main__":

    DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data")

    p = pd.read_csv(f"{DATA_PATH}/index_prices2.csv", index_col=0, parse_dates=True)
    all_prices = pd.concat([p["^GSPC"], p["^RUT"]], axis=1)
    all_prices = all_prices.loc["1987-10-09": "2024-12-31"]
    all_p = all_prices.pct_change().dropna()
    all_p['Diff'] = all_p['^GSPC'] - all_p["^RUT"]
    r = all_p['Diff']
    rebase_p = all_prices.loc["1987-10-09": "2024-12-31"].dropna()
    rebase_p = rebase_p / rebase_p.iloc[0]
    # rolling de la statistique R/S modifiée 6 mois
    rolling_modified_rs = r.rolling(120).apply(
        lambda window: ComputeRS.rs_modified_statistic(window, window_size=len(window), chin=False) / np.sqrt(
            len(window)),
        raw=False
    ).dropna()

    # --- 1. Calcul du momentum sur la Diff
    # 12 mois sans prendre en compte le dernier mois
    momentum = all_p['Diff'].shift(20) / all_p['Diff'].shift(252) - 1
    # momentum =  (1 +  all_p['Diff']).shift(20).rolling(window=252).apply(np.prod, raw=True) - 1

    # on shift de 1 pour avoir le momentum du jour précédent
    rolling_critical = rolling_modified_rs.shift(1)
    # --- 2. Définition des positions
    # Création d'un DataFrame positions avec une colonne pour SPX et une pour RUT.
    positions = pd.DataFrame(index=all_p.index, columns=['SPX', 'RUT'])

    # Par défaut, on est long sur les deux indices (position 0.5)
    positions['SPX'] = 0.5
    positions['RUT'] = 0.5

    # Condition : lorsque la rolling critical > 1.62
    condition = rolling_critical > 1.62

    # On aligne le DataFrame "positions" avec "rolling_critical" et "momentum".
    # On prend uniquement les dates communes
    common_dates = rolling_critical.index.intersection(momentum.index)
    positions = positions.loc[common_dates]
    all_p = all_p.loc[common_dates]
    momentum = momentum.loc[common_dates]
    rolling_critical = rolling_critical.loc[common_dates]

    # Pour les dates où la condition est remplie, on ajuste les positions selon le signe du momentum
    positions.loc[condition & (momentum > 0), 'SPX'] = 1
    positions.loc[condition & (momentum > 0), 'RUT'] = 0

    positions.loc[condition & (momentum < 0), 'SPX'] = 0
    positions.loc[condition & (momentum < 0), 'RUT'] = 1

    # --- 2bis. Application d'un minimum de 1 an de maintien avant reswitch ---
    min_holding_days = 360
    # On va créer une nouvelle série de positions qui respecte la contrainte
    final_positions = positions.copy()
    last_switch_date = positions.index[0]
    final_positions.loc[positions.index[0]] = positions.loc[positions.index[0]]

    for date in positions.index[1:]:
        # Si moins de 252 jours se sont écoulés depuis le dernier switch, on conserve la position précédente
        if (date - last_switch_date).days < min_holding_days:
            final_positions.loc[date] = final_positions.loc[last_switch_date]
        else:
            # Si le signal indique un changement par rapport à la dernière position maintenue, on met à jour
            if not (positions.loc[date] == final_positions.loc[last_switch_date]).all():
                final_positions.loc[date] = positions.loc[date]
                last_switch_date = date
            else:
                final_positions.loc[date] = final_positions.loc[last_switch_date]

    # Utilisation de final_positions pour le backtest
    portfolio_returns = final_positions['SPX'] * all_p['^GSPC'] + final_positions['RUT'] * all_p['^RUT']
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # Calcul des portefeuilles de comparaison
    sp500_returns = all_p['^GSPC']
    sp500_cumulative = (1 + sp500_returns).cumprod()

    russel_returns = all_p["^RUT"]
    russel_cumulative = (1 + russel_returns).cumprod()

    portfolio_50_50_returns = 0.5 * all_p['^GSPC'] + 0.5 * all_p["^RUT"]
    portfolio_50_50_cumulative = (1 + portfolio_50_50_returns).cumprod()


    ###################### Annotations pour le graphique ######################


    # --- 3. Détection des changements de position et création des annotations ---
    annotations = []
    # On se base sur final_positions pour détecter les changements, en itérant à partir du deuxième jour
    prev_pos = final_positions.iloc[0]
    for date in final_positions.index[1:]:
        curr_pos = final_positions.loc[date]
        if not (curr_pos == prev_pos).all():
            # Détermine le nouvel état sous forme de tuple (SPX, RUT)
            state = (curr_pos['SPX'], curr_pos['RUT'])
            if state == (1, 0):
                text = "Switch: Long SPX, Neutral RUT"
                arrowcolor = "green"
                ay = -30
            elif state == (0, 1):
                text = "Switch: Neutral SPX, Long RUT"
                arrowcolor = "red"
                ay = 30
            elif state == (0.5, 0.5):
                text = "Switch: Neutral"
                arrowcolor = "blue"
                ay = -20
            else:
                text = "Switch: Long SPX, Long RUT"
                arrowcolor = "purple"
                ay = 20

            annotations.append(dict(
                x=date,
                y=cumulative_returns.loc[date],
                xref="x",
                yref="y",
                text=text,
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=ay,
                arrowcolor=arrowcolor
            ))
            prev_pos = curr_pos
    # je suis obligé de le rajouter à la main sinon il n'affiche pas dans le graphe
    annotations.extend([
        dict(
            x="1991-05-20",
            y=1.3259,
            xref="x",
            yref="y",
            text="Switch: Neutral SPX, Long RUT",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=30,
            arrowcolor='red'
        ),
        dict(
            x="1992-05-14",
            y=1.5374,
            xref="x",
            yref="y",
            text="Switch: Neutral",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-20,
            arrowcolor='blue'
        )
    ])


    # --- 4. Visualisation des résultats avec comparaison des portefeuilles et annotations ---
    fig_backtest = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 subplot_titles=("Évolution des portefeuilles", "Rolling Critical"))

    # Trace de la stratégie long/short SPX vs Russel
    fig_backtest.add_trace(
        go.Scatter(x=cumulative_returns.index, y=cumulative_returns, mode='lines', name="Stratégie Long/Neutral", line=dict(color='blue')),
        row=1, col=1
    )

    # Portefeuille long only SP500
    fig_backtest.add_trace(
        go.Scatter(x=sp500_cumulative.index, y=sp500_cumulative, mode='lines', name="Long Only SP500", line=dict(color='purple')),
        row=1, col=1
    )

    # Portefeuille long only Russel
    fig_backtest.add_trace(
        go.Scatter(x=russel_cumulative.index, y=russel_cumulative, mode='lines', name="Long Only Russel", line=dict(color='magenta')),
        row=1, col=1
    )

    # Portefeuille 50/50 SP500 & Russel
    fig_backtest.add_trace(
        go.Scatter(x=portfolio_50_50_cumulative.index, y=portfolio_50_50_cumulative, mode='lines', name="50/50 SPX & Russel", line=dict(color='brown')),
        row=1, col=1
    )

    # Rolling modified RS
    fig_backtest.add_trace(
        go.Scatter(x=rolling_critical.index, y=rolling_critical, mode='lines', name="Rolling Critical", line=dict(color='green')),
        row=2, col=1
    )

    # Seuil de Rolling Critical
    fig_backtest.add_trace(
        go.Scatter(x=rolling_critical.index, y=[1.62]*len(rolling_critical), mode='lines', name="Seuil 1.62", line=dict(color='red', dash='dash')),
        row=2, col=1
    )


    # Ajout des annotations pour les changements de position
    fig_backtest.update_layout(annotations=annotations)

    fig_backtest.update_layout(title="Backtest SPX vs Russel & Comparaisons : Stratégie, Long Only, 50/50 (Minimum 1 an de maintien)",
                                 showlegend=True)
    fig_backtest.update_xaxes(title_text="Date", row=2, col=1)
    fig_backtest.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig_backtest.update_yaxes(title_text="Rolling Critical", row=2, col=1)
    fig_backtest.show()


    # 1) Statistiques sur la stratégie Long/Short
    ann_ret_l_s, ann_vol_l_s, sharpe_l_s, max_dd_l_s = compute_performance_stats(portfolio_returns)

    # 2) Statistiques S&P 500 (long only)
    ann_ret_spx, ann_vol_spx, sharpe_spx, max_dd_spx = compute_performance_stats(sp500_returns)

    # 3) Statistiques Russell 2000 (long only)
    ann_ret_rus, ann_vol_rus, sharpe_rus, max_dd_rus = compute_performance_stats(russel_returns)

    # 4) Statistiques Portefeuille 50/50
    ann_ret_5050, ann_vol_5050, sharpe_5050, max_dd_5050 = compute_performance_stats(portfolio_50_50_returns)


    df_results = pd.DataFrame({
        'Stratégie': [
            "Long/Neutral (Crit. & Mom)",
            "Long Only SPX",
            "Long Only Russell",
            "Portefeuille 50/50"
        ],
        'Annual Return (%)': [
            round(ann_ret_l_s * 100, 2),
            round(ann_ret_spx * 100, 2),
            round(ann_ret_rus * 100, 2),
            round(ann_ret_5050 * 100, 2)
        ],
        'Annual Vol (%)': [
            round(ann_vol_l_s * 100, 2),
            round(ann_vol_spx * 100, 2),
            round(ann_vol_rus * 100, 2),
            round(ann_vol_5050 * 100, 2)
        ],
        'Sharpe': [
            round(sharpe_l_s, 2),
            round(sharpe_spx, 2),
            round(sharpe_rus, 2),
            round(sharpe_5050, 2)
        ],
        'Max Drawdown (%)': [
            round(max_dd_l_s * 100, 2),
            round(max_dd_spx * 100, 2),
            round(max_dd_rus * 100, 2),
            round(max_dd_5050 * 100, 2)
        ]
    })

    print("=== Tableau Récapitulatif des Stratégies ===")
    print(df_results)
    df_results.to_csv(f"{DATA_PATH}/backtest_long_neutral_results.csv", index=False)

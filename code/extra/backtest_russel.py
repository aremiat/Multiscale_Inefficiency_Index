import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.RS import ComputeRS

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data")

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
    # Chargement des données pour '^RUT'
    ticker1 = "^RUT"
    data = pd.read_csv(os.path.join(DATA_PATH, "russel_stocks.csv"), index_col=0, parse_dates=True)[ticker1]
    log_prices = np.log(data).dropna()
    returns = log_prices.diff().dropna()

    # Calcul du rolling modified R/S statistic sur une fenêtre de 120 jours
    # rolling_modified_rs = returns.rolling(window=120).apply(
    #     lambda window: ComputeRS.rs_modified_statistic(window, len(window), chin=False) / np.sqrt(len(window)),
    #     raw=False
    # ).dropna()

    rolling_modified_rs = returns.rolling(window=120).apply(
        lambda window: np.log(ComputeRS.rs_statistic(window, len(window))) / np.log(len(window)),
        raw=False
    ).dropna()

    diff_shifted = log_prices.shift(20)

    # Calculer la moyenne mobile sur 220 jours à partir de la série décalée
    momentum = diff_shifted.rolling(window=220).mean()

    # momentum = all_p['Diff'].rolling(window=20).mean()

    # on shift de 1 pour avoir le momentum du jour précédent
    rolling_critical = rolling_modified_rs.shift(1)

    # --- 2. Définition des positions
    # Création d'un DataFrame positions avec une colonne pour SPX et une pour RUT.
    positions = pd.DataFrame(index=returns.index, columns=[ticker1])

    # Par défaut, on est long sur les deux indices (position 0.5)
    positions[ticker1] = 0
    # Condition : lorsque la rolling critical > 1.62
    condition = rolling_critical > 0.5

    # On aligne le DataFrame "positions" avec "rolling_critical" et "momentum".
    # On prend uniquement les dates communes
    common_dates = rolling_critical.index.intersection(momentum.index)
    positions = positions.loc[common_dates]
    all_p = returns.loc[common_dates]
    momentum = momentum.loc[common_dates]
    rolling_critical = rolling_critical.loc[common_dates]

    # Pour les dates où la condition est remplie, on ajuste les positions selon le signe du momentum
    positions.loc[condition & (momentum > 0), ticker1] = 1

    positions.loc[condition & (momentum < 0), ticker1] = 0

    # # --- 2bis. Application d'un minimum de 1 an de maintien avant reswitch ---
    # min_holding_days = 360
    # # On va créer une nouvelle série de positions qui respecte la contrainte
    # final_positions = positions.copy()
    # last_switch_date = positions.index[0]
    # final_positions.loc[positions.index[0]] = positions.loc[positions.index[0]]
    #
    # for date in positions.index[1:]:
    #     # Si moins de 252 jours se sont écoulés depuis le dernier switch, on conserve la position précédente
    #     if (date - last_switch_date).days < min_holding_days:
    #         final_positions.loc[date] = final_positions.loc[last_switch_date]
    #     else:
    #         # Si le signal indique un changement par rapport à la dernière position maintenue, on met à jour
    #         if not (positions.loc[date] == final_positions.loc[last_switch_date]).all():
    #             final_positions.loc[date] = positions.loc[date]
    #             last_switch_date = date
    #         else:
    #             final_positions.loc[date] = final_positions.loc[last_switch_date]

    # Utilisation de final_positions pour le backtest
    portfolio_returns = positions[ticker1] * all_p
    cumulative_returns = (1 + portfolio_returns).cumprod()

    index_returns = (1 + all_p).cumprod()


    # Visualisation : prix, rolling modified R/S statistic et momentum
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=(
                        "Russell Price", "Rolling Modified R/S Statistic", "Momentum Signal (20-day lag)"))

    fig.add_trace(go.Scatter(
        x=index_returns.index, y=index_returns, mode='lines',
        name='Russell cumulative return', line=dict(color='red')), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=rolling_modified_rs.index, y=rolling_modified_rs, mode='lines',
        name='Rolling Modified R/S', line=dict(color='green')), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=cumulative_returns.index, y=cumulative_returns, mode='lines',
        name='Long Neutral Strategy', line=dict(color='blue')), row=1, col=1)

    fig.update_layout(title="Backtest: Momentum sur Rolling Modified R/S Statistic", showlegend=True)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Rolling Modified R/S", row=2, col=1)

    fig.show()

    # 1) Statistiques sur la stratégie Long/Short
    ann_ret_l_s, ann_vol_l_s, sharpe_l_s, max_dd_l_s = compute_performance_stats(portfolio_returns)

    # 3) Statistiques Russell 2000 (long only)
    ann_ret_rus, ann_vol_rus, sharpe_rus, max_dd_rus = compute_performance_stats(index_returns)


    df_results = pd.DataFrame({
        'Stratégie': [
            "Long/Neutral (Crit. & Mom)",
            "Long Only SPX",
            "Long Only Russell",
            "Portefeuille 50/50"
        ],
        'Annual Return (%)': [
            round(ann_ret_l_s * 100, 2),
            round(ann_ret_rus * 100, 2),
        ],
        'Annual Vol (%)': [
            round(ann_vol_l_s * 100, 2),
            round(ann_vol_rus * 100, 2),
        ],
        'Sharpe': [
            round(sharpe_l_s, 2),
            round(sharpe_rus, 2),
        ],
        'Max Drawdown (%)': [
            round(max_dd_l_s * 100, 2),
            round(max_dd_rus * 100, 2),
        ]
    })

    print("=== Tableau Récapitulatif des Stratégies ===")
    print(df_results)

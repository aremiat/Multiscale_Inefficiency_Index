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


DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data")


p = pd.read_csv(f"{DATA_PATH}/index_prices2.csv", index_col=0, parse_dates=True)
all_prices = pd.concat([p["^GSPC"], p["^RUT"]], axis=1)
all_prices = all_prices.loc["1987-10-09": "2024-12-31"]
all_p = all_prices.pct_change().dropna()
all_p['Diff'] = all_p['^GSPC'] - all_p["^RUT"]
r = all_p['Diff']
rebase_p = all_prices.loc["1987-10-09": "2024-12-31"].dropna()
rebase_p = rebase_p / rebase_p.iloc[0]
rolling_critical = r.rolling(120).apply(
    lambda window: ComputeRS.rs_modified_statistic(window, window_size=len(window), chin=False) / np.sqrt(
        len(window)),
    raw=False
).dropna()

# --- 1. Calcul du momentum sur la Diff (ici, moyenne mobile sur 20 jours)
momentum = all_p['Diff'].rolling(window=20).mean()
momentum = momentum.shift(1)
rolling_critical = rolling_critical.shift(1)
# --- 2. Définition des positions
# Création d'un DataFrame positions avec une colonne pour SPX et une pour RUT.
positions = pd.DataFrame(index=all_p.index, columns=['SPX', 'RUT'])

# Par défaut, on est long sur les deux indices (position +1)
positions['SPX'] = 0.5
positions['RUT'] = 0.5

# Condition : lorsque la rolling critical > 1.62 (attention : rolling_critical commence à une date plus tard)
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
positions.loc[condition & (momentum > 0), 'RUT'] = -1

positions.loc[condition & (momentum < 0), 'SPX'] = -1
positions.loc[condition & (momentum < 0), 'RUT'] = 1

# (Si le momentum est exactement zéro, on peut rester long les deux, ici on ne change rien)

# --- 2bis. Application d'un minimum de 10 jours de maintien avant reswitch ---
min_holding_days = 30
# On va créer une nouvelle série de positions qui respecte la contrainte
final_positions = positions.copy()
last_switch_date = positions.index[0]
final_positions.loc[positions.index[0]] = positions.loc[positions.index[0]]

for date in positions.index[1:]:
    # Si moins de 10 jours se sont écoulés depuis le dernier switch, on conserve la position précédente
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

# --- 3. Détection des changements de position et création des annotations ---
annotations = []
# On se base sur final_positions pour détecter les changements, en itérant à partir du deuxième jour
prev_pos = final_positions.iloc[0]
for date in final_positions.index[1:]:
    curr_pos = final_positions.loc[date]
    if not (curr_pos == prev_pos).all():
        # Détermine le nouvel état sous forme de tuple (SPX, RUT)
        state = (curr_pos['SPX'], curr_pos['RUT'])
        if state == (1, -1):
            text = "Switch: Long SPX, Short RUT"
            arrowcolor = "green"
            ay = -30
        elif state == (-1, 1):
            text = "Switch: Short SPX, Long RUT"
            arrowcolor = "red"
            ay = 30
        elif state == (1, 1):
            text = "Switch: Long Only"
            arrowcolor = "blue"
            ay = -20
        else:
            text = f"Switch: {state}"
            arrowcolor = "black"
            ay = 0
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

# --- 4. Visualisation des résultats avec comparaison des portefeuilles et annotations ---
fig_backtest = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=("Évolution des portefeuilles", "Rolling Critical & Momentum"))

# Trace de la stratégie long/short SPX vs Russel (stratégie appliquée avec contrainte de 10 jours)
fig_backtest.add_trace(
    go.Scatter(x=cumulative_returns.index, y=cumulative_returns, mode='lines', name="Stratégie Long/Short", line=dict(color='blue')),
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

# Rolling Critical
fig_backtest.add_trace(
    go.Scatter(x=rolling_critical.index, y=rolling_critical, mode='lines', name="Rolling Critical", line=dict(color='green')),
    row=2, col=1
)

# Seuil de Rolling Critical
fig_backtest.add_trace(
    go.Scatter(x=rolling_critical.index, y=[1.62]*len(rolling_critical), mode='lines', name="Seuil 1.62", line=dict(color='red', dash='dash')),
    row=2, col=1
)

# Momentum (moyenne mobile sur 20 jours)
fig_backtest.add_trace(
    go.Scatter(x=momentum.index, y=momentum, mode='lines', name="Momentum (20j moy)", line=dict(color='orange')),
    row=2, col=1
)

# Ajout des annotations pour les changements de position
fig_backtest.update_layout(annotations=annotations)

fig_backtest.update_layout(title="Backtest SPX vs Russel & Comparaisons : Stratégie, Long Only, 50/50 (Minimum 10j de maintien)",
                             height=800, width=1000, showlegend=True)
fig_backtest.update_xaxes(title_text="Date", row=2, col=1)
fig_backtest.update_yaxes(title_text="Cumulative Return", row=1, col=1)
fig_backtest.update_yaxes(title_text="Rolling Critical / Momentum", row=2, col=1)
fig_backtest.show()

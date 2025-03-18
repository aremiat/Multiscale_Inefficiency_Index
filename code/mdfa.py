import os.path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.RS import ComputeRS
from plotly.subplots import make_subplots
import os

DATA_PATH = os.path.dirname(__file__) + "/../data"

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
    # Centrer et intégrer le signal (profil)
    signal = signal - np.mean(signal)
    Y = np.cumsum(signal)

    Fq = np.zeros((len(q_list), len(scales)))

    # Pour chaque échelle s
    for i, s in enumerate(scales):
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
        for j, q in enumerate(q_list):
            if np.abs(q) < 1e-6:
                # q = 0 : utilisation de la moyenne géométrique (limite q->0)
                Fq[j, i] = np.exp(0.5 * np.mean(np.log(F_seg)))
            else:
                Fq[j, i] = (np.mean(F_seg ** (q / 2))) ** (1 / q)

    return Fq

def compute_alpha_falpha(q_list, h_q):
    """
    Calcule alpha et f(alpha) via la transformation de Legendre
    à partir de h(q).
    """
    dq = q_list[1] - q_list[0]
    dh_dq = np.gradient(h_q, dq)  # dérivée numérique de h(q)
    alpha = h_q + q_list * dh_dq  # alpha(q) = h(q) + q * h'(q)
    f_alpha = q_list * (alpha - h_q) + 1  # f(alpha) = q * [alpha(q) - h(q)] + 1
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
        pd.Series contenant la largeur du spectre Delta alpha
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

        # 1) Calcul Fq(s) pour la fenêtre
        Fq = mfdfa(window_data, scales, q_list, order=order)

        # 2) Calcul de h(q) par régression linéaire log(Fq) vs log(s)
        h_q = []
        log_scales = np.log(scales)
        for j, q in enumerate(q_list):
            log_Fq = np.log(Fq[j, :])
            # Ajustement linéaire pour trouver la pente = h(q)
            coeffs = np.polyfit(log_scales, log_Fq, 1)
            h_q.append(coeffs[0])
        h_q = np.array(h_q)

        # 3) Transformation de Legendre -> alpha, f(alpha)
        alpha, f_alpha = compute_alpha_falpha(q_list, h_q)

        # 4) Largeur du spectre multifractal
        alpha_width = alpha.max() - alpha.min()

        alpha_widths.append(alpha_width)

        # On associe la valeur au dernier point de la fenêtre
        rolling_index.append(index_data[end - 1])

    return pd.Series(alpha_widths, index=rolling_index, name="alpha_width")

def plot_russell_and_critical_alpha(price_series, rolling_critical, alpha_width_series, threshold=1.620):
    """
    Affiche deux sous-graphiques (rows=2, shared_xaxes=True):
      1) Le cours du Russell 2000 (en haut).
      2) En bas, le Rolling Critical Value (axe de gauche) + la ligne de seuil,
         et Δα (axe de droite).
    """

    # Création d'une figure avec 2 rangées, la seconde ayant un axe secondaire
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,  # on partage l'axe x (la date)
        row_heights=[0.4, 0.6],            # proportion de hauteur pour row1 et row2
        vertical_spacing=0.06,             # espacement vertical entre les subplots
        specs=[
            [{"secondary_y": False}],      # row 1 => pas de second axe y
            [{"secondary_y": True}]        # row 2 => second axe y activé
        ],
        subplot_titles=("Russell 2000 Price", "Rolling Critical Value vs. Δα")
    )

    # ---- 1) Premier subplot : le cours (row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=price_series.index,
            y=price_series.values,
            mode='lines',
            name='Russell 2000 Price',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)

    # ---- 2) Deuxième subplot : Rolling Critical Value (axe de gauche) + ligne de seuil
    fig.add_trace(
        go.Scatter(
            x=rolling_critical.index,
            y=rolling_critical.values,
            mode='lines',
            name='Rolling Critical Value (R/S mod.)',
            line=dict(color='green')
        ),
        row=2, col=1, secondary_y=False
    )

    # Ligne de seuil horizontal
    fig.add_trace(
        go.Scatter(
            x=rolling_critical.index,
            y=[threshold]*len(rolling_critical),
            mode='lines',
            line=dict(dash='dash', color='red'),
            name=f'Seuil = {threshold}'
        ),
        row=2, col=1, secondary_y=False
    )

    # ---- 3) Δα (axe de droite)
    fig.add_trace(
        go.Scatter(
            x=alpha_width_series.index,
            y=alpha_width_series.values,
            mode='lines+markers',
            name='Δα (MF-DFA)',
            marker=dict(color='purple')
        ),
        row=2, col=1, secondary_y=True
    )

    # Mise en forme
    fig.update_layout(
        title="Russell 2000 Price + Rolling Critical Value vs. Δα",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98)
    )

    # Configurer les titres d'axes
    fig.update_yaxes(title_text='Critical Value (R/S)', row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text='Δα (MF-DFA)', row=2, col=1, secondary_y=True)
    fig.update_xaxes(title_text='Date', row=2, col=1)

    fig.show()

# --- Téléchargement des données et calcul des rendements ---

data = pd.read_csv(f"{DATA_PATH}/russell_2000.csv", index_col=0, parse_dates=True)

ticker = "^RUT"
# Calcul des rendements journaliers en log
returns = np.log(data).diff().dropna()
r_m = returns
# r_m = returns.resample('M').last()

window_size = 120  # ex. 120 mois (10 ans)
q_list = np.linspace(-5, 5, 21)
scales = np.unique(np.floor(np.logspace(np.log10(10), np.log10(80), 10)).astype(int))
scales_daily = np.unique(np.floor(np.logspace(np.log10(10), np.log10(120), 10)).astype(int))
#
# alpha = mfdfa(r_m.values, scales, q_list, order=1)
Fq = mfdfa(r_m.values, scales, q_list, order=1)

alpha_width_series = mfdfa_rolling(r_m, window_size, q_list, scales, order=1)

# rs_value = ComputeRS.rs_statistic(returns, window_size=len(returns))
# hurst_rs = np.log(rs_value) / np.log(len(returns))

rolling_critical = r_m.rolling(window_size).apply(
    lambda window: ComputeRS.rs_modified_statistic(window, len(window), chin=False)/np.sqrt(len(window)),
    raw=False
).dropna()

# rolling_critical = r_m.rolling(window_size).apply(
#     lambda w: np.log(ComputeRS.rs_statistic(w, len(w))) / np.log(len(w)),
#     raw=False
# ).dropna()

alpha_width_series.index = rolling_critical.index

alpha_rolling_price = pd.concat([data, alpha_width_series, rolling_critical], axis=1, join='inner')
alpha_rolling_price.columns = ['Price', 'Alpha Width', 'Critical Value']
alpha_rolling_price.to_csv(f"{DATA_PATH}/alpha_rolling_price.csv")

# window_size = 12
# # Calcul de la corrélation glissante
# rolling_corr = rolling_critical.rolling(window=window_size).corr(alpha_width_series)
# # Création de la figure Plotly
# fig = go.Figure()
# fig.add_trace(go.Scatter(
#     x=rolling_corr.index,
#     y=rolling_corr.values,
#     mode='lines',
#     name=f'Rolling Corr (window={window_size})'
# ))
# fig.update_layout(
#     title='Corrélation glissante entre rolling_critical et alpha_width_series',
#     xaxis_title='Date',
#     yaxis_title='Coefficient de corrélation'
# )
# fig.show()
data_rolling = data.loc["1997-08-31": "2025-02-28"]
rolling_series = rolling_critical[ticker].dropna()
price_series = data_rolling[ticker].dropna()
plot_russell_and_critical_alpha(price_series, rolling_series, alpha_width_series, threshold=1.2)

Fq = mfdfa(r_m.values, scales, q_list, order=1)

# --- Estimation des exposants h(q) ---
# Pour chaque q, ajuster une régression linéaire sur le log-log (log(Fq(s)) vs log(s))
h_q = []
log_scales = np.log(scales)
for j, q in enumerate(q_list):
    log_Fq = np.log(Fq[j, :])
    coeffs = np.polyfit(log_scales, log_Fq, 1)
    h_q.append(coeffs[0])
h_q = np.array(h_q)


hq_q = pd.concat([pd.Series(q_list, name='q'), pd.Series(h_q, name='h(q)')], axis=1)
hq_q.to_csv(f"{DATA_PATH}/multifractal_spectrum.csv", index=False)

fig_hq = go.Figure()
fig_hq.add_trace(go.Scatter(
    x=q_list,
    y=h_q,
    mode='lines+markers',
    marker=dict(size=6),
    name='h(q)'
))
fig_hq.update_layout(
    title=f"Multifractal Spectrum h(q) for the Russell 2000 returns",
    xaxis_title="q",
    yaxis_title="h(q)",
    template="plotly_white"
)
fig_hq.show()

# --- Calcul du spectre multifractal f(α) via la transformation de Legendre ---
dq = q_list[1] - q_list[0]
dh_dq = np.gradient(h_q, dq)
alpha = h_q + q_list * dh_dq
f_alpha = q_list * (alpha - h_q) + 1

# --- Tracé du spectre f(α) avec Plotly ---
fig_falpha = go.Figure()
fig_falpha.add_trace(go.Scatter(
    x=alpha,
    y=f_alpha,
    mode='lines+markers',
    marker=dict(size=6),
    name='f(α)'
))
fig_falpha.update_layout(
    title=f"Spectre multifractal f(α) pour les rendements de {ticker}",
    xaxis_title="α",
    yaxis_title="f(α)",
    template="plotly_white"
)
fig_falpha.show()

df = pd.DataFrame({
    'f_alpha': f_alpha,
    'alpha': alpha
})

# Sauvegarde au format CSV
df.to_csv(f'{DATA_PATH}/f_alpha_alpha.csv', index=False)
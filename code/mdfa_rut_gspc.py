import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.RS import ComputeRS

DATA_PATH = os.path.dirname(__file__) + "/../data"

# =============================================================================
# Fonctions MF-DFA
# =============================================================================

def mfdfa(signal, scales, q_list, order=1):
    """
    Calcule le MF-DFA pour une série temporelle.
    """
    N = len(signal)
    signal = signal - np.mean(signal)
    Y = np.cumsum(signal)
    Fq = np.zeros((len(q_list), len(scales)))

    for i, s in enumerate(scales):
        s = int(s)
        if s < 2:
            continue
        n_segments = N // s
        F_seg = []
        # Découpage non chevauchant depuis le début
        for v in range(n_segments):
            segment = Y[v*s:(v+1)*s]
            idx = np.arange(s)
            coeffs = np.polyfit(idx, segment, order)
            fit = np.polyval(coeffs, idx)
            F_seg.append(np.mean((segment - fit) ** 2))
        # Découpage depuis la fin
        for v in range(n_segments):
            segment = Y[N-(v+1)*s:N-v*s]
            idx = np.arange(s)
            coeffs = np.polyfit(idx, segment, order)
            fit = np.polyval(coeffs, idx)
            F_seg.append(np.mean((segment - fit) ** 2))

        F_seg = np.array(F_seg)
        F_seg[F_seg < 1e-10] = 1e-10
        for j, q in enumerate(q_list):
            if np.abs(q) < 1e-6:
                # q=0 : moyenne géométrique
                Fq[j, i] = np.exp(0.5 * np.mean(np.log(F_seg)))
            else:
                Fq[j, i] = (np.mean(F_seg ** (q/2))) ** (1/q)
    return Fq

def compute_alpha_falpha(q_list, h_q):
    """
    Calcule alpha et f(alpha) via la transformation de Legendre
    à partir de h(q).
    """
    dq = q_list[1] - q_list[0]
    dh_dq = np.gradient(h_q, dq)
    alpha = h_q + q_list * dh_dq
    f_alpha = q_list * (alpha - h_q) + 1
    return alpha, f_alpha

def mfdfa_rolling(series, window_size, q_list, scales, order=1):
    """
    Applique MF-DFA sur des fenêtres glissantes et renvoie Δα pour chaque fenêtre.
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
        Fq = mfdfa(window_data, scales, q_list, order=order)
        h_q = []
        log_scales = np.log(scales)
        for j, q in enumerate(q_list):
            log_Fq = np.log(Fq[j, :])
            coeffs = np.polyfit(log_scales, log_Fq, 1)
            h_q.append(coeffs[0])
        h_q = np.array(h_q)
        alpha, _ = compute_alpha_falpha(q_list, h_q)
        delta_alpha = alpha.max() - alpha.min()
        alpha_widths.append(delta_alpha)
        rolling_index.append(index_data[end - 1])

    return pd.Series(alpha_widths, index=rolling_index, name="alpha_width")

# =============================================================================
# Fonction de tracé
# =============================================================================

def plot_prices_and_critical_alpha(
    price_gspc, price_rut,
    rolling_critical, alpha_width_series,
    threshold=1.620
):
    """
    Affiche deux sous-graphiques (rows=2, shared_xaxes=True) :
      1) Les prix de GSPC et RUT (en haut).
      2) En bas, la Rolling Critical Value (axe gauche) + la ligne de seuil,
         et Δα (axe droit).
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.4, 0.6],
        vertical_spacing=0.06,
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": True}]
        ],
        subplot_titles=("GSPC & RUT Prices", "Rolling Critical Value vs. Δα")
    )

    # --- 1) Premier subplot : cours GSPC et RUT
    fig.add_trace(
        go.Scatter(
            x=price_gspc.index,
            y=price_gspc.values,
            mode='lines',
            name='GSPC Price',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=price_rut.index,
            y=price_rut.values,
            mode='lines',
            name='RUT Price',
            line=dict(color='orange')
        ),
        row=1, col=1
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)

    # --- 2) Deuxième subplot : Rolling Critical Value (axe gauche) + ligne de seuil
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

    # Ligne de seuil
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

    # --- 3) Δα (axe de droite)
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
        title="GSPC & RUT Prices + Rolling Critical Value vs. Δα",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98)
    )
    fig.update_yaxes(title_text='Critical Value (R/S)', row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text='Δα (MF-DFA)', row=2, col=1, secondary_y=True)
    fig.update_xaxes(title_text='Date', row=2, col=1)

    fig.show()

# =============================================================================
# Script principal
# =============================================================================

if __name__ == "__main__":

    # 1) Lecture des prix GSPC & RUT dans index_prices2.csv
    df_prices = pd.read_csv(
        os.path.join(DATA_PATH, "index_prices2.csv"),
        index_col=0, parse_dates=True
    )
    # On conserve ^GSPC et ^RUT
    df_prices = df_prices[['^GSPC','^RUT']].dropna()

    # 2) Calcul des rendements journaliers
    daily_returns = df_prices.pct_change().dropna()

    # 3) Différence de rendements
    diff_series = daily_returns['^GSPC'] - daily_returns['^RUT']
    diff_series = diff_series.dropna()

    # 4) Agrégation mensuelle (dernier point du mois)
    diff_monthly = diff_series.resample('M').last().dropna()

    # 5) Paramètres MF-DFA
    window_size = 120  # 120 mois
    q_list = np.linspace(-5, 5, 21)
    scales = np.unique(np.floor(np.logspace(np.log10(10), np.log10(80), 10)).astype(int))

    # 6) Calcul de Δα sur fenêtres roulantes
    alpha_width_series = mfdfa_rolling(diff_monthly, window_size, q_list, scales, order=1)

    # 7) Calcul de la Rolling Critical Value
    rolling_critical = diff_monthly.rolling(window_size).apply(
        lambda w: np.log(ComputeRS.rs_modified_statistic(w, len(w), chin=False)) / np.sqrt(len(w)),
        raw=False
    ).dropna()

    # rolling_critical = diff_monthly.rolling(window_size).apply(
    #     lambda w: ComputeRS.rs_modified_statistic(w, len(w), chin=False) / np.sqrt(len(w)),
    #     raw=False
    # ).dropna()

    # Aligner index
    alpha_width_series.index = rolling_critical.index

    # 8) Affichage
    # On prend la partie commune pour le tracé
    common_idx = rolling_critical.index.intersection(alpha_width_series.index)
    diff_monthly_aligned = diff_monthly.loc[common_idx]
    rolling_crit_aligned = rolling_critical.loc[common_idx]
    alpha_width_aligned = alpha_width_series.loc[common_idx]

    # Alignement des index
    alpha_width_series.index = rolling_critical.index
    common_idx = rolling_critical.index.intersection(alpha_width_series.index)

    # 4) Préparation des prix GSPC & RUT (mensuels ou journaliers ?)
    # Ici, on va simplement prendre les prix mensuels (dernier jour du mois)
    # pour l'affichage. On peut faire un .resample('M').last()
    df_prices = df_prices.loc["1996-08-31": "2025-02-28"]
    gspc_price = df_prices['^GSPC'].dropna()
    rut_price = df_prices['^RUT'].dropna()

    # Aligner sur common_idx si on veut
    # gspc_aligned = gspc_price_m.loc[common_idx].dropna()
    # rut_aligned = rut_price_m.loc[common_idx].dropna()

    # 5) Affichage
    # plot_prices_and_critical_alpha(
    #     gspc_aligned, rut_aligned,
    #     rolling_critical.loc[common_idx],
    #     alpha_width_series.loc[common_idx],
    #     threshold=1.620
    # )

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.4, 0.6],
        vertical_spacing=0.06,
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": True}]
        ],
        subplot_titles=("GSPC & RUT Prices", "Rolling Critical Value vs. Δα")
    )

    # --- 1) Premier subplot : cours GSPC et RUT
    fig.add_trace(
        go.Scatter(
            x=gspc_price.index,
            y=gspc_price.values,
            mode='lines',
            name='GSPC Price',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=rut_price.index,
            y=rut_price.values,
            mode='lines',
            name='RUT Price',
            line=dict(color='orange')
        ),
        row=1, col=1
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)

    # --- 2) Deuxième subplot : Rolling Critical Value (axe gauche) + ligne de seuil
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

    # Ligne de seuil
    fig.add_trace(
        go.Scatter(
            x=rolling_critical.index,
            y=[1.620]*len(rolling_critical),
            mode='lines',
            line=dict(dash='dash', color='red'),
            name=f'Seuil = {1.620}'
        ),
        row=2, col=1, secondary_y=False
    )

    # --- 3) Δα (axe de droite)
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
        title="GSPC & RUT Prices + Rolling Critical Value vs. Δα",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98)
    )
    fig.update_yaxes(title_text='Critical Value (R/S)', row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text='Δα (MF-DFA)', row=2, col=1, secondary_y=True)
    fig.update_xaxes(title_text='Date', row=2, col=1)

    fig.show()

    # 9) Calcul et tracé du spectre multifractal global
    #    (sur toute la série mensuelle, par exemple)
    from numpy import log
    Fq = mfdfa(diff_monthly.dropna().values, scales, q_list, order=1)

    # Estimation de h(q)
    h_q = []
    log_scales = np.log(scales)
    for j, q in enumerate(q_list):
        log_Fq = np.log(Fq[j, :])
        coeffs = np.polyfit(log_scales, log_Fq, 1)
        h_q.append(coeffs[0])
    h_q = np.array(h_q)

    # Plot h(q)
    fig_hq = go.Figure()
    fig_hq.add_trace(go.Scatter(
        x=q_list,
        y=h_q,
        mode='lines+markers',
        name='h(q)'
    ))
    fig_hq.update_layout(
        title="Multifractal Spectrum h(q) for (GSPC - RUT) returns difference",
        xaxis_title="q",
        yaxis_title="h(q)",
        template="plotly_white"
    )
    fig_hq.show()

    # Transformation de Legendre -> alpha, f(alpha)
    dq = q_list[1] - q_list[0]
    dh_dq = np.gradient(h_q, dq)
    alpha = h_q + q_list * dh_dq
    f_alpha = q_list * (alpha - h_q) + 1

    fig_falpha = go.Figure()
    fig_falpha.add_trace(go.Scatter(
        x=alpha,
        y=f_alpha,
        mode='lines+markers',
        name='f(α)'
    ))
    fig_falpha.update_layout(
        title="Spectre multifractal f(α) pour la différence GSPC - RUT",
        xaxis_title="α",
        yaxis_title="f(α)",
        template="plotly_white"
    )
    fig_falpha.show()

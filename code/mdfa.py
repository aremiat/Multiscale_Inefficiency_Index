import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.RS import ComputeRS
from plotly.subplots import make_subplots
import os
import plotly.express as px
from scipy.stats import norm, skew, kurtosis
from utils.MFDFA import ComputeMFDFA
import time
from typing import Sequence, Union


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
    Calcule alpha et f(alpha) via la transformation de Legendre
    à partir de h(q).
    """
    dq = q_list[1] - q_list[0]
    dh_dq = np.gradient(h_q, dq)  # dérivée numérique de h(q)
    alpha = h_q + q_list * dh_dq  # alpha(q) = h(q) + q * h'(q)
    f_alpha = q_list * (alpha - h_q) + 1  # f(alpha) = q * [alpha(q) - h(q)] + 1
    return alpha, f_alpha


def mfdfa_rolling_opti(
        series: Union[pd.Series, Sequence[float]],
        window_size: int,
        q_list: Sequence[float],
        scales: Sequence[int],
        order: int = 1
) -> pd.Series:
    """
    Apply rolling MF-DFA on a series and return the multifractal width (Delta alpha),
    with progress and ETA printed to stdout.
    """
    # Extraction des données et de l’index
    if isinstance(series, pd.Series):
        data = series.values.astype(float)
        idx = series.index
    else:
        data = np.asarray(series, dtype=float)
        idx = pd.RangeIndex(len(data))

    N = data.shape[0]
    num_win = N - window_size + 1
    if num_win <= 0:
        return pd.Series([], name="alpha_width")

    # Pré-calculs constants
    q = np.asarray(q_list, dtype=float)
    scales_arr = np.asarray(scales, dtype=float)
    log_s = np.log(scales_arr)
    x = log_s
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    # Préallocation du résultat
    alpha_widths = np.empty(num_win, dtype=float)

    # Horloge de départ
    t0 = time.time()

    # Boucle principale
    for i in range(num_win):
        window = data[i: i + window_size]
        Fq = ComputeMFDFA.mfdfa(window, scales_arr, q, order)
        logF = np.log(Fq)

        # régression vectorisée pour h(q)
        y_mean = logF.mean(axis=1)
        cov = ((x[None, :] - x_mean) * (logF - y_mean[:, None])).sum(axis=1)
        h_q = cov / denom

        # multifractal width
        alpha, _ = ComputeMFDFA.compute_alpha_falpha(q, h_q)
        alpha_widths[i] = alpha.max() - alpha.min()

        # affichage progression + ETA
        elapsed = time.time() - t0
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (num_win - i - 1)
        print(
            f"Window {i + 1}/{num_win} — "
            f"Elapsed: {elapsed:.1f}s — "
            f"ETA: {remaining:.1f}s",
            end="\r",
            flush=True
        )

    # passe à la ligne après la barre
    print()

    # Construction de la série pandas
    out_idx = idx[window_size - 1: window_size - 1 + num_win]
    return pd.Series(alpha_widths, index=out_idx, name="alpha_width")


def fa(x, scales, qs):
    """
    Fluctuation Analysis (FA) pour des séries stationnaires normalisées.

    Paramètres :
      x      : série temporelle (tableau 1D)
      scales : liste (ou tableau) des tailles d'échelle s.
               On suppose que pour chaque s, N est un multiple entier de s.
      qs     : tableau des valeurs q (q peut être négatif, positif ou zéro).

    Retourne :
      Fqs    : tableau 2D de forme (len(qs), len(scales)) contenant Fq(s) pour chaque q et chaque échelle s.
    """
    x = np.array(x)
    N = len(x)

    # Calcul du profil en ajoutant Y(0)=0
    Y = np.zeros(N + 1)
    Y[1:] = np.cumsum(x - np.mean(x))

    Fqs = np.zeros((len(qs), len(scales)))

    for j, s in enumerate(scales):
        s = int(s)
        # On suppose ici que N est un multiple entier de s.
        Ns = N // s

        # Calcul des fluctuations locales dans chaque segment
        # Pour κ=1,..., Ns, on a F_FA(s, v) = |Y(v*s) - Y((v-1)*s)|
        F_loc = np.empty(Ns)
        for v in range(1, Ns + 1):
            F_loc[v - 1] = np.abs(Y[v * s] - Y[(v - 1) * s])

        # Calcul de la fonction de fluctuation d'ordre q
        for i, q in enumerate(qs):
            if q == 0:
                # Pour q=0, utilisation de la moyenne logarithmique
                Fqs[i, j] = np.exp(np.mean(np.log(F_loc)))
            else:
                Fqs[i, j] = (np.mean(F_loc ** q)) ** (1 / q)

    return Fqs
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


def surrogate_gaussian_corr(series):
    N = len(series)
    # 1. FFT
    Xf = np.fft.rfft(series)
    amplitudes = np.abs(Xf)

    # 2. Phases aléatoires (sauf DC et Nyquist)
    random_phases = np.exp(2j * np.pi * np.random.rand(len(Xf)))
    random_phases[0] = 1.0  # conserve la composante moyenne
    if N % 2 == 0:
        random_phases[-1] = 1.0  # pour fréquence Nyquist si N pair

    # 3. Reconstruction du spectre
    Xf_surr = amplitudes * random_phases

    # 4. IFFT
    surrogate = np.fft.irfft(Xf_surr, n=N)
    return surrogate

# --- Téléchargement des données et calcul des rendements ---
# Paramètres
q_list = np.linspace(-3, 3, 13)
scales_rut = np.unique(np.floor(np.logspace(np.log10(10), np.log10(500), 10)).astype(int))
scales_gspc = np.unique(np.floor(np.logspace(np.log10(10), np.log10(500), 10)).astype(int))
tickers = ["^GSPC", "^RUT", "^FTSE", "^N225", "^GDAXI"]

if __name__ == "__main__":
    # Lecture des données pour '^RUT'

    data_equity = pd.read_csv(os.path.join(DATA_PATH, "index_prices2.csv"), index_col=0, parse_dates=True)
    data_multi_assets = pd.read_csv(os.path.join(DATA_PATH, "multi_assets.csv"), index_col=0, parse_dates=True)
    # data_multi_assets.columns = data_multi_assets.columns.droplevel(0)
    # # 2. si tu veux trier par ordre alphabétique des tickers :
    # data_multi_assets = data_multi_assets.sort_index(axis=1)
    # df_ticker = data.loc["1987-09-10":"2025-02-28"]
    data_multi_assets = data_multi_assets.loc["2014-09-17":]
    scales = scales_rut

    for tick in data_multi_assets.columns:
        data_tick = data_multi_assets[tick]
        returns = np.log(data_tick).diff().dropna()


    # alpha_width = mfdfa_rolling_opti(returns, 1008, q_list, scales, order=1)
    #
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=alpha_width.index, y=alpha_width.values, mode='lines+markers',
    #                          name='Δα (MF-DFA)', line=dict(color='purple')))
    # fig.update_layout(title=f'Δα (MF-DFA) for {name}',
    #                     xaxis_title='Date', yaxis_title='Δα', template='plotly_white')
    # fig.show()

    # # Calcul de F(q,s)
    # Fa = fa(returns.values, scales, q_list)  # Fq est un tableau de forme (len(q_list), len(scales))
    # h_q = []
    # log_scales = np.log(scales)
    # for i, q in enumerate(q_list):
    #     log_Fq = np.log(Fa[i, :])
    #     slope, _ = np.polyfit(log_scales, log_Fq, 1)
    #     h_q.append(slope)
    # h_q = np.array(h_q)
    # alpha, f_alpha = compute_alpha_falpha(q_list, h_q)
    #
    # fig_f = go.Figure()
    # fig_f.add_trace(go.Scatter(x=alpha, y=f_alpha, mode='lines+markers',
    #                            name='f(α) original', line=dict(color='blue')))
    # # fig_f.add_trace(go.Scatter(x=alpha_shuf, y=f_alpha_shuf, mode='lines+markers',
    # #                            name='f(α) shuffled', line=dict(color='orange')))
    # fig_f.update_layout(title=f'Spectre multifractal f(α): Original vs Shuffled, {name}',
    #                     xaxis_title='α', yaxis_title='f(α)', template='plotly_white')
    # fig_f.show()
    #
    # # Créer une grille pour le plot 3D :
    # Q, S = np.meshgrid(q_list, scales, indexing='ij')
    #
    # # Création de la surface avec Plotly :
    # fig = go.Figure(data=[go.Surface(x=S, y=Q, z=Fq)])
    # fig.update_layout(
    #     title='Surface de F(q,s)',
    #     scene=dict(
    #         xaxis_title='Échelle s',
    #         yaxis_title='Ordre q',
    #         zaxis_title='F(q,s)'
    #     )
    # )
    # fig.show()

    # Calcul de la moyenne et de l'écart-type des rendements
    # mu = returns.mean()
    # sigma = returns.std()
    #
    # # Création d'un histogramme interactif avec Plotly Express
    # fig = px.histogram(returns, nbins=100, opacity=0.75,
    #                    title=f"Histogramme des rendements de {name} et densité gaussienne",
    #                    labels={'value': "Rendements"})
    #
    # # Création de la densité théorique d'une gaussienne
    # x_values = np.linspace(returns.min(), returns.max(), 200)
    # pdf = norm.pdf(x_values, loc=mu, scale=sigma)
    #
    # # Normalisation : Pour superposer la densité sur l'histogramme,
    # # il faut l'ajuster au nombre d'observations et à la largeur des bins.
    # bin_width = (returns.max() - returns.min()) / 50
    # pdf_scaled = pdf * len(returns) * bin_width
    #
    # # Ajout de la courbe gaussienne sur l'histogramme
    # fig.add_trace(go.Scatter(x=x_values, y=pdf_scaled, mode='lines', name='Gaussienne théorique',
    #                          line=dict(color='red', width=2)))
    #
    # fig.update_layout(template="plotly_white", xaxis_title="Rendements", yaxis_title="Fréquence")
    # fig.show()

    # stats = {
    #     'Mean': returns.mean(),
    #     'Std Dev': returns.std(),
    #     'Skewness': skew(returns),
    #     'Kurtosis': kurtosis(returns, fisher=False)  # Fisher=False pour obtenir la kurtosis "classique"
    # }
    # # Création du DataFrame
    # stats_df = pd.DataFrame(stats, index=['Returns'])
    #
    # print(stats_df)

    # --- 1. Calcul pour la série originale ---

        Fq = mfdfa(returns.values, scales, q_list, order=1)
        h_q = []
        log_scales = np.log(scales)
        for i, q in enumerate(q_list):
            log_Fq = np.log(Fq[i, :])
            slope, _ = np.polyfit(log_scales, log_Fq, 1)
            h_q.append(slope)
        h_q = np.array(h_q)
        alpha, f_alpha = compute_alpha_falpha(q_list, h_q)

        # plot de la log variance against log s
        # fig_var = go.Figure()
        # for i, q in enumerate(q_list):
        #     log_Fq = np.log(Fq[i, :])
        #
        #     fig_var.add_trace(go.Scatter(
        #         x=log_scales,
        #         y=log_Fq,
        #         mode='markers+lines',
        #         name=f'q={q}'
        #     ))
        # fig_var.update_layout(
        #     title=f'Log-Variance vs Log-Scale pour {name}',
        #     xaxis_title='log(s)',
        #     yaxis_title='log(variance)',
        #     template='plotly_white'
        # )
        # fig_var.show()

        surogate_series = surrogate_gaussian_corr(returns.values)
        Fq_surrogate = mfdfa(surogate_series, scales, q_list, order=1)
        h_q_surrogate = []
        for i, q in enumerate(q_list):
            log_Fq_surrogate = np.log(Fq_surrogate[i, :])
            slope_surrogate, _ = np.polyfit(log_scales, log_Fq_surrogate, 1)
            h_q_surrogate.append(slope_surrogate)
        h_q_surrogate = np.array(h_q_surrogate)
        alpha_surrogate, f_alpha_surrogate = compute_alpha_falpha(q_list, h_q_surrogate)



        # # --- 2. Calcul pour la série mélangée (shuffle) ---
        returns_shuf = returns.sample(frac=1, random_state=42).reset_index(drop=True)
        new_returns_shuf = returns_shuf.sample(frac=1, random_state=56).reset_index(drop=True)
        new_new_returns_shuf = new_returns_shuf.sample(frac=1, random_state=25).reset_index(drop=True)
        Fq_shuf = mfdfa(new_new_returns_shuf.values, scales, q_list, order=1)
        h_q_shuf = []
        for i, q in enumerate(q_list):
            log_Fq_shuf = np.log(Fq_shuf[i, :])
            slope_shuf, _ = np.polyfit(log_scales, log_Fq_shuf, 1)
            h_q_shuf.append(slope_shuf)
        h_q_shuf = np.array(h_q_shuf)
        alpha_shuf, f_alpha_shuf = compute_alpha_falpha(q_list, h_q_shuf)

        # --- 3. Calcul du correcteur de Hurst dû aux corrélations ---
        h_cor = h_q - h_q_shuf

        # Calcul des différences pour alpha et f(alpha)
        alpha_diff = alpha - alpha_shuf
        f_alpha_diff = f_alpha - f_alpha_shuf

        # --- Affichage avec Plotly ---
        import plotly.graph_objects as go

        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=q_list,
            y=h_q,
            mode='lines+markers',
            name='h(q) - h(q)_shuf',
            line=dict(color='purple')
        ))
        fig_h.update_layout(
            title=f'Difference between hurst exponents : Original vs Shuffled, {tick}',
            xaxis_title='Ordre q',
            yaxis_title=r'h(q) - h_{shuf}(q)',
            template='plotly_white'
        )
        # fig_h.show()

        # Graphique 1 : Exposant de Hurst
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=q_list, y=h_q, mode='lines+markers',
                                   name='h(q) original', line=dict(color='blue')))
        fig_h.add_trace(go.Scatter(x=q_list, y=h_q_shuf, mode='lines+markers',
                                   name='h(q) shuffled', line=dict(color='orange')))
        fig_h.add_trace(go.Scatter(x=q_list, y=h_q_surrogate, mode='lines+markers',
                                      name='h(q) - h(q) shuffled', line=dict(color='green')))
        fig_h.update_layout(title=f'Hurst Exponent h(q): Original vs Shuffled {tick}',
                            xaxis_title='q', yaxis_title='h(q)', template='plotly_white')
        # fig_h.show()

        # Graphique 2 : Spectre multifractal f(α)
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=alpha, y=f_alpha, mode='lines+markers',
                                   name='f(α) original', line=dict(color='blue')))
        fig_f.add_trace(go.Scatter(x=alpha_shuf, y=f_alpha_shuf, mode='lines+markers',
                                   name='f(α) shuffled', line=dict(color='orange')))
        fig_f.add_trace(go.Scatter(x=alpha_surrogate, y=f_alpha_surrogate, mode='lines+markers',
                                      name='f(α) surrogate', line=dict(color='green')))
        fig_f.update_layout(title=f'Spectre multifractal f(α): Original vs Shuffled, {tick}',
                            xaxis_title='α', yaxis_title='f(α)', template='plotly_white')
        fig_f.show()

        # Graphique 3 : Différences dans α et f(α)
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Scatter(x=q_list, y=alpha_diff, mode='lines+markers',
                                      name='α - α_shuf', line=dict(color='green')))
        fig_diff.add_trace(go.Scatter(x=q_list, y=f_alpha_diff, mode='lines+markers',
                                      name='f(α) - f(α)_shuf', line=dict(color='red')))
        fig_diff.update_layout(title=f'Differences in α and f(α) between Original and Shuffled {tick}',
                               xaxis_title='q', yaxis_title='Différence', template='plotly_white')
        # fig_diff.show()
    #
    # hq_q = pd.concat([pd.Series(q_list, name='q'), pd.Series(h_q, name='h(q)'), pd.Series(h_q_shuf, name='h(q) shuffled')], axis=1)
    # # hq_q.to_csv(f"{DATA_PATH}/multifractal_spectrum_daily_{name}.csv", index=False)
    # df = pd.DataFrame({
    # 'f_alpha': f_alpha,
    # 'alpha': alpha,
    # 'f_alpha_shuf': f_alpha_shuf,
    # 'alpha_shuf': alpha_shuf,
    # })
        # df.to_csv(f'{DATA_PATH}/f_alpha_alpha_{name}.csv', index=False)

# data_price = pd.read_csv(f"{DATA_PATH}/russel_stocks.csv", index_col=0, parse_dates=True)

# def adf_test(series):
#     result = adfuller(series, autolag='AIC')
#     print('ADF Statistic: %f' % result[0])
#     print('p-value: %f' % result[1])
#     if result[1] < 0.05:
#         print("The series is stationary")
#         return result[1]
#     else:
#         print("The series is not stationary, differencing recommended")
#         return result[1]
#
# # for tik in data_price.columns:
# ticker = '^GSPC'
# data = data_price[ticker]
# # data = data.loc["1987-09-10":"2024-12-31"]
# # Calcul des rendements journaliers en log
# returns = np.log(data).diff().dropna()
# r_m = returns
# # r_m = returns.resample('M').last()
# r_m_shuffle = r_m.sample(frac=1, random_state=42).reset_index(drop=True)
#
# # adf_test(data.dropna())
#
# # adf_test(data)
#
#
# # window_size = 120  # ex. 120 mois (10 ans)
# q_list = np.linspace(-5, 5, 21)
# scales = np.unique(np.floor(np.logspace(np.log10(10), np.log10(2000), num=50)).astype(int))
# # alpha = mfdfa(r_m.values, scales, q_list, order=1)
# # Fq = mfdfa(r_m.values, scales, q_list, order=1)
# # Fq = mfdfa(r_m.values, scales, q_list, order=1)
# Fq = mfdfa(r_m_shuffle.values, scales, q_list)
#
# # --- Estimation des exposants h(q) ---
# # Pour chaque q, ajuster une régression linéaire sur le log-log (log(Fq(s)) vs log(s))
# h_q = []
# log_scales = np.log(scales)
# for j, q in enumerate(q_list):
#     log_Fq = np.log(Fq[j, :])
#     coeffs = np.polyfit(log_scales, log_Fq, 1)
#     h_q.append(coeffs[0])
# h_q = np.array(h_q)
#
# hq_q = pd.concat([pd.Series(q_list, name='q'), pd.Series(h_q, name='h(q)')], axis=1)
# hq_q.to_csv(f"{DATA_PATH}/multifractal_spectrum_daily_rut.csv", index=False)

# Estimation des exposants h(q) pour la série et la version mélangée
# log_scales = np.log(scales)
# h_q = []
# h_q_shuf = []
#
# for i, q in enumerate(q_list):
#     log_Fq = np.log(Fq[i, :])
#     log_Fq_shuf = np.log(Fq_shuf[i, :])
#     slope, _ = np.polyfit(log_scales, log_Fq, 1)
#     slope_shuf, _ = np.polyfit(log_scales, log_Fq_shuf, 1)
#     h_q.append(slope)
#     h_q_shuf.append(slope_shuf)
#
# h_q = np.array(h_q)
# h_q_shuf = np.array(h_q_shuf)

# Calcul du "correcteur" de Hurst dû aux corrélations longues
# h_cor = h_q - h_q_shuf
#
# Affichage du résultat
# plt.figure(figsize=(8, 6))
# plt.plot(q_list, h_cor, marker='o', linestyle='-', color='purple')
# plt.xlabel('Ordre q')
# plt.ylabel(r'$h(q) - h_{\rm shuf}(q)$')
# plt.title('Différence des exposants de Hurst: Original vs Shuffled')
# plt.grid(True)
# plt.show()

# fig_hq = go.Figure()
# fig_hq.add_trace(go.Scatter(
#     x=q_list,
#     y=h_q,
#     mode='lines+markers',
#     marker=dict(size=6),
#     name='h(q)'
# ))
# fig_hq.update_layout(
#     title=f"Multifractal Spectrum h(q) for {ticker} returns",
#     xaxis_title="q",
#     yaxis_title="h(q)",
#     template="plotly_white"
# )
# fig_hq.show()
#
# # --- Calcul du spectre multifractal f(α) via la transformation de Legendre ---
# dq = q_list[1] - q_list[0]
# dh_dq = np.gradient(h_q, dq)
# alpha = h_q + q_list * dh_dq
# f_alpha = q_list * (alpha - h_q) + 1
#
# # --- Tracé du spectre f(α) avec Plotly ---
# fig_falpha = go.Figure()
# fig_falpha.add_trace(go.Scatter(
# x=alpha,
# y=f_alpha,
# mode='lines+markers',
# marker=dict(size=6),
# name='f(α)'
# ))
# fig_falpha.update_layout(
# title=f"Multifractal Spectrum f(α) for {ticker} returns",
# xaxis_title="α",
# yaxis_title="f(α)",
# template="plotly_white"
# )
# fig_falpha.show()
#
# df = pd.DataFrame({
# 'f_alpha': f_alpha,
# 'alpha': alpha
# })

# Sauvegarde au format CSV
# df.to_csv(f'{DATA_PATH}/f_alpha_alpha_daily_rut.csv', index=False)



############### Rolling MF-DFA ####################


    # alpha_width_series = mfdfa_rolling(r_m, window_size, q_list, scales, order=1)

    # rolling_critical = r_m.rolling(window_size).apply(
    # lambda window: ComputeRS.rs_modified_statistic(window, len(window), chin=False)/np.sqrt(len(window)),
    # raw=False
    # ).dropna()
    #
    # rolling_and_prices = pd.concat([np.log(data), rolling_critical], axis=1, join='inner')
    # rolling_and_prices.columns = ['Price', 'Critical Value']
    # rolling_and_prices.to_csv(f"{DATA_PATH}/rolling_and_price.csv")

    # alpha_width_series.index = rolling_critical.index
    #
    # alpha_rolling_price = pd.concat([np.log(data), alpha_width_series, rolling_critical], axis=1, join='inner')
    # alpha_rolling_price.columns = ['Price', 'Alpha Width', 'Critical Value']
    # alpha_rolling_price.index.name = "Date"
    # alpha_rolling_price.to_csv(f"{DATA_PATH}/alpha_rolling_price_daily.csv", index=True)

    # data_rolling = data
    # rolling_series = rolling_critical
    # price_series = data_rolling
    # plot_russell_and_critical_alpha(np.log(price_series), rolling_series, alpha_width_series, threshold=1.6)

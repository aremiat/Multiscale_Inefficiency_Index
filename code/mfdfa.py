import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import os
from utils.MFDFA import ComputeMFDFA
import time
from typing import Sequence, Union
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis
from scipy.stats import jarque_bera


DATA_PATH = os.path.dirname(__file__) + "/../data"

# --- Téléchargement des données et calcul des rendements ---
# Paramètres
q_list = np.linspace(-3, 3, 13)
scales_rut = np.unique(np.floor(np.logspace(np.log10(10), np.log10(500), 10)).astype(int))
scales_gspc = np.unique(np.floor(np.logspace(np.log10(10), np.log10(500), 10)).astype(int))
tickers = ["^GSPC", "^RUT", "^FTSE", "^N225", "^GDAXI"]

if __name__ == "__main__":
    # Lecture des données pour '^RUT'
    np.random.seed(45)

    data_equity = pd.read_csv(os.path.join(DATA_PATH, "index_prices2.csv"), index_col=0, parse_dates=True)
    data_multi_assets = pd.read_csv(os.path.join(DATA_PATH, "multi_assets.csv"), index_col=0, parse_dates=True)
    df_ticker = data_equity.loc["1987-09-10":"2025-02-28"]
    # data_multi_assets = data_multi_assets.loc["2014-09-17":]
    scales = scales_rut

    for tick in tickers[:2]:
        data_tick = df_ticker[tick]
        returns = np.log(data_tick).diff().dropna()
        if tick == '^RUT':
            name = "Russel 2000"
        elif tick == '^GSPC':
            name = "SP500"
        elif tick == '^FTSE':
            name = "FTSE 100"
        elif tick == '^N225':
            name = "Nikkei 225"
        elif tick == '^GDAXI':
            name = "DAX"

    # --- 1. Calcul pour la série originale ---

        Fq = ComputeMFDFA.mfdfa(returns.values, scales, q_list, order=1)
        h_q = []
        log_scales = np.log(scales)
        for i, q in enumerate(q_list):
            log_Fq = np.log(Fq[i, :])
            slope, _ = np.polyfit(log_scales, log_Fq, 1)
            h_q.append(slope)
        h_q = np.array(h_q)
        alpha, f_alpha = ComputeMFDFA.compute_alpha_falpha(q_list, h_q)

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

        surogate_series = ComputeMFDFA.surrogate_gaussian_corr(returns.values)
        surogate_series = pd.Series(surogate_series, index=returns.index)
        surogate_series.name = name
        stat, p_value = jarque_bera(surogate_series)
        print(f"Jarque-Bera test for {name}: stat={stat}, p-value={p_value}")
        Fq_surrogate = ComputeMFDFA.mfdfa(surogate_series, scales, q_list, order=1)
        h_q_surrogate = []
        for i, q in enumerate(q_list):
            log_Fq_surrogate = np.log(Fq_surrogate[i, :])
            slope_surrogate, _ = np.polyfit(log_scales, log_Fq_surrogate, 1)
            h_q_surrogate.append(slope_surrogate)
        h_q_surrogate = np.array(h_q_surrogate)
        alpha_surrogate, f_alpha_surrogate = ComputeMFDFA.compute_alpha_falpha(q_list, h_q_surrogate)

        # # --- 2. Calcul pour la série mélangée (shuffle) ---
        returns_shuf = returns.sample(frac=1, random_state=42).reset_index(drop=True)
        Fq_shuf = ComputeMFDFA.mfdfa(returns_shuf.values, scales, q_list, order=1)
        h_q_shuf = []
        for i, q in enumerate(q_list):
            log_Fq_shuf = np.log(Fq_shuf[i, :])
            slope_shuf, _ = np.polyfit(log_scales, log_Fq_shuf, 1)
            h_q_shuf.append(slope_shuf)
        h_q_shuf = np.array(h_q_shuf)
        alpha_shuf, f_alpha_shuf = ComputeMFDFA.compute_alpha_falpha(q_list, h_q_shuf)

        alpha_width = alpha_shuf.max() - alpha_shuf.min()
        

        # --- 3. Calcul du correcteur de Hurst dû aux corrélations ---
        h_cor = h_q - h_q_shuf

        # Calcul des différences pour alpha et f(alpha)
        alpha_diff = alpha - alpha_shuf
        f_alpha_diff = f_alpha - f_alpha_shuf


        # Graphique 1 : Exposant de Hurst
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=q_list, y=h_q, mode='lines+markers',
                                   name='h(q) original', line=dict(color='blue')))
        fig_h.update_layout(title=f'Hurst Exponent h(q): Original vs Shuffled {name}',
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
        fig_f.update_layout(title=f'Spectre multifractal f(α): Original vs Shuffled, {name}',
                            xaxis_title='α', yaxis_title='f(α)', template='plotly_white')
        fig_f.show()

    #
        # hq_q = pd.concat([pd.Series(q_list, name='q'), pd.Series(h_q, name='h(q)'), pd.Series(h_q_shuf, name='h(q) shuffled')], axis=1)
        # hq_q.to_csv(f"{DATA_PATH}/multifractal_spectrum_daily_{name}.csv", index=False)
        df = pd.DataFrame({
        'f_alpha': f_alpha,
        'alpha': alpha,
        'f_alpha_shuf': f_alpha_shuf,
        'alpha_shuf': alpha_shuf,
        'f_alpha_surrogate': f_alpha_surrogate,
        'alpha_surrogate': alpha_surrogate,
        })
        df.to_csv(f'{DATA_PATH}/f_alpha_alpha_{name}.csv', index=False)

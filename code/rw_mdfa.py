import os.path

import numpy as np
import plotly.graph_objects as go
import os
import plotly.io as pio
from utils.MFDFA import ComputeMFDFA
import matplotlib as plt
import pandas as pd
from utils.ADF import adf_test



if __name__ == "__main__":
    IMG_PATH = os.path.dirname(__file__) + "/../img/"
    DATA_PATH = os.path.dirname(__file__) + "/../data/"
    # Generate a random walk and compute its cumulative profile
    np.random.seed(42)
    N = 10000
    increments = np.random.normal(0, 1, N)
    random_walk = np.cumsum(increments)
    # Use the differences (returns)
    returns = np.diff(random_walk)
    returns_centered = returns - np.mean(returns)
    Y = np.cumsum(returns_centered)
    q_list = np.linspace(-5, 5, 21)
    scales= np.unique(np.floor(np.logspace(np.log10(10), np.log10(500), 10)).astype(int))
    ticker = ['^RUT']
    data = pd.read_csv(os.path.join(DATA_PATH, "russel_stocks.csv"), index_col=0, parse_dates=True)[ticker]
    df_ticker = data.loc["1987-09-10":"2025-02-28"]
    df_ticker = df_ticker.dropna()
    returns = np.log(df_ticker).diff().dropna()
    # Compute the cumulative profile
    returns_centered = returns - np.mean(returns)
    Y = np.cumsum(returns_centered)
    # Normalize the returns
    Y = Y.values.flatten()
    # adf_test(Y)
    # Parameter: segment size (scale)
    s_example = 880
    n_segments = len(returns) // s_example

    dates = pd.date_range(start="1987-09-10", periods=len(returns), freq='B')

    # Create the Plotly figure
    fig = go.Figure()

    # Plot the cumulative profile
    fig.add_trace(go.Scatter(
        x=dates,
        y=Y,
        mode='lines',
        name='Cumulative Profile'
    ))

    # Cumulative profile
    Y = pd.Series(Y, index=dates)

    # For each segment, plot the segment boundaries and the linear fit
    for v in range(n_segments):
        start = v * s_example
        end = (v + 1) * s_example
        segment = Y.iloc[start:end]
        idx = segment.index

        # Fit linéaire
        coeffs = np.polyfit(np.arange(s_example), segment.values, 1)
        fit = np.polyval(coeffs, np.arange(s_example))

        # Tracé du fit
        fig.add_trace(go.Scatter(
            x=idx,
            y=fit + np.mean(returns),  # facultatif selon ce que tu veux montrer
            mode='lines',
            line=dict(dash='dash', width=5, color='black'),
            name=f"Segment {v}"
        ))

        # Ajout des bornes de segment avec dates
        fig.add_vline(x=idx[0], line=dict(color='gray', dash='dot'))
        fig.add_vline(x=idx[-1], line=dict(color='gray', dash='dot'))

    fig.update_layout(
        title=f"Cumulative Profile with Segment Partitioning (segment size = {s_example}) and Trend Removal, Random Walk",
        xaxis_title="Index",
        yaxis_title="Cumulative Profile"
    )
    fig.show()

    # for i in range(1, 2):
    #
    #     q_list = np.linspace(-3, 3, 13)
    #     scales = np.unique(np.logspace(np.log10(10), np.log10(1000), 10, dtype=int))
    #     order = 1
    #     # Compute MFDFA
    #     increments = np.random.randn(10000)
    #     x = np.cumsum(increments)
    #     df = pd.DataFrame(x, columns=['Price'])
    #     # returns = np.log(df['Price']).diff().dropna() / 100
    #     Fq = ComputeMFDFA.mfdfa(returns, scales, q_list, order=order)
    #
    #     h_q = []
    #     log_scales = np.log(scales)
    #     for i, q in enumerate(q_list):
    #         log_Fq = np.log(Fq[i, :])
    #         slope, _ = np.polyfit(log_scales, log_Fq, 1)
    #         h_q.append(slope)
    #     h_q = np.array(h_q)
    #     alpha, f_alpha = ComputeMFDFA.compute_alpha_falpha(q_list, h_q)
    #
    #     fig_f = go.Figure()
    #     fig_f.add_trace(go.Scatter(x=alpha, y=f_alpha, mode='lines+markers',
    #                                name='f(α) original', line=dict(color='blue')))
    #     fig_f.update_layout(title=f'Spectre multifractal f(α): Original vs Shuffled, random walk',
    #                         xaxis_title='α', yaxis_title='f(α)', template='plotly_white')
    #     # fig_f.show()
    #
    # # save the results of alpha and f_alpha
    # concat_results = pd.concat([pd.DataFrame(alpha), pd.DataFrame(f_alpha)], axis=1)
    # concat_results.columns = ['alpha', 'f_alpha']
    # # concat_results.to_csv(os.path.join(DATA_PATH, 'f_alpha_alpha_random_walk.csv'), index=False)
    #
    # returns = pd.DataFrame(returns, columns=['Returns'])
    # returns_shuf = returns.sample(frac=1, random_state=42).reset_index(drop=True)
    # Fq_shuf = ComputeMFDFA.mfdfa(returns_shuf.values, scales, q_list, order=1)
    # h_q_shuf = []
    # for i, q in enumerate(q_list):
    #     log_Fq_shuf = np.log(Fq_shuf[i, :])
    #     slope_shuf, _ = np.polyfit(log_scales, log_Fq_shuf, 1)
    #     h_q_shuf.append(slope_shuf)
    # h_q_shuf = np.array(h_q_shuf)
    # alpha_shuf, f_alpha_shuf = ComputeMFDFA.compute_alpha_falpha(q_list, h_q_shuf)
    #
    # fig_f = go.Figure()
    # fig_f.add_trace(go.Scatter(x=alpha, y=f_alpha, mode='lines+markers',
    #                            name='f(α) original', line=dict(color='blue')))
    # fig_f.add_trace(go.Scatter(x=alpha_shuf, y=f_alpha_shuf, mode='lines+markers',
    #                            name='f(α) shuffled', line=dict(color='orange')))
    # fig_f.update_layout(title=f'Spectre multifractal f(α): Original vs Shuffled, random walk',
    #                     xaxis_title='α', yaxis_title='f(α)', template='plotly_white')
    # fig_f.show()


    # pio.write_image(fig, f"{IMG_PATH}/fdm_autocorr.png", scale=5, width=1200, height=1000)
    # fig.write_image(f"{IMG_PATH}cumulative_profile_segment_partitioning.png")

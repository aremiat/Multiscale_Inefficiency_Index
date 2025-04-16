import os.path

import numpy as np
import plotly.graph_objects as go
import os
import plotly.io as pio
from utils.MFDFA import ComputeMFDFA
import fathon
from fathon import fathonUtils as fu
import matplotlib as plt
import pandas as pd
from utils.ADF import adf_test



if __name__ == "__main__":
    IMG_PATH = os.path.dirname(__file__) + "/../img/"
    DATA_PATH = os.path.dirname(__file__) + "/../data/"
    # Generate a random walk and compute its cumulative profile
    np.random.seed(42)
    N = 5000
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
    adf_test(Y)
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
            line=dict(dash='dash'),
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

    # q_list = np.linspace(-5, 5, 21)
    # scales = np.unique(np.logspace(np.log10(10), np.log10(900), 10, dtype=int))
    # order = 1
    # # Compute MFDFA
    # Fq = ComputeMFDFA.mfdfa(random_walk, scales, q_list, order=order)
    #
    # h_q = []
    # log_scales = np.log(scales)
    # for i, q in enumerate(q_list):
    #     log_Fq = np.log(Fq[i, :])
    #     slope, _ = np.polyfit(log_scales, log_Fq, 1) - 1
    #     h_q.append(slope)
    # h_q = np.array(h_q)
    # alpha, f_alpha = ComputeMFDFA.compute_alpha_falpha(q_list, h_q)
    #
    # fig_h = go.Figure()
    # fig_h.add_trace(go.Scatter(x=q_list, y=h_q, mode='lines+markers',
    #                            name='h(q) original', line=dict(color='blue')))
    # # fig_h.add_trace(go.Scatter(x=q_list, y=h_q_shuf, mode='lines+markers',
    # #                            name='h(q) shuffled', line=dict(color='orange')))
    #
    # fig_h.show()

    # pio.write_image(fig, f"{IMG_PATH}/fdm_autocorr.png", scale=5, width=1200, height=1000)
    # fig.write_image(f"{IMG_PATH}cumulative_profile_segment_partitioning.png")

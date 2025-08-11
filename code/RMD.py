import numpy as np
import plotly.graph_objects as go
from utils.MFDFA import ComputeMFDFA
from utils.ADF import adf_test
from scipy.stats import skew, kurtosis

# Random Midpoint Displacement
def random_midpoint_displacement(num_points, roughness=0.5, seed=43):
    """Generates a random midpoint displacement fractal curve."""
    if seed is not None:
        np.random.seed(seed)
    n = num_points - 1
    if not (n & (n - 1) == 0):
        raise ValueError("num_points should be equal to 2^k + 1")
    y = np.zeros(num_points)
    x = np.linspace(0, 1, num_points)
    scale = 1.0
    step = n
    while step > 1:
        half = step // 2
        for i in range(0, num_points - 1, step):
            mid = i + half
            avg = 0.5 * (y[i] + y[i + step])
            disp = np.random.randn() * scale
            y[mid] = avg + disp
        scale *= 2 ** (-roughness)
        step = half
    return x, y

num_points = 2**16 + 1
scales = np.unique(np.floor(np.logspace(np.log10(10), np.log10(2000), 20)).astype(int))
q_list = np.linspace(-5, 5, 21)

if __name__ == '__main__':
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 250)

    x, y = random_midpoint_displacement(num_points, roughness=0.7, seed=42)
    increments = np.diff(y)
    adf_test(y)
    print(skew(increments))
    print(kurtosis(increments))

    Fq = ComputeMFDFA.mfdfa(increments, scales, q_list, order=1)
    log_s = np.log(scales)
    h_q = np.array([(np.polyfit(log_s, np.log(Fq[j]), 1)[0]) for j in range(len(q_list))])
    alpha, f_alpha = ComputeMFDFA.compute_alpha_falpha(q_list, h_q)


    fig1 = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', line=dict(color='purple')))
    fig1.update_layout(title="Random Midpoint Displacement (H=0.7)", xaxis_title="Normalized Position", yaxis_title="Height")
    fig2 = go.Figure(data=go.Scatter(x=q_list, y=h_q, mode='markers+lines', marker=dict(color='blue')))
    fig2.update_layout(title="Exposant généralisé h(q) vs q", xaxis_title="q", yaxis_title="h(q)")
    fig3 = go.Figure(data=go.Scatter(x=alpha, y=f_alpha, mode='markers+lines', marker=dict(color='orange')))
    fig3.update_layout(title="Spectre multifractal f(α) vs α", xaxis_title="α", yaxis_title="f(α)")
    fig3.show()
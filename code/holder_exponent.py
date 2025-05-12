import numpy as np
import plotly.graph_objects as go

# Simulate a random walk time series
np.random.seed(42)
N = 1001
increments = np.random.normal(loc=0, scale=1, size=N-1)
series = np.concatenate([[0], np.cumsum(increments)])

# Time axis centered at t0
t = np.arange(N) - (N // 2)
f0 = series[N // 2]

# Compute envelope curves for different alphas
alphas = [0.2, 0.4, 0.6]
envelopes = {}
for alpha in alphas:
    mask = t != 0
    C = np.max(np.abs(series[mask] - f0) / (np.abs(t[mask])**alpha))
    envelopes[alpha] = C * np.abs(t)**alpha

# Create interactive Plotly figure
fig = go.Figure()

# Random walk trace
fig.add_trace(go.Scatter(
    x=t, y=series,
    mode='lines',
    name='Random Walk',
    line=dict(color='black', width=1)
))

# Envelope curves for each alpha
colors = ['red', 'green', 'blue']
for alpha, color in zip(alphas, colors):
    fig.add_trace(go.Scatter(
        x=t, y=envelopes[alpha],
        mode='lines',
        name=f'α = {alpha}',
        line=dict(color=color)
    ))
    fig.add_trace(go.Scatter(
        x=t, y=-envelopes[alpha],
        mode='lines',
        showlegend=False,
        line=dict(color=color)
    ))

# Vertical line at t0
fig.add_vline(x=0, line_dash='dash', line_color='black')

# Layout customization
fig.update_layout(
    title='Illustration of Local Hölder Exponents',
    xaxis_title='t - t₀',
    yaxis_title='f(t)',
    legend=dict(title='Envelopes')
)

# Show the figure
fig.show()

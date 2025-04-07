import os.path

import numpy as np
import plotly.graph_objects as go
import os

if __name__ == "__main__":
    IMG_PATH = os.path.dirname(__file__) + "/../img/"
    # Generate a random walk and compute its cumulative profile
    np.random.seed(42)
    N = 5000
    increments = np.random.normal(0, 1, N)
    random_walk = np.cumsum(increments)
    # Use the differences (returns)
    returns = np.diff(random_walk)
    returns_centered = returns - np.mean(returns)
    Y = np.cumsum(returns_centered)

    # Parameter: segment size (scale)
    s_example = 990
    n_segments = len(Y) // s_example

    # Create the Plotly figure
    fig = go.Figure()

    # Plot the cumulative profile
    fig.add_trace(go.Scatter(
        x=np.arange(len(Y)),
        y=Y,
        mode='lines',
        name='Cumulative Profile'
    ))

    # For each segment, plot the segment boundaries and the linear fit
    for v in range(n_segments):
        # Segment indices
        start = v * s_example
        end = (v + 1) * s_example
        idx = np.arange(start, end)
        segment = Y[start:end]

        # Linear fit (polynomial of order 1)
        coeffs = np.polyfit(np.arange(s_example), segment, 1)
        fit = np.polyval(coeffs, np.arange(s_example))

        fig.add_trace(go.Scatter(
            x=idx,
            y=fit + np.mean(returns),
            mode='lines',
            line=dict(dash='dash')
        ))


        # Add a vertical line to mark the beginning of the segment
        fig.add_vline(x=start, line=dict(color='gray', dash='dot'))
        # Add a vertical line to mark the end of the segment
        fig.add_vline(x=end, line=dict(color='gray', dash='dot'))

    fig.update_layout(
        title=f"Cumulative Profile with Segment Partitioning (segment size = {s_example}) and Trend Removal, Random Walk",
        xaxis_title="Index",
        yaxis_title="Cumulative Profile"
    )
    fig.show()
    fig.write_image(f"{IMG_PATH}cumulative_profile_segment_partitioning.png")

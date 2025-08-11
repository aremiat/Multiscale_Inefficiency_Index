import numpy as np
import pandas as pd


def simulate_msm(K, gamma1, b, T=10_000):
    """
    Compute the average duration of each level in a Markov Switching Model (MSM)

    Args:
        K (int): Number of levels in the MSM.
        gamma1 (float): Probability of switching from level 1 to higher levels.
        b (float): Base value for the switching probabilities.
        T (int): Total number of time steps to simulate.
    Returns:
        list: Average duration for each level in the MSM.
    """
    gamma_list = []
    for k in range(1, K + 1):
        val = 1.0 - (1.0 - gamma1) * (b ** (k - 1))
        gamma_list.append(val)

    states = np.random.randint(0, 2, size=K)

    duration_counts = np.zeros(K)
    switch_counts = np.zeros(K)

    current_duration = np.zeros(K)

    for t in range(T):
        for k in range(K):
            current_duration[k] += 1
            if np.random.rand() < gamma_list[k]:
                duration_counts[k] += current_duration[k]
                switch_counts[k] += 1
                current_duration[k] = 0
                states[k] = 1 - states[k]

    duration_counts += current_duration

    avg_duration_levels = []
    for k in range(K):
        if switch_counts[k] > 0:
            avg_duration = duration_counts[k] / switch_counts[k]
        else:
            avg_duration = T
        avg_duration_levels.append(avg_duration)

    return avg_duration_levels


if __name__ == "__main__":
    np.random.seed(43)
    K = 3
    gamma1 = 0.01
    T = 10_000

    b_values = [0.2, 0.5, 0.8, 0.95, 0.99, 1, 5, 10, 20, 100]
    results = []

    for b in b_values:
        avg_durations = simulate_msm(K, gamma1, b, T)
        results.append({
            'b': b,
            'avg_duration_level_1': avg_durations[0],
            'avg_duration_level_2': avg_durations[1],
            'avg_duration_level_3': avg_durations[2]
        })

    df_results = pd.DataFrame(results)
    print(df_results)

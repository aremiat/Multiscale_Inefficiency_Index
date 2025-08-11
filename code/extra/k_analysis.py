import numpy as np
import matplotlib.pyplot as plt


def simulate_msm_binomial(T, K, m0, gamma_list):
    """
    Simulate a Markov Switching Model (MSM) using a binomial tree approach.

    Args:
        T (int): Number of time steps.
        K (int): Number of levels in the MSM.
        m0 (float): Initial multiplier value.
        gamma_list (list): List of probabilities for switching between multipliers.
    Returns:
        M (np.ndarray): Array of multipliers at each time step for each level.
        vol (np.ndarray): Volatility calculated as the product of multipliers at each time step.
    """
    m_high = 2.0 - m0

    M = np.zeros((T, K))
    M[0, :] = m0

    for t in range(1, T):
        for k in range(K):
            if np.random.rand() < gamma_list[k]:
                if M[t - 1, k] == m0:
                    M[t, k] = m_high
                else:
                    M[t, k] = m0
            else:
                M[t, k] = M[t - 1, k]

    vol = np.prod(M, axis=1)

    return M, vol


if __name__ == "__main__":
    np.random.seed(43)

    T = 200
    K = 5
    m0 = 0.8
    m_high = 2 - m0

    for k in range(1, K + 1):
        gamma_list = [1 - (1 - 0.1) ** (2 ** (k - 1)) for k in range(1, K + 1)]


    M, vol = simulate_msm_binomial(T, K, m0, gamma_list)

    fig, axes = plt.subplots(K + 1, 1, figsize=(8, 8), sharex=True)

    time = np.arange(T)

    for k in range(K):
        axes[k].plot(time, M[:, k], label=f"level {k+1}", color='black')
        axes[k].legend(loc="upper right")
        axes[k].set_ylabel("Value")
        axes[k].grid(True)

    axes[K].plot(time, vol, label="Volatility (product of multipliers)", color='black')
    axes[K].legend(loc="upper right")
    axes[K].set_ylabel("Volatility")

    axes[K].set_xlabel("Time")
    plt.suptitle(f"MSM Binomial Simulation (K={K}, m0={m0}, high={m_high}, gamma={np.round(gamma_list,2)})")
    plt.tight_layout()
    plt.show()

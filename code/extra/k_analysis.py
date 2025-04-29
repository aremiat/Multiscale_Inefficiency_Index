import numpy as np
import matplotlib.pyplot as plt


def simulate_msm_binomial(T, K, m0, gamma_list):
    """
    Simule un modèle MSM binomial sur T périodes pour K niveaux.

    Paramètres:
    -----------
    T : int
        Nombre de pas de temps
    K : int
        Nombre de niveaux
    m0 : float
        Valeur bas (et la valeur haut sera 2 - m0)
    gamma_list : array-like (de taille K)
        Probabilité de switch pour chaque niveau (du 1er au K-ième niveau)

    Retourne:
    --------
    M : ndarray (T x K)
        M[t, k] = valeur du multiplicateur au niveau k à l'instant t
    vol : ndarray (T,)
        Volatilité totale (produit des K multiplicateurs) à l'instant t
    """
    # Valeur "haut"
    m_high = 2.0 - m0

    # Initialisation des états pour chaque niveau (on commence arbitrairement en "bas")
    M = np.zeros((T, K))
    M[0, :] = m0  # on commence tout en état "bas"

    for t in range(1, T):
        for k in range(K):
            # Test de switch
            if np.random.rand() < gamma_list[k]:
                # On change d'état
                if M[t - 1, k] == m0:
                    M[t, k] = m_high
                else:
                    M[t, k] = m0
            else:
                # On reste dans le même état
                M[t, k] = M[t - 1, k]

    # Calcul de la volatilité (produit des K multiplicateurs)
    vol = np.prod(M, axis=1)

    return M, vol


if __name__ == "__main__":
    np.random.seed(42)

    T = 200  # nombre de périodes
    K = 5  # nombre de niveaux
    m0 = 0.8  # valeur "bas"
    m_high = 2 - m0  # valeur "haut" = 1.2

    # Probabilité de switch pour chaque niveau (exemple simple):
    # Niveau 1 (rapide) : gamma_1 = 0.3
    # Niveau 2 (moyen) : gamma_2 = 0.1
    # Niveau 3 (lent)  : gamma_3 = 0.05
    for k in range(1, K + 1):
        gamma_list = [1 - (1 - 0.1) ** (2 ** (k - 1)) for k in range(1, K + 1)]


    # Simulation
    M, vol = simulate_msm_binomial(T, K, m0, gamma_list)

    # Affichage des séries pour chaque niveau
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

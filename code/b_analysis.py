import numpy as np
import pandas as pd


def simulate_msm(K, gamma1, b, T=10_000):
    """
    Simule un modèle MSM binomial avec K composantes, probabilité de switch
    gamma_k = 1 - (1 - gamma1)*b^(k-1) pour k=1..K, sur T pas de temps.

    Retourne :
    - avg_duration_levels : liste de taille K contenant la durée moyenne
      passée dans le même état pour chaque composante.
    """
    # Calcul des probabilités de switch pour chaque niveau k
    gamma_list = []
    for k in range(1, K + 1):
        val = 1.0 - (1.0 - gamma1) ** (b ** (k - 1))
        gamma_list.append(val)

    # Initialisation aléatoire des états (0 ou 1) pour chaque niveau
    # Chaque composante k est une chaîne de Markov binaire indépendante
    states = np.random.randint(0, 2, size=K)

    # Compteurs pour calculer les durées moyennes
    # duration_counts[k] va accumuler le total de durées passées dans un état
    # switch_counts[k] va compter le nombre de switches pour la composante k
    duration_counts = np.zeros(K)
    switch_counts = np.zeros(K)

    # On va suivre la "durée courante" depuis le dernier switch pour chaque k
    current_duration = np.zeros(K)

    for t in range(T):
        # Pour chaque composante k, on décide s'il y a switch
        for k in range(K):
            current_duration[k] += 1  # on prolonge la durée depuis le dernier switch
            # Tirage d'un nombre aléatoire pour savoir si on switch
            if np.random.rand() < gamma_list[k]:
                # On a un switch, on met à jour le compteur
                duration_counts[k] += current_duration[k]
                switch_counts[k] += 1
                # Reset de la durée
                current_duration[k] = 0
                # Changement d'état
                states[k] = 1 - states[k]

    # Après la simulation, on ajoute la dernière portion de durée
    # (si un switch n'est pas survenu juste avant la fin)
    duration_counts += current_duration

    # Calcul de la durée moyenne = total des durées / nombre de switches
    # On évite la division par zéro si jamais il n'y a pas eu de switch
    avg_duration_levels = []
    for k in range(K):
        if switch_counts[k] > 0:
            avg_duration = duration_counts[k] / switch_counts[k]
        else:
            # Si aucun switch, la durée moyenne est la longueur totale T
            avg_duration = T
        avg_duration_levels.append(avg_duration)

    return avg_duration_levels


if __name__ == "__main__":
    np.random.seed(123)
    K = 3  # Nombre de composantes
    gamma1 = 0.01
    T = 10_000  # Nombre de pas de temps pour la simulation

    b_values = [0.2, 0.5, 0.8, 0.95, 0.99, 1, 5, 10, 20, 100]  # Quelques valeurs de b
    results = []

    for b in b_values:
        avg_durations = simulate_msm(K, gamma1, b, T)
        results.append({
            'b': b,
            'avg_duration_level_1': avg_durations[0],
            'avg_duration_level_2': avg_durations[1],
            'avg_duration_level_3': avg_durations[2]
        })

    # Conversion en DataFrame pour un affichage propre
    df_results = pd.DataFrame(results)
    print(df_results)

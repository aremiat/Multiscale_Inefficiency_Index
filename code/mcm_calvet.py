import os.path

import numpy as np
import pandas as pd
from arch import arch_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from aeqlib.quantlib import volatility


###############################################################################
# 1) Construction de l'espace d'états, de la matrice de transition, etc.
###############################################################################

def build_state_space(K, m0):
    """
    Construit l'espace d'états pour K composantes (chacune pouvant être m0 ou 2 - m0).
    Retourne :
      - states : liste de tuples (ex: (m0, 2-m0, m0)) représentant chaque état
      - product_cache : dict qui associe l'index de l'état au produit des composantes
        (pour accélérer les calculs de densité)
    """
    # Pour K composantes binaires, on a 2^K états.
    # On génère toutes les combinaisons : 0 => m0, 1 => 2-m0
    d = 2 ** K # Nombre total d'états
    states = []
    product_cache = np.zeros(d)

    for i in range(d):
        # ex: i=3 => binaire = '011' => état = (m0, 2-m0, 2-m0) si on lit de gauche à droite
        bits = np.binary_repr(i, width=K)
        mults = []
        for bit in bits:
            if bit == '0':
                mults.append(m0)
            else:
                mults.append(2.0 - m0)
        states.append(tuple(mults))
        product_cache[i] = np.prod(mults)

    return states, product_cache


def build_transition_matrix(states, K, gamma_list):
    """
    Construit la matrice de transition globale A pour le vecteur M_t = (M_{1,t},...,M_{K,t}).
    On suppose que chaque composante k peut 'switcher' à la valeur alternative
    avec prob = gamma_list[k], et rester inchangée avec prob = 1-gamma_list[k].

    states : liste de tuples (ex: (m0, 2-m0, m0)) représentant les états
    K : nombre de composantes
    gamma_list : liste des gamma_k, k=1..K

    Retourne :
      - A : matrice de taille d x d (d=2^K)
    """
    d = len(states)
    A = np.zeros((d, d))

    # Pour chaque état courant i, on détermine la probabilité de passer à l'état j
    # en considérant chaque composante k = 0..K-1
    # Si composante k change => la valeur m0 <-> (2-m0)
    # Probability = prod_{k=1..K} [gamma_k si on change, (1-gamma_k) si on reste]
    # Mais on doit identifier la composante k qui diffère entre i et j
    # => on peut comparer bit par bit
    for i in range(d):
        s_i = states[i]
        for j in range(d):
            s_j = states[j]
            prob = 1.0
            # Pour chaque composante k
            for k in range(K):
                if s_i[k] == s_j[k]:
                    # composante identique => pas de switch => prob = (1-gamma_k)
                    prob *= (1.0 - gamma_list[k])
                else:
                    # composante différente => switch => prob = gamma_k
                    # On vérifie que c'est bien le bon switch (m0 <-> 2-m0)
                    # si s_i[k] = m0, s_j[k] doit être 2-m0, etc.
                    # s_i[k] + s_j[k] doit valoir 2 => (m0 + (2-m0) = 2)
                    if abs(s_i[k] + s_j[k] - 2.0) < 1e-10:
                        prob *= gamma_list[k]
                    else:
                        # si on tombe ici, c'est que c'est un switch invalide
                        prob = 0.0
                        break
            A[i, j] = prob
        # Normalisation par la somme, pour s'assurer que la ligne i somme à 1
        row_sum = A[i, :].sum()
        if row_sum > 1e-14:
            A[i, :] /= row_sum

    return A


###############################################################################
# 2) Filtrage et log-vraisemblance
###############################################################################

def msm_loglik(data, sigma, states, product_cache, A):
    """
    Calcule la log-vraisemblance du modèle MSM (binomial) pour la série 'data' (rendements).

    Hypothèse : x_t = sigma * sqrt( product(M_{k,t}) ) * eps_t, eps_t ~ N(0,1)

    data : array-like, rendements (T x 1)
    sigma : float
    states : liste de tuples
    product_cache : array de la taille d (d=2^K) donnant le produit des composantes
    A : matrice de transition d x d

    Retourne : log_likelihood (scalaire)
    """
    T = len(data)
    d = len(states)

    # Probabilité stationnaire (ergodique) comme vecteur initial
    # On peut la trouver en résolvant pi * A = pi.
    # Pour la simplicité, on prend la première valeur propre ou on fait un vecteur stationnaire approx
    eigvals, eigvecs = np.linalg.eig(A.T) # A.T car on veut les vecteurs propres à gauche
    # stationnaire = vecteur propre de valeur propre = 1
    idx_stat = np.argmin(np.abs(eigvals - 1.0)) # indice de la valeur propre la plus proche de 1
    pi0 = eigvecs[:, idx_stat].real # vecteur propre associé
    pi0 = pi0 / pi0.sum() # normalisation

    # Filtre : pi_t (1 x d)
    pi_t = pi0

    logL = 0.0

    for t in range(T):
        # Calcul de la densité conditionnelle f(x_t | M_t = s_j)
        # f(x_t) = 1/(sigma * sqrt(prod)) * N( x_t / [sigma * sqrt(prod}] )
        # => On calcule tout d'un coup
        fx = np.zeros(d)
        for j in range(d):
            vol = sigma * np.sqrt(product_cache[j])
            if vol < 1e-14:
                fx[j] = 0.0
            else:
                z = data[t] / vol # z = x_t / [sigma * sqrt(prod)]
                fx[j] = (1.0 / vol) * (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z ** 2) # densité

        # Mise à jour pi_{t+1} = [ pi_t * A ] . f(x_t) / ...
        # On fait pi_t A => shape (1 x d)
        piA = pi_t @ A
        # piA * fx => shape (1 x d) (Hadamard product)
        numer = piA * fx # pi_t * A . f(x_t)
        denom = numer.sum()
        if denom < 1e-14:
            # risque underflow => on peut stopper ou log(1e-300) ...
            logL += np.log(1e-300)
            # reinit
            pi_t = pi0
        else:
            logL += np.log(denom)
            pi_t = numer / denom

    return logL


###############################################################################
# 3) Exemple d'estimation par "grid search" sur (m0, gamma_1)
###############################################################################

def estimate_msm_binomial(
        data, K=3, b=2.0,  # param b pour gamma_k = 1 - (1-gamma_1)*b^(k-1)
        m0_grid=None, gamma1_grid=None, sigma_grid=None
):
    """
    Exemple d'estimation brute par grille 2D ou 3D (m0, gamma1, sigma).
    On fixe K et b, on balaie m0_grid, gamma1_grid, sigma_grid.

    Renvoie le meilleur paramètre (m0*, gamma1*, sigma*) et la logLik max.
    """
    if m0_grid is None:
        m0_grid = np.linspace(0.6, 1.4, 5)  # ex
    if gamma1_grid is None:
        gamma1_grid = np.linspace(0.1, 0.9, 5)
    if sigma_grid is None:
        sigma_grid = np.linspace(0.5, 2.0, 5)

    # Calcul du nombre total d'itérations
    total_iterations = len(m0_grid) * len(gamma1_grid) * len(sigma_grid)
    current_iter = 0

    best_logL = -1e15
    best_params = (None, None, None)

    data_arr = np.array(data)

    for m0 in m0_grid:
        states, product_cache = build_state_space(K, m0)
        for gamma1 in gamma1_grid:
            # Construction de gamma_k
            gamma_list = []
            for k in range(1, K + 1):
                gamma_k = 1.0 - (1.0 - gamma1) ** (b ** (k - 1))
                gamma_list.append(gamma_k)
            # Matrice de transition
            A = build_transition_matrix(states, K, gamma_list)
            for sigma in sigma_grid:
                # Incrémenter le compteur et afficher la progression
                current_iter += 1
                progress = 100.0 * current_iter / total_iterations
                print(f"Progress: {progress:.2f}% completed", end="\r")

                # Calcul de la log-vraisemblance
                ll = msm_loglik(data_arr, sigma, states, product_cache, A)
                if ll > best_logL:
                    best_logL = ll
                    best_params = (m0, gamma1, sigma)

    # Pour revenir à la ligne après la boucle
    print()
    return best_params, best_logL


def forward_filter(data, sigma_opt, states, product_cache, A):
    T = len(data)
    d = len(states)
    eigvals, eigvecs = np.linalg.eig(A.T)
    idx_stat = np.argmin(np.abs(eigvals - 1.0))
    pi0 = eigvecs[:, idx_stat].real
    pi0 = pi0 / pi0.sum()

    pi_t = pi0
    pi_history = np.zeros((T, d))
    loglik = 0.0
    for t in range(T):
        fx = np.zeros(d)
        for j in range(d):
            vol_j = sigma_opt * np.sqrt(product_cache[j])
            z = data[t] / vol_j
            fx[j] = (1.0 / vol_j) * (1.0 / np.sqrt(2*np.pi)) * np.exp(-0.5 * z**2)
        piA = pi_t @ A
        numer = piA * fx
        denom = numer.sum()
        if denom < 1e-14:
            denom = 1e-14
        loglik += np.log(denom)
        pi_t = numer / denom
        pi_history[t, :] = pi_t
    return pi_history, loglik

def fitted_volatility(pi_history, sigma_opt, product_cache):
    T, d = pi_history.shape
    vol_fit = np.zeros(T)
    for t in range(T):
        exp_var_t = sigma_opt**2 * np.sum(pi_history[t, :] * product_cache)
        vol_fit[t] = np.sqrt(exp_var_t)
    return vol_fit

###############################################################################
# 4) Exemple d'utilisation sur des données fictives
###############################################################################

if __name__ == "__main__":
    np.random.seed(1234)
    K = 5
    b = 2

    DATA_PATH = os.path.dirname(__file__) + "/../data"
    data_df = pd.read_csv(f"{DATA_PATH}/russell_2000.csv", index_col=0)

    # Suppose qu'il y a une colonne '^RUT' contenant les prix => calcul rendements
    returns = data_df['^RUT'].pct_change().dropna().values

    # 1) Split train/test
    T = len(returns)
    T_train = int(0.8 * T)
    train_data = returns[:T_train]
    test_data = returns[T_train:]

    # 2) Estimation sur l'échantillon d'entraînement
    m0_grid = np.linspace(0.1, 1.9, 20)
    gamma1_grid = np.linspace(0.01, 0.99, 20)
    sigma_grid = np.linspace(0.001, 0.1, 20)

    best_params, best_logL = estimate_msm_binomial(train_data,
                                                   K=K, b=b,
                                                   m0_grid=m0_grid,
                                                   gamma1_grid=gamma1_grid,
                                                   sigma_grid=sigma_grid)
    m0_opt, gamma1_opt, sigma_opt = best_params
    print("Best param (m0, gamma1, sigma) =", best_params, "LogLik =", best_logL)

    # AIC / BIC sur training
    n_train = len(train_data)
    k_params = 3
    AIC = 2 * k_params - 2 * best_logL
    BIC = k_params * np.log(n_train) - 2 * best_logL
    print(f"AIC (train) = {AIC:.2f}")
    print(f"BIC (train) = {BIC:.2f}")

    # 3) Construction de l'espace d'états + matrice de transition
    states, product_cache = build_state_space(K, m0_opt)
    gamma_list_opt = []
    for k_ in range(1, K+1):
        val = 1.0 - (1.0 - gamma1_opt)**(b**(k_-1))
        gamma_list_opt.append(max(0.0, min(1.0, val)))
    A = build_transition_matrix(states, K, gamma_list_opt)

    # 4) Filtrage in-sample sur le training pour récupérer pi_{T_train}
    pi_history_train, loglik_in_sample = forward_filter(train_data, sigma_opt, states, product_cache, A)
    pi_last_train = pi_history_train[-1, :]  # distribution à la fin du training

    # 5) Volatilité fittée in-sample
    vol_fit_train = fitted_volatility(pi_history_train, sigma_opt, product_cache)

    # 6) Projection out-of-sample
    T_test = len(test_data)
    pi_history_test = np.zeros((T_test, len(states)))
    vol_forecast = np.zeros(T_test)

    pi_t = pi_last_train.copy()  # On part de la distribution de fin de training
    for t in range(T_test):
        # 6.1) On fait un forecast de la distribution à t+1 => pi_{t+1} = pi_t @ A
        pi_pred  = pi_t @ A

        expected_var = sigma_opt**2 * np.sum(pi_pred * product_cache)
        vol_forecast[t] = np.sqrt(expected_var)

        # 6.3) Incorporation de l’observation test_data[t] pour réajuster pi_t
        fx = np.zeros(len(states))
        for j in range(len(states)):
            vol_j = sigma_opt * np.sqrt(product_cache[j])
            if vol_j < 1e-14:
                fx[j] = 1e-300
            else:
                z = test_data[t] / vol_j
                fx[j] = (1.0 / vol_j) * (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z ** 2)

        numer = pi_pred * fx
        denom = numer.sum()
        if denom < 1e-14:
            denom = 1e-14
        pi_t = numer / denom

        # Stocker la distribution filtrée (post-observation)
        pi_history_test[t, :] = pi_t

    # 7) Calcul de la volatilité réalisée sur la période test
    #    Par exemple, rolling 21 jours, ou l'écart-type sur x jours. Ici, on fait 21 jours :
    realized_vol_test = pd.Series(test_data).rolling(window=21).std().values
    realized_vol_test = np.where(np.isnan(realized_vol_test), 0.0, realized_vol_test)

    # 8) Mesure de la qualité de prévision sur la période de test
    #    On tronque vol_forecast pour aligner avec realized_vol_test décalée ?
    #    Ici, on va juste comparer sur les points "valides".
    valid_idx = np.where(realized_vol_test > 0)[0]  # indices où on a un calcul de vol
    vol_fore_oos = vol_forecast[valid_idx]
    vol_real_oos = realized_vol_test[valid_idx]

    rmse_oos = np.sqrt(np.mean((vol_fore_oos - vol_real_oos) ** 2))
    corr_oos = np.corrcoef(vol_fore_oos, vol_real_oos)[0, 1]
    print(f"Out-of-sample RMSE = {rmse_oos:.5f}")
    print(f"Out-of-sample Corr  = {corr_oos:.3f}")

    # 9) Visualisation Plotly
    fig = make_subplots(rows=1, cols=1, subplot_titles=["Out-of-sample forecast (MSM) vs. Realized Vol"])
    # Index de test => pour l'affichage, on va juste mettre un range simple
    test_range = np.arange(T_test)

    fig.add_trace(go.Scatter(
        x=test_range,
        y=vol_forecast,
        mode='lines',
        name='Forecasted MSM Vol'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=test_range,
        y=realized_vol_test,
        mode='lines',
        name='Realized Vol (21d rolling)'
    ), row=1, col=1)

    fig.update_layout(
        title=f"Out-of-sample MSM Vol Forecast vs. Realized Vol\n(RMSE={rmse_oos:.4f}, Corr={corr_oos:.3f})",
        xaxis_title="Test sample index",
        yaxis_title="Volatility"
    )
    fig.show()

    # Best param (m0, gamma1, sigma) = (0.5736842105263157, 0.01, 0.011421052631578946) LogLik = 23826.9989722232
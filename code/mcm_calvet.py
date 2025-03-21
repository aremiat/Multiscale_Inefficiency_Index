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

    best_logL = -1e15
    best_params = (None, None, None)

    T = len(data)
    # On convertit data en np.array
    data_arr = np.array(data)

    for m0 in m0_grid:
        # On construit l'espace d'états pour K composantes
        states, product_cache = build_state_space(K, m0)
        d = len(states)
        for gamma1 in gamma1_grid:
            # On calcule gamma_k = 1 - (1-gamma1)**b^(k-1)
            gamma_list = []
            for k in range(1, K + 1):
                gamma_k = 1.0 - (1.0 - gamma1) ** (b ** (k - 1))
                gamma_list.append(gamma_k)
            # Matrice de transition
            A = build_transition_matrix(states, K, gamma_list)
            for sigma in sigma_grid:
                # log-likelihood
                ll = msm_loglik(data_arr, sigma, states, product_cache, A)
                if ll > best_logL:
                    best_logL = ll
                    best_params = (m0, gamma1, sigma)

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
    # Paramètres optimaux trouvés
    # m0_opt = 1.0
    # gamma1_opt = 0.01
    # sigma_opt = 0.012

    K = 5 # K composantes binaires max 10
    b = 2 # b entre 1 et +infini

    # Chargement des données (rendements Russell 2000)
    DATA_PATH = os.path.dirname(__file__) + "/../data"
    data_df = pd.read_csv(f"{DATA_PATH}/russell_2000.csv", index_col=0)
    # On suppose qu'il y a une colonne '^RUT' contenant les prix
    returns = data_df['^RUT'].pct_change().dropna().values
    m0_grid = np.linspace(0.1, 1.9, 10)  # 20 points entre 0.1 et 2 multiplicateur de volatilité minimal
    gamma1_grid = np.linspace(0.01, 0.99, 10)  # 20 points entre 0.1 et 0.99, proba entre 0 et 1
    sigma_grid = np.linspace(0.001, 0.1, 10)  # 20 points entre 0.001 et 0.1 (volatilité daily)


    # Estimation brute par grille
    best_params, best_logL = estimate_msm_binomial(returns, m0_grid=m0_grid, gamma1_grid=gamma1_grid,
                                                   sigma_grid=sigma_grid, K=K, b=b)
    print("Best param (m0, gamma1, sigma) =", best_params)
    print("LogLik =", best_logL)

    m0_opt = best_params[0]
    gamma1_opt = best_params[1]
    sigma_opt = best_params[2]

    # Calcul des critères AIC et BIC
    # Hypothèse : 3 paramètres estimés (m0, gamma1, sigma)
    n = len(returns)  # nombre d'observations
    k_params = 3      # nombre de paramètres libres estimés
    AIC = 2 * k_params - 2 * best_logL
    BIC = k_params * np.log(n) - 2 * best_logL

    print(f"AIC = {AIC:.2f}")
    print(f"BIC = {BIC:.2f}")


    # m0_opt = 0.39473684210526316
    # gamma1_opt = 0.01
    # sigma_opt = 0.01663157894736842

    # Construction de l'espace d'états
    states, product_cache = build_state_space(K, m0_opt)

    # Calcul des gamma_k = 1 - (1 - gamma1_opt)*b^(k-1), en s'assurant 0 <= gamma_k <= 1
    gamma_list_opt = []
    for k_ in range(1, K+1):
        val = 1.0 - (1.0 - gamma1_opt)**(b**(k_-1))
        gamma_list_opt.append(max(0.0, min(1.0, val)))

    # Matrice de transition
    A = build_transition_matrix(states, K, gamma_list_opt)

    # Filtrage
    pi_history, loglik_in_sample = forward_filter(returns, sigma_opt, states, product_cache, A)

    # Volatilité fittée in-sample
    vol_fit = fitted_volatility(pi_history, sigma_opt, product_cache)
    #
    # # Comparaison via Plotly

    #
    # abs_ret = np.abs(returns)
    vol = pd.Series(returns).rolling(window=21).std().values # éviter les divisions par 0
    # vol = volatility(returns)
    vol = [v if v > 1e-14 else 1e-14 for v in vol]
    #
    rmse = np.sqrt(np.mean((vol_fit - vol) ** 2))
    print("RMSE between fitted volatility and realized volatility =", rmse)

    corr_coef = np.corrcoef(vol_fit, vol)[0, 1]
    print("Correlation coefficient =", corr_coef)

    fig = make_subplots(rows=1, cols=1, subplot_titles=["In-sample: MSM Volatility vs. Volatility |Returns|"])
    fig.add_trace(go.Scatter(
        x=np.arange(len(vol_fit)),
        y=vol_fit,
        mode='lines',
        name='Fitted MSM Vol'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.arange(len(vol)),
        y=vol,
        mode='lines',
        name='|Returns|'
    ), row=1, col=1)

    fig.update_layout(
        title=f"LogLik in-sample = {loglik_in_sample:.2f}",
        xaxis_title="Time index",
        yaxis_title="Value"
    )
    fig.show()
    #
    # garch_model = arch_model(pd.Series(returns), vol='GARCH', p=1, q=1, dist='normal')
    # garch_fit = garch_model.fit(disp='off')
    #
    # # La volatilité conditionnelle quotidienne (généralement en échelle "daily") est obtenue comme suit :
    # garch_vol = garch_fit.conditional_volatility
    #
    # rmse_garch = np.sqrt(np.mean((garch_vol - vol) ** 2))
    # print("RMSE between GARCH volatility and realized volatility =", rmse_garch)
    #
    # corr_coef_garch = np.corrcoef(garch_vol, vol)[0, 1]
    # print("Correlation coefficient GARCH =", corr_coef_garch)
    #
    # fig = make_subplots(rows=1, cols=1, subplot_titles=["In-sample: GARCH Volatility vs. Volatility |Returns|"])
    # fig.add_trace(go.Scatter(
    #     x=np.arange(len(garch_vol)),
    #     y=garch_vol,
    #     mode='lines',
    #     name='Fitted GARCH Vol'
    # ), row=1, col=1)
    #
    # fig.add_trace(go.Scatter(
    #     x=np.arange(len(vol)),
    #     y=vol,
    #     mode='lines',
    #     name='|Returns|'
    # ), row=1, col=1)
    #
    # fig.update_layout(
    #     title=f"LogLik in-sample = {garch_fit.loglikelihood:.2f}",
    #     xaxis_title="Time index",
    #     yaxis_title="Value"
    # )
    # fig.show()

    # Best
    # param(m0, gamma1, sigma) = (0.46842105263157896, 0.01, 0.01663157894736842)
    # LogLik = 29402.418073294393
    # RMSE
    # between
    # fitted
    # volatility and realized
    # volatility = 0.0032690964001980878
    # Correlation
    # coefficient = 0.9085561692670849
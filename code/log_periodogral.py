import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.fft import fft
from scipy import stats
import os

LOADER_PATH = os.path.dirname(__file__) + "/Loader"


def perST(fBm=None, m1=1, m2=None, llplot=1):
    """
    Estimation du paramètre d'auto-similarité H en utilisant le log-périodogramme
    et une régression linéaire des log-périodogrammes sur les log-fréquences.

    Args:
        fBm : Séries temporelles modélisées par un mouvement brownien fractionnaire.
        m1, m2 : Plage de fréquences utilisée pour le calcul de la régression.
        llplot : Si égal à 1, un graphique log-log sera tracé.

    Returns:
        H : Estimation du paramètre d'auto-similarité.
    """
    if fBm is None:
        fBm = fbm(T=1)  # Fonction de simulation d'un fBm
    if m2 is None:
        m2 = len(fBm) // 2  # Plage de fréquences par défaut

    # Calcul de la différence pour obtenir un bruit blanc fractionnaire (fGn)
    fGn = np.diff(fBm)
    n = len(fGn)

    # Calcul du périodogramme et des fréquences de Fourier
    I_lambda = np.abs(fft(fGn)) ** 2 / (2 * np.pi * n)
    I_lambda = I_lambda[m1:m2]
    lambda_freq = (2 * np.pi * np.arange(m1, m2)) / n

    # Régression des log-périodogrammes sur les log-fréquences
    log_lambda = np.log(lambda_freq)
    log_I_lambda = np.log(I_lambda)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_lambda, log_I_lambda)

    # Estimation de H
    Hest = 0.5 * (1 - slope)

    # Affichage du graphique log-log avec Plotly si llplot == 1
    if llplot == 1:
        Hchar = f"{Hest:.4f}"
        Hleg = f"Hest = {Hchar}"

        fig = go.Figure()

        # Ajouter le log-périodogramme
        fig.add_trace(go.Scatter(x=log_lambda, y=log_I_lambda, mode='markers', name='Log-periodogram'))

        # Ajouter la régression linéaire
        fig.add_trace(
            go.Scatter(x=log_lambda, y=intercept + slope * log_lambda, mode='lines', name=Hleg, line=dict(color='red')))

        # Titre et labels
        fig.update_layout(
            title="Régression du log-périodogramme sur les log-fréquences",
            xaxis_title="log(Frequencies)",
            yaxis_title="log(Periodogram)",
            showlegend=True
        )

        # Afficher le graphique
        fig.show()

    return Hest


# Fonction simulant un mouvement brownien fractionnaire
def circFBM(n=500, H=0.6):
    """
    Fonction pour générer un mouvement brownien fractionnaire circulaire (fBm) en utilisant
    une méthode de simulation basée sur un bruit gaussien.

    Args:
        n : Nombre de points à générer.
        H : Paramètre d'auto-similarité du fBm.

    Returns:
        Une série temporelle de fBm.
    """
    t = np.arange(0, n)
    d = np.random.randn(n)  # Bruit blanc gaussien
    fBm = np.cumsum(d)  # Mouvement brownien fractionnaire simple

    # Appliquer un filtre pour obtenir un fBm avec le paramètre H
    # Ce n'est qu'un exemple de génération de fBm, il existe des méthodes plus précises
    for i in range(1, len(fBm)):
        fBm[i] += (fBm[i - 1] - fBm[i - 1]) * (i ** (H - 0.5)) / (i + 1)

    return fBm

def fbm(T, H = 0.3, N=1000):
    times = np.linspace(0, T, N)
    cov_matrix = np.array(
        [[0.5 * (t1 ** (2 * H) + t2 ** (2 * H) - abs(t1 - t2) ** (2 * H)) for t1 in times] for t2 in times])
    L = np.linalg.cholesky(cov_matrix + 1e-10 * np.eye(N))  # Cholesky matrix
    W = np.random.randn(N)  # Gaussian noise
    X = np.dot(L, W)  # Fractional Brownian motion
    return X



if __name__ == "__main__":
    tickers = ["^GSPC", "^FTSE", "^SBF250", "^TOPX", "^GSPTSE"]

    for ticker in tickers:
        p = pd.read_csv(LOADER_PATH + "/index_prices.csv", index_col=0, parse_dates=True)
        p = p[ticker].dropna()

        log_p = np.log(p)
        r = log_p.diff().dropna()

        H = perST(r)
        print(f"Estimation du paramètre d'auto-similarité H : {H}")
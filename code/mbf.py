import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from statsmodels.tsa.stattools import acf
import plotly.io as pio
import os


IMAGE_PATH = os.path.dirname(__file__) + "/../img"

np.random.seed(42)

# Fonction pour générer un fBm par incréments de Cholesky
def fbm(T, H, N=1000):
    times = np.linspace(0, T, N)
    cov_matrix = np.array([[0.5 * (t1**(2*H) + t2**(2*H) - abs(t1-t2)**(2*H)) for t1 in times] for t2 in times])
    L = np.linalg.cholesky(cov_matrix + 1e-10 * np.eye(N))  # Matrice de Cholesky
    W = np.random.randn(N)  # Bruit gaussien
    X = np.dot(L, W)  # Mouvement brownien fractionnaire
    return times, X

# Valeurs de Hurst à tester
H_values = [0.2, 0.3, 0.5, 0.6, 0.8]

# Création d'une figure avec plusieurs sous-graphiques
fig = sp.make_subplots(rows=2, cols=5, subplot_titles=[f"H = {H}" for H in H_values] + ["Autocorrelation"]*5)

# Génération et affichage des fBm et de leurs autocorrélations
for i, H in enumerate(H_values):
    times, X = fbm(T=1, H=H, N=1000)
    autocorr = acf(X, nlags=40, fft=True)

    # Ajout du fBm à la première ligne de subplots
    fig.add_trace(go.Scatter(x=times, y=X, mode='lines', name=f"H={H}"), row=1, col=i+1)

    # Ajout de l'autocorrélation à la deuxième ligne de subplots
    fig.add_trace(go.Bar(x=np.arange(len(autocorr)), y=autocorr, name=f"ACF H={H}"), row=2, col=i+1)

# Mise en page
fig.update_layout(title="Fractional Brownian Motion & Autocorrelation",
                  height=800, width=1200, showlegend=False)

pio.write_image(fig, f"{IMAGE_PATH}/fdm_autocorr.png", scale=5, width=1000, height=1000)

# Affichage
fig.show()

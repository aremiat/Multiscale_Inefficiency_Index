import time

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# --- Définition de la fonction MF-DFA ---
def mfdfa(signal, scales, q_list, order=1):
    """
    Calcule le MF-DFA pour une série temporelle.

    Paramètres:
      signal : tableau numpy, la série (ex. rendements) à analyser.
      scales : liste des échelles (tailles de segments).
      q_list : liste des ordres q pour lesquels calculer F_q(s).
      order : ordre du polynôme pour le detrending (1 = linéaire par défaut).

    Retourne:
      Fq : matrice de taille (len(q_list), len(scales)) contenant F_q(s).
    """
    N = len(signal)
    # Centrer et intégrer le signal (profil)
    signal = signal - np.mean(signal)
    Y = np.cumsum(signal)
    Fq = np.zeros((len(q_list), len(scales)))

    for j, s in enumerate(scales):
        s = int(s)
        if s < 2:
            continue
        n_segments = N // s
        F_seg = []
        # Découpage non chevauchant depuis le début
        for v in range(n_segments):
            segment = Y[v * s:(v + 1) * s]
            idx = np.arange(s)
            coeffs = np.polyfit(idx, segment, order)
            fit = np.polyval(coeffs, idx)
            F_seg.append(np.mean((segment - fit) ** 2))
        # Découpage depuis la fin pour couvrir la totalité de la série
        for v in range(n_segments):
            segment = Y[N - (v + 1) * s:N - v * s]
            idx = np.arange(s)
            coeffs = np.polyfit(idx, segment, order)
            fit = np.polyval(coeffs, idx)
            F_seg.append(np.mean((segment - fit) ** 2))
        F_seg = np.array(F_seg)
        F_seg[F_seg < 1e-10] = 1e-10
        for k, q in enumerate(q_list):
            if np.abs(q) < 1e-6:
                Fq[k, j] = np.exp(0.5 * np.mean(np.log(F_seg)))
            else:
                Fq[k, j] = (np.mean(F_seg ** (q / 2))) ** (1 / q)
    return Fq


def compute_hq(Fq, scales, q_list):
    """
    Pour chaque q, réalise une régression linéaire log-log (log(Fq) vs log(s))
    pour obtenir h(q).
    """
    h_q = []
    log_scales = np.log(scales)
    for i in range(len(q_list)):
        log_Fq = np.log(Fq[i, :])
        slope, _ = np.polyfit(log_scales, log_Fq, 1)
        h_q.append(slope)
    return np.array(h_q)


def compute_alpha_falpha(q_list, h_q):
    """
    Calcule alpha et f(alpha) via la transformation de Legendre :
      alpha = h(q) + q * h'(q)
      f(alpha) = q*(alpha - h(q)) + 1
    """
    dq = q_list[1] - q_list[0]
    dh_dq = np.gradient(h_q, dq)
    alpha = h_q + q_list * dh_dq
    f_alpha = q_list * (alpha - h_q) + 1
    return alpha, f_alpha


def compute_multifractal_metrics(signal, scales, q_list, order=1):
    """
    Pour une série, calcule :
      - h(q) (les exposants de fluctuation)
      - alpha via la transformation de Legendre
      - delta_alpha = max(alpha) - min(alpha)
    """
    Fq = mfdfa(signal, scales, q_list, order=order)
    h_q = compute_hq(Fq, scales, q_list)
    alpha, f_alpha = compute_alpha_falpha(q_list, h_q)
    delta_alpha = np.max(alpha) - np.min(alpha)
    return {'h_q': h_q, 'alpha': alpha, 'f_alpha': f_alpha, 'delta_alpha': delta_alpha, 'h_mean': np.mean(h_q)}


# --- Récupération des données de Yahoo Finance ---
# Liste de 20 tickers représentatifs (big et small caps)
tickers = [
    "^GSPC",   # S&P 500
    "^DJI",    # Dow Jones Industrial Average
    "^IXIC",   # NASDAQ Composite
    "^RUT",    # Russell 2000
    "^FTSE",   # FTSE 100
    "^GDAXI",  # DAX
    "^FCHI",   # CAC 40
    "^N225",   # Nikkei 225
    "^HSI",    # Hang Seng
    "000001.SS"  # Shanghai Composite
]

# Période d'analyse (par exemple, 5 dernières années)
start_date = "1987-09-10"
end_date = "2024-12-31"

# Pour stocker les indicateurs
results = {}

for ticker in tickers:
    print(f"Traitement de {ticker}")
    data = yf.download(ticker, start=start_date, end=end_date)
    time.sleep(1)
    # On utilise la colonne 'Adj Close' pour les rendements
    if data.empty:
        print(f"Aucune donnée pour {ticker}.")
        continue
    prices = data['Close']
    returns = np.log(prices).diff().dropna().values  # rendements en log
    # Paramètres du MF-DFA
    scales = np.unique(np.floor(np.logspace(np.log10(10), np.log10(80), 10)).astype(int))
    q_list = np.linspace(-5, 5, 21)
    metrics = compute_multifractal_metrics(returns, scales, q_list, order=1)
    results[ticker] = metrics

# Construire un DataFrame avec les métriques clés
df_metrics = pd.DataFrame({
    'Ticker': list(results.keys()),
    'delta_alpha': [results[t]['delta_alpha'] for t in results],
    'h_mean': [results[t]['h_mean'] for t in results]
})

print(df_metrics)

# --- Clustering ---
# On va standardiser et utiliser KMeans pour regrouper selon delta_alpha et h_mean
X = df_metrics[['delta_alpha', 'h_mean']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choix du nombre de clusters (ici 2, à ajuster en fonction de l'analyse)
kmeans = KMeans(n_clusters=2, random_state=42)
df_metrics['cluster'] = kmeans.fit_predict(X_scaled)

print(df_metrics)

# Visualisation du clustering
plt.figure(figsize=(8, 6))
plt.scatter(df_metrics['delta_alpha'], df_metrics['h_mean'], c=df_metrics['cluster'], cmap='viridis', s=100)
for i, ticker in enumerate(df_metrics['Ticker']):
    plt.annotate(ticker, (df_metrics['delta_alpha'][i], df_metrics['h_mean'][i]))
plt.xlabel('Delta Alpha (Largeur du spectre multifractal)')
plt.ylabel('Moyenne de h(q)')
plt.title('Clustering des trading pairs selon la multifractalité')
plt.show()

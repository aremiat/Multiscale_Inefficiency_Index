import os
import numpy as np
import pandas as pd
import yfinance as yf
from utils.RS import ComputeRS
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

# Liste de tickers à analyser
tickers = [
    "^GSPC",
    "^RUT",
    "^FTSE",
    "^N225",
    "^GSPTSE",
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corporation
    "AMZN",  # Amazon.com, Inc.
    "GOOGL",  # Alphabet Inc.
    "CALM",  # Cal-Maine Foods, Inc.
    "NARI",  # Inari Medical, Inc.
    "FARO",  # FARO Technologies, Inc.
    "ALGT",  # Allegiant Travel Company
    "SMTC",  # SMTC Corporation
    "AGYS"  # Agilysys, Inc.
]

if __name__ == "__main__":
    # Configuration des options d'affichage
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 250)

    # Chemin vers le dossier data
    DATA_PATH = os.path.join(os.path.dirname(__file__), "../data")

    # Dictionnaire pour stocker les résultats de chaque ticker
    results_summary = {}

    # Choix de la taille de la fenêtre glissante (ici ~2520 jours = environ 10 ans)
    window = 120

    for ticker in tickers[1:2]:
        print("Traitement de :", ticker)
        # Téléchargement des prix de clôture

        try:
            data = pd.read_csv(os.path.join(DATA_PATH, "russel_stocks.csv"), index_col=0, parse_dates=True)[ticker]
        except:
            data = yf.download(ticker)["Close"]

        data = data.loc[:'2015-02-28']


        # Calcul des rendements journaliers en log
        log_prices = np.log(data)
        returns = log_prices.diff().dropna()
        returns = returns.resample('M').last().dropna()

        # Calcul de la volatilité (écart-type sur la fenêtre)
        vol = returns.rolling(window=window).std().dropna()

        # Calcul de l'exposant de Hurst via la statistique R/S :
        # h = log(ComputeRS.rs_statistic(window)) / log(len(window))
        rolling_hurst = returns.rolling(window=window).apply(
            lambda window: np.log(ComputeRS.rs_statistic(window, len(window))) / np.log(len(window)),
            raw=False
        ).dropna()
        df = pd.concat([vol, rolling_hurst], axis=1)

        df.columns = ["Volatilité", "Hurst"]

        # Calcul de la corrélation entre volatilité et exposant de Hurst
        correlation = df["Volatilité"].corr(df["Hurst"])
        print(f"Corrélation entre volatilité et exposant de Hurst pour {ticker} : {correlation:.4f}")

        # Régression linéaire simple
        X = df["Volatilité"].values.reshape(-1, 1)
        y = df["Hurst"].values

        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        y_pred_lin = lin_reg.predict(X)

        idx_sort = np.argsort(X.flatten())
        X_sorted = X.flatten()[idx_sort]
        y_lin_sorted = y_pred_lin[idx_sort]

        fig_lin = go.Figure()
        # Nuage de points
        fig_lin.add_trace(go.Scatter(
            x=X.flatten(),
            y=y,
            mode='markers',
            name='Données',
            marker=dict(size=5, opacity=0.5)
        ))
        # Courbe de régression linéaire
        fig_lin.add_trace(go.Scatter(
            x=X_sorted,
            y=y_lin_sorted,
            mode='lines',
            name='Linear Regression',
            line=dict(color='red', width=2)
        ))
        fig_lin.update_layout(
            title=f"Linear regression for russel 2000",
            xaxis_title="Volatility (rolling 10 years)",
            yaxis_title="Hurst Exponent (rolling 10 years)"
        )
        fig_lin.show()

        # ---- 2) Régression polynomiale (degré 2) ----
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        poly_reg = LinearRegression()
        poly_reg.fit(X_poly, y)
        y_pred_poly = poly_reg.predict(X_poly)

        # Trie pour le tracé
        y_poly_sorted = y_pred_poly[idx_sort]

        fig_poly = go.Figure()
        # Nuage de points
        fig_poly.add_trace(go.Scatter(
            x=X.flatten(),
            y=y,
            mode='markers',
            name='Données',
            marker=dict(size=5, opacity=0.5)
        ))
        # Courbe polynomiale
        fig_poly.add_trace(go.Scatter(
            x=X_sorted,
            y=y_poly_sorted,
            mode='lines',
            name='Polynomial Regression (order=2)',
            line=dict(color='green', width=2)
        ))
        fig_poly.update_layout(
            title=f"Polynomial regression for russel 2000",
            xaxis_title="Volatility (rolling 10 years)",
            yaxis_title="Hurst Exponent (rolling 10 years)"
        )
        fig_poly.show()


        # Sauvegarde des résultats résumés
        results_summary[ticker] = {
            "Corrélation": correlation,
            "Coef Linéaire": lin_reg.coef_[0],
            "Intercept Linéaire": lin_reg.intercept_
        }

    # Sauvegarde du résumé dans un CSV
    summary_df = pd.DataFrame(results_summary).T
    summary_csv_path = os.path.join(DATA_PATH, "volatilite_hurst_regression_summary.csv")
    summary_df.to_csv(summary_csv_path)
    print("Résumé sauvegardé dans", summary_csv_path)

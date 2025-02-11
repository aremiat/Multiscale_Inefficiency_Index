from aeqlib import utils, lisa, portfolio_construction
import numpy as np
import os
import pandas as pd
from aeqlib.quantlib import volatility
import plotly.graph_objects as go
import plotly.figure_factory as ff
import yfinance as yf


ticker = ["AAPL"]


aapl = yf.download(ticker, progress=False)['Close']
r_daily_appl = aapl.pct_change().dropna()
r_weekly_appl = r_daily_appl.resample("W").last()
r_monthly_appl = r_daily_appl.resample("M").last()
r_quaterly_appl = r_daily_appl.resample("Q").last()
r_annual_appl = r_daily_appl.resample("Y").last()
daily_rolling_vol = r_daily_appl.rolling(21).apply(lambda x : volatility(x)).dropna()
weekly_rolling_vol = r_weekly_appl.rolling(21).apply(lambda x : volatility(x)).dropna()
monthly_rolling_vol = r_monthly_appl.rolling(21).apply(lambda x : volatility(x)).dropna()
quaterly_rolling_vol = r_quaterly_appl.rolling(21).apply(lambda x : volatility(x)).dropna()
annually_rolling_vol = r_annual_appl.rolling(5).apply(lambda x : volatility(x)).dropna()

fig = go.Figure()

# Trace de la volatilité quotidienne
fig.add_trace(go.Scatter(
    x=daily_rolling_vol.index,
    y=daily_rolling_vol['AAPL'],
    mode='lines',
    name='Volatilité quotidienne',
    line=dict(color='light blue', width=1),
    opacity=0.5
))

# Trace de la volatilité hebdomadaire
fig.add_trace(go.Scatter(
    x=weekly_rolling_vol.index,
    y=weekly_rolling_vol['AAPL'],
    mode='lines',
    name='Volatilité hebdomadaire',
    line=dict(color='red', width=2)
))

# Trace de la volatilité mensuelle
fig.add_trace(go.Scatter(
    x=monthly_rolling_vol.index,
    y=monthly_rolling_vol['AAPL'],
    mode='lines',
    name='Volatilité mensuel',
    line=dict(color='green', width=2)
))

fig.add_trace(go.Scatter(
    x=quaterly_rolling_vol.index,
    y=quaterly_rolling_vol['AAPL'],
    mode='lines',
    name='Volatilité quaterly',
    line=dict(color='yellow', width=2)
))

fig.add_trace(go.Scatter(
    x=annually_rolling_vol.index,
    y=annually_rolling_vol['AAPL'],
    mode='lines',
    name='Volatilité annuel',
    line=dict(color='pink', width=2)
))


# Mise en forme du graphique
fig.update_layout(
    title="Évolution de la volatilité quotidienne et hebdomadaire de AAPL",
    xaxis_title="Date",
    yaxis_title="Volatilité",
    template="plotly_dark",
    showlegend=True
)

# Affichage du graphique
fig.show()

# Préparation des données pour les densités
data = [
    daily_rolling_vol['AAPL'].dropna(),
    weekly_rolling_vol['AAPL'].dropna(),
    monthly_rolling_vol['AAPL'].dropna(),
    quaterly_rolling_vol['AAPL'].dropna(),
    annually_rolling_vol['AAPL'].dropna()
]

labels = ["Daily Vol", "Weekly Vol", "Monthly Vol", "Quarterly Vol", "Annual Vol"]
colors = ["lightblue", "red", "green", "yellow", "pink"]

# Création des courbes de densité
fig = ff.create_distplot(
    data,
    labels,
    show_hist=False,  # Désactive l'histogramme pour ne garder que la courbe de densité
    colors=colors
)

# Mise en forme du graphique
fig.update_layout(
    title="Densités des différentes volatilities de AAPL",
    xaxis_title="Volatilité",
    yaxis_title="Densité",
    template="plotly_dark",
    showlegend=True
)

# Affichage du graphique
fig.show()



print("done")
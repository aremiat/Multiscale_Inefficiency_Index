"""
loader_assets.py
~~~~~~~~~~~~~~~~
Télécharge et empile 6 classes d’actifs (3 tickers chacune) depuis Yahoo Finance.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Literal
import time

# -------------------------------------------------------------------
# 1) Liste centrale des actifs : modifie ici si tu veux d'autres codes
# -------------------------------------------------------------------
ASSET_MAP: Dict[str, List[str]] = {
    "equities":        ["^GSPC", "^STOXX50E", "^N225"],      # actions/indices
    "gov_bonds":       ["IEF", "IEGA.L", "CJGB.L"],          # ETF trésor US, OAT €, JGB
    "commodities":     ["GC=F", "BZ=F", "HG=F"],             # or, pétrole Brent, cuivre
    "fx":              ["EURUSD=X", "USDJPY=X", "GBPUSD=X"], # devises spot
    "reits":           ["VNQ", "IYR", "XLRE"],               # immobilier coté
    "crypto":          ["BTC-USD", "ETH-USD", "SOL-USD"],    # cryptomonnaies
}


# -------------------------------------------------------------------
# 2) Fonction utilitaire de téléchargement
# -------------------------------------------------------------------
def download_yf(
    tickers: List[str],
    start: str = "2015-01-01",
    end:   str | None = None,
    field: Literal["Adj Close", "Close"] = "Adj Close",
    auto_adjust: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Télécharge les prix d'une liste de tickers via yfinance et renvoie
    un DataFrame date x ticker.

    Parameters
    ----------
    tickers : list[str]
        Codes Yahoo Finance.
    start, end : str | None
        Bornes temporelles (format ISO YYYY-MM-DD).
    field : "Adj Close" | "Close"
        Colonne à retenir dans les données Yahoo.
    auto_adjust : bool
        Ajuste automatiquement dividendes/splits (équiv. à Adj Close).
    progress : bool
        Affiche ou non la barre de progression yfinance.
    """
    print(tickers)
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=progress,
        group_by="ticker",
        threads=True,
    )
    time.sleep(10)
    # yfinance renvoie multi‑colonnes : niveau 0 = champ (Open, High…)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(field, level=data.columns.names.index('Price'), axis=1)

    # Bascule en datetime index propre
    data.index = pd.to_datetime(data.index)

    return data


# -------------------------------------------------------------------
# 3) Loader principal multi‑classes
# -------------------------------------------------------------------
def load_asset_universe(
    start: str = "2015-01-01",
    end:   str | None = None,
    field: Literal["Adj Close", "Close"] = "Close",
) -> pd.DataFrame:
    """
    Combine toutes les classes d'actifs définies dans ASSET_MAP.

    Returns
    -------
    prices : pd.DataFrame
        Index = date ; colonnes = MultiIndex (class, ticker)
        Les trous de calendrier sont forward‑filled pour chaque ticker.
    """
    frames: List[Tuple[str, pd.DataFrame]] = []

    for asset_class, tickers in ASSET_MAP.items():
        df = download_yf(tickers, start=start, end=end, field=field)
        # Harmonise noms des colonnes pour retraite facile
        df.columns = pd.MultiIndex.from_product([[asset_class], df.columns])
        frames.append((asset_class, df))

    # Concatène sur l'axe colonnes, trie second niveau
    prices = pd.concat([f for _, f in frames], axis=1).sort_index(axis=1, level=1)

    # Fréquence business day + forward fill pour éviter NaN isolés
    prices = prices.asfreq("B").ffill()

    return prices


# -------------------------------------------------------------------
# 4) Exemple d'utilisation
# -------------------------------------------------------------------
if __name__ == "__main__":
    px = load_asset_universe(start='1987-09-10', end='2025-02-28')
    print(px.head())
    # Exemple : extraire les cryptos seulement
    cryptos = px["crypto"]
    print("\nCrypto sub‑frame\n", cryptos.tail())
    px.to_csv("mutli_assets.csv")

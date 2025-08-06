from __future__ import annotations

import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Literal
import time

ASSET_MAP: Dict[str, List[str]] = {
    "equities":        ["^GSPC", "^STOXX50E", "^N225"],      # actions/indices
    "gov_bonds":       ["IEF", "IEGA.L", "CJGB.L"],          # ETF trésor US, OAT €, JGB
    "commodities":     ["GC=F", "BZ=F", "HG=F"],             # or, pétrole Brent, cuivre
    "fx":              ["EURUSD=X", "USDJPY=X", "GBPUSD=X"], # devises spot
    "reits":           ["VNQ", "IYR", "XLRE"],               # immobilier coté
    "crypto":          ["BTC-USD", "ETH-USD", "SOL-USD"],    # cryptomonnaies
}


def download_yf(
    tickers: List[str],
    start: str = "2015-01-01",
    end:   str | None = None,
    field: Literal["Adj Close", "Close"] = "Adj Close",
    auto_adjust: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Download historical prices for a list of tickers from Yahoo Finance.
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
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(field, level=data.columns.names.index('Price'), axis=1)

    data.index = pd.to_datetime(data.index)

    return data


def load_asset_universe(
    start: str = "2015-01-01",
    end:   str | None = None,
    field: Literal["Adj Close", "Close"] = "Close",
) -> pd.DataFrame:
    """
    Compile a DataFrame of historical prices for multiple asset classes.
    """
    frames: List[Tuple[str, pd.DataFrame]] = []

    for asset_class, tickers in ASSET_MAP.items():
        df = download_yf(tickers, start=start, end=end, field=field)
        df.columns = pd.MultiIndex.from_product([[asset_class], df.columns])
        frames.append((asset_class, df))

    prices = pd.concat([f for _, f in frames], axis=1).sort_index(axis=1, level=1)
    prices = prices.asfreq("B").ffill()

    return prices

if __name__ == "__main__":
    px = load_asset_universe(start='1987-09-10', end='2025-02-28')
    print(px.head())
    cryptos = px["crypto"]
    print("\nCrypto sub‑frame\n", cryptos.tail())
    px.to_csv("mutli_assets.csv")

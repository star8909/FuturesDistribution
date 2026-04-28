"""yfinance 기반 선물 데이터 로더 (parquet 캐시).

연속 컨트랙트 (front-month auto-roll) 사용:
  - `CL=F` (WTI), `GC=F` (Gold), `ES=F` (S&P 500 e-mini) 등
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from .config import CACHE_DIR

warnings.filterwarnings("ignore", category=FutureWarning)


def _cache_path(symbol: str, interval: str = "1d") -> Path:
    safe = symbol.replace("=", "_").replace("/", "_").replace("^", "")
    return CACHE_DIR / f"{safe}__{interval}.parquet"


def load_ticker(symbol: str, interval: str = "1d", since: str | None = "2010-01-01") -> pd.DataFrame:
    """심볼 OHLCV 로드. 캐시 우선, 없으면 yfinance 다운.

    반환 컬럼: open, high, low, close, adj_close, volume (DatetimeIndex).
    """
    p = _cache_path(symbol, interval)
    if p.exists():
        df = pd.read_parquet(p)
        if since:
            df = df[df.index >= pd.Timestamp(since)]
        return df

    import yfinance as yf
    print(f"[data_loader] fetch {symbol} {interval}")
    raw = yf.download(symbol, interval=interval, start=since, auto_adjust=False, progress=False, threads=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in raw.columns]
    else:
        raw.columns = [str(c).lower() for c in raw.columns]
    rename = {"adj close": "adj_close"}
    raw = raw.rename(columns=rename)
    raw.index.name = "date"
    raw.to_parquet(p)
    return raw


def load_close(symbols: list[str], since: str = "2010-01-01") -> pd.DataFrame:
    """여러 심볼의 adj_close (없으면 close) DataFrame."""
    cols = {}
    for s in symbols:
        df = load_ticker(s, since=since)
        if df.empty:
            continue
        c = df["adj_close"] if "adj_close" in df.columns else df["close"]
        c = c[c.index >= pd.Timestamp(since)]
        if len(c) < 100:
            continue
        cols[s] = c
    return pd.DataFrame(cols)

"""yfinance 기반 선물 데이터 로더 (parquet 캐시).

연속 컨트랙트 (front-month auto-roll) 사용:
  - `CL=F` (WTI), `GC=F` (Gold), `ES=F` (S&P 500 e-mini) 등

주의: =F continuous futures는 롤오버 시 가격 점프 발생.
  - ZC=F July 15: -23.56% (2020), -17.39% (2019) 등 가짜 수익률
  - clean_roll_jumps()로 필터 적용 필수 (|Δ|>15% → NaN → 0)
  - 지수선물(ES, NQ, YM)은 현금결제라 점프 없음. 농산물이 주 문제.

ETF 대체 매핑 (roll cost 자동 반영):
  AGRI_ETF_MAP: ZC→CORN, ZW→WEAT, ZS→SOYB, KC→JO, SB→CANE
  DBA: 농산물 바스켓 ETF
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

# 농산물 선물 → ETF 대체 매핑 (roll cost 내재화)
AGRI_ETF_MAP: dict[str, str] = {
    "ZC=F": "CORN",   # Corn → Teucrium Corn Fund
    "ZW=F": "WEAT",   # Wheat → Teucrium Wheat Fund
    "ZS=F": "SOYB",   # Soybean → Teucrium Soybean Fund
    "KC=F": "JO",     # Coffee → iPath Coffee
    "SB=F": "CANE",   # Sugar → Teucrium Sugar Fund
    "CT=F": "BAL",    # Cotton → iPath Cotton
    "CC=F": "NIB",    # Cocoa → iPath Cocoa
}

# roll jump가 심각한 선물 카테고리 (지수/통화/금리는 현금결제라 문제 없음)
ROLL_JUMP_RISK_SYMBOLS = {s for s in AGRI_ETF_MAP} | {
    "CL=F", "NG=F", "HO=F", "RB=F", "BZ=F",   # 에너지 (실물 인도)
    "SI=F", "PL=F", "PA=F",                     # 귀금속 일부
}

ROLL_JUMP_THRESHOLD = 0.12  # |일간 수익률| > 12% → 롤오버 점프 의심


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


def clean_roll_jumps(prices: pd.Series, symbol: str = "",
                     threshold: float = ROLL_JUMP_THRESHOLD) -> pd.Series:
    """롤오버 점프 제거: |일간 수익률| > threshold → 해당 날 수익률 0으로 대체.

    방법: 의심 날의 price를 전날 price로 대체 (수익률 0 처리).
    지수선물(ES, NQ, YM, RTY, NIY) / 통화 / 금리는 skip (현금결제).
    """
    if not any(k in symbol for k in ["=F"]):
        return prices
    is_risky = symbol in ROLL_JUMP_RISK_SYMBOLS
    if not is_risky:
        return prices

    rets = prices.pct_change()
    jump_mask = rets.abs() > threshold
    n_jumps = int(jump_mask.sum())
    if n_jumps == 0:
        return prices

    prices_clean = prices.copy().astype(float)
    for dt in prices.index[jump_mask]:
        loc = prices.index.get_loc(dt)
        if loc > 0:
            prices_clean.iloc[loc] = prices_clean.iloc[loc - 1]

    return prices_clean


def load_close(symbols: list[str], since: str = "2010-01-01",
               clean_jumps: bool = True) -> pd.DataFrame:
    """여러 심볼의 adj_close (없으면 close) DataFrame.

    clean_jumps=True (기본): roll jump 필터 적용.
    """
    cols = {}
    for s in symbols:
        df = load_ticker(s, since=since)
        if df.empty:
            continue
        c = df["adj_close"] if "adj_close" in df.columns else df["close"]
        c = c[c.index >= pd.Timestamp(since)]
        if len(c) < 100:
            continue
        if clean_jumps:
            c = clean_roll_jumps(c, symbol=s)
        cols[s] = c
    return pd.DataFrame(cols)


def load_close_etf(symbols: list[str], since: str = "2010-01-01") -> pd.DataFrame:
    """=F 선물을 ETF 대체로 로드 (가능한 것만, 없으면 원본 사용).

    ETF는 roll cost가 자동으로 NAV에 반영되므로 정확한 비용 모델.
    AGRI_ETF_MAP에 없는 심볼(ES=F, GC=F 등)은 clean_roll_jumps 적용 원본 사용.
    """
    cols = {}
    for s in symbols:
        etf_sym = AGRI_ETF_MAP.get(s)
        if etf_sym:
            df = load_ticker(etf_sym, since=since)
            if not df.empty:
                c = df["adj_close"] if "adj_close" in df.columns else df["close"]
                c = c[c.index >= pd.Timestamp(since)]
                if len(c) >= 100:
                    cols[s] = c
                    continue
        df = load_ticker(s, since=since)
        if df.empty:
            continue
        c = df["adj_close"] if "adj_close" in df.columns else df["close"]
        c = c[c.index >= pd.Timestamp(since)]
        if len(c) < 100:
            continue
        c = clean_roll_jumps(c, symbol=s)
        cols[s] = c
    return pd.DataFrame(cols)

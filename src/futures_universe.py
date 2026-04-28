"""한국투자증권 거래 가능 해외선물 universe.

거래소: CME / EUREX / ICE / HKEX / SGX 등 10개
yfinance 연속 컨트랙트 (=F suffix, front-month auto-roll) 사용.

Mini/Micro 선물은 주석으로 표시 (실거래 시 retail 권장).
"""
from __future__ import annotations


# ─── 지수선물 (Index Futures) ─────────────────────────────────────
INDEX_FUTURES = [
    "ES=F",     # E-mini S&P 500     (Micro: MES=F)
    "NQ=F",     # E-mini Nasdaq 100  (Micro: MNQ=F)
    "YM=F",     # E-mini Dow         (Micro: MYM=F)
    "RTY=F",    # E-mini Russell 2000 (Micro: M2K=F)
    "NIY=F",    # Nikkei 225 (USD)
    # "^HSI",   # Hang Seng (HKEX) — yfinance index but limited futures data
    # "^GDAXI", # DAX
    # "^STOXX50E", # EuroStoxx 50
]


# ─── 통화선물 (Currency Futures) ──────────────────────────────────
CURRENCY_FUTURES = [
    "6E=F",     # Euro FX
    "6J=F",     # Japanese Yen
    "6B=F",     # British Pound
    "6A=F",     # Australian Dollar
    "6C=F",     # Canadian Dollar
    "6S=F",     # Swiss Franc
    "DX=F",     # US Dollar Index
]


# ─── 금리선물 (Rate Futures) ──────────────────────────────────────
RATE_FUTURES = [
    "ZT=F",     # US 2-Year T-Note
    "ZF=F",     # US 5-Year T-Note
    "ZN=F",     # US 10-Year T-Note
    "ZB=F",     # US 30-Year T-Bond
    # German Bund 등 EUREX 종목은 yfinance 데이터 약함
]


# ─── 에너지선물 (Energy Futures) ──────────────────────────────────
ENERGY_FUTURES = [
    "CL=F",     # WTI Crude Oil       (Micro: MCL=F)
    "BZ=F",     # Brent Crude
    "NG=F",     # Natural Gas         (Micro: MNG=F)
    "HO=F",     # Heating Oil
    "RB=F",     # RBOB Gasoline
]


# ─── 금속선물 (Metal Futures) ─────────────────────────────────────
METAL_FUTURES = [
    "GC=F",     # Gold                (Micro: MGC=F)
    "SI=F",     # Silver              (Mini:  SIL=F)
    "HG=F",     # Copper              (Micro: MHG=F)
    "PL=F",     # Platinum
    "PA=F",     # Palladium
]


# ─── 농산물선물 (Agriculture Futures) ─────────────────────────────
AGRI_FUTURES = [
    "ZC=F",     # Corn
    "ZW=F",     # Wheat
    "ZS=F",     # Soybean
    "ZL=F",     # Soybean Oil
    "ZM=F",     # Soybean Meal
    "KC=F",     # Coffee
    "SB=F",     # Sugar #11
    "CT=F",     # Cotton
    "CC=F",     # Cocoa
    "OJ=F",     # Orange Juice
]


# ─── Micro 선물 (소액증거금, retail $10k 권장) ───────────────────
MICRO_FUTURES = [
    "MES=F",    # Micro E-mini S&P 500 (1/10 ES)
    "MNQ=F",    # Micro E-mini Nasdaq  (1/10 NQ)
    "MYM=F",    # Micro E-mini Dow     (1/10 YM)
    "M2K=F",    # Micro E-mini Russell (1/10 RTY)
    "MGC=F",    # Micro Gold           (1/10 GC)
    "MCL=F",    # Micro Crude Oil      (1/10 CL)
    "MNG=F",    # Micro Natural Gas    (1/10 NG)
    "MHG=F",    # Micro Copper         (1/10 HG)
]


def all_futures() -> dict[str, list[str]]:
    """카테고리별 선물 universe 딕셔너리."""
    return {
        "index": INDEX_FUTURES,
        "currency": CURRENCY_FUTURES,
        "rate": RATE_FUTURES,
        "energy": ENERGY_FUTURES,
        "metal": METAL_FUTURES,
        "agri": AGRI_FUTURES,
    }


def all_futures_flat() -> list[str]:
    """모든 선물 심볼 1차원 list (중복 제거)."""
    out, seen = [], set()
    for lst in all_futures().values():
        for t in lst:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def micro_futures() -> list[str]:
    """Retail 권장 Micro 선물만."""
    return list(MICRO_FUTURES)

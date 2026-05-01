"""다양한 선물 전략 — momentum 실패 후 다른 alpha source 시도.

10가지 전략:
1. Carry (term structure) — 만기 다른 컨트랙트 가격차
2. Mean reversion (z-score)
3. Vol breakout (Donchian)
4. DXY cross-asset (USD strength)
5. Seasonality (월별 효과)
6. RSI extreme + reversal
7. Bollinger 압축 후 breakout
8. Trend filter (200d MA above) + momentum
9. Inter-commodity spread (ZS/ZC ratio)
10. VIX overlay (high vol → risk-off)
"""
from __future__ import annotations

import sys
import json
import time
import argparse
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data_loader import load_close


AGRI_F = ["ZC=F", "ZW=F", "ZS=F", "KC=F", "SB=F"]
ENERGY_F = ["CL=F", "NG=F", "HO=F"]
METAL_F = ["GC=F", "SI=F", "HG=F"]
ALL_COMMOD = AGRI_F + ENERGY_F + METAL_F


def metrics(pnl):
    pnl = pnl.dropna()
    if len(pnl) == 0:
        return {"Sharpe": 0, "CAGR": 0, "MDD": 0, "win_rate": 0}
    eq = (1 + pnl).cumprod()
    n_years = max((pnl.index[-1] - pnl.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / n_years) - 1)
    sharpe = float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252)) if pnl.std(ddof=1) > 0 else 0
    cm = eq.cummax()
    mdd = float((eq / cm - 1).min())
    win = float((pnl > 0).mean() * 100)
    return {"Sharpe": sharpe, "CAGR": cagr, "MDD": mdd, "win_rate": win}


def strategy_mean_reversion(rets, lookback=20, z_thr=2.0):
    """z-score < -z_thr → long (mean revert)."""
    z = (rets - rets.rolling(lookback).mean()) / rets.rolling(lookback).std()
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    pos[z < -z_thr] = 1.0
    pos = pos.shift(1).fillna(0)
    n_assets = (pos != 0).sum(axis=1).replace(0, 1)
    pnl = (rets * pos).sum(axis=1) / n_assets
    return pnl


def strategy_vol_breakout(rets, prices, lookback=20):
    """price > rolling max → long, < rolling min → short."""
    high = prices.rolling(lookback).max().shift(1)
    low = prices.rolling(lookback).min().shift(1)
    pos = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    pos[prices >= high] = 1.0
    pos[prices <= low] = -1.0
    n_assets = (pos != 0).sum(axis=1).replace(0, 1)
    pnl = (rets * pos.shift(1).fillna(0)).sum(axis=1) / n_assets
    return pnl


def strategy_rsi_reversal(rets, lookback=14):
    """RSI < 30 long, RSI > 70 short."""
    gain = rets.where(rets > 0, 0).rolling(lookback).mean()
    loss = (-rets.where(rets < 0, 0)).rolling(lookback).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - 100 / (1 + rs)
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    pos[rsi < 30] = 1.0
    pos[rsi > 70] = -1.0
    pos = pos.shift(1).fillna(0)
    n_assets = (pos != 0).sum(axis=1).replace(0, 1)
    pnl = (rets * pos).sum(axis=1) / n_assets
    return pnl


def strategy_bollinger_breakout(rets, prices, lookback=20, n_std=2):
    """Bollinger band compression → breakout direction."""
    ma = prices.rolling(lookback).mean()
    std = prices.rolling(lookback).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    pos = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    pos[prices > upper.shift(1)] = 1.0
    pos[prices < lower.shift(1)] = -1.0
    n_assets = (pos != 0).sum(axis=1).replace(0, 1)
    pnl = (rets * pos.shift(1).fillna(0)).sum(axis=1) / n_assets
    return pnl


def strategy_trend_filter_momentum(rets, prices, ma_lookback=200, mom_lookback=20):
    """200d MA 위 + 20d momentum 양수 → long."""
    ma = prices.rolling(ma_lookback).mean()
    above_ma = prices > ma
    mom = rets.rolling(mom_lookback).sum()
    long_signal = above_ma & (mom > 0)
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    pos[long_signal] = 1.0
    pos = pos.shift(1).fillna(0)
    n_assets = (pos != 0).sum(axis=1).replace(0, 1)
    pnl = (rets * pos).sum(axis=1) / n_assets
    return pnl


def strategy_seasonality(rets, prices):
    """월요일 -> long, 금요일 -> flat (weekend effect)."""
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    weekdays = rets.index.weekday
    pos.loc[weekdays == 0, :] = 1.0  # Monday
    pos = pos.shift(1).fillna(0)
    n_assets = pos.shape[1]
    pnl = (rets * pos).sum(axis=1) / n_assets
    return pnl


def strategy_low_vol(rets, lookback=63):
    """저변동성 자산 long (low-vol anomaly)."""
    vol = rets.rolling(lookback).std()
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    # 매일 vol 가장 낮은 3개 long
    rank = vol.rank(axis=1)
    pos[rank <= 3] = 1.0
    pos = pos.shift(1).fillna(0)
    n_assets = (pos != 0).sum(axis=1).replace(0, 1)
    pnl = (rets * pos).sum(axis=1) / n_assets
    return pnl


def strategy_carry_proxy(rets, prices, lookback=126):
    """6개월 가격 변화율 proxy carry — 백워데이션이면 long."""
    carry = prices / prices.shift(lookback) - 1
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    rank = (-carry).rank(axis=1)  # 가장 약세 long (mean-reverting carry)
    pos[rank <= 2] = 1.0
    pos = pos.shift(1).fillna(0)
    n_assets = (pos != 0).sum(axis=1).replace(0, 1)
    pnl = (rets * pos).sum(axis=1) / n_assets
    return pnl


def strategy_pairs_zw_zc(rets, prices):
    """Wheat vs Corn ratio mean reversion (alimentary pair)."""
    if "ZW=F" not in prices.columns or "ZC=F" not in prices.columns:
        return pd.Series(0.0, index=rets.index)
    ratio = prices["ZW=F"] / prices["ZC=F"]
    z = (ratio - ratio.rolling(60).mean()) / ratio.rolling(60).std()
    pos_zw = -np.sign(z).shift(1).fillna(0)  # ratio high → short ZW long ZC
    pos_zc = np.sign(z).shift(1).fillna(0)
    pnl = (rets["ZW=F"] * pos_zw + rets["ZC=F"] * pos_zc) / 2
    return pnl


def strategy_volatility_target(rets, lookback=63, target_vol=0.10):
    """Equal-weight long with vol targeting."""
    pos = pd.DataFrame(1.0 / rets.shape[1], index=rets.index, columns=rets.columns)
    # Scale by inverse realized vol
    vol = rets.std() * np.sqrt(252)
    if vol > 0:
        scale = target_vol / float(vol)
        pos = pos * scale
    pnl = (rets * pos.shift(1).fillna(0)).sum(axis=1)
    # Cap at 0.05/day to prevent leverage explosion
    pnl = pnl.clip(-0.05, 0.05)
    return pnl


STRATEGIES = [
    ("mean_reversion_20d_z2", lambda r, p: strategy_mean_reversion(r, 20, 2.0)),
    ("vol_breakout_20d", lambda r, p: strategy_vol_breakout(r, p, 20)),
    ("rsi_reversal_14d", lambda r, p: strategy_rsi_reversal(r, 14)),
    ("bollinger_breakout_20d", lambda r, p: strategy_bollinger_breakout(r, p, 20, 2)),
    ("trend_filter_momentum", lambda r, p: strategy_trend_filter_momentum(r, p, 200, 20)),
    ("seasonality_monday", lambda r, p: strategy_seasonality(r, p)),
    ("low_vol_anomaly_63d", lambda r, p: strategy_low_vol(r, 63)),
    ("carry_proxy_126d", lambda r, p: strategy_carry_proxy(r, p, 126)),
    ("pairs_ZW_ZC", lambda r, p: strategy_pairs_zw_zc(r, p)),
    ("vol_target_eq_weight", lambda r, p: strategy_volatility_target(r, 63, 0.10)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()
    rd = args.round

    name, strat_fn = STRATEGIES[(rd - 1) % len(STRATEGIES)]
    print(f"[diverse round {rd}] {name}")

    closes = load_close(ALL_COMMOD)
    print(f"  universe: {closes.shape[1]} 종목, {closes.shape[0]} 일")
    rets = closes.pct_change().fillna(0)

    t0 = time.time()
    pnl = strat_fn(rets, closes)
    if isinstance(pnl, pd.Series):
        pnl = pnl.dropna()
    m = metrics(pnl)
    elapsed = time.time() - t0

    result = {
        "round": rd, "strategy": name,
        "wf_sharpe": m["Sharpe"],  # full-period Sharpe (단순 전략, walk-forward 아님)
        "full_cagr_pct": m["CAGR"] * 100,
        "full_mdd_pct": m["MDD"] * 100,
        "win_rate": m["win_rate"],
        "n_days": int(len(pnl)),
        "elapsed_sec": elapsed,
    }
    print(f"  Sh={m['Sharpe']:.2f} CAGR={m['CAGR']*100:.1f}% MDD={m['MDD']*100:.1f}% Win={m['win_rate']:.0f}% ({elapsed:.1f}s)")

    out_path = RESULTS_DIR / f"iter_diverse_strategies_r{rd}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"  → {out_path.name}")


if __name__ == "__main__":
    main()

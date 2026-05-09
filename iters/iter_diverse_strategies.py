"""다양한 선물 전략 — momentum 실패 후 다른 alpha source 시도.

20가지 전략:
1-10. mean reversion / breakout / RSI / Bollinger / trend filter / seasonality /
      low-vol / carry / pairs ZW-ZC / vol target
11-20. Donchian-50 / multi-timeframe trend / MACD / Gold-Silver pair /
      Energy spread / quantile breakout / KAMA / cross-section momentum /
      vol-weighted basket / weekly mean revert
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
    """Equal-weight long with vol targeting (per-asset, scalar avg)."""
    n = rets.shape[1]
    pos = pd.DataFrame(1.0 / n, index=rets.index, columns=rets.columns)
    # Average across assets, then daily Sharpe-style scaling
    portfolio_vol = float((rets.mean(axis=1)).std() * np.sqrt(252))
    if portfolio_vol > 0:
        scale = target_vol / portfolio_vol
        pos = pos * scale
    pnl = (rets * pos.shift(1).fillna(0)).sum(axis=1)
    pnl = pnl.clip(-0.05, 0.05)
    return pnl


def strategy_donchian_long(rets, prices, lookback=50):
    """Turtle-style: 50d high breakout long-only (no shorts)."""
    high = prices.rolling(lookback).max().shift(1)
    pos = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    pos[prices >= high] = 1.0
    pos = pos.shift(1).fillna(0)
    n_assets = (pos != 0).sum(axis=1).replace(0, 1)
    pnl = (rets * pos).sum(axis=1) / n_assets
    return pnl


def strategy_multi_tf_trend(rets, prices, lookbacks=(8, 21, 63)):
    """모든 lookback에서 momentum 양수일 때만 long (강한 합의)."""
    signals = []
    for lb in lookbacks:
        mom = prices / prices.shift(lb) - 1
        signals.append(mom > 0)
    agree = signals[0]
    for s in signals[1:]:
        agree = agree & s
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    pos[agree] = 1.0
    pos = pos.shift(1).fillna(0)
    n_assets = (pos != 0).sum(axis=1).replace(0, 1)
    pnl = (rets * pos).sum(axis=1) / n_assets
    return pnl


def strategy_macd(rets, prices, fast=12, slow=26, signal_p=9):
    """MACD crossover signal."""
    ema_f = prices.ewm(span=fast).mean()
    ema_s = prices.ewm(span=slow).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=signal_p).mean()
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    pos[macd > signal] = 1.0
    pos[macd < signal] = -1.0
    pos = pos.shift(1).fillna(0)
    n_assets = (pos != 0).sum(axis=1).replace(0, 1)
    pnl = (rets * pos).sum(axis=1) / n_assets
    return pnl


def strategy_gold_silver_pair(rets, prices):
    """GC/SI 비율 mean reversion. ratio 높음 → SI long GC short."""
    if "GC=F" not in prices.columns or "SI=F" not in prices.columns:
        return pd.Series(0.0, index=rets.index)
    ratio = prices["GC=F"] / prices["SI=F"]
    z = (ratio - ratio.rolling(60).mean()) / ratio.rolling(60).std()
    pos_gc = -np.sign(z).shift(1).fillna(0)
    pos_si = np.sign(z).shift(1).fillna(0)
    pnl = (rets["GC=F"] * pos_gc + rets["SI=F"] * pos_si) / 2
    return pnl


def strategy_energy_spread(rets, prices):
    """Crack spread proxy: HO-CL (heating oil over crude)."""
    if "CL=F" not in prices.columns or "HO=F" not in prices.columns:
        return pd.Series(0.0, index=rets.index)
    spread = prices["HO=F"] - prices["CL=F"]
    z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
    pos_ho = -np.sign(z).shift(1).fillna(0)
    pos_cl = np.sign(z).shift(1).fillna(0)
    pnl = (rets["HO=F"] * pos_ho + rets["CL=F"] * pos_cl) / 2
    return pnl


def strategy_quantile_breakout(rets, lookback=126, q=0.90):
    """Past returns 90th percentile 돌파 → momentum long."""
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    for col in rets.columns:
        thresh = rets[col].rolling(lookback).quantile(q).shift(1)
        pos[col] = (rets[col] > thresh).astype(float)
    pos = pos.shift(1).fillna(0)
    n_assets = (pos != 0).sum(axis=1).replace(0, 1)
    pnl = (rets * pos).sum(axis=1) / n_assets
    return pnl


def strategy_kama_trend(rets, prices, lookback=10):
    """KAMA-like adaptive trend: efficiency ratio 가중."""
    change = (prices - prices.shift(lookback)).abs()
    volatility = rets.abs().rolling(lookback).sum()
    er = change / (volatility * prices + 1e-9)  # efficiency ratio
    # 추세가 강한(er 큰) 자산 long
    rank = er.rank(axis=1, ascending=False)
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    pos[rank <= 3] = 1.0
    pos = pos.shift(1).fillna(0)
    n_assets = (pos != 0).sum(axis=1).replace(0, 1)
    pnl = (rets * pos).sum(axis=1) / n_assets
    return pnl


def strategy_xsection_momentum(rets, lookback=63):
    """Cross-sectional: top 30% long, bottom 30% short."""
    cum = rets.rolling(lookback).sum()
    rank = cum.rank(axis=1, pct=True).shift(1)
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    pos[rank >= 0.7] = 1.0
    pos[rank <= 0.3] = -1.0
    n_long = (pos > 0).sum(axis=1).replace(0, 1)
    n_short = (pos < 0).sum(axis=1).replace(0, 1)
    long_pnl = (rets * pos.where(pos > 0, 0)).sum(axis=1) / n_long
    short_pnl = (rets * pos.where(pos < 0, 0)).sum(axis=1) / n_short
    return (long_pnl + short_pnl) / 2


def strategy_inv_vol_basket(rets, lookback=63):
    """Inverse-volatility weighted basket (risk parity long-only)."""
    vol = rets.rolling(lookback).std()
    inv = 1 / (vol + 1e-9)
    w = inv.div(inv.sum(axis=1), axis=0).shift(1).fillna(0)
    pnl = (rets * w).sum(axis=1)
    return pnl


def strategy_weekly_revert(rets, lookback=5, z_thr=1.5):
    """5일 누적 음수 후 long (단기 mean revert)."""
    cum5 = rets.rolling(lookback).sum()
    z = (cum5 - cum5.rolling(60).mean()) / cum5.rolling(60).std()
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    pos[z < -z_thr] = 1.0
    pos = pos.shift(1).fillna(0)
    n_assets = (pos != 0).sum(axis=1).replace(0, 1)
    pnl = (rets * pos).sum(axis=1) / n_assets
    return pnl


STRATEGIES = [
    # 1-10 기존
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
    # 11-20 새 전략
    ("donchian_50d_long_only", lambda r, p: strategy_donchian_long(r, p, 50)),
    ("multi_tf_trend_8_21_63", lambda r, p: strategy_multi_tf_trend(r, p, (8, 21, 63))),
    ("macd_12_26_9", lambda r, p: strategy_macd(r, p, 12, 26, 9)),
    ("pairs_GC_SI", lambda r, p: strategy_gold_silver_pair(r, p)),
    ("crack_spread_HO_CL", lambda r, p: strategy_energy_spread(r, p)),
    ("quantile_breakout_q90", lambda r, p: strategy_quantile_breakout(r, 126, 0.90)),
    ("kama_top3_trend", lambda r, p: strategy_kama_trend(r, p, 10)),
    ("xsection_mom_63d_LS", lambda r, p: strategy_xsection_momentum(r, 63)),
    ("inv_vol_basket_63d", lambda r, p: strategy_inv_vol_basket(r, 63)),
    ("weekly_mean_revert_5d", lambda r, p: strategy_weekly_revert(r, 5, 1.5)),
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

"""iter21: Pair / spread trading (Gold-Silver, Crude-Brent, Soy-Corn).

가설: 정상 비율에서 z-score > 2 이탈 시 mean reversion (long 약한 자산 short 강한 자산).
- Au/Ag 비율 (gold/silver)
- WTI/Brent (Crude vs European Crude)
- Soy/Corn (ZS/ZC)
- Heating Oil/Crude (HO/CL)

특히 Au/Ag 비율은 macro proxy로 유명.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data_loader import load_close
from src.backtest import metrics, wf_metrics


def pair_backtest(p1, p2, lookback=126, z_thr=2.0, fee_per_change=0.005):
    """Cointegration-based pair: log_ratio z-score > thr → mean reversion."""
    df = pd.concat([p1.rename("a"), p2.rename("b")], axis=1).dropna()
    if len(df) < lookback * 2:
        return None
    df["log_a"] = np.log(df["a"])
    df["log_b"] = np.log(df["b"])
    df["spread"] = df["log_a"] - df["log_b"]
    df["mean"] = df["spread"].rolling(lookback).mean()
    df["std"] = df["spread"].rolling(lookback).std()
    df["z"] = (df["spread"] - df["mean"]) / df["std"]

    # ret a - ret b (long a short b portfolio = a outperform - b outperform)
    df["ret_a"] = df["a"].pct_change()
    df["ret_b"] = df["b"].pct_change()
    df["spread_ret"] = df["ret_a"] - df["ret_b"]

    # signal: z > thr → short spread (b > a, so long b short a = -spread_ret)
    #         z < -thr → long spread (a > b, so long a short b = +spread_ret)
    df["pos"] = 0
    df.loc[df["z"] > z_thr, "pos"] = -1  # short spread (long b)
    df.loc[df["z"] < -z_thr, "pos"] = 1   # long spread (long a)
    df["pos"] = df["pos"].shift(1).fillna(0)
    # exit when |z| < 0.5
    in_pos = pd.Series(0.0, index=df.index)
    last_pos = 0.0
    for i in range(len(df)):
        z_now = df["z"].iloc[i]
        sig_now = df["pos"].iloc[i]
        if pd.isna(z_now):
            in_pos.iloc[i] = last_pos
            continue
        if last_pos != 0 and abs(z_now) < 0.5:
            last_pos = 0
        elif sig_now != 0:
            last_pos = sig_now
        in_pos.iloc[i] = last_pos

    df["pos_eff"] = in_pos
    df["pnl"] = df["pos_eff"] * df["spread_ret"]
    # transaction cost on flips
    df["flip"] = df["pos_eff"].diff().abs()
    df["pnl"] -= df["flip"] * fee_per_change
    return df


def main():
    print("[iter21] Spread / pair trading")

    pairs = [
        ("Gold-Silver (GC-SI)", "GC=F", "SI=F"),
        ("Crude-Brent (CL-BZ)", "CL=F", "BZ=F"),
        ("Soy-Corn (ZS-ZC)", "ZS=F", "ZC=F"),
        ("Heating-Crude (HO-CL)", "HO=F", "CL=F"),
        ("Soy-Wheat (ZS-ZW)", "ZS=F", "ZW=F"),
        ("Corn-Wheat (ZC-ZW)", "ZC=F", "ZW=F"),
        ("Gas-Crude (NG-CL)", "NG=F", "CL=F"),
    ]

    syms = list(set([s for _, p1, p2 in pairs for s in (p1, p2)]))
    closes = load_close(syms)
    closes = closes.fillna(method='ffill')
    print(f"  종목 로드: {list(closes.columns)}")

    print(f"\n=== Pair trading 결과 ===")
    print(f"  {'Pair':25s} {'z_thr':>5} {'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'flips':>6}")
    for name, p1, p2 in pairs:
        if p1 not in closes.columns or p2 not in closes.columns:
            print(f"  {name}: 데이터 없음 ({p1} or {p2})")
            continue
        for z_thr in [1.5, 2.0, 2.5]:
            res = pair_backtest(closes[p1], closes[p2], lookback=126, z_thr=z_thr)
            if res is None:
                continue
            pnl = res["pnl"].dropna()
            m = metrics(pnl)
            n_flips = int(res["flip"].sum() / 2) if "flip" in res.columns else 0
            marker = "🚀" if m['Sharpe'] > 1.5 else "✅" if m['Sharpe'] > 0.8 else ""
            print(f"  {name:25s} {z_thr:>5.1f} {m['Sharpe']:>7.2f} {m['CAGR']*100:>+6.1f}% {m['MDD']*100:>+6.1f}% {n_flips:>6} {marker}")

    out_path = RESULTS_DIR / "iter21_spread_pairs.json"
    out_path.write_text("{}", encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()

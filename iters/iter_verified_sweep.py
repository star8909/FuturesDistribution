"""verified sweep — 50 configs (Futures)."""
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
from src.backtest import metrics, wf_metrics


AGRI_F = ["ZC=F", "ZW=F", "ZS=F", "ZL=F", "ZM=F", "KC=F", "SB=F", "CT=F", "CC=F", "OJ=F"]
AGRI_ETF = ["DBA", "CORN", "WEAT", "SOYB"]
ENERGY_F = ["CL=F", "BZ=F", "NG=F", "HO=F", "RB=F"]
METAL_F = ["GC=F", "SI=F", "HG=F", "PL=F", "PA=F"]
ALL_F = AGRI_F + ENERGY_F + METAL_F


def champ_wf_simple(rets, top_k=2, cvar_alpha=1.0, dd_stop=-0.10, lock_days=63,
                    train=504, test=252, step=126,
                    momentum_periods=((8, 1.0),)):
    n = len(rets)
    pnl = pd.Series(0.0, index=rets.index)
    used = pd.Series(False, index=rets.index)
    win_pnls = []
    s = 0
    while s + train + test <= n:
        if s + train < 504:
            s += step
            continue
        end_idx = s + train
        composite = pd.Series(0.0, index=rets.columns)
        for period, pw in momentum_periods:
            rec = rets.iloc[end_idx - period:end_idx]
            cum = (1 + rec.fillna(0)).prod() - 1
            if cum.std() == 0:
                continue
            z = (cum - cum.mean()) / cum.std()
            composite = composite.add(z * pw, fill_value=0)
        composite = composite.reindex(rets.columns).fillna(0)
        window = rets.iloc[end_idx - 63:end_idx]
        cvar_abs = pd.Series(0.001, index=window.columns)
        for col in window.columns:
            ret = window[col].dropna()
            if len(ret) >= 20:
                tail = ret[ret <= np.percentile(ret, 5)]
                if len(tail) > 0:
                    cvar_abs[col] = abs(tail.mean()) + 1e-6
        score = composite / (cvar_abs ** cvar_alpha + 1e-6)
        topk = min(top_k, len(score[score > 0]))
        if topk == 0:
            s += step
            continue
        top = list(score[score > 0].sort_values(ascending=False).head(topk).index)
        w = pd.Series(0.0, index=rets.columns)
        for c in top:
            w[c] = 1.0 / len(top)
        test_idx = rets.iloc[s + train:s + train + test]
        win_pnl = pd.Series(0.0, index=test_idx.index)
        locked_until = -1
        for i in range(len(test_idx)):
            ts = test_idx.index[i]
            current_idx = s + train + i
            if locked_until > current_idx:
                w_eff = pd.Series(0.0, index=rets.columns)
            else:
                lookback_pnl = win_pnl.iloc[max(0, i-126):i]
                if len(lookback_pnl) > 30:
                    eq_lb = (1 + lookback_pnl).cumprod()
                    cm_lb = eq_lb.cummax()
                    cur_dd = float((eq_lb.iloc[-1] / cm_lb.iloc[-1] - 1)) if cm_lb.iloc[-1] > 0 else 0
                    if cur_dd < dd_stop:
                        locked_until = current_idx + lock_days
                        w_eff = pd.Series(0.0, index=rets.columns)
                    else:
                        w_eff = w
                else:
                    w_eff = w
            r = float((test_idx.iloc[i] * w_eff).sum())
            pnl.loc[ts] = r
            used.loc[ts] = True
            win_pnl.iloc[i] = r
        win_pnls.append(win_pnl)
        s += step
    return pnl[used], win_pnls


CONFIGS = []
# Universe × momentum (10 each universe × 5 universes = 50)
for u_name, syms in [("Agri_F", AGRI_F), ("Agri_ETF", AGRI_ETF),
                     ("Energy_F", ENERGY_F), ("Metal_F", METAL_F),
                     ("All_F", ALL_F)]:
    for m_name, mom, alpha in [
        ("8d_a1", ((8, 1.0),), 1.0),
        ("8d_a2", ((8, 1.0),), 2.0),
        ("8d_a0", ((8, 1.0),), 0.0),
        ("21d_a1", ((21, 1.0),), 1.0),
        ("21d_a2", ((21, 1.0),), 2.0),
        ("comp_a1", ((21, 0.3), (8, 0.7)), 1.0),
        ("comp_a2", ((21, 0.3), (8, 0.7)), 2.0),
        ("63d_a1", ((63, 1.0),), 1.0),
        ("5d_a1", ((5, 1.0),), 1.0),
        ("21+5_a1", ((21, 0.5), (5, 0.5)), 1.0),
    ]:
        CONFIGS.append({"name": f"{u_name}_{m_name}", "syms": syms,
                        "top_k": 2 if u_name in ("Agri_ETF", "Energy_F", "Metal_F") else 3,
                        "alpha": alpha, "mom": mom})

assert len(CONFIGS) == 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()
    rd = args.round

    cfg = CONFIGS[(rd - 1) % len(CONFIGS)]
    print(f"[round {rd}] {cfg['name']}")

    closes = load_close(cfg["syms"])
    print(f"  universe: {closes.shape[1]}")
    rets = closes.pct_change().fillna(0)

    t0 = time.time()
    pnl, win_pnls = champ_wf_simple(rets, top_k=cfg["top_k"],
                                      cvar_alpha=cfg["alpha"],
                                      momentum_periods=cfg["mom"])
    if not win_pnls:
        return
    m = wf_metrics(pnl, win_pnls)
    elapsed = time.time() - t0

    result = {
        "round": rd, "config": cfg["name"],
        "params": {k: v for k, v in cfg.items() if k != "syms"},
        "wf_sharpe": m["mean_sharpe"],
        "median_sharpe": m.get("median_sharpe"),
        "full_cagr_pct": m["CAGR"] * 100,
        "full_mdd_pct": m["MDD"] * 100,
        "n_windows": m["n_windows"],
        "neg_windows": m["neg_windows"],
        "win_rate": (m["n_windows"] - m["neg_windows"]) / m["n_windows"] * 100,
        "elapsed_sec": elapsed,
    }
    print(f"  Sh={m['mean_sharpe']:.2f} CAGR={m['CAGR']*100:.0f}% MDD={m['MDD']*100:.1f}% "
          f"Win={result['win_rate']:.0f}% ({elapsed:.0f}s)")

    out_path = RESULTS_DIR / f"iter_verified_sweep_r{rd}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

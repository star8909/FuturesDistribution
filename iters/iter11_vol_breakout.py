"""iter11: Volatility breakout (Bollinger band 돌파).

가설: 가격이 20일 평균 + 2σ 돌파 시 → trend 시작 → long.
+ DD-10% lock으로 catastrophic 차단.

iter04 logic은 momentum z-score 기반.
iter11은 volatility breakout 기반 — 다른 angle.
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
from src.futures_universe import AGRI_FUTURES, ENERGY_FUTURES, METAL_FUTURES, INDEX_FUTURES
from src.backtest import metrics, wf_metrics


def bb_breakout_wf(closes, cash_rets,
                   bb_lookback=20, bb_z=2.0,
                   top_k=2, dd_stop=-0.10, lock_days=63,
                   train=504, test=252, step=126,
                   fee_per_change=0.0050):
    """Bollinger band breakout: 가격이 MA + z*sigma 돌파 종목만 long."""
    rets = closes.pct_change().fillna(0)
    full = pd.concat([rets, cash_rets], axis=1).fillna(0)
    n = len(full)
    pnl = pd.Series(0.0, index=full.index)
    used = pd.Series(False, index=full.index)
    s = 0
    max_p = 1008

    def compute(end_idx):
        if end_idx < max_p:
            return None
        # Bollinger band 계산 (20일 MA + z*sigma)
        rec_close = closes.iloc[end_idx - bb_lookback:end_idx]
        ma = rec_close.mean()
        std = rec_close.std()
        upper = ma + bb_z * std
        # 현재 가격이 upper 돌파한 종목
        current = closes.iloc[end_idx - 1]
        breakout = (current > upper)
        breakout_score = ((current - ma) / std).where(breakout, 0)  # z-score above MA
        candidates = breakout_score[breakout_score > 0].sort_values(ascending=False)
        if candidates.empty:
            return None
        topk = min(top_k, len(candidates))
        top = list(candidates.head(topk).index)
        w = pd.Series(0.0, index=full.columns)
        for c in top:
            w.loc[c] = 1.0 / len(top)
        return w

    locked_until = -1
    window_pnls = []
    while s + train + test <= n:
        if s + train < max_p:
            s += step
            continue
        w = compute(s + train)
        if w is None:
            w = pd.Series(0.0, index=full.columns)
            for c in cash_rets.columns:
                w[c] = 1.0 / len(cash_rets.columns)
        test_idx = full.iloc[s + train:s + train + test]
        win_pnl = pd.Series(0.0, index=test_idx.index)
        for i in range(len(test_idx)):
            ts = test_idx.index[i]
            cost = 0.0
            current_idx = s + train + i
            if locked_until > current_idx:
                w_eff = pd.Series(0.0, index=full.columns)
                for c in cash_rets.columns:
                    w_eff[c] = 1.0 / len(cash_rets.columns)
            else:
                lookback_pnl = pnl.iloc[max(0, current_idx-252):current_idx]
                if len(lookback_pnl) > 30:
                    eq_lb = (1 + lookback_pnl).cumprod()
                    cm_lb = eq_lb.cummax()
                    current_dd = float((eq_lb.iloc[-1] / cm_lb.iloc[-1] - 1)) if cm_lb.iloc[-1] > 0 else 0
                    if current_dd < dd_stop:
                        locked_until = current_idx + lock_days
                        w_eff = pd.Series(0.0, index=full.columns)
                        for c in cash_rets.columns:
                            w_eff[c] = 1.0 / len(cash_rets.columns)
                    else:
                        w_eff = w
                else:
                    w_eff = w
            if i > 0 and i % 2 == 0 and locked_until <= current_idx:
                end_pos = s + train + i
                new_w = compute(end_pos)
                if new_w is not None:
                    turnover = (new_w - w).abs().sum()
                    if turnover < 0.5:
                        pass
                    else:
                        cost = turnover * fee_per_change
                        w = new_w
                        w_eff = w
            r = float((test_idx.iloc[i] * w_eff).sum()) - cost
            pnl.loc[ts] = r
            used.loc[ts] = True
            win_pnl.iloc[i] = r
        window_pnls.append(win_pnl)
        s += step
    return pnl[used], window_pnls


def run_cat(name, syms, cash):
    closes = load_close(syms, since="2010-01-01")
    if closes.empty or closes.shape[1] < 2:
        return {}
    print(f"\n=== {name} BB breakout sweep ===")
    results = {}
    for lb, z in [(20, 2.0), (20, 1.5), (10, 2.0), (50, 2.0), (50, 1.5)]:
        try:
            pnl, win_pnls = bb_breakout_wf(closes, cash, bb_lookback=lb, bb_z=z, top_k=2)
            m = wf_metrics(pnl, win_pnls)
            results[f"lb{lb}_z{z}"] = m
            neg = m['neg_windows']; nw = m['n_windows']
            color = "🚀" if m['mean_sharpe'] > 2.0 and m['MDD'] > -0.20 else \
                    "✅" if m['mean_sharpe'] > 1.0 else \
                    "⚠️" if m['mean_sharpe'] > 0.3 else "❌"
            print(f"  {color} lb={lb} z={z}: Sharpe={m['mean_sharpe']:.2f} (win {nw-neg}/{nw}) CAGR={m['CAGR']*100:.1f}% MDD={m['MDD']*100:.1f}%")
        except Exception as e:
            print(f"  lb{lb} z{z}: ERROR {e}")
    return results


def main():
    print("[iter11] Volatility breakout (Bollinger band)")
    cash = load_close(["TLT", "GLD"], since="2010-01-01").pct_change().fillna(0)

    out = {}
    out["agri"] = run_cat("Agri", AGRI_FUTURES, cash)
    out["energy"] = run_cat("Energy", ENERGY_FUTURES, cash)
    out["metal"] = run_cat("Metal", METAL_FUTURES, cash)
    out["index"] = run_cat("Index", INDEX_FUTURES, cash)

    out_path = RESULTS_DIR / "iter11_vol_breakout.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()

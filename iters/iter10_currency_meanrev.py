"""iter10: 통화/금리 mean-reversion (momentum 부적합 자산엔 반대 logic).

iter01 결과:
- Currency (6E/6J/6B/...): Sharpe -2.66 — momentum 음수
- Rate (ZT/ZF/ZN/ZB): Sharpe -2.80 — momentum 음수

가설: 이런 자산들은 mean-reverting → momentum 반대 (high → short, low → long).
Z-score가 +2σ면 short, -2σ면 long.
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
from src.futures_universe import CURRENCY_FUTURES, RATE_FUTURES


def metrics(pnl):
    pnl = pnl.dropna()
    if len(pnl) == 0:
        return {"CAGR": 0, "Sharpe": 0, "MDD": 0}
    eq = (1 + pnl).cumprod()
    n_years = max((pnl.index[-1] - pnl.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / n_years) - 1)
    sharpe = float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252)) if pnl.std(ddof=1) > 0 else 0
    cm = eq.cummax()
    return {"CAGR": cagr, "Sharpe": sharpe, "MDD": float((eq / cm - 1).min())}


def mean_rev_wf(rets, cash_rets, lookback=21, z_thr=1.5,
                top_k=2, dd_stop=-0.10, lock_days=63,
                train=504, test=252, step=126,
                fee_per_change=0.0050):
    """Mean reversion: z-score 극단 → 반대 베팅."""
    full = pd.concat([rets, cash_rets], axis=1).fillna(0)
    n = len(full)
    pnl = pd.Series(0.0, index=full.index)
    used = pd.Series(False, index=full.index)
    s = 0
    max_p = 1008

    def compute(end_idx):
        if end_idx < max_p:
            return None
        # Z-score 계산
        rec = rets.iloc[end_idx - lookback:end_idx]
        cum = (1 + rec.fillna(0)).prod() - 1
        z = (cum - cum.mean()) / cum.std()

        # Mean reversion: z 극단 → 반대 (z<0 → long, z>0 → short)
        # 하지만 long-only 가정 (한투 선물도 short 가능하지만 단순화)
        # → z < -z_thr인 종목만 long (반등 기대)
        candidates = z[z < -z_thr].sort_values()  # 가장 떨어진 것
        if len(candidates) == 0:
            return None

        topk = min(top_k, len(candidates))
        top = list(candidates.head(topk).index)
        w = pd.Series(0.0, index=full.columns)
        for c in top:
            w.loc[c] = 1.0 / len(top)
        return w

    locked_until = -1
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
        s += step
    return pnl[used]


def main():
    print("[iter10] Currency/Rate mean-reversion")
    cash = load_close(["TLT", "GLD"], since="2010-01-01").pct_change().fillna(0)

    cur_rets = load_close(CURRENCY_FUTURES, since="2010-01-01").pct_change().fillna(0)
    rate_rets = load_close(RATE_FUTURES, since="2010-01-01").pct_change().fillna(0)
    print(f"  Currency: {cur_rets.shape[1]} 종목, Rate: {rate_rets.shape[1]} 종목")

    print(f"\n=== Currency mean-reversion sweep ===")
    cur_results = {}
    for lb, z_thr in [(21, 1.0), (21, 1.5), (21, 2.0), (42, 1.5), (63, 2.0)]:
        try:
            pnl = mean_rev_wf(cur_rets, cash, lookback=lb, z_thr=z_thr, top_k=2)
            m = metrics(pnl)
            cur_results[f"lb{lb}_z{z_thr}"] = m
            color = "🚀" if m['Sharpe'] > 2 else "✅" if m['Sharpe'] > 1 else "⚠️" if m['Sharpe'] > 0 else "❌"
            print(f"  {color} lb={lb} z>{z_thr}: Sharpe={m['Sharpe']:.2f} CAGR={m['CAGR']*100:.1f}% MDD={m['MDD']*100:.1f}%")
        except Exception as e:
            print(f"  lb{lb} z{z_thr}: ERROR {e}")

    print(f"\n=== Rate mean-reversion sweep ===")
    rate_results = {}
    for lb, z_thr in [(21, 1.0), (21, 1.5), (21, 2.0), (42, 1.5), (63, 2.0)]:
        try:
            pnl = mean_rev_wf(rate_rets, cash, lookback=lb, z_thr=z_thr, top_k=2)
            m = metrics(pnl)
            rate_results[f"lb{lb}_z{z_thr}"] = m
            color = "🚀" if m['Sharpe'] > 2 else "✅" if m['Sharpe'] > 1 else "⚠️" if m['Sharpe'] > 0 else "❌"
            print(f"  {color} lb={lb} z>{z_thr}: Sharpe={m['Sharpe']:.2f} CAGR={m['CAGR']*100:.1f}% MDD={m['MDD']*100:.1f}%")
        except Exception as e:
            print(f"  lb{lb} z{z_thr}: ERROR {e}")

    out = {"currency": cur_results, "rate": rate_results}
    out_path = RESULTS_DIR / "iter10_currency_meanrev.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()

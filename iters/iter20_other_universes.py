"""iter20: 다른 universe 시도 (Sugar/Coffee/Cocoa 단독, Soft commodities, etc).

iter17 baseline: Agri 10종목 → Sharpe 4.69
iter20: 다른 카테고리 sub-universe 시도 — soft (KC/SB/CC/CT/OJ/SS) vs grain (ZC/ZW/ZS/ZL/ZM).
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
from hmmlearn.hmm import GaussianHMM

from src.config import RESULTS_DIR
from src.data_loader import load_close


GRAIN = ["ZC=F", "ZW=F", "ZS=F", "ZL=F", "ZM=F"]
SOFT = ["KC=F", "SB=F", "CT=F", "CC=F", "OJ=F"]
ENERGY_METAL = ["CL=F", "GC=F", "SI=F", "HG=F", "NG=F"]
GOLD_SILVER = ["GC=F", "SI=F"]


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


def champ_wf(rets, cash_rets,
             cvar_alpha=1.0, top_k=2, dd_stop=-0.10, lock_days=63,
             hmm_high=0.4, hmm_lookback=1008,
             train=504, test=252, step=126,
             fee_per_change=0.0050):
    full = pd.concat([rets, cash_rets], axis=1).fillna(0)
    n = len(full)
    pnl = pd.Series(0.0, index=full.index)
    used = pd.Series(False, index=full.index)
    s = 0
    max_p = max(hmm_lookback, 1008)
    market_rets = rets.mean(axis=1).dropna()

    def detect_hmm(end_idx):
        try:
            window = market_rets.iloc[end_idx - hmm_lookback:end_idx].dropna()
            if len(window) < 100:
                return 0
            X = window.values.reshape(-1, 1)
            hmm = GaussianHMM(n_components=2, covariance_type='full', random_state=42, n_iter=50)
            hmm.fit(X)
            variances = np.array([np.diag(c).item() if c.shape == (1,1) else c.flatten()[0] for c in hmm.covars_])
            state_order = np.argsort(variances)
            return list(state_order).index(hmm.predict(X)[-1])
        except Exception:
            return 0

    def compute(end_idx):
        if end_idx < max_p:
            return None
        rec = rets.iloc[end_idx - 8:end_idx]
        rec = rec.loc[:, rec.notna().sum() >= 4]
        if rec.empty:
            return None
        cum = (1 + rec.fillna(0)).prod() - 1
        z = (cum - cum.mean()) / cum.std()
        composite = z.dropna()
        if composite.empty:
            return None
        window = rets.iloc[end_idx - 63:end_idx]
        cvar_5 = pd.Series(index=window.columns, dtype=float)
        for col in window.columns:
            ret = window[col].dropna()
            if len(ret) < 20:
                cvar_5[col] = 0.001
                continue
            threshold = np.percentile(ret, 5)
            tail = ret[ret <= threshold]
            cvar_5[col] = tail.mean() if len(tail) > 0 else 0.001
        cvar_abs = cvar_5.abs().reindex(composite.index).fillna(0.001)
        composite = composite / (cvar_abs ** cvar_alpha + 0.001)
        scores = composite[composite > 0].dropna()
        if scores.empty:
            return None
        topk = min(top_k, len(scores))
        top = list(scores.sort_values(ascending=False).head(topk).index)
        w = pd.Series(0.0, index=full.columns)
        vol_window = rets.iloc[end_idx - 21:end_idx]
        sigmas = vol_window[top].std(ddof=1).fillna(0.01).clip(lower=0.001)
        inv = 1.0 / sigmas
        inv = inv / inv.sum()
        for c in top:
            w.loc[c] = inv[c]
        for _ in range(8):
            over = w[top] > 0.30
            if not over.any():
                break
            excess = (w[top][over] - 0.30).sum()
            w.loc[w[top].index[over]] = 0.30
            rem = w[top][~over]
            if rem.sum() > 0:
                w.loc[rem.index] = rem + excess * (rem / rem.sum())
        if w[top].sum() > 0:
            w.loc[top] = w[top] / w[top].sum()
        state = detect_hmm(end_idx)
        size = 1.0 if state == 0 else hmm_high
        if size < 1.0:
            stock_total = w[top].sum()
            spare = stock_total * (1 - size)
            for c in top:
                w[c] *= size
            cash_cols = [c for c in cash_rets.columns]
            for c in cash_cols:
                w[c] = spare / len(cash_cols)
            if w.sum() > 0:
                w = w / w.sum()
        return w

    locked_until = -1
    while s + train + test <= n:
        if s + train < max_p:
            s += step
            continue
        w = compute(s + train)
        if w is None:
            s += step
            continue
        test_idx = full.iloc[s + train:s + train + test]
        for i in range(len(test_idx)):
            ts = test_idx.index[i]
            cost = 0.0
            current_idx = s + train + i
            if locked_until > current_idx:
                w_eff = pd.Series(0.0, index=full.columns)
                cash_cols = [c for c in cash_rets.columns]
                for c in cash_cols:
                    w_eff[c] = 1.0 / len(cash_cols)
            else:
                lookback_pnl = pnl.iloc[max(0, current_idx-252):current_idx]
                if len(lookback_pnl) > 30:
                    eq_lb = (1 + lookback_pnl).cumprod()
                    cm_lb = eq_lb.cummax()
                    current_dd = float((eq_lb.iloc[-1] / cm_lb.iloc[-1] - 1)) if cm_lb.iloc[-1] > 0 else 0
                    if current_dd < dd_stop:
                        locked_until = current_idx + lock_days
                        w_eff = pd.Series(0.0, index=full.columns)
                        cash_cols = [c for c in cash_rets.columns]
                        for c in cash_cols:
                            w_eff[c] = 1.0 / len(cash_cols)
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
    print("[iter20] 다른 sub-universe 시도")
    cash = load_close(["TLT", "GLD"], since="2010-01-01").pct_change().fillna(0)

    print(f"\n=== Sub-universe sweep ===")
    for name, syms, k in [
        ("Grain (ZC/ZW/ZS/ZL/ZM)",     GRAIN, 2),
        ("Soft (KC/SB/CT/CC/OJ)",       SOFT, 2),
        ("Energy+Metal (CL/GC/SI/HG/NG)", ENERGY_METAL, 2),
        ("Gold/Silver (GC/SI)",          GOLD_SILVER, 1),
    ]:
        try:
            closes = load_close(syms, since="2010-01-01")
            rets = closes.pct_change().fillna(0)
            if rets.empty or rets.shape[1] < 2:
                print(f"  {name}: 데이터 없음")
                continue
            pnl = champ_wf(rets, cash, top_k=k)
            m = metrics(pnl)
            color = "🚀" if m['Sharpe'] > 4.69 else "✅" if m['Sharpe'] > 3 else "⚠️" if m['Sharpe'] > 1.5 else "❌"
            print(f"  {color} {name} (k={k}): Sharpe={m['Sharpe']:.2f} CAGR={m['CAGR']*100:.1f}% MDD={m['MDD']*100:.1f}%")
        except Exception as e:
            print(f"  {name}: ERROR {e}")

    out_path = RESULTS_DIR / "iter20_other_universes.json"
    out_path.write_text("{}", encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()

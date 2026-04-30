"""iter08: Multi-asset DD-stop ensemble.

iter04 (Agri DD-10% lock 63d): Sharpe 3.94, MDD -14.3% — 메인 챔.
iter08: Agri + Energy + Metal 각 카테고리별 DD-10% lock 적용 → 분산 효과로
   더 안정 + Sharpe 유지 가능.

가설:
- 한 카테고리 DD 발동 시 다른 카테고리 영향 X (분리 운용)
- 각 카테고리 1/3 자본 → portfolio MDD ≈ avg(category MDDs) / sqrt(3)
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
from src.futures_universe import AGRI_FUTURES, ENERGY_FUTURES, METAL_FUTURES
from src.backtest import metrics, wf_metrics


def champ_wf_dd(rets, cash_rets,
                top_k=3, dd_stop=-0.10, lock_days=63,
                hmm_high=0.4, hmm_lookback=1008,
                train=504, test=252, step=126,
                fee_per_change=0.0050):
    """iter04 logic 그대로 — 한 카테고리에서 DD-10% lock 63d."""
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
        composite = pd.Series(0.0)
        all_z = {}
        for period, pw in [(21, 0.3), (8, 0.7)]:
            rec = rets.iloc[end_idx - period:end_idx]
            rec = rec.loc[:, rec.notna().sum() >= max(period // 2, 5)]
            if rec.empty:
                continue
            cum = (1 + rec.fillna(0)).prod() - 1
            z = (cum - cum.mean()) / cum.std()
            all_z[period] = z
            composite = composite.add(z * pw, fill_value=0)
        composite = composite.dropna()
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
        composite = composite / (cvar_abs ** 2.0 + 0.001)

        mask = pd.Series(True, index=composite.index)
        for p, z in all_z.items():
            z_aligned = z.reindex(composite.index)
            mask = mask & (z_aligned > 0)
        scores = composite[mask].dropna()
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
    window_pnls = []
    while s + train + test <= n:
        if s + train < max_p:
            s += step
            continue
        w = compute(s + train)
        if w is None:
            s += step
            continue
        test_idx = full.iloc[s + train:s + train + test]
        win_pnl = pd.Series(0.0, index=test_idx.index)
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
            win_pnl.iloc[i] = r
        window_pnls.append(win_pnl)
        s += step
    return pnl[used], window_pnls


def run_cat(name, syms, cash, top_k=3):
    closes = load_close(syms, since="2010-01-01")
    rets = closes.pct_change().fillna(0)
    if rets.empty:
        return None
    pnl, _win_pnls = champ_wf_dd(rets, cash, top_k=top_k, dd_stop=-0.10, lock_days=63)
    return pnl


def main():
    print("[iter08] Multi-asset DD-stop ensemble (Agri + Energy + Metal)")
    cash = load_close(["TLT", "GLD"], since="2010-01-01").pct_change().fillna(0)

    print("\n=== 카테고리별 DD-10% lock 63d ===")
    pnls = {}
    for name, syms, k in [
        ("Agri", AGRI_FUTURES, 3),
        ("Energy", ENERGY_FUTURES, 2),
        ("Metal", METAL_FUTURES, 2),
    ]:
        pnl = run_cat(name, syms, cash, top_k=k)
        if pnl is None or pnl.empty:
            print(f"  {name}: 데이터 없음")
            continue
        m = metrics(pnl)
        pnls[name] = pnl
        print(f"  {name} (k={k}): Sharpe={m['Sharpe']:.2f} CAGR={m['CAGR']*100:.1f}% MDD={m['MDD']*100:.1f}%")

    # 합치기 (1/3씩)
    print("\n=== Equal-weight ensemble (1/3 Agri + 1/3 Energy + 1/3 Metal) ===")
    if len(pnls) >= 2:
        ensemble = pd.concat(pnls.values(), axis=1).fillna(0).mean(axis=1)
        m = metrics(ensemble)
        color = "🚀" if m['Sharpe'] > 4 and m['MDD'] > -0.15 else \
                "✅" if m['Sharpe'] > 2 and m['MDD'] > -0.20 else \
                "⚠️" if m['Sharpe'] > 1 else "❌"
        print(f"  {color} Ensemble: Sharpe={m['Sharpe']:.2f} CAGR={m['CAGR']*100:.1f}% MDD={m['MDD']*100:.1f}%")
        print(f"\n  iter04 (Agri only) 비교: Sharpe 3.94, MDD -14.3%")
        if m['Sharpe'] > 3.94 and m['MDD'] > -0.14:
            print(f"  🚀 iter04 능가! 새 챔피언")

    # Agri vol 다양 + DD-10% 합치기
    print("\n=== Agri × Energy 50/50 ensemble (DD-10%) ===")
    if "Agri" in pnls and "Energy" in pnls:
        e2 = pd.concat([pnls["Agri"], pnls["Energy"]], axis=1).fillna(0).mean(axis=1)
        m2 = metrics(e2)
        color2 = "🚀" if m2['Sharpe'] > 4 and m2['MDD'] > -0.15 else \
                 "✅" if m2['Sharpe'] > 2 and m2['MDD'] > -0.20 else \
                 "⚠️" if m2['Sharpe'] > 1 else "❌"
        print(f"  {color2} Agri+Energy: Sharpe={m2['Sharpe']:.2f} CAGR={m2['CAGR']*100:.1f}% MDD={m2['MDD']*100:.1f}%")

    out = {
        "categories": {n: metrics(pnls[n]) for n in pnls},
        "ensemble_3way": metrics(ensemble) if len(pnls) >= 2 else None,
        "ensemble_agri_energy": metrics(e2) if "Agri" in pnls and "Energy" in pnls else None,
    }
    out_path = RESULTS_DIR / "iter08_multi_dd.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()

"""iter02: Agri 단독 + 안전형 (MDD 줄이기).

iter01 결과:
- Agri (n=10, k=3): Sharpe 3.11, CAGR 103% but MDD -86.8% (망함)
- 가설: 농산물 momentum 살아있지만 MDD 폭발이 문제

iter02 안전형:
1. Agri 단독 universe (10종목)
2. Top-K = 2~5 sweep
3. Inverse-vol weighting (1/σ_21d)
4. Position cap 0.25 (단일 종목 최대 25%)
5. Cash 비중 ↑ (HMM high state size = 0.3~0.5)
6. 다양한 fee 검증
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
from src.futures_universe import AGRI_FUTURES
from src.backtest import metrics, wf_metrics


def champ_wf(rets, cash_rets,
             top_k=3, weight_mode="invvol", weight_cap=0.25,
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

        if weight_mode == "equal":
            for c in top:
                w.loc[c] = 1.0 / len(top)
        elif weight_mode == "invvol":
            vol_window = rets.iloc[end_idx - 21:end_idx]
            sigmas = vol_window[top].std(ddof=1).fillna(0.01)
            sigmas = sigmas.replace(0, 0.01).clip(lower=0.001)
            inv = 1.0 / sigmas
            inv = inv / inv.sum()
            for c in top:
                w.loc[c] = inv[c]
            # cap
            if weight_cap is not None:
                for _ in range(8):
                    over = w[top] > weight_cap
                    if not over.any():
                        break
                    excess = (w[top][over] - weight_cap).sum()
                    w.loc[w[top].index[over]] = weight_cap
                    rem = w[top][~over]
                    if rem.sum() > 0:
                        w.loc[rem.index] = rem + excess * (rem / rem.sum())
                if w[top].sum() > 0:
                    w.loc[top] = w[top] / w[top].sum()

        # HMM mild — 더 강한 방어
        state = detect_hmm(end_idx)
        size = 1.0 if state == 0 else hmm_high
        if size < 1.0:
            stock_total = w[top].sum()
            spare = stock_total * (1 - size)
            for c in top:
                w[c] *= size
            cash_cols = [c for c in cash_rets.columns]
            cash_total = w[cash_cols].sum()
            if cash_total > 0:
                for c in cash_cols:
                    w[c] *= (1 + spare / cash_total)
            else:
                for c in cash_cols:
                    w[c] = spare / len(cash_cols)
            if w.sum() > 0:
                w = w / w.sum()
        return w

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
            if i > 0 and i % 2 == 0:
                end_pos = s + train + i
                new_w = compute(end_pos)
                if new_w is not None:
                    turnover = (new_w - w).abs().sum()
                    if turnover < 0.5:
                        pass
                    else:
                        cost = turnover * fee_per_change
                        w = new_w
            r = float((test_idx.iloc[i] * w).sum()) - cost
            pnl.loc[ts] = r
            used.loc[ts] = True
            win_pnl.iloc[i] = r
        window_pnls.append(win_pnl)
        s += step
    return pnl[used], window_pnls


def main():
    print("[iter02] Agri 단독 + 안전형 (MDD 줄이기)")
    print(f"  Universe: {AGRI_FUTURES}")

    closes = load_close(AGRI_FUTURES, since="2010-01-01")
    rets = closes.pct_change().fillna(0)
    print(f"  Loaded: {rets.shape[1]} agri futures")

    cash = load_close(["TLT", "GLD"], since="2010-01-01").pct_change().fillna(0)
    print(f"  Cash assets: {list(cash.columns)}")

    # configs: top_k × weight × cap × hmm_high
    configs = [
        ("baseline equal k=3 hmm0.6",         3, "equal",  None, 0.6),
        ("invvol k=3 cap=0.25 hmm0.4",        3, "invvol", 0.25, 0.4),
        ("invvol k=3 cap=0.30 hmm0.4",        3, "invvol", 0.30, 0.4),
        ("invvol k=2 cap=0.50 hmm0.3",        2, "invvol", 0.50, 0.3),
        ("invvol k=4 cap=0.25 hmm0.3",        4, "invvol", 0.25, 0.3),
        ("invvol k=5 cap=0.20 hmm0.4",        5, "invvol", 0.20, 0.4),
        ("invvol k=3 cap=0.25 hmm0.5",        3, "invvol", 0.25, 0.5),
        ("invvol k=3 cap=0.25 hmm0.2 (defensive)", 3, "invvol", 0.25, 0.2),
    ]
    print("\n=== Agri 안전형 sweep ===")
    results = {}
    for name, k, mode, cap, h in configs:
        try:
            pnl, win_pnls = champ_wf(rets, cash, top_k=k, weight_mode=mode, weight_cap=cap, hmm_high=h)
            m = wf_metrics(pnl, win_pnls)
            results[name] = m
            mdd_ok = m['MDD'] > -0.30
            neg = m['neg_windows']; nw = m['n_windows']
            color = "🚀" if mdd_ok and m['mean_sharpe'] > 2.0 else \
                    "✅" if mdd_ok and m['mean_sharpe'] > 1.0 else \
                    "⚠️" if m['mean_sharpe'] > 0.3 else "❌"
            print(f"  {color} {name}: Sharpe={m['mean_sharpe']:.2f} (win {nw-neg}/{nw}) CAGR={m['CAGR']*100:.1f}% MDD={m['MDD']*100:.1f}%")
        except Exception as e:
            print(f"  {name}: ERROR {e}")
            results[name] = {"error": str(e)}

    # Cost stress on best
    print("\n=== Cost Stress on baseline equal k=3 ===")
    cost_results = {}
    for fee_bps in [50, 100, 150]:
        pnl_c, win_pnls_c = champ_wf(rets, cash, top_k=3, weight_mode="equal", weight_cap=None,
                         hmm_high=0.6, fee_per_change=fee_bps/1e4)
        mc = wf_metrics(pnl_c, win_pnls_c)
        cost_results[fee_bps] = mc
        color = "✅" if mc['mean_sharpe'] > 1.0 else "⚠️" if mc['mean_sharpe'] > 0.3 else "❌"
        print(f"  {color} {fee_bps}bps: Sharpe={mc['mean_sharpe']:.2f} MDD={mc['MDD']*100:.1f}%")

    # 종합
    best_name = max(results, key=lambda k: results[k].get('mean_sharpe', 0) if isinstance(results[k], dict) and 'error' not in results[k] else -999)
    best = results[best_name]
    print(f"\n=== iter02 종합 ===")
    print(f"  최고: {best_name}")
    print(f"  Sharpe={best.get('mean_sharpe', 0):.2f} CAGR={best.get('CAGR', 0)*100:.1f}% MDD={best.get('MDD', 0)*100:.1f}%")
    if best.get('MDD', -1) > -0.30:
        print(f"  ✅ MDD < -30% — 실거래 가능 영역")
    elif best.get('MDD', -1) > -0.50:
        print(f"  ⚠️ MDD < -50% — 위험하지만 retail 시도 가능")
    else:
        print(f"  ❌ MDD > -50% — 실거래 불가, 추가 조정 필요")

    out = RESULTS_DIR / "iter02_agri_safe.json"
    out.write_text(json.dumps({
        "results": results,
        "cost_stress": cost_results,
        "best": best_name,
    }, indent=2, ensure_ascii=False))
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()

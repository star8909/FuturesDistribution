"""iter05: Cash-heavy portfolio (의무 cash 비중 50%+).

iter02 결과로 HMM/cap/invvol은 MDD에 무력 확인.
iter05 가설: notional 자체를 줄이면 MDD 비례적으로 줄어듬.

50% cash + 30% Agri + 20% bonds → MDD ≈ -42% 예상 (Agri 단독 -85% × 0.5)
70% cash + 20% Agri + 10% bonds → MDD ≈ -17% 예상

목표: Sharpe 1.0+ + MDD < -25%
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
             top_k=2, agri_alloc=0.30, cash_alloc=0.70,
             hmm_high=0.5, hmm_lookback=1008,
             train=504, test=252, step=126,
             fee_per_change=0.0050):
    """agri_alloc + cash_alloc = 1.0 강제. agri_alloc 비중만 momentum logic."""
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

        # Inverse-vol weighting within top-K
        vol_window = rets.iloc[end_idx - 21:end_idx]
        sigmas = vol_window[top].std(ddof=1).fillna(0.01).clip(lower=0.001)
        inv = 1.0 / sigmas
        inv = inv / inv.sum()

        # 강제 cash allocation 분리
        w = pd.Series(0.0, index=full.columns)
        # Agri 비중: agri_alloc
        for c in top:
            w.loc[c] = inv[c] * agri_alloc

        # Cash 비중: cash_alloc (TLT, GLD 균등)
        cash_cols = [c for c in cash_rets.columns]
        for c in cash_cols:
            w[c] = cash_alloc / len(cash_cols)

        # HMM mild — high state면 agri → cash로 추가 이동
        state = detect_hmm(end_idx)
        if state == 1:
            for c in top:
                spare = w[c] * (1 - hmm_high)
                w[c] *= hmm_high
                # spare를 cash에 분산
                for cc in cash_cols:
                    w[cc] += spare / len(cash_cols)

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
    print("[iter05] Cash-heavy portfolio (Agri 30~50% + Cash 70~50%)")
    cash = load_close(["TLT", "GLD"], since="2010-01-01").pct_change().fillna(0)
    closes = load_close(AGRI_FUTURES, since="2010-01-01")
    rets = closes.pct_change().fillna(0)
    print(f"  Agri: {rets.shape[1]} 종목, Cash: {list(cash.columns)}")

    # (name, agri_alloc, cash_alloc, top_k, hmm_high)
    configs = [
        ("Agri 50% + Cash 50%, k=2",      0.50, 0.50, 2, 0.5),
        ("Agri 40% + Cash 60%, k=2",      0.40, 0.60, 2, 0.5),
        ("Agri 30% + Cash 70%, k=2",      0.30, 0.70, 2, 0.5),
        ("Agri 20% + Cash 80%, k=2",      0.20, 0.80, 2, 0.5),
        ("Agri 30% + Cash 70%, k=3",      0.30, 0.70, 3, 0.5),
        ("Agri 30% + Cash 70%, k=2 hmm0.3", 0.30, 0.70, 2, 0.3),
        ("Agri 25% + Cash 75%, k=2 hmm0.3", 0.25, 0.75, 2, 0.3),
        ("Agri 15% + Cash 85%, k=1 (극보수)", 0.15, 0.85, 1, 0.3),
    ]
    results = {}
    for name, ag, ca, k, h in configs:
        try:
            pnl, win_pnls = champ_wf(rets, cash, top_k=k, agri_alloc=ag, cash_alloc=ca, hmm_high=h)
            m = wf_metrics(pnl, win_pnls)
            results[name] = m
            neg = m['neg_windows']; nw = m['n_windows']
            color = "🚀" if m['mean_sharpe'] > 2.0 and m['MDD'] > -0.20 else \
                    "✅" if m['mean_sharpe'] > 1.0 and m['MDD'] > -0.25 else \
                    "⚠️" if m['mean_sharpe'] > 0.3 and m['MDD'] > -0.40 else "❌"
            print(f"  {color} {name}: Sharpe={m['mean_sharpe']:.2f} (win {nw-neg}/{nw}) CAGR={m['CAGR']*100:.1f}% MDD={m['MDD']*100:.1f}%")
        except Exception as e:
            print(f"  {name}: ERROR {e}")
            results[name] = {"error": str(e)}

    print("\n=== iter05 종합 ===")
    best_name = max(results, key=lambda k: results[k].get('mean_sharpe', -999) if isinstance(results[k], dict) and 'error' not in results[k] else -999)
    best = results[best_name]
    print(f"  최고 Sharpe: {best_name} → {best.get('mean_sharpe', 0):.2f} MDD={best.get('MDD', 0)*100:.1f}%")

    # MDD가 -25% 안에 있는 best 찾기
    safe = {k: v for k, v in results.items() if isinstance(v, dict) and 'error' not in v and v.get('MDD', -1) > -0.25}
    if safe:
        safe_best = max(safe, key=lambda k: safe[k]['mean_sharpe'])
        print(f"  MDD -25% 이내 최고: {safe_best} → Sharpe {safe[safe_best]['mean_sharpe']:.2f} MDD {safe[safe_best]['MDD']*100:.1f}%")
        if safe[safe_best]['mean_sharpe'] > 1.0:
            print(f"  ✅ 실거래 가능 영역 진입!")
    else:
        print(f"  ❌ 모든 config가 MDD -25% 초과")

    out = RESULTS_DIR / "iter05_cash_heavy.json"
    out.write_text(json.dumps({"results": results, "best": best_name}, indent=2, ensure_ascii=False))
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()

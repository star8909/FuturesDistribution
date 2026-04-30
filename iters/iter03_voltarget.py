"""iter03: Vol-targeting (연 10% 변동성 목표) + Agri 단독.

iter01 문제: MDD -77~-100% 폭발 (raw notional 1x).
iter03 해결: 포지션 크기를 vol target으로 자동 조절.

Vol targeting:
  position_size = target_vol / realized_vol
  if realized_vol = 30% (선물 평균), target = 10% → size = 0.33 (1/3 노출)

목표: Sharpe 1.5+ 유지 + MDD < -25%
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


def champ_wf(rets, cash_rets,
             top_k=3, target_vol=0.10, vol_lookback=21,
             max_size=1.0,
             hmm_high=0.4, hmm_lookback=1008,
             train=504, test=252, step=126,
             fee_per_change=0.0050):
    """Vol-targeting 적용. position_size = target_vol / realized_portfolio_vol."""
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

        # Inverse-vol weighting (within top-K)
        vol_window = rets.iloc[end_idx - vol_lookback:end_idx]
        sigmas = vol_window[top].std(ddof=1).fillna(0.01).clip(lower=0.001)
        inv = 1.0 / sigmas
        inv = inv / inv.sum()

        # 포트폴리오 vol estimate
        # 단순 가정: portfolio_vol ≈ sqrt(sum(w_i^2 * sigma_i^2))  (correlation 무시)
        portfolio_vol_daily = np.sqrt(((inv ** 2) * (sigmas ** 2)).sum())
        portfolio_vol_annual = portfolio_vol_daily * np.sqrt(252)

        # Vol target sizing: scale to target_vol
        size = min(target_vol / max(portfolio_vol_annual, 0.01), max_size)

        # HMM mild
        state = detect_hmm(end_idx)
        if state == 1:
            size *= hmm_high  # high state면 추가 축소

        w = pd.Series(0.0, index=full.columns)
        for c in top:
            w.loc[c] = inv[c] * size

        # Spare → cash
        spare = 1.0 - w.sum()
        cash_cols = [c for c in cash_rets.columns]
        if spare > 0:
            for c in cash_cols:
                w[c] = spare / len(cash_cols)
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


def run_universe(name, syms, cash):
    print(f"\n=== {name} (n={len(syms)}) ===")
    closes = load_close(syms, since="2010-01-01")
    rets = closes.pct_change().fillna(0)
    if rets.empty:
        return {}
    results = {}
    targets = [(0.05, "vol 5%"), (0.08, "vol 8%"), (0.10, "vol 10%"), (0.15, "vol 15%"), (0.20, "vol 20%")]
    for target, label in targets:
        try:
            pnl, win_pnls = champ_wf(rets, cash, top_k=3, target_vol=target)
            m = wf_metrics(pnl, win_pnls)
            results[label] = m
            neg = m['neg_windows']; nw = m['n_windows']
            color = "🚀" if m['mean_sharpe'] > 2.0 and m['MDD'] > -0.25 else \
                    "✅" if m['mean_sharpe'] > 1.0 and m['MDD'] > -0.30 else \
                    "⚠️" if m['mean_sharpe'] > 0.3 else "❌"
            print(f"  {color} {label}: Sharpe={m['mean_sharpe']:.2f} (win {nw-neg}/{nw}) CAGR={m['CAGR']*100:.1f}% MDD={m['MDD']*100:.1f}%")
        except Exception as e:
            print(f"  {label}: ERROR {e}")
            results[label] = {"error": str(e)}
    return results


def main():
    print("[iter03] Vol-targeting (연 10% 목표) + 카테고리 sweep")
    cash = load_close(["TLT", "GLD"], since="2010-01-01").pct_change().fillna(0)
    print(f"  Cash assets: {list(cash.columns)}")

    out = {}
    out["agri"] = run_universe("Agri (10종)", AGRI_FUTURES, cash)
    out["energy"] = run_universe("Energy (5종)", ENERGY_FUTURES, cash)
    out["metal"] = run_universe("Metal (5종)", METAL_FUTURES, cash)

    # 종합 결론
    print("\n=== iter03 Vol-target 종합 ===")
    best_each = {}
    for cat, results in out.items():
        if not results:
            continue
        best = max(results.items(), key=lambda kv: kv[1].get("mean_sharpe", -999) if isinstance(kv[1], dict) and 'error' not in kv[1] else -999)
        best_each[cat] = (best[0], best[1])
        print(f"  {cat}: 최고 = {best[0]}: Sharpe {best[1].get('mean_sharpe', 0):.2f} MDD {best[1].get('MDD', 0)*100:.1f}%")

    # 가장 우수한 카테고리/타겟
    best_cat = max(best_each.items(), key=lambda kv: kv[1][1].get('mean_sharpe', -999))
    print(f"\n  🏆 전체 최고: {best_cat[0]} {best_cat[1][0]} → Sharpe {best_cat[1][1]['mean_sharpe']:.2f}, MDD {best_cat[1][1]['MDD']*100:.1f}%")
    if best_cat[1][1].get('MDD', -1) > -0.25:
        print(f"  ✅ MDD < -25% 달성! 실거래 시도 영역")
    elif best_cat[1][1].get('MDD', -1) > -0.50:
        print(f"  ⚠️ MDD < -50% — 위험하지만 retail micro 선물 가능")
    else:
        print(f"  ❌ MDD > -50% — 추가 안전장치 필요")

    out_path = RESULTS_DIR / "iter03_voltarget.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()

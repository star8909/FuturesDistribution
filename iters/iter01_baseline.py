"""iter01: 해외선물 universe + R70 logic baseline.

EquityDistribution R70 (composite z 21d+8d + CVaR α=2.0 + HMM 4y mild + single-K) 을
한투 해외선물 universe (~30개) 에 그대로 이식.

가설: 선물은 trend-following이 정통이라 momentum logic 통할 가능성 ↑.
But 효율적 시장이라 retail 알파 적을 듯. Sharpe 1~3 예상.

검증:
1. 전체 선물 universe baseline
2. 카테고리별 분리 (index/energy/metal/agri 등)
3. Cost stress (50/100/150bps — 선물은 비용 더 높음)
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
from src.futures_universe import all_futures, all_futures_flat, INDEX_FUTURES, ENERGY_FUTURES, METAL_FUTURES, AGRI_FUTURES, CURRENCY_FUTURES, RATE_FUTURES
from src.backtest import metrics, wf_metrics


def champ_wf(rets, cash_rets,
             top_k=4, periods=(21, 8), weights_p=(0.3, 0.7),
             cvar_alpha=2.0, cvar_lookback=63,
             hmm_high=0.6, hmm_lookback=1008,
             train=504, test=252, step=126,
             fee_per_change=0.0050):  # 50bps default — 선물은 더 높음
    """R70 logic 적용. cash_rets = 현금성 (TLT나 0).

    Universe 작아서 (30개) top_k=4 (8 → 4 비례 축소).
    """
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
        for period, pw in zip(periods, weights_p):
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

        window = rets.iloc[end_idx - cvar_lookback:end_idx]
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

        # strict_pos
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
        for c in top:
            w.loc[c] = 1.0 / len(top)

        # HMM mild — high vol → cash로 spare
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
    print("[iter01] FuturesDistribution baseline (R70 logic 이식)")
    universe = all_futures_flat()
    print(f"  Universe: {len(universe)} futures")

    # 데이터 로드
    closes = load_close(universe, since="2010-01-01")
    rets = closes.pct_change().fillna(0)
    print(f"  Loaded: {rets.shape[1]} 선물 (cache hit)")

    # 현금성: TLT (long bond) + GLD (안전자산)
    cash = load_close(["TLT", "GLD"], since="2010-01-01").pct_change().fillna(0)
    print(f"  Cash assets: {list(cash.columns)}")

    # 1. 전체 universe + R70 logic
    print("\n=== R70 logic on Futures (전체 universe) ===")
    pnl, win_pnls = champ_wf(rets, cash, top_k=4)
    m = wf_metrics(pnl, win_pnls)
    neg = m['neg_windows']; nw = m['n_windows']
    color = "🚀" if m['mean_sharpe'] > 2.0 else "✅" if m['mean_sharpe'] > 1.0 else "⚠️" if m['mean_sharpe'] > 0.3 else "❌"
    print(f"  {color} 전체 + top_k=4: Sharpe={m['mean_sharpe']:.2f} (win {nw-neg}/{nw}) CAGR={m['CAGR']*100:.1f}% MDD={m['MDD']*100:.1f}%")

    # 2. 카테고리별
    print("\n=== 카테고리별 R70 logic ===")
    categories = {
        "Index": INDEX_FUTURES,
        "Energy": ENERGY_FUTURES,
        "Metal": METAL_FUTURES,
        "Agri": AGRI_FUTURES,
        "Currency": CURRENCY_FUTURES,
        "Rate": RATE_FUTURES,
    }
    cat_results = {}
    for cat_name, syms in categories.items():
        sub_rets = rets[[s for s in syms if s in rets.columns]]
        if sub_rets.empty or sub_rets.shape[1] < 3:
            print(f"  {cat_name}: skip (too few)")
            continue
        # top_k = min(2, len(syms)//3)
        top_k_cat = max(2, sub_rets.shape[1] // 3)
        sub_pnl, sub_win_pnls = champ_wf(sub_rets, cash, top_k=top_k_cat)
        sm = wf_metrics(sub_pnl, sub_win_pnls)
        cat_results[cat_name] = sm
        neg = sm['neg_windows']; nw = sm['n_windows']
        color = "🚀" if sm['mean_sharpe'] > 2.0 else "✅" if sm['mean_sharpe'] > 1.0 else "⚠️" if sm['mean_sharpe'] > 0.3 else "❌"
        print(f"  {color} {cat_name} (n={sub_rets.shape[1]}, k={top_k_cat}): Sharpe={sm['mean_sharpe']:.2f} (win {nw-neg}/{nw}) CAGR={sm['CAGR']*100:.1f}% MDD={sm['MDD']*100:.1f}%")

    # 3. Cost stress (선물 더 높음)
    print("\n=== Cost Stress (선물 50/100/150/200 bps) ===")
    cost_results = {}
    for fee_bps in [50, 100, 150, 200]:
        pnl_c, win_pnls_c = champ_wf(rets, cash, top_k=4, fee_per_change=fee_bps/1e4)
        mc = wf_metrics(pnl_c, win_pnls_c)
        cost_results[fee_bps] = mc
        color = "✅" if mc['mean_sharpe'] > 1.0 else "⚠️" if mc['mean_sharpe'] > 0.3 else "❌"
        print(f"  {color} {fee_bps}bps: Sharpe={mc['mean_sharpe']:.2f} CAGR={mc['CAGR']*100:.1f}% MDD={mc['MDD']*100:.1f}%")

    # 종합
    print("\n=== 선물 baseline 종합 ===")
    full_sharpe = m['mean_sharpe']
    if full_sharpe > 2.0:
        print(f"  🚀 전체 universe Sharpe {full_sharpe:.2f} — 주식만큼 강함!")
    elif full_sharpe > 1.0:
        print(f"  ✅ 전체 universe Sharpe {full_sharpe:.2f} — 의미 있음, 분산 효과 가능")
    elif full_sharpe > 0.3:
        print(f"  ⚠️ 전체 universe Sharpe {full_sharpe:.2f} — 약함, 추가 시그널 필요")
    else:
        print(f"  ❌ 전체 universe Sharpe {full_sharpe:.2f} — R70 logic 부적합 (선물 특화 시그널 필요)")

    print(f"  Asia 주식 R70 비교: BT 11.74, 75bps 8.88")

    out = RESULTS_DIR / "iter01_baseline.json"
    out.write_text(json.dumps({
        "universe": m,
        "categories": cat_results,
        "cost_stress": cost_results,
    }, indent=2, ensure_ascii=False))
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()

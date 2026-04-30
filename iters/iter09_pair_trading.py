"""iter09: Pair trading (Agri long + Index short — 시장 중립).

iter04 (Agri only): Sharpe 3.94, MDD -14.3%
iter08 multi-asset: Energy/Metal 추가 시 Sharpe ↓

iter09 가설: Agri momentum long + Index short으로 시장 베타 제거.
   - Agri long: R70 logic + DD-10% lock
   - Index short: ES (S&P 500) 정적 short hedge

장점: 시장 폭락 시 Agri 손실 → Index short 수익 보전
단점: 강세장에서 Index short 손실 (Agri 수익 일부 상쇄)
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


def champ_wf_dd(rets, cash_rets,
                top_k=3, dd_stop=-0.10, lock_days=63,
                hmm_high=0.4, hmm_lookback=1008,
                train=504, test=252, step=126,
                fee_per_change=0.0050):
    """iter04 Agri logic 그대로."""
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


def main():
    print("[iter09] Pair trading — Agri long + ES short")
    cash = load_close(["TLT", "GLD"], since="2010-01-01").pct_change().fillna(0)
    agri_rets = load_close(AGRI_FUTURES, since="2010-01-01").pct_change().fillna(0)

    # ES (S&P 500 e-mini)
    es = load_close(["ES=F"], since="2010-01-01").pct_change().fillna(0)
    if es.empty:
        # SPY 폴백
        es = load_close(["SPY"], since="2010-01-01").pct_change().fillna(0)
    print(f"  Agri: {agri_rets.shape[1]} 종목 / Hedge: {list(es.columns)}")

    # Agri long PnL (iter04)
    pnl_agri, win_pnls_agri = champ_wf_dd(agri_rets, cash, top_k=3, dd_stop=-0.10, lock_days=63)
    m_agri = wf_metrics(pnl_agri, win_pnls_agri)
    print(f"\n=== Agri only (iter04) ===")
    print(f"  Sharpe={m_agri['mean_sharpe']:.2f} CAGR={m_agri['CAGR']*100:.1f}% MDD={m_agri['MDD']*100:.1f}%")

    # Index short pair (다양한 hedge ratio)
    print(f"\n=== Pair: Agri long + ES short (다양한 hedge ratio) ===")
    if 'ES=F' in es.columns or 'SPY' in es.columns:
        es_col = 'ES=F' if 'ES=F' in es.columns else 'SPY'
        es_rets_aligned = es[es_col].reindex(pnl_agri.index).fillna(0)

        for hedge_ratio in [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00]:
            pair_pnl = pnl_agri - hedge_ratio * es_rets_aligned
            m_pair = metrics(pair_pnl)
            color = "🚀" if m_pair['Sharpe'] > 4 and m_pair['MDD'] > -0.12 else \
                    "✅" if m_pair['Sharpe'] > 3 and m_pair['MDD'] > -0.18 else \
                    "⚠️" if m_pair['Sharpe'] > 2 else "❌"
            print(f"  {color} Hedge {hedge_ratio*100:>3.0f}%: Sharpe={m_pair['Sharpe']:.2f} "
                  f"CAGR={m_pair['CAGR']*100:.1f}% MDD={m_pair['MDD']*100:.1f}%")
    else:
        print(f"  ❌ ES/SPY 데이터 없음")

    out = {
        "agri_only": m_agri,
        "pair_results": {},
    }

    if 'ES=F' in es.columns or 'SPY' in es.columns:
        es_col = 'ES=F' if 'ES=F' in es.columns else 'SPY'
        es_rets_aligned = es[es_col].reindex(pnl_agri.index).fillna(0)
        for hedge_ratio in [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00]:
            pair_pnl = pnl_agri - hedge_ratio * es_rets_aligned
            m_pair = metrics(pair_pnl)
            out["pair_results"][f"hedge_{int(hedge_ratio*100)}"] = m_pair

    out_path = RESULTS_DIR / "iter09_pair_trading.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()

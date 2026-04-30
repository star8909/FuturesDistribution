"""iter04: Equity curve 기반 stop (DD -15% 도달 시 cash 1년 lock).

iter01 문제: MDD -77% (등락 누적).
iter04 해결: 누적 DD 모니터링 + DD>-15% 시 강제 cash 1년.

가설: 큰 drawdown은 잠시 멈추면 회복. cash 동안 alpha 잃지만 catastrophic 손실 방지.
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
             top_k=3, dd_stop=-0.15, lock_days=63,
             hmm_high=0.4, hmm_lookback=1008,
             train=504, test=252, step=126,
             fee_per_change=0.0050):
    """DD stop이 도달하면 lock_days만큼 cash 강제."""
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

        # Cap 0.30
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

    locked_until = -1  # DD stop locked 종료 시점 (인덱스)
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

            # DD stop 체크: 직전 252일 누적 PnL
            current_idx = s + train + i
            if locked_until > current_idx:
                # cash 강제
                w_eff = pd.Series(0.0, index=full.columns)
                cash_cols = [c for c in cash_rets.columns]
                for c in cash_cols:
                    w_eff[c] = 1.0 / len(cash_cols)
            else:
                # DD 측정 (직전 252일)
                lookback_pnl = pnl.iloc[max(0, current_idx-252):current_idx]
                if len(lookback_pnl) > 30:
                    eq_lb = (1 + lookback_pnl).cumprod()
                    cm_lb = eq_lb.cummax()
                    current_dd = float((eq_lb.iloc[-1] / cm_lb.iloc[-1] - 1)) if cm_lb.iloc[-1] > 0 else 0
                    if current_dd < dd_stop:
                        # Stop 발동
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
    print("[iter04] Equity curve stop (DD -15% → cash 63일)")
    cash = load_close(["TLT", "GLD"], since="2010-01-01").pct_change().fillna(0)
    closes = load_close(AGRI_FUTURES, since="2010-01-01")
    rets = closes.pct_change().fillna(0)
    print(f"  Agri: {rets.shape[1]} 종목")

    # DD stop level × lock days 조합
    configs = [
        ("baseline (no stop)", -1.0, 0),
        ("DD-10% lock 63d (3개월)", -0.10, 63),
        ("DD-15% lock 63d", -0.15, 63),
        ("DD-15% lock 126d (6개월)", -0.15, 126),
        ("DD-15% lock 252d (1년)", -0.15, 252),
        ("DD-20% lock 126d", -0.20, 126),
        ("DD-25% lock 126d", -0.25, 126),
        ("DD-15% + lock 30d (빠른 복귀)", -0.15, 30),
    ]
    results = {}
    for name, dd, lock in configs:
        try:
            pnl, win_pnls = champ_wf(rets, cash, dd_stop=dd, lock_days=lock)
            m = wf_metrics(pnl, win_pnls)
            results[name] = m
            sh = m['mean_sharpe']
            neg = m['neg_windows']
            n_win = m['n_windows']
            color = "🚀" if sh > 2.0 else "✅" if sh > 1.0 else "⚠️" if sh > 0.3 else "❌"
            print(f"  {color} {name}: Sharpe={sh:.2f} (win {n_win-neg}/{n_win}) CAGR={m['CAGR']*100:.1f}% MDD={m['MDD']*100:.1f}%")
        except Exception as e:
            print(f"  {name}: ERROR {e}")
            results[name] = {"error": str(e)}

    print("\n=== iter04 종합 ===")
    best_name = max(results, key=lambda k: results[k].get('mean_sharpe', -999) if isinstance(results[k], dict) and 'error' not in results[k] else -999)
    best = results[best_name]
    print(f"  최고: {best_name}")
    print(f"  Sharpe={best.get('mean_sharpe', 0):.2f} MDD={best.get('MDD', 0)*100:.1f}%")

    out = RESULTS_DIR / "iter04_equity_stop.json"
    out.write_text(json.dumps({"results": results, "best": best_name}, indent=2, ensure_ascii=False))
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()

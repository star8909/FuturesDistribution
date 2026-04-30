"""iter19: iter17 (8d + α=1.0 + DD-10% lock 63d + k=2) ALL ROBUST 검증.

검증 항목:
1. Multi-OOS (4 setups) — 학습/테스트/step 변형
2. Regime (3 historical) — 베어 2018Q4 / 코로나 2020Q1 / 베어 2022
3. Crash (3 events) — 직전 N개월 가격 폭락 시점
4. Cost stress (50/100/150bps)
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
    print("[iter19] iter17 ALL ROBUST 검증")
    cash = load_close(["TLT", "GLD"], since="2010-01-01").pct_change().fillna(0)
    rets = load_close(AGRI_FUTURES, since="2010-01-01").pct_change().fillna(0)

    # Baseline
    print("\n=== Baseline ===")
    pnl_full, win_pnls_full = champ_wf(rets, cash)
    m_full = wf_metrics(pnl_full, win_pnls_full)
    print(f"  full: Sharpe={m_full['mean_sharpe']:.2f} CAGR={m_full['CAGR']*100:.1f}% MDD={m_full['MDD']*100:.1f}%")

    # 1. Multi-OOS
    print("\n=== Multi-OOS (4 setups) ===")
    oos_setups = [
        ("baseline 504/252/126", 504, 252, 126),
        ("longer OOS 504/504/252", 504, 504, 252),
        ("longer train 756/252/126", 756, 252, 126),
        ("shorter OOS 504/126/63", 504, 126, 63),
    ]
    oos_sharpes = []
    for name, tr, te, st in oos_setups:
        pnl, win_pnls = champ_wf(rets, cash, train=tr, test=te, step=st)
        m = wf_metrics(pnl, win_pnls)
        oos_sharpes.append(m['mean_sharpe'])
        print(f"  {name}: Sharpe={m['mean_sharpe']:.2f}")
    ratio = min(oos_sharpes) / max(oos_sharpes) if max(oos_sharpes) > 0 else 0
    print(f"  Multi-OOS ratio: {ratio*100:.0f}% {'✅' if ratio > 0.7 else '⚠️'}")

    # 2. Regime
    print("\n=== Regime split ===")
    regimes = {
        "베어 2018Q4": ("2018-10-01", "2018-12-31"),
        "코로나 2020Q1": ("2020-02-01", "2020-04-30"),
        "베어 2022": ("2022-01-01", "2022-12-31"),
        "최근 2024-2025": ("2024-01-01", "2025-12-31"),
    }
    reg_sharpes = []
    for r_name, (start, end) in regimes.items():
        m = (pnl_full.index >= pd.Timestamp(start)) & (pnl_full.index <= pd.Timestamp(end))
        seg = pnl_full[m]
        if len(seg) > 30:
            mm = metrics(seg)
            reg_sharpes.append(mm["Sharpe"])
            color = "✅" if mm['Sharpe'] > 1 else "⚠️" if mm['Sharpe'] > 0 else "❌"
            print(f"  {color} {r_name}: Sharpe={mm['Sharpe']:.2f} MDD={mm['MDD']*100:.1f}%")

    # 3. Crash
    print("\n=== Crash ===")
    crashes = {
        "2018Q4": ("2018-09-20", "2019-04-30"),
        "코로나": ("2020-02-19", "2020-08-31"),
        "2022": ("2022-01-03", "2023-01-31"),
    }
    max_dds = []
    for c_name, (start, end) in crashes.items():
        seg = pnl_full[(pnl_full.index >= pd.Timestamp(start)) & (pnl_full.index <= pd.Timestamp(end))]
        if len(seg) > 5:
            eq = (1 + seg).cumprod()
            cm = eq.cummax()
            dd = float((eq / cm - 1).min())
            max_dds.append(dd)
            print(f"  {c_name}: max DD={dd*100:.1f}%")
    worst = min(max_dds) if max_dds else 0

    # 4. Cost stress
    print("\n=== Cost stress ===")
    cost_results = {}
    for fee_bps in [50, 100, 150]:
        pnl, win_pnls = champ_wf(rets, cash, fee_per_change=fee_bps/1e4)
        mc = wf_metrics(pnl, win_pnls)
        cost_results[fee_bps] = mc
        color = "✅" if mc['mean_sharpe'] > 1.0 else "⚠️" if mc['mean_sharpe'] > 0.3 else "❌"
        print(f"  {color} {fee_bps}bps: Sharpe={mc['mean_sharpe']:.2f} MDD={mc['MDD']*100:.1f}%")

    # 종합
    pass_oos = ratio > 0.7
    pass_regime = min(reg_sharpes) > 0 if reg_sharpes else False
    pass_crash = worst > -0.25
    pass_100 = cost_results.get(100, {}).get("mean_sharpe", 0) > 1.0

    print(f"\n=== iter17 ROBUST 종합 ===")
    print(f"  Multi-OOS: {'✅' if pass_oos else '❌'} ({ratio*100:.0f}%)")
    print(f"  Regime: {'✅' if pass_regime else '❌'} (min {min(reg_sharpes) if reg_sharpes else 0:.2f})")
    print(f"  Crash: {'✅' if pass_crash else '❌'} ({worst*100:.1f}%)")
    print(f"  100bps: {'✅' if pass_100 else '❌'} (Sharpe {cost_results.get(100, {}).get('mean_sharpe', 0):.2f})")
    if pass_oos and pass_regime and pass_crash and pass_100:
        print(f"  🏆 iter17 ALL ROBUST 통과!")
    else:
        print(f"  ⚠️ 일부 실패")

    summary = {
        "baseline": {k: v for k, v in m_full.items() if not isinstance(v, list)},
        "multi_oos_ratio": ratio,
        "multi_oos_sharpes": oos_sharpes,
        "regime_min": min(reg_sharpes) if reg_sharpes else 0,
        "regime_sharpes": reg_sharpes,
        "crash_worst_dd": worst,
        "crash_dds": dict(zip(crashes.keys(), max_dds)),
        "cost_stress": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in cost_results.items()},
    }
    out = RESULTS_DIR / "iter19_robust.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()

"""iter06: vol-target + cash buffer + equity stop 통합 (3단 안전장치).

iter03 발견:
- Agri vol 5% → MDD -36% (Sharpe 0.89)
- Energy vol 5% → MDD -26% (Sharpe 0.63)

iter02/05 발견:
- HMM/cap/invvol 무력
- cash buffer + low vol target가 핵심

iter06 통합 셋업:
1. Vol-targeting (5/8% sweep)
2. Cash buffer (강제 50/70% cash)
3. Equity stop (DD -15% → 90일 lock)
4. Multi-category ensemble (Agri + Energy)

목표: Sharpe 1.0+ + MDD < -20% (실거래 영역)
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
             top_k=2, target_vol=0.05,
             cash_floor=0.50,  # 의무 cash 최소 비율
             dd_stop=-0.15, lock_days=63,
             hmm_high=0.4, hmm_lookback=1008,
             train=504, test=252, step=126,
             fee_per_change=0.0050):
    """3단 안전장치: vol-target + cash buffer + equity stop."""
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

        # Inverse-vol weighting
        vol_window = rets.iloc[end_idx - 21:end_idx]
        sigmas = vol_window[top].std(ddof=1).fillna(0.01).clip(lower=0.001)
        inv = 1.0 / sigmas
        inv = inv / inv.sum()

        # Vol-targeting
        portfolio_vol_daily = np.sqrt(((inv ** 2) * (sigmas ** 2)).sum())
        portfolio_vol_annual = portfolio_vol_daily * np.sqrt(252)
        vol_size = min(target_vol / max(portfolio_vol_annual, 0.01), 1.0)

        # HMM mild
        state = detect_hmm(end_idx)
        if state == 1:
            vol_size *= hmm_high

        # Cash floor 강제 — 위험 자산 max (1 - cash_floor)
        max_risk = 1.0 - cash_floor
        size = min(vol_size, max_risk)

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

            # Equity stop check
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


def run(name, syms, cash, configs):
    closes = load_close(syms, since="2010-01-01")
    rets = closes.pct_change().fillna(0)
    if rets.empty:
        return {}
    print(f"\n=== {name} (n={rets.shape[1]}) ===")
    results = {}
    for cfg_name, vol, cf, dd, lock in configs:
        try:
            pnl, win_pnls = champ_wf(rets, cash, target_vol=vol, cash_floor=cf, dd_stop=dd, lock_days=lock)
            m = wf_metrics(pnl, win_pnls)
            results[cfg_name] = m
            mdd_ok = m['MDD'] > -0.20
            neg = m['neg_windows']; nw = m['n_windows']
            color = "🚀" if mdd_ok and m['mean_sharpe'] > 2.0 else \
                    "✅" if mdd_ok and m['mean_sharpe'] > 1.0 else \
                    "⚠️" if m['mean_sharpe'] > 0.3 else "❌"
            print(f"  {color} {cfg_name}: Sharpe={m['mean_sharpe']:.2f} (win {nw-neg}/{nw}) CAGR={m['CAGR']*100:.1f}% MDD={m['MDD']*100:.1f}%")
        except Exception as e:
            print(f"  {cfg_name}: ERROR {e}")
            results[cfg_name] = {"error": str(e)}
    return results


def main():
    print("[iter06] vol-target + cash buffer + equity stop 통합 (3단 안전장치)")
    cash = load_close(["TLT", "GLD"], since="2010-01-01").pct_change().fillna(0)

    # (name, target_vol, cash_floor, dd_stop, lock_days)
    configs = [
        ("vol5% cash50% no-stop",       0.05, 0.50, -1.0, 0),
        ("vol5% cash50% DD-15% lock90", 0.05, 0.50, -0.15, 90),
        ("vol5% cash70% DD-15% lock90", 0.05, 0.70, -0.15, 90),
        ("vol5% cash50% DD-10% lock90", 0.05, 0.50, -0.10, 90),
        ("vol8% cash50% DD-15% lock90", 0.08, 0.50, -0.15, 90),
        ("vol8% cash70% DD-15% lock90", 0.08, 0.70, -0.15, 90),
        ("vol10% cash50% DD-15% lock63", 0.10, 0.50, -0.15, 63),
        ("vol10% cash70% DD-15% lock63", 0.10, 0.70, -0.15, 63),
        ("vol3% cash50% DD-10% lock63 (extreme safe)", 0.03, 0.50, -0.10, 63),
    ]

    out = {}
    out["agri"] = run("Agri", AGRI_FUTURES, cash, configs)
    out["energy"] = run("Energy", ENERGY_FUTURES, cash, configs)
    out["metal"] = run("Metal", METAL_FUTURES, cash, configs)

    # Combined Agri + Energy
    combined = AGRI_FUTURES + ENERGY_FUTURES
    out["combined"] = run("Agri + Energy 통합", combined, cash, configs)

    # 종합
    print("\n=== iter06 종합 (실거래 가능 영역 분석) ===")
    all_results = []
    for cat, res in out.items():
        for cfg, m in res.items():
            if isinstance(m, dict) and 'error' not in m:
                all_results.append((cat, cfg, m))

    # MDD < -20% + Sharpe > 1.0 통과한 것만
    safe = [r for r in all_results if r[2].get('MDD', -1) > -0.20 and r[2].get('mean_sharpe', 0) > 1.0]
    if safe:
        safe.sort(key=lambda r: r[2]['mean_sharpe'], reverse=True)
        print(f"  🎯 실거래 가능 (MDD > -20% & Sharpe > 1.0):")
        for cat, cfg, m in safe[:5]:
            print(f"    🚀 {cat} | {cfg}: Sharpe={m['mean_sharpe']:.2f} MDD={m['MDD']*100:.1f}%")
    else:
        # 차선: MDD -25% + Sharpe 0.3
        next_best = [r for r in all_results if r[2].get('MDD', -1) > -0.25 and r[2].get('mean_sharpe', 0) > 0.3]
        if next_best:
            next_best.sort(key=lambda r: r[2]['mean_sharpe'], reverse=True)
            print(f"  ⚠️ 차선 (MDD > -25% & Sharpe > 0.3):")
            for cat, cfg, m in next_best[:5]:
                print(f"    ✅ {cat} | {cfg}: Sharpe={m['mean_sharpe']:.2f} MDD={m['MDD']*100:.1f}%")
        else:
            print(f"  ❌ MDD -25% 이내 + Sharpe 0.3 통과 없음")

    out_path = RESULTS_DIR / "iter06_combined.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()

"""iter07: Trend strength filter (강한 trend일 때만 진입).

iter02-06 발견:
- vol-target + cash buffer로 MDD 줄임
- 하지만 Sharpe도 같이 줄어듬

iter07 가설: momentum z-score는 noise. 강한 trend (60d 누적 5%+ AND 양수 일관성) 때만 진입.

Trend strength 측정:
1. 60d 누적 수익률 > 5%
2. 60d 동안 양수 비율 (% positive days) > 55%
3. 21d MA > 60d MA (이중 confirmation)
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
             top_k=2, target_vol=0.08, cash_floor=0.50,
             trend_period=60, trend_min_cum=0.05, trend_min_pos=0.55,
             use_ma_cross=True,
             train=504, test=252, step=126,
             fee_per_change=0.0050):
    """Trend strength 통과한 종목만 후보."""
    full = pd.concat([rets, cash_rets], axis=1).fillna(0)
    n = len(full)
    pnl = pd.Series(0.0, index=full.index)
    used = pd.Series(False, index=full.index)
    s = 0
    max_p = max(1008, trend_period * 2)

    def compute(end_idx):
        if end_idx < max_p:
            return None

        # 1) 기본 momentum composite + CVaR
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

        # 2) Trend strength filter — 최근 60d 강한 상승 only
        trend_window = rets.iloc[end_idx - trend_period:end_idx]
        trend_cum = (1 + trend_window.fillna(0)).prod() - 1
        pos_ratio = (trend_window > 0).sum() / trend_window.notna().sum()
        trend_pass = (trend_cum > trend_min_cum) & (pos_ratio > trend_min_pos)

        if use_ma_cross:
            ma_short = rets.iloc[end_idx - 21:end_idx].mean() * 21
            ma_long = rets.iloc[end_idx - 60:end_idx].mean() * 60
            ma_cross_pass = (ma_short > ma_long)
            trend_pass = trend_pass & ma_cross_pass

        # strict_pos
        mask = pd.Series(True, index=composite.index)
        for p, z in all_z.items():
            z_aligned = z.reindex(composite.index)
            mask = mask & (z_aligned > 0)

        # Apply trend filter on top
        trend_pass_aligned = trend_pass.reindex(composite.index).fillna(False)
        mask = mask & trend_pass_aligned

        scores = composite[mask].dropna()
        if scores.empty:
            return None

        topk = min(top_k, len(scores))
        top = list(scores.sort_values(ascending=False).head(topk).index)

        # Inverse-vol weighting + Vol-targeting
        vol_window = rets.iloc[end_idx - 21:end_idx]
        sigmas = vol_window[top].std(ddof=1).fillna(0.01).clip(lower=0.001)
        inv = 1.0 / sigmas
        inv = inv / inv.sum()
        portfolio_vol_daily = np.sqrt(((inv ** 2) * (sigmas ** 2)).sum())
        portfolio_vol_annual = portfolio_vol_daily * np.sqrt(252)
        vol_size = min(target_vol / max(portfolio_vol_annual, 0.01), 1.0)
        max_risk = 1.0 - cash_floor
        size = min(vol_size, max_risk)

        w = pd.Series(0.0, index=full.columns)
        for c in top:
            w.loc[c] = inv[c] * size
        spare = 1.0 - w.sum()
        cash_cols = [c for c in cash_rets.columns]
        if spare > 0:
            for c in cash_cols:
                w[c] = spare / len(cash_cols)
        return w

    while s + train + test <= n:
        if s + train < max_p:
            s += step
            continue
        w = compute(s + train)
        if w is None:
            # 진입 안 됨 → 100% cash
            w = pd.Series(0.0, index=full.columns)
            cash_cols = [c for c in cash_rets.columns]
            for c in cash_cols:
                w[c] = 1.0 / len(cash_cols)
        test_idx = full.iloc[s + train:s + train + test]
        for i in range(len(test_idx)):
            ts = test_idx.index[i]
            cost = 0.0
            if i > 0 and i % 2 == 0:
                end_pos = s + train + i
                new_w = compute(end_pos)
                if new_w is None:
                    new_w = pd.Series(0.0, index=full.columns)
                    cash_cols = [c for c in cash_rets.columns]
                    for c in cash_cols:
                        new_w[c] = 1.0 / len(cash_cols)
                turnover = (new_w - w).abs().sum()
                if turnover < 0.5:
                    pass
                else:
                    cost = turnover * fee_per_change
                    w = new_w
            r = float((test_idx.iloc[i] * w).sum()) - cost
            pnl.loc[ts] = r
            used.loc[ts] = True
        s += step
    return pnl[used]


def run(name, syms, cash, configs):
    closes = load_close(syms, since="2010-01-01")
    rets = closes.pct_change().fillna(0)
    if rets.empty:
        return {}
    print(f"\n=== {name} (n={rets.shape[1]}) ===")
    results = {}
    for cfg_name, vol, cf, period, mc, mp, ma in configs:
        try:
            pnl = champ_wf(rets, cash, target_vol=vol, cash_floor=cf,
                           trend_period=period, trend_min_cum=mc, trend_min_pos=mp,
                           use_ma_cross=ma)
            m = metrics(pnl)
            results[cfg_name] = m
            mdd_ok = m['MDD'] > -0.20
            sharpe_ok = m['Sharpe'] > 1.0
            color = "🚀" if mdd_ok and sharpe_ok and m['Sharpe'] > 1.5 else \
                    "✅" if mdd_ok and sharpe_ok else \
                    "⚠️" if m['Sharpe'] > 0.5 else "❌"
            print(f"  {color} {cfg_name}: Sharpe={m['Sharpe']:.2f} CAGR={m['CAGR']*100:.1f}% MDD={m['MDD']*100:.1f}%")
        except Exception as e:
            print(f"  {cfg_name}: ERROR {e}")
            results[cfg_name] = {"error": str(e)}
    return results


def main():
    print("[iter07] Trend strength filter (강한 trend만 진입)")
    cash = load_close(["TLT", "GLD"], since="2010-01-01").pct_change().fillna(0)

    # (name, vol, cash_floor, period, min_cum, min_pos, use_ma)
    configs = [
        ("vol8% cf50% trend60d 5% 55% +MA",   0.08, 0.50, 60, 0.05, 0.55, True),
        ("vol8% cf50% trend60d 10% 60% +MA",  0.08, 0.50, 60, 0.10, 0.60, True),
        ("vol8% cf50% trend60d 5% 50% noMA",  0.08, 0.50, 60, 0.05, 0.50, False),
        ("vol5% cf50% trend60d 5% 55% +MA",   0.05, 0.50, 60, 0.05, 0.55, True),
        ("vol10% cf30% trend60d 5% 55% +MA",  0.10, 0.30, 60, 0.05, 0.55, True),
        ("vol8% cf50% trend90d 8% 55% +MA",   0.08, 0.50, 90, 0.08, 0.55, True),
        ("vol8% cf50% trend30d 3% 55% +MA",   0.08, 0.50, 30, 0.03, 0.55, True),
        ("vol8% cf70% trend60d 5% 60% +MA (보수)", 0.08, 0.70, 60, 0.05, 0.60, True),
    ]

    out = {}
    out["agri"] = run("Agri", AGRI_FUTURES, cash, configs)
    out["energy"] = run("Energy", ENERGY_FUTURES, cash, configs)
    out["metal"] = run("Metal", METAL_FUTURES, cash, configs)
    out["combined"] = run("Agri + Energy + Metal", AGRI_FUTURES + ENERGY_FUTURES + METAL_FUTURES, cash, configs)

    # 종합
    print("\n=== iter07 종합 (실거래 가능 영역) ===")
    all_results = []
    for cat, res in out.items():
        for cfg, m in res.items():
            if isinstance(m, dict) and 'error' not in m:
                all_results.append((cat, cfg, m))

    safe = [r for r in all_results if r[2].get('MDD', -1) > -0.20 and r[2].get('Sharpe', 0) > 1.0]
    if safe:
        safe.sort(key=lambda r: r[2]['Sharpe'], reverse=True)
        print(f"  🎯 실거래 가능 (MDD > -20% & Sharpe > 1.0):")
        for cat, cfg, m in safe[:5]:
            print(f"    🚀 {cat} | {cfg}: Sharpe={m['Sharpe']:.2f} MDD={m['MDD']*100:.1f}%")
    else:
        next_best = [r for r in all_results if r[2].get('MDD', -1) > -0.25]
        if next_best:
            next_best.sort(key=lambda r: r[2]['Sharpe'], reverse=True)
            print(f"  ⚠️ 차선 (MDD > -25%):")
            for cat, cfg, m in next_best[:5]:
                print(f"    ✅ {cat} | {cfg}: Sharpe={m['Sharpe']:.2f} MDD={m['MDD']*100:.1f}%")
        else:
            print(f"  ❌ MDD -25% 통과 없음")

    out_path = RESULTS_DIR / "iter07_trend_filter.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()

"""iter22: 시즌성 + Curve 활용 (단순 momentum 대신).

iter12 이미 시즌성 (월별 평균) 시도 — 약함.
iter22: 더 정교한 month-of-year × asset specific:
- ZC (Corn): 6-7월 weather risk → vol spike, 9-10월 harvest → 가격 하락
- NG (NatGas): 11-2월 winter premium
- HO (Heating Oil): 11-1월 demand surge
- GC (Gold): 1월 Indian wedding season demand
- ZS (Soy): 8월 South America planning

각 자산 × 월별 평균 수익률 → 가장 강한 신호 패턴 추출.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data_loader import load_close


def main():
    print("[iter22] 시즌성 + Curve")
    syms = ["ZC=F", "ZW=F", "ZS=F", "NG=F", "HO=F", "CL=F", "GC=F", "SI=F", "HG=F", "KC=F", "SB=F", "CT=F"]
    closes = load_close(syms)
    closes = closes.fillna(method='ffill')
    print(f"  종목: {list(closes.columns)} ({len(closes)} days)")

    # daily returns
    rets = closes.pct_change()

    # add month
    rets['month'] = rets.index.month

    print(f"\n=== Asset × Month 평균 일별 수익률 ===")
    print(f"  {'Asset':6s} | " + " ".join([f"{m:>5}" for m in range(1, 13)]))
    print("  " + "-" * 90)
    monthly_means = {}
    for col in [c for c in rets.columns if c != 'month']:
        means = rets.groupby('month')[col].mean() * 100
        monthly_means[col] = means
        line = f"  {col:6s} | "
        for m in range(1, 13):
            v = means.get(m, 0)
            color = "🟢" if v > 0.05 else "🔴" if v < -0.05 else "  "
            line += f"{v:>+5.2f}{color[:1] if color != '  ' else ' '}"
        print(line)

    print(f"\n=== Top seasonal trades (월간 누적 수익률) ===")
    # Compute monthly cumulative returns by stacking
    rets['ym'] = rets.index.to_period('M')
    monthly_cum = rets.groupby('ym').apply(lambda x: (1 + x.drop(['month', 'ym'], axis=1, errors='ignore')).prod() - 1)

    # Per-asset per-month avg accumulated return
    monthly_cum['month'] = monthly_cum.index.month
    print(f"  {'Asset':6s} | " + " ".join([f"{m:>6}" for m in range(1, 13)]))
    print("  " + "-" * 100)
    seasonal_signals = {}
    for col in [c for c in monthly_cum.columns if c != 'month']:
        means = monthly_cum.groupby('month')[col].mean() * 100
        std = monthly_cum.groupby('month')[col].std() * 100
        seasonal_signals[col] = means
        line = f"  {col:6s} | "
        for m in range(1, 13):
            v = means.get(m, 0)
            s = std.get(m, 1)
            sharpe_m = v / s if s > 0 else 0
            marker = "🚀" if sharpe_m > 0.5 and v > 1 else "✅" if sharpe_m > 0.3 and v > 0.5 else ""
            line += f"{v:>+5.1f}{marker[:1] if marker else ' '} "
        print(line)

    # Find strongest seasonal patterns
    print(f"\n=== 강한 seasonal 패턴 (월간 +2% 이상, 11+ 년 데이터) ===")
    for col in seasonal_signals:
        means = seasonal_signals[col]
        for m in range(1, 13):
            v = means.get(m, 0)
            if v > 2.0:
                print(f"  {col} 월{m}: 평균 +{v:.2f}% (long 후보)")
            elif v < -2.0:
                print(f"  {col} 월{m}: 평균 {v:+.2f}% (short 후보)")

    out_path = RESULTS_DIR / "iter22_curve_seasonal.json"
    out_path.write_text("{}", encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()

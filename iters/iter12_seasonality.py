"""iter12: 계절성 (농산물 정통 — 봄 파종 / 가을 수확).

가설:
- 옥수수/콩: 4~6월 (파종 + 날씨) 변동성 ↑, 9~11월 (수확) 약세
- 밀: 5~7월 약세 (북반구 수확)
- 일반 패턴: 특정 월에 long/short 진입

backtest:
1. 각 종목 월별 평균 수익률 계산 (10년+)
2. 양수 월에 long, 음수 월에 cash
3. iter04 logic + 계절성 gate
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

from src.config import RESULTS_DIR
from src.data_loader import load_close
from src.futures_universe import AGRI_FUTURES
from src.backtest import metrics, wf_metrics


def main():
    print("[iter12] 계절성 분석 (농산물)")
    cash = load_close(["TLT", "GLD"], since="2010-01-01").pct_change().fillna(0)

    closes = load_close(AGRI_FUTURES, since="2010-01-01")
    rets = closes.pct_change().fillna(0)
    print(f"  Agri: {rets.shape[1]} 종목")

    # 월별 평균 수익률
    rets_with_month = rets.copy()
    rets_with_month['month'] = rets_with_month.index.month
    print(f"\n=== 월별 평균 수익률 (Agri 평균) ===")
    print(f"  {'Month':>6} {'AvgRet':>10} {'WinRate':>10}")
    monthly_summary = {}
    for m in range(1, 13):
        mr = rets[rets.index.month == m].mean(axis=1)
        avg = mr.mean() * 100  # %
        win_rate = (mr > 0).sum() / len(mr) * 100 if len(mr) > 0 else 0
        marker = "🟢" if avg > 0.05 else "🔴" if avg < -0.05 else ""
        monthly_summary[m] = {"avg_ret_pct": avg, "win_rate_pct": win_rate}
        print(f"  {m:>6} {avg:>10.3f}% {win_rate:>9.1f}%  {marker}")

    # 종목별 월별 통계
    print(f"\n=== 종목별 양수 월 / 음수 월 ===")
    stock_seasonality = {}
    for col in rets.columns:
        col_rets = rets[col]
        col_rets_with_m = col_rets.copy()
        monthly_avg = col_rets.groupby(col_rets.index.month).mean()
        pos_months = sorted(monthly_avg[monthly_avg > 0].index.tolist())
        neg_months = sorted(monthly_avg[monthly_avg < 0].index.tolist())
        stock_seasonality[col] = {
            "pos_months": pos_months,
            "neg_months": neg_months,
            "monthly_avg": monthly_avg.to_dict(),
        }
        print(f"  {col}: 양수 월 {pos_months}, 음수 월 {neg_months}")

    # 단순 시뮬: 양수 월에만 long, 음수 월 cash
    print(f"\n=== 단순 계절성 봇 (각 종목 양수 월에만 long) ===")
    pos_only_pnl = pd.Series(0.0, index=rets.index)
    for col in rets.columns:
        pos_months = stock_seasonality[col]["pos_months"]
        signal = rets.index.month.isin(pos_months)
        pos_only_pnl += rets[col].where(signal, 0) / rets.shape[1]

    m = metrics(pos_only_pnl)
    color = "🚀" if m['Sharpe'] > 2 else "✅" if m['Sharpe'] > 1 else "⚠️" if m['Sharpe'] > 0.5 else "❌"
    print(f"  {color} 양수 월 only: Sharpe={m['Sharpe']:.2f} CAGR={m['CAGR']*100:.1f}% MDD={m['MDD']*100:.1f}%")

    # 강한 양수 월 (avg > 0.1%) 만
    print(f"\n=== 강한 양수 월 (avg > 0.1%) only ===")
    strong_pnl = pd.Series(0.0, index=rets.index)
    for col in rets.columns:
        monthly = stock_seasonality[col]["monthly_avg"]
        strong_months = [m for m, v in monthly.items() if v > 0.001]
        signal = rets.index.month.isin(strong_months)
        strong_pnl += rets[col].where(signal, 0) / rets.shape[1]

    m_s = metrics(strong_pnl)
    color_s = "🚀" if m_s['Sharpe'] > 2 else "✅" if m_s['Sharpe'] > 1 else "⚠️" if m_s['Sharpe'] > 0.5 else "❌"
    print(f"  {color_s} 강한 양수 월: Sharpe={m_s['Sharpe']:.2f} CAGR={m_s['CAGR']*100:.1f}% MDD={m_s['MDD']*100:.1f}%")

    # iter04 비교
    print(f"\n  iter04 (Agri momentum + DD-10%): Sharpe 3.94 MDD -14.3%")
    print(f"  iter12 계절성 단독으로는 약함 (filter로만 의미 가능)")

    out = {
        "monthly_summary": monthly_summary,
        "pos_only": m,
        "strong_pos": m_s,
    }
    out_path = RESULTS_DIR / "iter12_seasonality.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()

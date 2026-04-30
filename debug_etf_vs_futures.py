"""ETF vs =F continuous futures 비교 검증.

가설: 백테스트 수익률 차이 = roll 점프로 인한 가짜 알파.

1. yfinance =F continuous (현재 방식) — roll 점프 포함
2. 농산물 ETF (DBA, CORN, WEAT, SOYB 등) — roll cost 내재화
3. 같은 8d momentum + Top-K logic으로 백테스트
4. CAGR / Sharpe 차이 측정
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.data_loader import load_close


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


def simple_8d_momentum(rets, top_k=2, dd_stop=-0.10, lock_days=63, fee_per_change=0.005):
    """8d momentum + Top-K + DD stop — iter17 simplified."""
    rets = rets.fillna(0)
    n = len(rets)
    pnl = pd.Series(0.0, index=rets.index)
    used = pd.Series(False, index=rets.index)

    locked_until = -1
    last_w = None
    for i in range(60, n - 1):
        # signal: 8d momentum z-score
        rec = rets.iloc[i - 8:i]
        cum = (1 + rec).prod() - 1
        z = (cum - cum.mean()) / (cum.std() + 1e-9)
        scores = z[z > 0].dropna()
        if scores.empty:
            continue
        topk = min(top_k, len(scores))
        top = list(scores.sort_values(ascending=False).head(topk).index)
        w = pd.Series(0.0, index=rets.columns)
        for c in top:
            w.loc[c] = 1.0 / len(top)

        # DD-10% lock
        if i > 252 and locked_until <= i:
            lookback_pnl = pnl.iloc[max(0, i-252):i]
            eq_lb = (1 + lookback_pnl).cumprod()
            cm_lb = eq_lb.cummax()
            current_dd = float((eq_lb.iloc[-1] / cm_lb.iloc[-1] - 1)) if cm_lb.iloc[-1] > 0 else 0
            if current_dd < dd_stop:
                locked_until = i + lock_days

        if locked_until > i:
            r = 0  # cash
        else:
            cost = 0
            if last_w is not None:
                turnover = (w - last_w).abs().sum()
                cost = turnover * fee_per_change
            r = float((rets.iloc[i + 1] * w).sum()) - cost
            last_w = w

        pnl.iloc[i + 1] = r
        used.iloc[i + 1] = True

    return pnl[used]


def main():
    print("[debug] ETF vs =F futures 비교")

    # =F continuous (current data)
    AGRI_FUTURES = ["ZC=F", "ZW=F", "ZS=F", "ZL=F", "ZM=F", "KC=F", "SB=F", "CT=F", "CC=F", "OJ=F"]
    rets_F = load_close(AGRI_FUTURES, since="2010-01-01").pct_change().dropna(how='all').fillna(0)
    print(f"\n  =F continuous: {rets_F.shape[1]}종목, {len(rets_F)} days")

    # 농산물 ETF (roll cost 자동 반영)
    AGRI_ETFS = ["DBA", "CORN", "WEAT", "SOYB", "JO", "CANE", "BAL", "NIB", "JJG", "RJA"]
    rets_ETF = load_close(AGRI_ETFS, since="2010-01-01").pct_change().dropna(how='all').fillna(0)
    print(f"  농산물 ETF: {rets_ETF.shape[1]}종목, {len(rets_ETF)} days")

    print(f"\n=== 비교: 8d momentum + Top-K=2 + DD-10% ===")
    print(f"  {'데이터셋':30s} {'Sharpe':>7} {'CAGR':>8} {'MDD':>8}")

    pnl_F = simple_8d_momentum(rets_F, top_k=2)
    m_F = metrics(pnl_F)
    print(f"  {'=F continuous (yfinance)':30s} {m_F['Sharpe']:>7.2f} {m_F['CAGR']*100:>+7.1f}% {m_F['MDD']*100:>+7.1f}%")

    pnl_ETF = simple_8d_momentum(rets_ETF, top_k=2)
    m_ETF = metrics(pnl_ETF)
    print(f"  {'ETF (roll cost 반영)':30s} {m_ETF['Sharpe']:>7.2f} {m_ETF['CAGR']*100:>+7.1f}% {m_ETF['MDD']*100:>+7.1f}%")

    print(f"\n=== Buy & Hold 비교 (단순 보유) ===")
    bh_F = rets_F.mean(axis=1)
    bh_ETF = rets_ETF.mean(axis=1)
    print(f"  =F equal-weight B&H: Sharpe={metrics(bh_F)['Sharpe']:.2f} CAGR={metrics(bh_F)['CAGR']*100:+.1f}%")
    print(f"  ETF equal-weight B&H: Sharpe={metrics(bh_ETF)['Sharpe']:.2f} CAGR={metrics(bh_ETF)['CAGR']*100:+.1f}%")

    # 점프 분석: 큰 일간 변화 빈도 비교
    print(f"\n=== 큰 일간 변화 빈도 (가짜 점프 의심) ===")
    for name, rets in [("=F continuous", rets_F), ("ETF", rets_ETF)]:
        flat = rets.values.flatten()
        flat = flat[~np.isnan(flat) & (flat != 0)]
        n = len(flat)
        big5 = (np.abs(flat) > 0.05).sum()
        big10 = (np.abs(flat) > 0.10).sum()
        print(f"  {name:20s}: total {n}, |Δ|>5% = {big5} ({big5/n*100:.2f}%), |Δ|>10% = {big10}")

    # 결론
    print(f"\n=== 결론 ===")
    diff_sharpe = m_F['Sharpe'] - m_ETF['Sharpe']
    diff_cagr = (m_F['CAGR'] - m_ETF['CAGR']) * 100
    print(f"  Sharpe 차이: {diff_sharpe:+.2f} (=F가 {'더 높음' if diff_sharpe > 0 else '더 낮음'})")
    print(f"  CAGR 차이: {diff_cagr:+.1f}%p")
    if diff_sharpe > 1.0:
        print(f"  ⚠️  =F가 ETF보다 Sharpe 1+ 더 높음 → roll 점프 가짜 알파 의심")


if __name__ == "__main__":
    main()

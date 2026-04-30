"""yfinance =F continuous futures의 roll 처리 검증.

농산물 선물의 raw close 데이터에서:
1. 일별 returns 분포 (큰 점프 빈도)
2. 큰 점프가 월말/분기말 시점인지 (= roll 시점)
3. adj_close vs close 차이
4. 이 점프들이 백테스트 returns에 어떻게 영향?
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.data_loader import load_ticker
from src.futures_universe import AGRI_FUTURES


def analyze_symbol(symbol):
    df = load_ticker(symbol, since="2010-01-01")
    if df.empty:
        print(f"  {symbol}: ❌ 데이터 없음")
        return None

    print(f"\n=== {symbol} ===")
    print(f"  데이터: {len(df)} days, {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"  컬럼: {list(df.columns)}")

    close = df["close"]
    adj = df.get("adj_close", close)

    # adj_close vs close 차이
    diff = (close - adj) / close
    if diff.abs().max() > 0.001:
        print(f"  ⚠️  adj_close ≠ close — max diff: {diff.abs().max()*100:.2f}%")
    else:
        print(f"  adj_close ≈ close (선물은 dividend 없으니 정상)")

    # 일별 returns
    rets = close.pct_change().dropna()
    print(f"  Returns: mean={rets.mean()*100:.3f}%, std={rets.std()*100:.2f}%")
    print(f"  큰 점프 (|Δ| > 5%): {(rets.abs() > 0.05).sum()}일 ({(rets.abs() > 0.05).sum() / len(rets) * 100:.1f}%)")
    print(f"  큰 점프 (|Δ| > 10%): {(rets.abs() > 0.10).sum()}일")
    print(f"  Max return: +{rets.max()*100:.2f}% on {rets.idxmax().date()}")
    print(f"  Min return: {rets.min()*100:.2f}% on {rets.idxmin().date()}")

    # roll 시점 (= 농산물의 contract expiry month) 확인
    # ZC, ZW, ZS: contract months = Mar, May, Jul, Sep, Dec
    # KC, SB, CT: 다른 월
    big_jumps = rets[rets.abs() > 0.05]
    print(f"\n  큰 점프 발생 시기 (월별 분포):")
    by_month = big_jumps.groupby(big_jumps.index.month).count()
    for m in range(1, 13):
        n = by_month.get(m, 0)
        marker = " ⚠️ roll 의심" if n > 5 else ""
        print(f"    {m}월: {n}회{marker}")

    # 가장 큰 점프 5개의 정확한 날짜 + 다음날 회복 여부
    print(f"\n  Top 5 일간 |Δ|:")
    top5 = rets.abs().sort_values(ascending=False).head(5)
    for d, v in top5.items():
        ret_d = rets.loc[d]
        # 다음 날 가격 회복 (roll 점프 vs 진짜 sustained move)
        try:
            next_d = rets.loc[d:].iloc[1:6]
            next_5d = next_d.sum() * 100
        except Exception:
            next_5d = np.nan
        print(f"    {d.date()}: {ret_d*100:+.2f}% / next 5d sum: {next_5d:+.2f}%")

    return {
        "symbol": symbol,
        "n_days": len(df),
        "big_jump_5pct": int((rets.abs() > 0.05).sum()),
        "big_jump_10pct": int((rets.abs() > 0.10).sum()),
        "max_ret": float(rets.max()),
        "min_ret": float(rets.min()),
    }


def main():
    print("[debug] yfinance =F continuous futures roll 검증")
    results = []
    for sym in AGRI_FUTURES:
        r = analyze_symbol(sym)
        if r:
            results.append(r)

    print(f"\n=== 종합 ===")
    print(f"  {'Symbol':10s} {'N':>5} {'>5% jumps':>10} {'>10%':>5} {'max ret':>8} {'min ret':>8}")
    for r in results:
        print(f"  {r['symbol']:10s} {r['n_days']:>5} {r['big_jump_5pct']:>10} {r['big_jump_10pct']:>5} {r['max_ret']*100:>+7.1f}% {r['min_ret']*100:>+7.1f}%")


if __name__ == "__main__":
    main()

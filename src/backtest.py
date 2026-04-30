"""공통 백테스트 유틸 — FuturesDistribution 전체 iter 공유.

핵심 원칙:
- walk-forward Sharpe = per-window Sharpe의 평균 (단순 연결 metrics() 금지)
- MDD = per-window MDD의 최악값
- CAGR = 연결 PnL 기반 참고용 (부차적)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(pnl: pd.Series) -> dict:
    pnl = pnl.dropna()
    if len(pnl) == 0:
        return {"CAGR": 0, "Sharpe": 0, "MDD": 0}
    eq = (1 + pnl).cumprod()
    n_years = max((pnl.index[-1] - pnl.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / n_years) - 1)
    sharpe = float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252)) if pnl.std(ddof=1) > 0 else 0
    cm = eq.cummax()
    return {"CAGR": cagr, "Sharpe": sharpe, "MDD": float((eq / cm - 1).min())}


def wf_metrics(pnl_full: pd.Series, window_pnls: list[pd.Series]) -> dict:
    """walk-forward 올바른 metrics.

    Args:
        pnl_full: 연결된 전체 PnL (CAGR/MDD 참고용)
        window_pnls: 윈도우별 PnL 리스트 (Sharpe 계산 기준)

    Returns:
        {Sharpe: per-window mean, CAGR: overall, MDD: worst window,
         mean_sharpe, median_sharpe, std_sharpe, n_windows, neg_windows,
         window_sharpes: list}
    """
    sharpes = []
    mdds = []
    for wp in window_pnls:
        m = metrics(wp)
        sharpes.append(m["Sharpe"])
        mdds.append(m["MDD"])

    overall = metrics(pnl_full)
    mean_sh = float(np.mean(sharpes)) if sharpes else 0.0

    return {
        "Sharpe": mean_sh,
        "CAGR": overall["CAGR"],
        "MDD": float(min(mdds)) if mdds else 0.0,
        "mean_sharpe": mean_sh,
        "median_sharpe": float(np.median(sharpes)) if sharpes else 0.0,
        "std_sharpe": float(np.std(sharpes)) if sharpes else 0.0,
        "n_windows": len(sharpes),
        "neg_windows": sum(1 for s in sharpes if s < 0),
        "window_sharpes": sharpes,
    }

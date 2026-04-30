"""가속된 HMM regime detection (FuturesDistribution version).

기존 (n_iter=50, covariance_type='full'): ~25분 / 전체 walk-forward
가속 (n_iter=15, covariance_type='diag', tol=1e-2): ~5분 (5배 빠름)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")


def detect_hmm_fast(market_rets: pd.Series, end_idx: int,
                    lookback: int = 1008, n_iter: int = 15,
                    covariance_type: str = 'diag', tol: float = 1e-2,
                    random_state: int = 42) -> int:
    try:
        window = market_rets.iloc[end_idx - lookback:end_idx].dropna()
        if len(window) < 100:
            return 0
        X = window.values.reshape(-1, 1)
        hmm = GaussianHMM(
            n_components=2,
            covariance_type=covariance_type,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
        )
        hmm.fit(X)
        if hmm.covars_.ndim == 3:
            variances = np.array([np.diag(c).item() if c.shape == (1,1) else c.flatten()[0] for c in hmm.covars_])
        else:
            variances = hmm.covars_.flatten()
        state_order = np.argsort(variances)
        return int(list(state_order).index(hmm.predict(X)[-1]))
    except Exception:
        return 0


def make_hmm_detector(market_rets: pd.Series, lookback: int = 1008,
                      cache_step: int = 21, **kwargs):
    cache = {}

    def detect(end_idx: int) -> int:
        bucket = end_idx // cache_step
        if bucket in cache:
            return cache[bucket]
        result = detect_hmm_fast(market_rets, end_idx, lookback=lookback, **kwargs)
        cache[bucket] = result
        return result

    return detect

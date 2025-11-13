# ai_rtc/data_gen.py
from dataclasses import dataclass
from typing import Tuple
import math
import numpy as np
from numpy.typing import NDArray

@dataclass
class ScenarioConfig:
    d: int = 10             # 변수(센서) 차원
    N0: int = 1500          # 참조데이터(Phase I) 크기
    T: int = 300            # Phase II 길이 (한 에피소드 길이)
    shift_time: int = 100   # t >= shift_time 부터 OOC (ARL1 평가는 0으로 재설정)
    sigma: float = 1.0      # 공분산은 I (단위분산) 가정


def make_cov(d: int) -> NDArray:
    return np.eye(d)


def gen_reference_data(cfg: ScenarioConfig, rng: np.random.RandomState) -> NDArray:
    return rng.multivariate_normal(mean=np.zeros(cfg.d), cov=make_cov(cfg.d), size=cfg.N0)


def make_phase2_series(cfg: ScenarioConfig, rng: np.random.RandomState, scenario: str, lam: float) -> Tuple[NDArray, NDArray]:
    """Returns X (T x d), labels_ic (T,) where labels_ic[t]=1 if in-control else 0."""
    X = rng.multivariate_normal(np.zeros(cfg.d), make_cov(cfg.d), size=cfg.T)
    labels_ic = np.ones(cfg.T, dtype=np.int64)

    if scenario.upper() == 'I':
        # Scenario I: exactly ONE variable is shifted by lam
        delta = np.zeros(cfg.d, dtype=float)
        delta[0] = lam
    else:
        # Scenario II: k = round(lam^2) variables are shifted
        k = max(1, int(round(lam**2)))
        k = min(k, cfg.d)
        a = lam / math.sqrt(k)     # keep total shift magnitude consistent
        delta = np.zeros(cfg.d, dtype=float)
        delta[:k] = a

    # apply mean shift at t >= shift_time
    if cfg.shift_time < cfg.T:
        X[cfg.shift_time:] += delta
        labels_ic[cfg.shift_time:] = 0

    return X.astype(np.float32), labels_ic

# cl_calib.py

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state
from tqdm import tqdm

from rtc_backend import compute_pS0_stat
from utils import _np2d, _np1d

@dataclass
class WindowCalib:
    """윈도우별 관리한계(CL)와 부트스트랩 표준편차를 담는 구조체."""
    CL: float   # 관리한계 (상한선)
    std: float  # 부트스트랩 std

def estimate_CL_for_window(
    S0: NDArray,
    d: int,
    window: int,
    n_boot: int,
    n_estimators: int,
    seed: int,
    backend: str = 'sklearn',
) -> WindowCalib:
    """
    주어진 window 크기에 대해, 부트스트랩으로 CL(상한)을 추정.
    반환값: WindowCalib(CL, std)
    """
    if n_boot <= 0:
        raise ValueError("estimate_CL_for_window: n_boot must be >= 1 (CL 스킵은 main에서 로드 분기를 사용).")
    
    rng = check_random_state(seed)
    alpha = 1.0 / 200.0   # ARL0 ≈ 200 을 맞추기 위한 상한 분위수
    stats = []
    N0 = len(S0)

    pbar = tqdm(range(n_boot), desc=f"  CL Boot (w={window})", leave=False, dynamic_ncols=True)
    for _ in pbar:
        start = 0 if N0 - window <= 0 else rng.randint(0, N0 - window)
        Sw = S0[start:start + window]

        # 정상(S0) vs 비정상(Sw)
        X = np.vstack([S0, Sw])
        y = np.hstack([
            np.zeros(len(S0), dtype=int),
            np.ones(len(Sw), dtype=int),
        ])

        # 항상 넘파이로 통일 (DataFrame → ndarray 혼용 방지)
        X = _np2d(X, dtype=np.float32)
        y = _np1d(y, dtype=np.int32)

        pS0 = compute_pS0_stat(
            X, y, np.arange(len(S0)),
            d=d, n_estimators=n_estimators,
            seed=rng.randint(1_000_000),
            backend=backend,
        )
        stats.append(pS0)

    stats = np.asarray(stats)
    CL = np.quantile(stats, 1 - alpha)
    std_boot = float(np.std(stats, ddof=1))

    return WindowCalib(CL=CL, std=std_boot)

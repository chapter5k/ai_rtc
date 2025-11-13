
# ai_rtc/benchmark.py
# benchmark.py
import os
import time
from typing import Literal  # Literal 안 쓰면 생략해도 됨

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from rtc_backend import compute_pS0_stat


def run_backend_benchmark(
    S0_ref: NDArray,
    d: int,
    n_estimators: int,
    seed: int,
    backend: str,
) -> float:
    """단일 분류기 호출 시간을 벤치마킹 (초)"""
    rng_bench = check_random_state(seed)
    w_bench = 10
    start_idx = rng_bench.randint(0, len(S0_ref) - w_bench)
    Sw = S0_ref[start_idx: start_idx + w_bench]

    # S0_ref (정상) vs Sw (비정상) 구분 학습
    X = np.vstack([S0_ref, Sw])
    y = np.hstack([
        np.zeros(len(S0_ref), dtype=int),
        np.ones(len(Sw), dtype=int),
    ])

    t0 = time.perf_counter()

    if backend in ["sklearn", "cuml_cv"]:
        # RF 경로: 기존 로직 그대로 사용
        compute_pS0_stat(
            X, y, np.arange(len(S0_ref)),
            d=d, n_estimators=n_estimators,
            seed=rng_bench.randint(1_000_000),
            backend=backend
        )

    elif backend == "lgbm":
        # ✅ LGBM 경로: fit/predict 모두 넘파이로 통일 + GPU/CPU 자동 선택
        from lightgbm import LGBMClassifier
        # 1) feature names 경고 방지: 항상 ndarray 사용
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int32)
        X_np = np.ascontiguousarray(X_np)
        y_np = np.ascontiguousarray(y_np)
        # 2) 장치 선택 (main()에서 os.environ["AI_RTC_DEVICE"]=args.device 설정 필요)
        use_gpu = os.environ.get("AI_RTC_DEVICE", "cpu").lower() == "cuda"
        model = LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            verbose=-1,
            n_jobs=-1,
            device_type=("gpu" if use_gpu else "cpu"),
        )
        model.fit(X_np, y_np)
        _ = model.predict_proba(X_np)  # 예측까지 포함해 ETA가 실제에 가깝게

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return time.perf_counter() - t0

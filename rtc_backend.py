import os
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state

from utils import sqrt_int, _np2d, _np1d

# -------------------------- p(S0,t) 계산 ----------------------------

BackendName = Literal["sklearn", "cuml_cv", "lgbm"]

def compute_pS0_stat(
    X: NDArray,
    y: NDArray,
    idx_S0: NDArray,
    d: int,
    n_estimators: int,
    seed: int,
    backend: BackendName = "sklearn",
    kfold: int = 5,
) -> float:
    if backend == 'sklearn':
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=sqrt_int(d),
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=seed,
        )
        rf.fit(X, y)
        oob = getattr(rf, 'oob_decision_function_', None)
        if oob is None:
            # 샘플이 너무 적어 oob가 없으면 안전 fallback
            X_S0_np = X[idx_S0]
            if X_S0_np.ndim == 1:
                X_S0_np = X_S0_np.reshape(1, -1)
            proba = rf.predict_proba(X_S0_np)
            return float(np.mean(proba[:, 0]))
        p0 = oob[idx_S0, 0]
        return float(np.nanmean(p0))

    elif backend == 'lgbm':
        # LightGBM OOF 확률 기반 p(S0,t) 추정
        try:
            import numpy as np
            from sklearn.model_selection import StratifiedKFold
            from lightgbm import LGBMClassifier
        except Exception as e:
            raise RuntimeError("LightGBM이 설치되어 있어야 backend='lgbm'를 사용할 수 있습니다. pip install lightgbm") from e
        
        use_gpu = os.environ.get("AI_RTC_DEVICE", "cpu").lower() == "cuda"

        # ✅ 1) 무조건 넘파이로 캐스팅 (DataFrame → ndarray)
        #    astype(...)는 DF를 여전히 DF로 유지하므로 경고 원인이 됨
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int32)

        # (선택) 메모리 연속성 확보 – 일부 환경에서 약간의 이득
        X_np = np.ascontiguousarray(X_np)
        y_np = np.ascontiguousarray(y_np)

        # --- 핵심 가드: 소수 클래스 개수에 맞춰 n_splits를 안전하게 조정 ---
        class_counts = np.bincount(y_np, minlength=2)
        min_class = int(class_counts.min())

        # --- Case A: 소수 클래스<2 → CV 불가 → 공정한 백업(sklearn OOB) ---
        if min_class < 2:
            from sklearn.ensemble import RandomForestClassifier as skRF
            sk = skRF(
                n_estimators=n_estimators,
                max_features=max(1, int(np.sqrt(X_np.shape[1]))),
                bootstrap=True,
                oob_score=True,
                random_state=seed,
                n_jobs=-1,
            )
            sk.fit(X_np, y_np)
            oob = getattr(sk, "oob_decision_function_", None)
            if oob is not None:
                return float(np.nanmean(oob[idx_S0, 0]))
            X_S0_np = X_np[idx_S0]
            if X_S0_np.ndim == 1:
                X_S0_np = X_S0_np.reshape(1, -1)
            proba = sk.predict_proba(X_S0_np)
            return float(np.mean(proba[:, 0]))

        # --- Case B: CV 가능 → StratifiedKFold로 안정화 ---
        try:
            n_splits = max(2, min(kfold, min_class))  # 외부 kfold 설정이 있으면 사용
        except NameError:
            n_splits = max(3, min(5, min_class))      # 백업 디폴트(3~5)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        # 불균형 처리 가중치: scale_pos_weight = (#neg / #pos)
        n_pos = int((y_np == 1).sum())
        n_neg = int((y_np == 0).sum())
        scale_pos_weight = float(n_neg) / float(max(1, n_pos))

        # ✅ 2) scikit API 사용 (fit/predict 모두 ndarray로 통일)
        lgbm_params = dict(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=200,          # RF와 분리된 합리적 preset
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            scale_pos_weight=max(1.0, scale_pos_weight),
        )
        
        lgbm_params["device_type"] = "gpu" if use_gpu else "cpu"

        p0_vals = []
        for tr_idx, te_idx in skf.split(X_np, y_np):
            y_tr_np = y_np[tr_idx]
            if np.unique(y_tr_np).size < 2:
                continue  # 학습세트가 단일클래스면 건너뜀

            model = LGBMClassifier(**lgbm_params)
            model.fit(X_np[tr_idx], y_tr_np)  # ✅ fit: ndarray

            # 테스트 폴드 중 S0 위치만 확률 취합
            te_S0_idx = te_idx[np.isin(te_idx, idx_S0)]
            if te_S0_idx.size == 0:
                continue

            X_te_np = X_np[te_S0_idx]
            if X_te_np.ndim == 1:
                X_te_np = X_te_np.reshape(1, -1)

            proba = model.predict_proba(X_te_np)  # ✅ predict: ndarray
            p0_vals.append(proba[:, 0])           # class-0 확률이 p(S0,t)

            # 메모리 정리(큰 실험에서 유용)
            try:
                del model, proba
            except Exception:
                pass

        if not p0_vals:
            # 극단 케이스: 모든 폴드 스킵 → sklearn OOB로 백업
            from sklearn.ensemble import RandomForestClassifier as skRF
            sk = skRF(
                n_estimators=n_estimators,
                max_features=max(1, int(np.sqrt(X_np.shape[1]))),
                bootstrap=True,
                oob_score=True,
                random_state=seed,
                n_jobs=-1,
            )
            sk.fit(X_np, y_np)
            oob = getattr(sk, "oob_decision_function_", None)
            if oob is not None:
                return float(np.nanmean(oob[idx_S0, 0]))
            X_S0_np = X_np[idx_S0]
            if X_S0_np.ndim == 1:
                X_S0_np = X_S0_np.reshape(1, -1)
            proba = sk.predict_proba(X_S0_np)
            return float(np.mean(proba[:, 0]))

        return float(np.mean(np.concatenate(p0_vals)))

    elif backend == 'cuml_cv':
        try:
            import cupy as cp
            from cuml.ensemble import RandomForestClassifier as cuRF
            from sklearn.model_selection import StratifiedKFold
            mempool = cp.get_default_memory_pool() # <-- (기존 수정안에 이미 있어야 함)
        except Exception as e:
            raise RuntimeError("cuML이 설치되어 있어야 backend='cuml_cv'를 사용할 수 있습니다. conda로 RAPIDS를 설치하세요.") from e

        X_np = X.astype(np.float32)
        y_np = y.astype(np.int32)

        # --- 핵심 가드: 소수 클래스 개수에 맞춰 n_splits를 안전하게 조정 ---
        class_counts = np.bincount(y_np, minlength=2)
        min_class = int(class_counts.min())

        # --- Case A: 소수 클래스<2 → CV 불가 → 공정한 백업(sklearn OOB) ---
        if min_class < 2:
            from sklearn.ensemble import RandomForestClassifier as skRF
            sk = skRF(
                n_estimators=n_estimators,
                max_features=max(1, int(np.sqrt(X_np.shape[1]))),
                bootstrap=True,
                oob_score=True,
                random_state=seed,
                n_jobs=-1,
            )
            sk.fit(X_np, y_np)
            oob = getattr(sk, "oob_decision_function_", None)
            if oob is not None:
                p0 = oob[idx_S0, 0]
                return float(np.nanmean(p0))
            X_S0_np = X_np[idx_S0]
            if X_S0_np.ndim == 1:
                X_S0_np = X_S0_np.reshape(1, -1)
            proba = sk.predict_proba(X_S0_np)
            return float(np.mean(proba[:, 0]))

        # --- Case B: CV 가능 → StratifiedKFold로 안정화 ---
        n_splits = max(2, min(kfold, min_class))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        p0_vals = []
        for tr_idx, te_idx in skf.split(X_np, y_np):
            y_tr_np = y_np[tr_idx]
            if np.unique(y_tr_np).size < 2:
                continue  # 학습세트가 단일클래스면 건너뜀
            # 학습(cuML)
            X_tr = cp.asarray(X_np[tr_idx]); y_tr = cp.asarray(y_tr_np)
            model = cuRF(
                n_estimators=n_estimators,
                max_features=max(1, int(np.sqrt(X_np.shape[1]))),
                bootstrap=True,
                n_streams=1,  # 재현성 강화
                random_state=seed,
            )
            model.fit(X_tr, y_tr)

            # 테스트에서 S0 위치만 확률 취합
            te_S0_mask = np.isin(te_idx, idx_S0)
            te_S0_idx = te_idx[te_S0_mask]
            if te_S0_idx.size == 0:
                continue
            X_te_np = X_np[te_S0_idx]
            if X_te_np.ndim == 1:
                X_te_np = X_te_np.reshape(1, -1)
            X_te = cp.asarray(X_te_np)
            proba = model.predict_proba(X_te)
            p0_vals.append(cp.asnumpy(proba)[:, 0])

# --- [수정] VRAM 누수 방지를 위한 명시적 메모리 해제 (패치 C) ---
            try:
                del model, X_tr, y_tr, X_te, proba
            except Exception:
                pass
            try:
                mempool.free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
            # -----------------------------------------------------


        if not p0_vals:
            # 극단 케이스: 모든 폴드 스킵 → sklearn OOB로 백업
            from sklearn.ensemble import RandomForestClassifier as skRF
            sk = skRF(
                n_estimators=n_estimators,
                max_features=max(1, int(np.sqrt(X_np.shape[1]))),
                bootstrap=True,
                oob_score=True,
                random_state=seed,
                n_jobs=-1,
            )
            sk.fit(X_np, y_np)
            oob = getattr(sk, "oob_decision_function_", None)
            if oob is not None:
                return float(np.nanmean(oob[idx_S0, 0]))
            X_S0_np = X_np[idx_S0]
            if X_S0_np.ndim == 1:
                X_S0_np = X_S0_np.reshape(1, -1)
            proba = sk.predict_proba(X_S0_np)
            return float(np.mean(proba[:, 0]))

        return float(np.mean(np.concatenate(p0_vals)))

    else:
        raise ValueError(f"Unknown backend: {backend}")


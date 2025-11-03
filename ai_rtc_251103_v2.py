"""
Reinforcement Learning-Based Real-Time Contrasts Control Chart Using an Adaptive Window Size
논문 실험 재현 — v5 (CV 가드/공정 백업 + t<w 판정 스킵 + 정적 ETA + tqdm + GPU 옵션)

- 데이터 생성 (시나리오 I/II, d=10)
- RTC 모니터링 통계 p(S0, t) 계산 (RandomForest)
  * CPU 정확 재현: scikit-learn RF + OOB
  * GPU 가속 근사: RAPIDS cuML RF + StratifiedKFold(OOB 근사) — 단, 소수클래스<2면 sklearn OOB로 공정 백업
- ARL0=200을 만족하도록 CL(제어한계) 부트스트랩으로 산출
- Policy Gradient + CNN 정책 네트워크(논문 Table 1 구성)로 윈도우 선택
- ARL1 평가 (시나리오 I/II)
- (1) 시작 시 벤치마크로 "전체 파이프라인 예상 완료 시각" 1회 추정
- (2) tqdm으로 단계별 진행률/ETA 표시

필요 패키지:
  pip install numpy scikit-learn torch tqdm
GPU 가속(RAPIDS) 사용 시:
  conda install -c rapidsai -c conda-forge -c nvidia cuml cupy -y
"""
from __future__ import annotations
import argparse
import math
import os
import random
import time
from dataclasses import dataclass, replace as _replace
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional

import numpy as np
import pickle
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
import gc
import torch

# ----------------------------- 경고 필터 -----------------------------
# (메시지 패턴 대신 모듈+카테고리로 확실히 억제)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.model_selection._split")
# 필요시 추가
# warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

# ----------------------------- 유틸리티 -----------------------------

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sqrt_int(x: float) -> int:
    return max(1, int(round(math.sqrt(x))))

def save_policy(policy: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = policy.module.state_dict() if isinstance(policy, nn.DataParallel) else policy.state_dict()
    torch.save(state, path)

def load_policy(path: str, d: int, num_actions: int, device: str) -> nn.Module:
    policy = PolicyCNN(d=d, num_actions=num_actions)
    state = torch.load(path, map_location=device)
    policy.load_state_dict(state)
    policy.to(device)
    policy.eval()
    return policy

# ------------------------- 데이터 생성부 -----------------------------
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

# -------------------------- p(S0,t) 계산 ----------------------------

def compute_pS0_stat(
    X: NDArray,
    y: NDArray,
    idx_S0: NDArray,
    d: int,
    n_estimators: int,
    seed: int,
    backend: str = 'sklearn',
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
            device_type="gpu" if use_gpu else "cpu",   # ✅ GPU 활성화
        )

        p0_vals = []
        for tr_idx, te_idx in skf.split(X_np, y_np):
            y_tr_np = y_np[tr_idx]
            if np.unique(y_tr_np).size < 2:
                continue  # 학습세트가 단일클래스면 건너뜀

            model = LGBMClassifier(**lgbm_params, device_type="gpu" if use_gpu else "cpu",)
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




@dataclass
class CLCalib:
    CL: float
    std_boot: float


def estimate_CL_for_window(
    S0: NDArray,
    d: int,
    window: int,
    n_boot: int,
    n_estimators: int,
    seed: int,
    backend: str = 'sklearn',
) -> CLCalib:
    rng = check_random_state(seed)
    alpha = 1.0 / 200.0
    stats = []
    N0 = len(S0)
    pbar = tqdm(range(n_boot), desc=f"  CL Boot (w={window})", leave=False, dynamic_ncols=True)
    for _ in pbar:
        start = 0 if N0 - window <= 0 else rng.randint(0, N0 - window)
        Sw = S0[start:start + window]
        X = np.vstack([S0, Sw])
        y = np.hstack([np.zeros(len(S0), dtype=int), np.ones(len(Sw), dtype=int)])
        pS0 = compute_pS0_stat(X, y, np.arange(len(S0)), d=d, n_estimators=n_estimators, seed=rng.randint(1_000_000), backend=backend)
        stats.append(pS0)
    stats = np.asarray(stats)
    CL = np.quantile(stats, 1 - alpha)
    std_boot = float(np.std(stats, ddof=1))
    return CLCalib(CL=CL, std_boot=std_boot)


# -------------------------- RL 정책 네트워크 --------------------------
class PolicyCNN(nn.Module):
    """논문 Table 1 구성: Conv(1->16,k=4x4)->Sigmoid -> Conv(16->8,k=4x4)->Sigmoid -> Flatten(288) -> FC(288)->Sigmoid -> FC(#actions)"""
    def __init__(self, d: int, num_actions: int):
        super().__init__()
        if d != 10:
            raise ValueError("이 CNN 구조는 논문 기준 d=10에 맞춰져 있습니다.")
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(4, 4))
        self.act1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(16, 8, kernel_size=(4, 4))
        self.act2 = nn.Sigmoid()
        H, W = 9, 4  # (15x10)->(12x7)->(9x4)
        flatten = 8 * H * W  # 288
        self.fc1 = nn.Linear(flatten, 288)
        self.act3 = nn.Sigmoid()
        self.fc2 = nn.Linear(288, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.act1(self.conv1(x))
        z = self.act2(self.conv2(z))
        z = torch.flatten(z, 1)
        z = self.act3(self.fc1(z))
        logits = self.fc2(z)
        return logits


@dataclass
class RLConfig:
    action_set: Tuple[int, int, int] = (5, 10, 15)
    lr: float = 1e-3
    gamma: float = 0.99
    episodes: int = 50
    device: str = 'cpu'


# ------------------------ 상태 구성 & 보상 계산 ------------------------

def make_state_tensor(windowed: NDArray, d: int, L: int = 15) -> torch.Tensor:
    w = windowed.shape[0]
    if w >= L:
        data = windowed[-L:]
    else:
        pad = np.zeros((L - w, d), dtype=windowed.dtype)
        data = np.vstack([pad, windowed])
    t = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return t


@dataclass
class WindowCalib:
    CL: float
    std: float
    size: int


def reward_by_algorithm1(action_idx: int, distances: List[float], in_control: bool) -> float:
    # distances: 큰 것이 IC에서 선호되도록 설계됨
    order = np.argsort(distances)[::-1]
    rank = int(np.where(order == action_idx)[0][0])
    return ({0: 1.0, 1: -1.0, 2: -2.0}[rank] if in_control else {0: -2.0, 1: -1.0, 2: 1.0}[rank])


# ----------------------------- 학습 루프 ------------------------------

def train_rl_policy(
    policy: PolicyCNN,
    optimizer: optim.Optimizer,
    cfg: RLConfig,
    scen: ScenarioConfig,
    calib_map: Dict[int, WindowCalib],
    S0_ref: NDArray,
    seed: int,
    rf_backend: str = 'sklearn',
    n_estimators_eval: int = 150,
) -> PolicyCNN:
    rng = check_random_state(seed)
    policy.train()
    device = cfg.device
    if device.startswith('cuda') and torch.cuda.device_count() > 1:
        print("    > [RL] nn.DataParallel 활성화 (멀티 GPU 사용)")
        policy = nn.DataParallel(policy)
    policy.to(device)
    if hasattr(torch, 'compile'):
        try:
            print("    > [RL] torch.compile() 적용 (PyTorch 2.x)")
            policy = torch.compile(policy)
        except Exception:
            pass
    actions = list(cfg.action_set)

    pbar_ep = tqdm(range(cfg.episodes), desc="[RL] Training", dynamic_ncols=True)
    for ep in pbar_ep:
        lam_choices_I = [math.sqrt(x) for x in [0.25, 0.5, 1, 2, 3, 5, 7, 9]]
        lam_choices_II = [math.sqrt(x) for x in [2, 3, 5, 7, 9]]
        if rng.rand() < 0.5:
            scenario = 'I'; lam = float(rng.choice(lam_choices_I))
        else:
            scenario = 'II'; lam = float(rng.choice(lam_choices_II))
        X, labels_ic = make_phase2_series(scen, rng, scenario, lam)

        logps: List[torch.Tensor] = []
        rewards: List[float] = []
        a_trace: List[int] = []

        for t in range(1, scen.T + 1):
            # --- 유효 행동 마스크(t>=w) ---
            valid = [t >= w for w in actions]
            if not any(valid):
                # 아직 어떤 창도 꽉 차지 못한 시점이면 상태만 갱신
                Lmax = 15
                w_for_state = min(max(actions), t)
                state = make_state_tensor(X[t - w_for_state:t], scen.d, L=Lmax).to(device)
                with torch.no_grad():
                    _ = policy(state)
                continue

            # --- 유효 행동만 통계/거리 계산 ---
            ms_list: List[float] = []
            D_list: List[float] = []
            valid_indices: List[int] = []
            for a_idx_tmp, w in enumerate(actions):
                if not valid[a_idx_tmp]:
                    continue
                Sw = X[t - w:t]
                Xrf = np.vstack([S0_ref, Sw])
                yrf = np.hstack([np.zeros(len(S0_ref), dtype=int), np.ones(len(Sw), dtype=int)])
                pS0 = compute_pS0_stat(
                    Xrf, yrf, np.arange(len(S0_ref)),
                    d=scen.d, n_estimators=n_estimators_eval,
                    seed=rng.randint(1_000_000), backend=rf_backend
                )
                ms_list.append(pS0)
                calib = calib_map[w]
                D = (calib.CL - pS0) / max(1e-8, calib.std)
                D_list.append(float(D))
                valid_indices.append(a_idx_tmp)

            # --- 정책 분포 (유효하지 않은 행동 마스킹) ---
            Lmax = 15
            w_for_state = min(max(actions), t)
            state = make_state_tensor(X[t - w_for_state:t], scen.d, L=Lmax).to(device)
            logits = policy(state)
            logits_masked = logits.clone()
            for idx, ok in enumerate(valid):
                if not ok:
                    logits_masked[0, idx] = -1e9
            probs = torch.softmax(logits_masked, dim=-1)
            m = torch.distributions.Categorical(probs=probs)
            a_idx_tensor = m.sample()
            a_idx = int(a_idx_tensor.item())

            a_trace.append(actions[a_idx])
            logps.append(m.log_prob(a_idx_tensor))

            # --- 보상(유효 행동의 로컬 인덱스 기준) ---
            if a_idx not in valid_indices:
                rewards.append(-2.0)  # 방어적 패널티(원칙적으로 발생하지 않음)
            else:
                local_idx = valid_indices.index(a_idx)
                ic = bool(labels_ic[t-1] == 1)
                rewards.append(float(reward_by_algorithm1(local_idx, D_list, in_control=ic)))

        # --- Policy Gradient 업데이트 ---
        pbar_ep.set_postfix_str("Updating policy.")
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + cfg.gamma * G
            returns.append(G)
        returns = list(reversed(returns))
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        loss = 0.0
        for logp, Gt in zip(logps, returns_t):
            loss = loss - logp * Gt
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # --- 10 에피소드마다 요약 로그 ---
        if every(10, ep):
            avg_r = float(np.mean(rewards)) if len(rewards) else 0.0
            if len(a_trace):
                vals, counts = np.unique(a_trace, return_counts=True)
                pi_hist = {int(v): f"{float(c)/float(len(a_trace)):.2f}" for v, c in zip(vals, counts)}
            else:
                pi_hist = {}
            print(f"\n[ep {ep+1}] avg_reward={avg_r:.3f}  pi(w)={pi_hist}")

    if isinstance(policy, nn.DataParallel):
        policy = policy.module
    return policy


# ----------------------------- 평가 (ARL1) ----------------------------

def run_length_until_alarm(
    X: NDArray,
    S0_ref: NDArray,
    policy: PolicyCNN,
    actions: List[int],
    calib_map: Dict[int, WindowCalib],
    d: int,
    rf_backend: str = 'sklearn',
    n_estimators_eval: int = 150,  # <-- 매개변수 추가 (기본값 150)
) -> int:
    policy.eval()
    device = next(policy.parameters()).device
    T = len(X)
    for t in range(1, T + 1):
        Lmax = 15
        w_for_state = min(max(actions), t)
        state = make_state_tensor(X[t - w_for_state:t], d, L=Lmax).to(device)
        with torch.no_grad():
            logits = policy(state)
            a_idx = int(torch.argmax(logits, dim=-1).item())

        w = actions[a_idx]
        # 아직 창이 다 안 찼으면 알람 판단을 미룹니다.
        if t < w:
            continue
        Sw = X[t - w:t]
        Xrf = np.vstack([S0_ref, Sw])
        yrf = np.hstack([np.zeros(len(S0_ref), dtype=int), np.ones(len(Sw), dtype=int)])
        pS0 = compute_pS0_stat(
            Xrf, yrf, np.arange(len(S0_ref)),
            d=d, n_estimators=n_estimators_eval, seed=42, backend=rf_backend
        )
        calib = calib_map[w]
        if pS0 > calib.CL:
            return t
    return T


def every(n, i):  # i: 0-based index
    return (i + 1) % n == 0


def evaluate_arl1(
    scen_cfg: ScenarioConfig,
    lam_list: List[float],
    scenario: str,
    policy: PolicyCNN,
    actions: List[int],
    calib_map: Dict[int, WindowCalib],
    S0_ref: NDArray,
    R: int,
    seed: int,
    rf_backend: str = 'sklearn',
    n_estimators_eval: int = 150,  # <-- 매개변수 추가
) -> Tuple[List[float], List[float]]:
    arl_means: List[float] = []
    arl_stds: List[float] = []
    rng = check_random_state(seed)
    # ARL1: 시프트 즉시 발생을 가정하므로 shift_time=0으로 복사
    scen_cfg_eval = _replace(scen_cfg, shift_time=0)
    for lam in lam_list:
        print(f"  Evaluating lam^2={lam**2:.2f} (lam={lam:.4f}) (R={R} reps).")
        RLs = []
        pbar_R = tqdm(range(R), desc="    Reps", leave=False, dynamic_ncols=True)
        for _ in pbar_R:
            X, _ = make_phase2_series(scen_cfg_eval, rng, scenario, lam)
            rl = run_length_until_alarm(
                X, S0_ref, policy, actions, calib_map, scen_cfg.d,
                rf_backend=rf_backend, n_estimators_eval=n_estimators_eval
            )
            RLs.append(rl)
        arl_means.append(float(np.mean(RLs)))
        arl_stds.append(float(np.std(RLs, ddof=1)))
    return arl_means, arl_stds


# ----------------------------- (신규) 벤치마크 헬퍼 ----------------------------

def _run_benchmark_rf(
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
        # ✅ LGBM 경로: fit/predict 모두 넘파이로 통일 (feature names 경고 방지)
        from lightgbm import LGBMClassifier

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int32)
        X_np = np.ascontiguousarray(X_np)
        y_np = np.ascontiguousarray(y_np)
        
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
            device_type="gpu" if use_gpu else "cpu",
            # scale_pos_weight 등은 벤치마크 목적상 생략 (속도만 잼)
        )
        model.fit(X_np, y_np)
        _ = model.predict_proba(X_np)  # 예측까지 포함해 ETA가 실제에 가깝게

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return time.perf_counter() - t0

# ------------------------------- 메인 -------------------------------

def main():
    import torch
    start_time = datetime.now()

    # ---------- 인자 파싱 ----------
    parser = argparse.ArgumentParser(description="RL-RTC 논문 재현 스크립트 (v5: CV 가드/공정 백업 + t<w 판정 스킵)")
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="PyTorch CNN 연산 장치 ('cuda' 또는 'cpu')")
    parser.add_argument('--episodes', type=int, default=30, help='RL 학습 에피소드 수 (권장 100~300 이상)')
    parser.add_argument('--n_boot', type=int, default=800, help='CL 부트스트랩 반복 (권장 2000~10000)')
    parser.add_argument('--action_set', type=str, default='5,10,15', help='윈도우 크기 옵션 (예: 5,10,15 또는 3,10,17)')
    parser.add_argument('--R', type=int, default=100, help='ARL1 평가 반복 (논문 500)')
    parser.add_argument('--rf_backend', type=str, default='sklearn', 
                        choices=['sklearn','cuml_cv','lgbm'], 
                        help="분류기 백엔드: 'sklearn'(CPU,OOB), 'cuml_cv'(GPU,K-fold OOB), 'lgbm'(CPU/GPU)")
    parser.add_argument('--guess_arl1', type=int, default=15, help='ARL1 평균 추정치(정적 ETA 계산용)')
    parser.add_argument('--n_estimators_eval', type=int, default=150, help='평가(ARL1) 단계에서 사용할 RF 트리 수 (기존 300 → 축소)')
    parser.add_argument('--policy_in', type=str, default=None, help='불러올 정책 가중치(.pt). 지정 시 해당 가중치로 시작')
    parser.add_argument('--policy_out', type=str, default=None, help='학습 후 저장할 정책 경로(.pt). 미지정 시 자동 결정')    
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device
    os.environ["AI_RTC_DEVICE"] = device  # ★ 추가
    
    action_set = tuple(int(x) for x in args.action_set.split(','))
    actions = list(action_set)
    scen = ScenarioConfig()

    # outputs 디렉토리/정책 경로 준비
    outputs_dir = os.path.abspath('./outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    action_str = "-".join(map(str, actions))
    if args.policy_out is None:
        # 에피소드/액션셋 기준 고정 이름 (원하면 timestamp로 변경 가능)
        args.policy_out = os.path.join(outputs_dir, f'policy_{action_str}.pt')

    # ---------- 설정 출력 ----------
    print("="*60)
    print("RL-RTC 논문 재현 시뮬레이션 시작 (v5)")
    print("="*60)
    print(f"Device (CNN): {device}")
    print(f"RF Backend:   {args.rf_backend}")
    print(f"Action Set:   {args.action_set}")
    print(f"RL Episodes:  {args.episodes}")
    print(f"CL n_boot:    {args.n_boot}")
    print(f"ARL1 R:       {args.R}")
    print(f"Seed:         {args.seed}")
    print(f"Started at:   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # -----------------------------------------------------------------
    # 정적 ETA 추정 (서로 다른 단위 작업의 비용 차이를 반영)
    # -----------------------------------------------------------------
    print("[ETA] 전체 파이프라인 예상 소요 시간 벤치마크 수행 중.")

    rng_s0 = check_random_state(args.seed)
    S0_ref_bench = gen_reference_data(scen, rng_s0)

    # <<< 여기에 추가 >>>
    t_unit = _run_benchmark_rf(S0_ref_bench, scen.d, args.n_estimators_eval, args.seed, args.rf_backend)
    print(f"[ETA] Benchmark ({args.rf_backend}, n_estimators={args.n_estimators_eval}): {t_unit:.3f} sec/call")


    N_BENCH_RUNS = 20
    bench_times_300 = []
    pbar_bench = tqdm(range(N_BENCH_RUNS), desc="  Bench RF(n=300)", leave=False, dynamic_ncols=True)
    for i in pbar_bench:
        bench_times_300.append(_run_benchmark_rf(S0_ref_bench, scen.d, n_estimators=300, seed=args.seed + i, backend=args.rf_backend))
    avg_rf_300_time = float(np.mean(bench_times_300))
    # n=200 RF는 n=300 RF 시간의 (200/300) 근사
    avg_rf_200_time = avg_rf_300_time * (200.0 / 300.0)

    lam_I = [math.sqrt(x) for x in [0.25, 0.5, 1, 2, 3, 5, 7, 9]]
    lam_II = [math.sqrt(x) for x in [2, 3, 5, 7, 9]]

    calls_cl = len(actions) * args.n_boot
    time_cl = calls_cl * avg_rf_300_time

    calls_rl = args.episodes * scen.T * len(actions)
    time_rl = calls_rl * avg_rf_200_time

    total_lam_runs = len(lam_I) + len(lam_II)
    calls_arl1 = total_lam_runs * args.R * args.guess_arl1
    
    # 평가에서 n_estimators_eval 사용 시 ETA 보정 (단순 선형 근사)
    n_eval = args.n_estimators_eval
    time_arl1 = calls_arl1 * (avg_rf_300_time * (n_eval / 300.0))

    total_estimated_seconds = time_cl + time_rl + time_arl1
    finish_time_eta = start_time + timedelta(seconds=total_estimated_seconds)

    print("" + "-"*60)
    print(f"[ETA] 벤치마크 완료 (Backend: {args.rf_backend})")
    print(f"  - RF(n=300) 1회 평균 시간: {avg_rf_300_time:.4f} 초")
    print(f"  - RF(n=200) 1회 평균 시간: {avg_rf_200_time:.4f} 초")
    print("[ETA] 예상 작업량 (RF 호출 횟수)")
    print(f"  - (1) CL 보정 : {calls_cl:10,d} 회 (예상 {timedelta(seconds=time_cl)})")
    print(f"  - (2) RL 학습 : {calls_rl:10,d} 회 (예상 {timedelta(seconds=time_rl)})")
    print(f"  - (3) ARL1 평가: {calls_arl1:10,d} 회 (가정 ARL1={args.guess_arl1}) (예상 {timedelta(seconds=time_arl1)})")
    print("."*60)
    print(f"[ETA] 총 예상 소요 시간 : {timedelta(seconds=total_estimated_seconds)}")
    print(f"[ETA] 예상 완료 시간 (ETA): {finish_time_eta.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "")

    # -----------------------------------------------------------------
    # Phase I 참조 데이터 (벤치마크에서 생성한 S0 재사용)
    # -----------------------------------------------------------------
    S0_ref = S0_ref_bench

    # -------------------- CL 보정 --------------------
    print(f"[작업 1/3] CL(ARL0=200) 보정 시작 (n_boot={args.n_boot}).")
    calib_map: Dict[int, WindowCalib] = {}
    for w in actions:
        calib = estimate_CL_for_window(S0_ref, scen.d, window=w, n_boot=args.n_boot, n_estimators=300, seed=rng_s0.randint(1_000_000), backend=args.rf_backend)
        calib_map[w] = WindowCalib(CL=calib.CL, std=calib.std_boot, size=w)
        print(f"  [CL] window={w:2d} -> CL={calib.CL:.5f}, std={calib.std_boot:.5f}")
    print("[작업 1/3] CL 보정 완료.")
    
    np.save(os.path.join(outputs_dir, "S0_ref.npy"), S0_ref)
    with open(os.path.join(outputs_dir, "calib_map.pkl"), "wb") as f:
        pickle.dump(calib_map, f)
    print("[저장 완료] S0_ref.npy, calib_map.pkl")

    # -------------------- RL 학습 --------------------

    policy = PolicyCNN(d=scen.d, num_actions=len(actions))
    # 사전 가중치 로드 (있으면)
    if args.policy_in and os.path.isfile(args.policy_in):
        print(f"[작업 2/3] 기존 정책 로드: {args.policy_in}")
        state = torch.load(args.policy_in, map_location=device)
        policy.load_state_dict(state)
        policy.to(device)
        policy.eval()
    else:
        if args.policy_in:
            print(f"[작업 2/3] 경고: --policy_in 경로가 존재하지 않습니다: {args.policy_in}")

    # 학습 수행 여부
    if args.episodes > 0:
        print(f"[작업 2/3] RL 정책 학습 시작 (Episodes={args.episodes})...")
        optimizer = optim.Adam(policy.parameters(), lr=1e-3)
        rl_cfg = RLConfig(action_set=tuple(actions), device=device, episodes=args.episodes)
        policy = train_rl_policy(policy, optimizer, rl_cfg, scen, calib_map, S0_ref, seed=args.seed, rf_backend=args.rf_backend, n_estimators_eval=args.n_estimators_eval)
        print("[작업 2/3] RL 정책 학습 완료.")
        # 학습 후 저장
        try:
            save_policy(policy, args.policy_out)
            print(f"[작업 2/3] 정책 저장 완료: {args.policy_out}")
        except Exception as e:
            print(f"[작업 2/3] 정책 저장 실패: {e}")
    else:
        print("[작업 2/3] RL 학습 스킵 (episodes=0). 기존/초기화된 정책으로 평가 진행.")    
    
    print("[작업 2/3] RL 정책 학습 완료.")

    # -------------------- ARL1 평가 --------------------
    print(f"[작업 3/3 - A] Scenario I ARL1 평가 시작 (R={args.R})...")
    mean_I, std_I = evaluate_arl1(
        scen, lam_I, 'I', policy, actions, calib_map, S0_ref,
        R=args.R, seed=args.seed + 7, rf_backend=args.rf_backend,
        n_estimators_eval=args.n_estimators_eval  # <-- args 값 전달
    )
    print("[작업 3/3 - A] Scenario I 완료.")

    print(f"[작업 3/3 - B] Scenario II ARL1 평가 시작 (R={args.R})...")
    mean_II, std_II = evaluate_arl1(
        scen, lam_II, 'II', policy, actions, calib_map, S0_ref,
        R=args.R, seed=args.seed + 13, rf_backend=args.rf_backend,
        n_estimators_eval=args.n_estimators_eval  # <-- args 값 전달
    )
    print("[작업 3/3 - B] Scenario II 완료.")

    # -------------------- 결과 출력/저장 --------------------
    def fmt_table(lams, means, stds, title):
        print("" + title)
        print("lambda^2\tlambda\t\tARL1 (mean [std])")
        for lam, m, s in zip(lams, means, stds):
            print(f"{lam**2:<8.2f}\t{lam:<10.4f}\t{m:.2f} [{s:.2f}]")

    fmt_table(lam_I, mean_I, std_I, f"Scenario I (action_set={actions})")
    fmt_table(lam_II, mean_II, std_II, f"Scenario II (action_set={actions})")

    out_dir = os.path.abspath('./outputs')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    action_str = "-".join(map(str, actions))
    np.savetxt(os.path.join(out_dir, f'arl1_scenI_{action_str}_{ts}.csv'), np.c_[lam_I, mean_I, std_I], delimiter=',', header='lambda,arl1_mean,arl1_std', comments='')
    np.savetxt(os.path.join(out_dir, f'arl1_scenII_{action_str}_{ts}.csv'), np.c_[lam_II, mean_II, std_II], delimiter=',', header='lambda,arl1_mean,arl1_std', comments='')
    print(f"[저장 완료] outputs/arl1_scenI_*.csv, outputs/arl1_scenII_*.csv")

    end_time = datetime.now()
    # ------- quiet cleanup for cuML (optional) -------
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    gc.collect()
    
    print("" + "="*60)
    print(f"시뮬레이션 종료. 총 소요 시간: {end_time - start_time}")
    print(f"종료 시각: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


if __name__ == '__main__':
    main()

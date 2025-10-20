#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reinforcement Learning-Based Real-Time Contrasts Control Chart Using an Adaptive Window Size
논문 실험 재현 — v5 (표준화 분모 가드 + 로그/아카이빙 + 학습루프 비용 완화)

- 데이터 생성 (시나리오 I/II, d=10)
- RTC 모니터링 통계 p(S0, t) 계산 (RandomForest)
  * CPU 정확 재현: scikit-learn RF + OOB
  * GPU 가속 근사: RAPIDS cuML RF + K-fold 교차검증으로 OOB 근사
- ARL0=200을 만족하도록 CL(제어한계) 부트스트랩으로 산출
- Policy Gradient + CNN 정책 네트워크(논문 Table 1 구성)로 윈도우 선택
- ARL1 평가 (시나리오 I/II)

[이번 버전의 주요 개선]
1) 표준화 분모 가드 보강: std≈0일 때 MAD 기반 백업 표준편차로 대체
2) 로그/아카이빙: CL 테이블, RL 학습 통계, 정책 스냅샷(.pt), ARL1 CSV/요약 JSON 저장
3) 학습루프 비용 완화: --cheap_mode에서 RL 단계의 RF를 "경량 설정"으로 학습(트리 수 축소, 부분 특성 사용, warm-start)

작성: 2025-10-20 (Asia/Seoul)
"""
import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from tqdm import tqdm

# ------------------------ 유틸 ------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sqrt_int(x: int) -> int:
    return max(1, int(math.sqrt(x)))

def mad_std_fallback(x: NDArray, eps: float = 1e-8) -> float:
    """
    x의 표준편차가 eps보다 작으면, MAD*1.4826을 백업 표준편차로 사용.
    """
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    if std >= eps:
        return std
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    robust = 1.4826 * mad
    return max(robust, eps)

# ------------------------ 시나리오/데이터 ------------------------

@dataclass
class ScenarioConfig:
    d: int = 10
    N0: int = 1500          # 참조데이터(Phase I) 크기
    T: int = 300            # Phase II 길이 (한 에피소드 길이)
    shift_time: int = 100   # t >= shift_time 부터 OOC (ARL1에서는 0으로 재설정)
    sigma: float = 1.0

def make_cov(d: int) -> NDArray:
    return np.eye(d)

def gen_reference_data(cfg: ScenarioConfig, rng: np.random.RandomState) -> NDArray:
    return rng.multivariate_normal(mean=np.zeros(cfg.d), cov=make_cov(cfg.d), size=cfg.N0)

def apply_mean_shift(X: NDArray, delta: NDArray, t0: int) -> NDArray:
    Y = X.copy()
    if t0 < len(Y):
        Y[t0:] += delta
    return Y

def make_phase2_series(cfg: ScenarioConfig, rng: np.random.RandomState,
                       scenario: str, lam: float) -> Tuple[NDArray, NDArray]:
    """
    Returns X (T x d), labels_ic (T,) where labels_ic[t]=1 if in-control else 0.
    Scenario I: 1번 변수에 평균 이동 lam
    Scenario II: 모든 변수에 평균 이동 lam/√d (분산 유지)
    """
    X = rng.multivariate_normal(np.zeros(cfg.d), make_cov(cfg.d), size=cfg.T)
    labels_ic = np.ones(cfg.T, dtype=np.int64)
    if scenario.upper() == 'I':
        delta = np.zeros(cfg.d); delta[0] = lam
    else:
        delta = np.ones(cfg.d) * (lam / math.sqrt(cfg.d))
    X = apply_mean_shift(X, delta, cfg.shift_time)
    labels_ic[cfg.shift_time:] = 0
    return X, labels_ic

# ------------------------ RF 기반 p(S0,t) ------------------------

def build_window_matrix(X: NDArray, w: int) -> NDArray:
    """
    (n, d) -> (n - w + 1, w, d)
    각 윈도우를 그대로 CNN 입력으로 사용할 수도 있고,
    RF에는 평탄화((w*d,) 벡터)로 사용.
    """
    n, d = X.shape
    if w > n: return np.empty((0, w, d), dtype=X.dtype)
    blocks = np.lib.stride_tricks.sliding_window_view(X, (w, d))[:,0,0,:]
    # blocks: (n-w+1, w, d)
    return blocks

def flatten_windows(blocks: NDArray) -> NDArray:
    return blocks.reshape(blocks.shape[0], -1)

@dataclass
class RFConfig:
    backend: str = 'sklearn'     # 'sklearn' | 'cuml_cv'
    n_estimators_full: int = 500
    n_estimators_train: int = 100
    feature_frac_train: float = 1.0   # RL 단계에서 사용(<=1.0)
    kfold: int = 5                    # cuML 경로에서 OOB 근사
    n_jobs: int = -1
    cheap_mode: bool = False          # RL 단계에서 경량 RF 사용

def estimate_pS0(
    S0: NDArray, Sw: NDArray, w: int, d: int, seed: int, rf_cfg: RFConfig
) -> float:
    """
    S0로부터 (class 0), Sw로부터 (class 1) 윈도우를 만들고 RF로 class0 확률 p0를 얻음.
    최종 모니터링 통계는 S0 윈도우들에서의 p0 평균.
    """
    S0_blocks = build_window_matrix(S0, w)   # (n0, w, d)
    Sw_blocks = build_window_matrix(Sw, w)   # (n1, w, d)
    if len(S0_blocks) == 0 or len(Sw_blocks) == 0:
        return 0.5  # 중립
    
    X0 = flatten_windows(S0_blocks)
    X1 = flatten_windows(Sw_blocks)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(len(X0), dtype=np.int64),
                        np.ones(len(X1), dtype=np.int64)])
    idx_S0 = np.arange(len(X0))

    # RL 단계에서 비용 완화를 위해 경량 설정 선택
    n_estimators = rf_cfg.n_estimators_train if rf_cfg.cheap_mode else rf_cfg.n_estimators_full
    max_features = sqrt_int(X.shape[1])
    if rf_cfg.cheap_mode and rf_cfg.feature_frac_train < 1.0:
        max_features = max(1, int(max_features * rf_cfg.feature_frac_train))

    if rf_cfg.backend == 'sklearn':
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=True,
            oob_score=True,
            n_jobs=rf_cfg.n_jobs,
            random_state=seed,
            warm_start=True if rf_cfg.cheap_mode else False,
        )
        rf.fit(X, y)
        if rf.oob_decision_function_ is None:
            raise RuntimeError("oob_decision_function_ is None")
        p0 = np.mean(rf.oob_decision_function_[idx_S0, 0])
        return float(p0)

    elif rf_cfg.backend == 'cuml_cv':
        try:
            import cupy as cp
            from cuml.ensemble import RandomForestClassifier as cuRF
            from sklearn.model_selection import KFold
        except Exception as e:
            raise RuntimeError("cuML 사용 불가: RAPIDS가 설치되어야 합니다.") from e

        X_np = X.astype(np.float32)
        y_np = y.astype(np.int32)
        kf = KFold(n_splits=rf_cfg.kfold, shuffle=True, random_state=seed)
        p0_vals = []
        for tr_idx, te_idx in kf.split(X_np):
            X_tr = cp.asarray(X_np[tr_idx]); y_tr = cp.asarray(y_np[tr_idx])
            model = cuRF(
                n_estimators=n_estimators,
                max_features=max_features,
                bootstrap=True,
                n_streams=8,
                random_state=seed,
            )
            model.fit(X_tr, y_tr)
            te_S0_mask = np.isin(te_idx, idx_S0)
            if not np.any(te_S0_mask):  # 이번 fold에 S0가 없으면 skip
                continue
            X_te = cp.asarray(X_np[te_idx])
            proba = model.predict_proba(X_te).get()
            p0_vals.append(np.mean(proba[te_S0_mask, 0]))
        if not p0_vals:
            return 0.5
        return float(np.mean(p0_vals))

    else:
        raise ValueError("rf_cfg.backend must be 'sklearn' or 'cuml_cv'")

# ------------------------ CNN 정책 ------------------------

class PolicyCNN(nn.Module):
    def __init__(self, d: int, num_actions: int):
        super().__init__()
        # 입력: (B, 1, L, d)  — L은 15로 고정 패딩
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, d), stride=(1, d)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(8 * 13, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        logits = self.fc(self.conv(x))
        return torch.log_softmax(logits, dim=-1)

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

@dataclass
class RLConfig:
    action_set: Tuple[int, int, int] = (5, 10, 15)
    lr: float = 1e-3
    gamma: float = 0.99
    episodes: int = 50
    device: str = 'cpu'

def reward_by_algorithm1(action_idx: int, distances: List[float], in_control: bool) -> float:
    order = np.argsort(distances)[::-1]  # 큰 D가 앞
    rank = int(np.where(order == action_idx)[0][0])
    # 순위 기반 보상 (간단화): 상위일수록 +, 하위일수록 -
    K = len(distances)
    base = (K - 1 - rank) / (K - 1) if K > 1 else 0.0  # 0~1
    return float(base if in_control else -base)

# ------------------------ CL 추정 ------------------------

@dataclass
class CLCalibResult:
    CL: float
    std_boot: float

def estimate_CL_for_window(
    S0: NDArray,
    d: int,
    window: int,
    n_boot: int = 1000,
    seed: int = 42,
    backend: str = 'sklearn',
    rf_full_estimators: int = 500,
    kfold: int = 5,
) -> CLCalibResult:
    """
    부트스트랩으로 p(S0,t) 분포를 얻고, ARL0=200 조건 (상위 1-1/200 분위수) CL 산출.
    표준화 분모는 std_boot, 단 std≈0일 때 MAD 백업 사용.
    """
    rng = np.random.RandomState(seed)
    pvals = []
    rf_cfg = RFConfig(backend=backend, n_estimators_full=rf_full_estimators,
                      n_estimators_train=rf_full_estimators, kfold=kfold, cheap_mode=False)
    for b in tqdm(range(n_boot), desc=f"[CL] w={window}", leave=False):
        # 부트스트랩 S0*, Sw* (IC) 생성
        idx0 = rng.randint(0, len(S0), size=len(S0))
        S0_b = S0[idx0]
        # IC에서 Sw는 S0_b와 같은 분포. 길이를 window*2 정도로 생성 후 마지막 window를 사용
        Sw_b = rng.multivariate_normal(np.zeros(d), make_cov(d), size=window*2)
        p0 = estimate_pS0(S0_b, Sw_b, window, d, seed=rng.randint(1_000_000), rf_cfg=rf_cfg)
        pvals.append(p0)
    pvals = np.array(pvals, dtype=float)
    alpha = 1.0 - 1.0/200.0  # ARL0=200
    CL = float(np.quantile(pvals, alpha))
    std_boot = mad_std_fallback(pvals, eps=1e-8)  # 개선 1) 적용
    return CLCalibResult(CL=CL, std_boot=std_boot)

# ------------------------ RL 학습 ------------------------

def train_rl_policy(
    policy: PolicyCNN,
    optimizer: optim.Optimizer,
    cfg: RLConfig,
    scen: ScenarioConfig,
    calib_map: Dict[int, WindowCalib],
    S0_ref: NDArray,
    seed: int,
    rf_cfg_train: RFConfig,
    log_dir: str,
) -> PolicyCNN:
    device = torch.device(cfg.device)
    policy.to(device)

    rng = np.random.RandomState(seed + 123)
    actions = list(cfg.action_set)

    # 로그 아카이빙 준비
    os.makedirs(log_dir, exist_ok=True)
    rl_log_path = os.path.join(log_dir, "rl_training_log.csv")
    with open(rl_log_path, "w", encoding="utf-8") as f:
        f.write("episode,total_reward,mean_reward,chosen_w_counts,loss\n")

    for ep in tqdm(range(cfg.episodes), desc="[RL] Episodes", leave=True):
        X, labels_ic = make_phase2_series(scen, rng, 'I', lam=0.0)  # RL은 IC 구간 포함 전체에서 학습
        rewards: List[float] = []
        logps: List[torch.Tensor] = []
        chosen_counts = {w:0 for w in actions}

        for t in range(scen.T):
            # 상태 구성
            # 각 행동 후보 w마다, Sw는 [0:t+1] 구간에서 최근 w
            distances = []
            logp_vec = None

            # CNN 입력은 "현재까지 관측 X[:t+1]"의 마지막 max(actions) 길이 부분
            w_max = max(actions)
            start_idx = max(0, t+1 - w_max)
            windowed = X[start_idx:t+1]
            state = make_state_tensor(windowed, scen.d).to(device)

            log_probs = policy(state)                   # (1, |A|)
            probs = torch.exp(log_probs).squeeze(0)     # (|A|,)
            dist = torch.distributions.Categorical(probs=probs)
            a_idx = int(dist.sample().item())
            logps.append(dist.log_prob(torch.tensor(a_idx, device=device)))

            w = actions[a_idx]
            chosen_counts[w] += 1

            # 모니터링 통계 계산 (경량 모드 적용)
            rf_cfg_local = rf_cfg_train
            rf_cfg_local.cheap_mode = True  # 개선 3) RL 단계에서는 경량 RF
            # Sw는 현재까지의 마지막 w 구간 (w가 안 채워졌으면 가능한 만큼)
            w_eff = min(w, t+1)
            Sw = X[t+1-w_eff:t+1]
            p0 = estimate_pS0(S0_ref, Sw, w_eff, scen.d, seed=rng.randint(1_000_000), rf_cfg=rf_cfg_local)

            # 표준화 거리 D = (CL - MS)/std  (논문 알고리즘1의 방향성)
            calib = calib_map[w]
            D = (calib.CL - p0) / max(calib.std, 1e-8)

            r = reward_by_algorithm1(a_idx, [D for _ in actions], bool(labels_ic[t]))
            rewards.append(r)

        # REINFORCE 업데이트
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
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # 로그 저장
        total_reward = float(np.sum(rewards))
        mean_reward = float(np.mean(rewards))
        with open(rl_log_path, "a", encoding="utf-8") as f:
            f.write(f"{ep},{total_reward:.6f},{mean_reward:.6f},{json.dumps(chosen_counts, ensure_ascii=False)},{float(loss):.6f}\n")

    # 정책 스냅샷 저장
    torch.save(policy.state_dict(), os.path.join(log_dir, "policy_snapshot.pt"))
    return policy

# ------------------------ 평가 (ARL1) ------------------------

def run_length_until_alarm(
    X: NDArray,
    S0_ref: NDArray,
    policy: PolicyCNN,
    actions: List[int],
    calib_map: Dict[int, WindowCalib],
    d: int,
    rf_cfg_eval: RFConfig,
) -> int:
    policy.eval()
    device = next(policy.parameters()).device
    for t in range(len(X)):
        w_max = max(actions)
        start_idx = max(0, t+1 - w_max)
        windowed = X[start_idx:t+1]
        state = make_state_tensor(windowed, d).to(device)

        with torch.no_grad():
            log_probs = policy(state)
            probs = torch.exp(log_probs).squeeze(0)
            a_idx = int(torch.argmax(probs).item())  # 평가 시 탐욕적
        w = actions[a_idx]

        w_eff = min(w, t+1)
        Sw = X[t+1-w_eff:t+1]
        rf_cfg_local = rf_cfg_eval
        rf_cfg_local.cheap_mode = False  # 평가/CL에서는 full RF
        p0 = estimate_pS0(S0_ref, Sw, w_eff, d, seed=12345 + t, rf_cfg=rf_cfg_local)

        calib = calib_map[w]
        if p0 > calib.CL:   # 경보
            return t+1
    return len(X)

def evaluate_arl1(
    scen: ScenarioConfig,
    lambdas: List[float],
    scenario: str,
    policy: PolicyCNN,
    action_set: List[int],
    calib_map: Dict[int, WindowCalib],
    R: int,
    seed: int,
    rf_cfg_eval: RFConfig,
) -> Tuple[List[float], List[float]]:
    rng = np.random.RandomState(seed)
    means = []; stds = []
    for lam in tqdm(lambdas, desc=f"[ARL1] Scenario {scenario}", leave=True):
        RLs = []
        for r in range(R):
            X, _ = make_phase2_series(scen, rng, scenario, lam)
            # ARL1에서는 변화가 t=0부터 시작되도록 설정
            X = apply_mean_shift(X, (np.ones(scen.d) * (lam if scenario=='I' else lam/math.sqrt(scen.d))), 0)
            RL = run_length_until_alarm(
                X, S0_ref=S0_global, policy=policy, actions=action_set,
                calib_map=calib_map, d=scen.d, rf_cfg_eval=rf_cfg_eval
            )
            RLs.append(RL)
        means.append(float(np.mean(RLs))); stds.append(float(np.std(RLs, ddof=1)))
    return means, stds

# ------------------------ 메인 ------------------------

def main():
    parser = argparse.ArgumentParser(description="RL-RTC 논문 재현 스크립트 (v5)")
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--episodes', type=int, default=30, help='RL 학습 에피소드 수')
    parser.add_argument('--n_boot', type=int, default=800, help='CL 부트스트랩 반복')
    parser.add_argument('--action_set', type=str, default='5,10,15', help='윈도우 크기 옵션 (예: 5,10,15)')
    parser.add_argument('--R', type=int, default=100, help='ARL1 평가 반복 (논문: 500)')
    parser.add_argument('--rf_backend', type=str, default='sklearn', help="RF 백엔드: 'sklearn' 또는 'cuml_cv'")
    # 학습루프 비용 완화 파라미터
    parser.add_argument('--cheap_mode', action='store_true', help='RL 단계에서 경량 RF 사용')
    parser.add_argument('--rf_train_estimators', type=int, default=100, help='RL 단계 RF n_estimators')
    parser.add_argument('--rf_full_estimators', type=int, default=500, help='CL/평가 RF n_estimators')
    parser.add_argument('--rf_feature_frac', type=float, default=1.0, help='RL 단계 feature 사용 비율(0<frac<=1)')
    parser.add_argument('--kfold', type=int, default=5, help='cuML backend 시 K-fold')
    parser.add_argument('--out_dir', type=str, default='outputs', help='결과 저장 디렉토리')
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device
    action_set = tuple(int(x) for x in args.action_set.split(','))
    actions = list(action_set)
    scen = ScenarioConfig()

    print("="*60)
    print("RL-RTC 논문 재현 시뮬레이션 시작 (v5)")
    print("="*60)
    print(f"Device (CNN): {device}")
    print(f"RF Backend:   {args.rf_backend}")
    print(f"Action Set:   {args.action_set}")
    print(f"Episodes:     {args.episodes}")
    print(f"n_boot:       {args.n_boot}")
    print(f"R (ARL1):     {args.R}")
    print(f"cheap_mode:   {args.cheap_mode}")
    print("="*60)

    # -------------------- 참조데이터 S0 --------------------
    rng_s0 = np.random.RandomState(args.seed + 1)
    global S0_global
    S0_global = gen_reference_data(scen, rng_s0)

    # -------------------- CL 보정 --------------------
    print(f"[작업 1/3] CL(ARL0=200) 보정 시작 (n_boot={args.n_boot})...")
    calib_map: Dict[int, WindowCalib] = {}
    cl_rows = []
    for w in actions:
        calib = estimate_CL_for_window(
            S0=S0_global, d=scen.d, window=w, n_boot=args.n_boot,
            seed=rng_s0.randint(1_000_000), backend=args.rf_backend,
            rf_full_estimators=args.rf_full_estimators, kfold=args.kfold
        )
        # 개선 1) std_boot는 MAD 백업이 들어간 상태
        calib_map[w] = WindowCalib(CL=calib.CL, std=calib.std_boot, size=w)
        print(f"  [CL] window={w:2d} -> CL={calib.CL:.6f}, std={calib.std_boot:.6f}")
        cl_rows.append((w, calib.CL, calib.std_boot))
    print("[작업 1/3] CL 보정 완료.")

    # CL 테이블 저장 (개선 2)
    os.makedirs(args.out_dir, exist_ok=True)
    cl_path = os.path.join(args.out_dir, f'cl_table_actions-{"-".join(map(str,actions))}.csv')
    with open(cl_path, "w", encoding="utf-8") as f:
        f.write("window,CL,std\n")
        for w, cl, sd in cl_rows:
            f.write(f"{w},{cl:.8f},{sd:.8f}\n")
    print(f"[저장] {cl_path}")

    # -------------------- RL 학습 --------------------
    print(f"[작업 2/3] RL 정책 학습 시작 (Episodes={args.episodes})...")
    policy = PolicyCNN(d=scen.d, num_actions=len(actions))
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    rl_cfg = RLConfig(action_set=tuple(actions), device=device, episodes=args.episodes)

    rf_cfg_train = RFConfig(
        backend=args.rf_backend,
        n_estimators_full=args.rf_full_estimators,
        n_estimators_train=args.rf_train_estimators,
        feature_frac_train=max(0.05, min(1.0, args.rf_feature_frac)),
        kfold=args.kfold,
        cheap_mode=args.cheap_mode,
        n_jobs=-1,
    )
    policy = train_rl_policy(
        policy, optimizer, rl_cfg, scen, calib_map, S0_ref=S0_global,
        seed=args.seed, rf_cfg_train=rf_cfg_train, log_dir=args.out_dir
    )
    print("[작업 2/3] RL 정책 학습 완료.")

    # -------------------- ARL1 평가 --------------------
    lam_I = [0.5, 1.0, 1.5, 2.0]
    lam_II = [0.5, 1.0, 1.5, 2.0]

    print(f"[작업 3/3 - A] Scenario I ARL1 평가 시작 (R={args.R})...")
    rf_cfg_eval = RFConfig(
        backend=args.rf_backend,
        n_estimators_full=args.rf_full_estimators,
        n_estimators_train=args.rf_full_estimators,
        kfold=args.kfold,
        cheap_mode=False,
        n_jobs=-1,
    )
    mean_I, std_I = evaluate_arl1(
        scen, lam_I, 'I', policy, actions, calib_map, R=args.R, seed=args.seed + 7, rf_cfg_eval=rf_cfg_eval
    )
    print("[작업 3/3 - A] Scenario I 완료.")

    print(f"[작업 3/3 - B] Scenario II ARL1 평가 시작 (R={args.R})...")
    mean_II, std_II = evaluate_arl1(
        scen, lam_II, 'II', policy, actions, calib_map, R=args.R, seed=args.seed + 13, rf_cfg_eval=rf_cfg_eval
    )
    print("[작업 3/3 - B] Scenario II 완료.")

    # -------------------- 결과 저장 --------------------
    def save_arl1_csv(lams, means, stds, scen_tag):
        path = os.path.join(args.out_dir, f'arl1_scen{scen_tag}_actions-{"-".join(map(str,actions))}.csv')
        with open(path, "w", encoding="utf-8") as f:
            f.write("lambda,arl1_mean,arl1_std\n")
            for lam, m, s in zip(lams, means, stds):
                f.write(f"{lam},{m:.6f},{s:.6f}\n")
        print(f"[저장] {path}")
        return path

    pI = save_arl1_csv(lam_I, mean_I, std_I, "I")
    pII = save_arl1_csv(lam_II, mean_II, std_II, "II")

    # 요약 JSON (개선 2)
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "episodes": args.episodes,
        "action_set": actions,
        "n_boot": args.n_boot,
        "R": args.R,
        "rf_backend": args.rf_backend,
        "cheap_mode": args.cheap_mode,
        "rf_train_estimators": args.rf_train_estimators,
        "rf_full_estimators": args.rf_full_estimators,
        "rf_feature_frac": args.rf_feature_frac,
        "cl_table": cl_path,
        "rl_training_log": os.path.join(args.out_dir, "rl_training_log.csv"),
        "policy_snapshot": os.path.join(args.out_dir, "policy_snapshot.pt"),
        "arl1_csv": {"scenario_I": pI, "scenario_II": pII},
        "arl1": {
            "scenario_I": {"lambda": lam_I, "mean": mean_I, "std": std_I},
            "scenario_II": {"lambda": lam_II, "mean": mean_II, "std": std_II},
        },
    }
    sum_path = os.path.join(args.out_dir, "summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[저장] {sum_path}")

    print("="*60)
    print("시뮬레이션 종료")
    print("="*60)

if __name__ == '__main__':
    main()

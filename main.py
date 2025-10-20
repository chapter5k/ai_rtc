"""
Reinforcement Learning-Based Real-Time Contrasts Control Chart Using an Adaptive Window Size
논문 실험 재현(Python 단일 파일)

- 데이터 생성 (시나리오 I/II, d=10)
- RTC 모니터링 통계 p(S0, t) 계산 (RandomForest + OOB 확률)
- ARL0=200이 되도록 CL 산출(부트스트랩)
- RL-RTC (Policy Gradient, CNN 정책네트)로 윈도우 크기 {5,10,15} 또는 {3,10,17} 적응 선택
- ARL1 평가 및 표 생성(논문 Table 2–5 재현용)

참고: 이 스크립트는 재현성 높은 기본값으로 구성되어 있으나, 전체 반복(예: R=500)과
대규모 부트스트랩을 수행하면 시간이 많이 걸립니다. CLI 인자로 규모를 조절하세요.
"""
from __future__ import annotations
import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------- 유틸리티 -----------------------------

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sqrt_int(x: float) -> int:
    return max(1, int(round(math.sqrt(x))))


# ------------------------- 데이터 생성부 -----------------------------
@dataclass
class ScenarioConfig:
    d: int = 10                 # 변수(센서) 차원
    N0: int = 1500              # 참조데이터(Phase I) 크기
    T: int = 300                # Phase II 길이 (한 에피소드 길이)
    shift_time: int = 100       # t >= shift_time 부터 OOC
    sigma: float = 1.0          # 공분산은 I (단위분산) 가정


def make_cov(d: int) -> NDArray:
    return np.eye(d)


def gen_reference_data(cfg: ScenarioConfig, rng: np.random.RandomState) -> NDArray:
    return rng.multivariate_normal(mean=np.zeros(cfg.d), cov=make_cov(cfg.d), size=cfg.N0)


def apply_mean_shift(X: NDArray, delta: NDArray, t0: int) -> NDArray:
    """t >= t0 구간에 평균 시프트(delta) 적용"""
    Y = X.copy()
    if t0 < len(Y):
        Y[t0:] += delta
    return Y


def make_phase2_series(cfg: ScenarioConfig, rng: np.random.RandomState,
                       scenario: str, lam: float) -> Tuple[NDArray, NDArray]:
    """
    Returns X (T x d), labels_ic (T,) where labels_ic[t]=1 if in-control at time t else 0.
    scenario I: 한 변수만 시프트 (|delta| = sqrt(lam^2) -> lam 정의는 sqrt(k) 꼴을 직접 받음)
    scenario II: lambda^2 = sum(delta^T Sigma^{-1} delta) 에서 동일분산 가정하 1씩 k개 변수 shift.
    """
    X = rng.multivariate_normal(np.zeros(cfg.d), make_cov(cfg.d), size=cfg.T)
    labels_ic = np.ones(cfg.T, dtype=np.int64)
    # 시나리오별 delta 구성
    if scenario.upper() == 'I':
        # 한 변수만 shift, 크기 lam
        delta = np.zeros(cfg.d)
        j = 0  # 첫 번째 변수에 시프트
        delta[j] = lam
    else:
        # 여러 변수에 동일크기 분배: lam^2 = k*(a^2) => a = lam/sqrt(k)
        # k는 lam과 함께 실험에서 정해지지만 논문과 같은 셋업을 위해 k ∈ {1,2,3,5,7,9} 중 lam^2에 대응
        # 여기서는 근사적으로 k = round(lam^2) 사용
        k = max(1, int(round(lam**2)))
        k = min(k, cfg.d)
        a = lam / math.sqrt(k)
        delta = np.zeros(cfg.d)
        delta[:k] = a
    X = apply_mean_shift(X, delta, cfg.shift_time)
    labels_ic[cfg.shift_time:] = 0  # OOC
    return X, labels_ic


# ---------------------- RTC (RandomForest + OOB) ----------------------

def train_rf_with_oob(X: NDArray, y: NDArray, n_estimators: int, mtry: int, seed: int) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=mtry,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X, y)
    return rf


def rtc_monitoring_stat_pS0(rf: RandomForestClassifier, idx_S0: NDArray) -> float:
    """
    p(S0,t) = S0 집합에 대해 OOB 확률의 class=0 평균 (논문에서 S0 평균 사용)
    sklearn의 oob_decision_function_은 각 학습 샘플에 대해 OOB 확률 제공
    """
    oob = getattr(rf, 'oob_decision_function_', None)
    if oob is None:
        raise RuntimeError('RandomForest must be fit with oob_score=True')
    # class 0(=S0) 확률 평균
    p0 = oob[idx_S0, 0]  # (n_samples_of_S0,)
    return float(np.mean(p0))


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
) -> CLCalib:
    """
    ARL0 = 200 -> 단위시점 false alarm 확률 alpha = 1/200
    부트스트랩으로 p(S0,t) 분포를 추정하여 상위 (1-alpha) 분위수로 CL 설정
    """
    rng = check_random_state(seed)
    alpha = 1.0 / 200.0
    stats = []
    # S0 vs Sw(t): Sw는 S0에서 임의 연속구간 window 만큼 샘플링(정상상태 가정)
    N0 = len(S0)
    # 인덱스 캐시
    idx_all = np.arange(N0 + window)
    for b in range(n_boot):
        # 연속구간 추출 (순서 보전). 시점 선택
        if N0 - window <= 0:
            start = 0
        else:
            start = rng.randint(0, N0 - window)
        Sw = S0[start:start + window]
        # 학습 데이터 구성: S0 vs Sw
        X = np.vstack([S0, Sw])
        y = np.hstack([np.zeros(len(S0), dtype=int), np.ones(len(Sw), dtype=int)])
        rf = train_rf_with_oob(X, y, n_estimators=n_estimators, mtry=sqrt_int(d), seed=rng.randint(1_000_000))
        # S0 인덱스는 처음 N0
        pS0 = rtc_monitoring_stat_pS0(rf, np.arange(len(S0)))
        stats.append(pS0)
    stats = np.asarray(stats)
    CL = np.quantile(stats, 1 - alpha)
    std_boot = float(np.std(stats, ddof=1))
    return CLCalib(CL=CL, std_boot=std_boot)


# -------------------------- RL 정책 네트워크 --------------------------
class PolicyCNN(nn.Module):
    """
    입력: (B, 1, L=15, d)
    논문 Table 1 구성에 맞춰 Conv2d(1->16)->Conv2d(16->8) 후 Flatten=288(=9*4*8) -> FC(288) -> Output
    d는 논문 기준 10으로 고정.
    """
    def __init__(self, d: int, num_actions: int):
        super().__init__()
        if d != 10:
            raise ValueError("이 CNN 구조는 논문 기준 d=10에 맞춰져 있습니다.")
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(4, 4))
        self.act1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(16, 8, kernel_size=(4, 4))  # 논문 Table 1: 8 filters
        self.act2 = nn.Sigmoid()
        H, W = 9, 4  # (15x10) -> (12x7) -> (9x4)
        flatten_size = 8 * H * W  # 288
        self.fc1 = nn.Linear(flatten_size, 288)
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
    action_set: Tuple[int, int, int] = (5, 10, 15)  # 또는 (3,10,17)
    lr: float = 1e-3
    gamma: float = 0.99
    episodes: int = 50
    device: str = 'cpu'
    action_set: Tuple[int, int, int] = (5, 10, 15)  # 또는 (3,10,17)
    lr: float = 1e-3
    gamma: float = 0.99
    episodes: int = 50
    device: str = 'cpu'


# ------------------------ 상태 구성 & 보상 계산 ------------------------

def make_state_tensor(windowed: NDArray, d: int, L: int = 15) -> torch.Tensor:
    """windowed: (w, d) -> (1,1,L,d) 제로패딩/크롭"""
    w = windowed.shape[0]
    if w >= L:
        data = windowed[-L:]
    else:
        pad = np.zeros((L - w, d), dtype=windowed.dtype)
        data = np.vstack([pad, windowed])
    t = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,L,d)
    return t


@dataclass
class WindowCalib:
    CL: float
    std: float
    size: int


def reward_by_algorithm1(
    action_idx: int,
    distances: List[float],
    in_control: bool,
) -> float:
    # distances = [D1, D2, D3], 큰 값이 유리(IC) / 불리(OOC)
    order = np.argsort(distances)[::-1]  # 내림차순 정렬 인덱스
    rank = int(np.where(order == action_idx)[0][0])  # 0,1,2
    if in_control:
        return {0: 1.0, 1: -1.0, 2: -2.0}[rank]
    else:
        return {0: -2.0, 1: -1.0, 2: 1.0}[rank]


# ----------------------------- 학습 루프 ------------------------------

def train_rl_policy(
    policy: PolicyCNN,
    optimizer: optim.Optimizer,
    cfg: RLConfig,
    scen: ScenarioConfig,
    calib_map: Dict[int, WindowCalib],
    S0_ref: NDArray,
    seed: int,
) -> PolicyCNN:
    rng = check_random_state(seed)
    policy.train()
    device = cfg.device
    policy.to(device)
    actions = list(cfg.action_set)

    for ep in range(cfg.episodes):
        # 한 에피소드: 무작위 lambda 선택 (소형 학습), 시나리오 I과 II 섞기
        lam_choices_I = [math.sqrt(x) for x in [0.25, 0.5, 1, 2, 3, 5, 7, 9]]
        lam_choices_II = [math.sqrt(x) for x in [2, 3, 5, 7, 9]]
        if rng.rand() < 0.5:
            scenario = 'I'
            lam = float(rng.choice(lam_choices_I))
        else:
            scenario = 'II'
            lam = float(rng.choice(lam_choices_II))
        X, labels_ic = make_phase2_series(scen, rng, scenario, lam)
        # 누적 손실 (REINFORCE)
        logps: List[torch.Tensor] = []
        rewards: List[float] = []
        # 시점 순회
        for t in range(1, scen.T + 1):
            # 모든 후보 윈도우에 대해 모니터링 통계 계산
            ms_list = []
            D_list = []
            for i, w in enumerate(actions):
                w_eff = min(w, t)
                Sw = X[t - w_eff:t]
                # 고정된 Phase-I 참조 S0_ref 사용
                Xrf = np.vstack([S0_ref, Sw])
                yrf = np.hstack([np.zeros(len(S0_ref), dtype=int), np.ones(len(Sw), dtype=int)])
                rf = train_rf_with_oob(Xrf, yrf, n_estimators=200, mtry=sqrt_int(scen.d), seed=rng.randint(1_000_000))
                pS0 = rtc_monitoring_stat_pS0(rf, np.arange(len(S0_ref)))
                ms_list.append(pS0)
                calib = calib_map[w]
                D = (calib.CL - pS0) / max(1e-8, calib.std)
                D_list.append(float(D))
            # 상태 텐서 구성 (가장 큰 후보 L=15 기준)
            Lmax = 15
            w_for_state = min(max(actions), t)
            state = make_state_tensor(X[t - w_for_state:t], scen.d, L=Lmax).to(device)
            logits = policy(state)
            probs = torch.softmax(logits, dim=-1)
            m = torch.distributions.Categorical(probs=probs)
            a_idx = int(m.sample().item())
            logps.append(m.log_prob(torch.tensor(a_idx, device=device)))
            # 보상 계산
            ic = bool(labels_ic[t-1] == 1)
            r = reward_by_algorithm1(a_idx, D_list, in_control=ic)
            rewards.append(float(r))
        # REINFORCE 손실 계산 (reward-to-go)
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + cfg.gamma * G
            returns.append(G)
        returns = list(reversed(returns))
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        # 정규화(분산 감소)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        loss = 0.0
        for logp, Gt in zip(logps, returns_t):
            loss = loss - logp * Gt
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
    return policy


# ----------------------------- 평가 (ARL1) ----------------------------

def run_length_until_alarm(
    X: NDArray,
    S0_ref: NDArray,
    policy: PolicyCNN,
    actions: List[int],
    calib_map: Dict[int, WindowCalib],
    d: int,
) -> int:
    """첫 신호 발생 시까지 관측 수 (없으면 T 반환)"""
    policy.eval()
    device = next(policy.parameters()).device
    T = len(X)
    for t in range(1, T + 1):
        # 상태에서 정책으로 액션 선택
        Lmax = 15
        w_for_state = min(max(actions), t)
        state = make_state_tensor(X[t - w_for_state:t], d, L=Lmax).to(device)
        with torch.no_grad():
            logits = policy(state)
            a_idx = int(torch.argmax(logits, dim=-1).item())  # 평가 시에는 greedy
        w = actions[a_idx]
        w_eff = min(w, t)
        Sw = X[t - w_eff:t]
        Xrf = np.vstack([S0_ref, Sw])
        yrf = np.hstack([np.zeros(len(S0_ref), dtype=int), np.ones(len(Sw), dtype=int)])
        rf = RandomForestClassifier(n_estimators=300, max_features=sqrt_int(d), bootstrap=True, oob_score=True, n_jobs=-1, random_state=42)
        rf.fit(Xrf, yrf)
        pS0 = rtc_monitoring_stat_pS0(rf, np.arange(len(S0_ref)))
        calib = calib_map[w]
        if pS0 > calib.CL:
            return t
    return T


def evaluate_arl1(
    scen_cfg: ScenarioConfig,
    lam_list: List[float],
    scenario: str,
    policy: PolicyCNN,
    actions: List[int],
    calib_map: Dict[int, WindowCalib],
    R: int,
    seed: int,
) -> Tuple[List[float], List[float]]:
    rng = check_random_state(seed)
    arl_means = []
    arl_stds = []
    # 참조 S0는 고정 (논문은 N0=1500 별도 Phase I)
    S0 = gen_reference_data(scen_cfg, rng)
    # ARL1: OOC가 즉시 시작하도록 shift_time=0으로 평가 설정
    from dataclasses import replace as _replace
    scen_cfg_eval = _replace(scen_cfg, shift_time=0)
    for lam in lam_list:
        RLs = []
        for k in range(R):
            X, labels_ic = make_phase2_series(scen_cfg_eval, rng, scenario, lam)
            rl = run_length_until_alarm(X, S0, policy, actions, calib_map, scen_cfg.d)
            RLs.append(rl)
        arl_means.append(float(np.mean(RLs)))
        arl_stds.append(float(np.std(RLs, ddof=1)))
    return arl_means, arl_stds


# ------------------------------- 메인 -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--episodes', type=int, default=30, help='RL 학습 에피소드 수 (권장 100~300 이상)')
    parser.add_argument('--n_boot', type=int, default=800, help='CL 부트스트랩 반복 (권장 2000~10000)')
    parser.add_argument('--action_set', type=str, default='5,10,15', help='예: 5,10,15 또는 3,10,17')
    parser.add_argument('--R', type=int, default=100, help='ARL1 반복 (논문 500)')
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device
    action_set = tuple(int(x) for x in args.action_set.split(','))

    # 설정
    scen = ScenarioConfig()
    rl_cfg = RLConfig(action_set=action_set, device=device, episodes=args.episodes)

    # Phase I 참조데이터 고정 및 CL 보정 (각 윈도우에 대해)
    rng = check_random_state(args.seed)
    S0_ref = gen_reference_data(scen, rng)

    calib_map: Dict[int, WindowCalib] = {}
    for w in action_set:
        calib = estimate_CL_for_window(S0_ref, scen.d, window=w, n_boot=args.n_boot, n_estimators=300, seed=rng.randint(1_000_000))
        calib_map[w] = WindowCalib(CL=calib.CL, std=calib.std_boot, size=w)
        print(f"[CL] window={w:2d} -> CL={calib.CL:.5f}, std={calib.std_boot:.5f}")

    # RL 정책 초기화 & 학습
    policy = PolicyCNN(d=scen.d, num_actions=3)
    optimizer = optim.Adam(policy.parameters(), lr=rl_cfg.lr)
    policy = train_rl_policy(policy, optimizer, rl_cfg, scen, calib_map, S0_ref, seed=args.seed)

    actions = list(action_set)

    # 평가 세트 (논문 값과 동일한 루트값 리스트)
    lam_I = [math.sqrt(x) for x in [0.25, 0.5, 1, 2, 3, 5, 7, 9]]
    lam_II = [math.sqrt(x) for x in [2, 3, 5, 7, 9]]

    # 시나리오 I 평가
    mean_I, std_I = evaluate_arl1(scen, lam_I, 'I', policy, actions, calib_map, R=args.R, seed=args.seed + 7)
    # 시나리오 II 평가
    mean_II, std_II = evaluate_arl1(scen, lam_II, 'II', policy, actions, calib_map, R=args.R, seed=args.seed + 13)

    # 결과 출력 (테이블 포맷 유사)
    def fmt_table(lams, means, stds, title):
        print("\n" + title)
        print("lambda\tARL1 (mean [std])")
        for lam, m, s in zip(lams, means, stds):
            print(f"{lam:.5f}\t{m:.2f} [{s:.2f}]")

    fmt_table(lam_I, mean_I, std_I, f"Scenario I (action_set={action_set})")
    fmt_table(lam_II, mean_II, std_II, f"Scenario II (action_set={action_set})")

    # CSV 저장
    out_dir = os.path.abspath('./outputs')
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, f'arl1_scenI_{"-".join(map(str,action_set))}.csv'),
               np.c_[lam_I, mean_I, std_I], delimiter=',', header='lambda,arl1_mean,arl1_std', comments='')
    np.savetxt(os.path.join(out_dir, f'arl1_scenII_{"-".join(map(str,action_set))}.csv'),
               np.c_[lam_II, mean_II, std_II], delimiter=',', header='lambda,arl1_mean,arl1_std', comments='')
    print(f"\n[저장 완료] outputs/arl1_scenI_*.csv, outputs/arl1_scenII_*.csv")


if __name__ == '__main__':
    main()

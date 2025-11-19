# ai_rtc/rl_pg.py
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from utils import _np2d, _np1d
from data_gen import ScenarioConfig, make_phase2_series
from cl_calib import WindowCalib
from rtc_backend import compute_pS0_stat
from reward_morl import MorlConfig, MorlStats, MorlEpisodeState, calc_reward_morl, calc_reward_alg1_wrapper



@dataclass
class RLConfig:
    action_set: Tuple[int, int, int] = (5, 10, 15)
    lr: float = 1e-3
    gamma: float = 0.99
    episodes: int = 50
    device: str = 'cpu'
    reward: str = "alg1"
    

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



def every(n, i):  # i: 0-based index
    return (i + 1) % n == 0

def train_rl_policy(
    policy: nn.Module,
    optimizer: optim.Optimizer,
    cfg: RLConfig,
    scen: ScenarioConfig,
    calib_map: Dict[int, WindowCalib],
    S0_ref: NDArray,
    seed: int,
    rf_backend: str = 'sklearn',
    n_estimators_eval: int = 150,
) -> nn.Module:
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

    morl_cfg = MorlConfig()
    morl_stats = MorlStats()

    pbar_ep = tqdm(range(cfg.episodes), desc="[RL] Training", dynamic_ncols=True)
    for ep in pbar_ep:

        morl_ep_state = MorlEpisodeState()        
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
                Xrf = _np2d(Xrf, dtype=np.float32)
                yrf = _np1d(yrf, dtype=np.int32)
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
            # --- 보상 계산 ---
            ic = bool(labels_ic[t-1] == 1)

            if cfg.reward == "alg1":
                r_t = calc_reward_alg1_wrapper(
                    action_global_idx=a_idx,
                    valid_indices=valid_indices,
                    D_list=D_list,
                    in_control=ic,
                )
                rewards.append(float(r_t))
            else:  # cfg.reward == "morl"
                # 선택된 행동의 D 값(selected_D) 추출
                if a_idx not in valid_indices:
                    # 이론상 거의 없음. 방어적 처리.
                    selected_D = 0.0
                else:
                    local_idx = valid_indices.index(a_idx)
                    selected_D = float(D_list[local_idx])

                r_t, comps = calc_reward_morl(
                    action_global_idx=a_idx,
                    valid_indices=valid_indices,
                    D_list=D_list,
                    in_control=ic,
                    t=t,
                    shift_time=scen.shift_time,
                    selected_D=selected_D,
                    episode_state=morl_ep_state,
                    stats=morl_stats,
                    cfg=morl_cfg,
                )
                # comps(dict)는 원하면 디버그/로그에 활용 가능
                rewards.append(float(r_t))


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


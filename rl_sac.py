"""
rl_sac.py

3단계: Policy Gradient → Discrete SAC (행동 = 윈도우 크기 {5, 10, 15} 유지)

- 환경/보상/상태 구성은 rl_pg.py 와 동일하게 사용
- 정책 네트워크는 policy_nets.py 의 CNN / CNN-LSTM 그대로 사용
- 이 모듈은 "알고리즘만" 교체 (off-policy SAC + 리플레이 버퍼 + 2x 크리틱 + α 자동 튜닝)

필수 의존 모듈:
  - data_gen.ScenarioConfig, make_phase2_series
  - cl_calib.WindowCalib
  - rtc_backend.compute_pS0_stat
  - rl_pg.make_state_tensor, rl_pg.reward_by_algorithm1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# --- 프로젝트 내 모듈 import (이름만 맞게 조정해주면 됨) ---
from data_gen import ScenarioConfig, make_phase2_series
from cl_calib import WindowCalib
from rtc_backend import compute_pS0_stat
from rl_pg import make_state_tensor, reward_by_algorithm1


# ---------------------------------------------------------------------------
# 1. SAC 전용 설정값
# ---------------------------------------------------------------------------

@dataclass
class SACConfig:
    """
    SAC 이산 버전(Discrete SAC)용 설정값.

    - action_set     : 행동으로 사용할 윈도우 크기 튜플 (예: (5, 10, 15))
    - episodes       : 학습 에피소드 수
    - gamma          : 할인율
    - device         : 'cuda' or 'cpu'
    - actor_lr       : 정책 네트워크 learning rate
    - critic_lr      : Q-네트워크 learning rate
    - alpha_lr       : temperature(α) learning rate
    - buffer_size    : 리플레이 버퍼 최대 크기
    - batch_size     : 미니배치 크기
    - tau            : 타깃 네트워크 soft update 계수
    - initial_random_steps : 이 step 수 전까지는 정책 대신 랜덤 행동 수행
    - updates_per_step     : 환경에서 1 step 생성할 때마다 몇 번 업데이트할지
    - target_entropy       : 엔트로피 타깃 (기본값 None이면 -log(#actions))
    """
    action_set: Tuple[int, int, int] = (5, 10, 15)
    episodes: int = 50
    gamma: float = 0.99
    device: str = "cpu"

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    buffer_size: int = 100_000
    batch_size: int = 64
    tau: float = 0.005

    initial_random_steps: int = 1_000
    updates_per_step: int = 1

    target_entropy: float | None = None  # None이면 -log(num_actions) 로 설정


# ---------------------------------------------------------------------------
# 2. 리플레이 버퍼
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    (s, a, r, s', done, valid_mask, next_valid_mask)를 저장하는 간단한 버퍼.
    - state, next_state : numpy float32 배열 (state 텐서를 .cpu().numpy()로 저장)
    - action            : int (행동 index)
    - reward            : float
    - done              : bool
    - valid_mask        : shape (num_actions,) bool
    - next_valid_mask   : shape (num_actions,) bool
    """

    def __init__(self, capacity: int, num_actions: int):
        self.capacity = capacity
        self.num_actions = num_actions

        self.states: List[np.ndarray] = []
        self.next_states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.valid_masks: List[np.ndarray] = []
        self.next_valid_masks: List[np.ndarray] = []

        self._idx = 0
        self._full = False

    def __len__(self) -> int:
        return len(self.states)

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        valid_mask: np.ndarray,
        next_valid_mask: np.ndarray,
    ) -> None:
        # state, next_state: (1, 1, L, d) 텐서라고 가정
        s = state.detach().cpu().numpy()
        s_next = next_state.detach().cpu().numpy()

        if len(self.states) < self.capacity and not self._full:
            self.states.append(s)
            self.next_states.append(s_next)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
            self.valid_masks.append(valid_mask.astype(bool))
            self.next_valid_masks.append(next_valid_mask.astype(bool))
        else:
            self.states[self._idx] = s
            self.next_states[self._idx] = s_next
            self.actions[self._idx] = action
            self.rewards[self._idx] = reward
            self.dones[self._idx] = done
            self.valid_masks[self._idx] = valid_mask.astype(bool)
            self.next_valid_masks[self._idx] = next_valid_mask.astype(bool)

            self._idx = (self._idx + 1) % self.capacity
            if self._idx == 0:
                self._full = True

    def sample(self, batch_size: int, device: str):
        idxs = np.random.randint(0, len(self.states), size=batch_size)

        states = torch.from_numpy(np.concatenate([self.states[i] for i in idxs], axis=0)).to(device)
        next_states = torch.from_numpy(np.concatenate([self.next_states[i] for i in idxs], axis=0)).to(device)
        actions = torch.tensor([self.actions[i] for i in idxs], dtype=torch.long, device=device)
        rewards = torch.tensor([self.rewards[i] for i in idxs], dtype=torch.float32, device=device)
        dones = torch.tensor([self.dones[i] for i in idxs], dtype=torch.float32, device=device)
        valid_masks = torch.from_numpy(np.stack([self.valid_masks[i] for i in idxs])).to(device)
        next_valid_masks = torch.from_numpy(np.stack([self.next_valid_masks[i] for i in idxs])).to(device)

        return states, actions, rewards, next_states, dones, valid_masks, next_valid_masks


# ---------------------------------------------------------------------------
# 3. Q-네트워크(크리틱) 정의
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    간단한 MLP 기반 Q 네트워크.
    - 입력: state 텐서 (B, 1, L, d)
    - 출력: Q(s, a) 벡터 (B, num_actions)

    정책 CNN/CNN-LSTM 과는 별도로, 여기서는 상태를 단순히 flatten 해서 사용.
    """

    def __init__(self, num_actions: int, L: int = 15, d: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.num_actions = num_actions
        self.L = L
        self.d = d

        in_dim = L * d   # (1, 1, L, d) -> flatten
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L, d)
        if x.dim() != 4:
            raise ValueError(f"QNetwork expects (B,1,L,d) but got {x.shape}")
        b, c, L, d = x.shape
        x = x.view(b, -1)   # (B, L*d)
        return self.net(x)


# ---------------------------------------------------------------------------
# 4. SAC 학습 루프
# ---------------------------------------------------------------------------

@torch.no_grad()
def _select_action_with_mask(
    policy: nn.Module,
    state: torch.Tensor,
    valid_mask: np.ndarray,
    device: str,
    rng: np.random.RandomState,
    step: int,
    cfg: SACConfig,
) -> Tuple[int, torch.Tensor]:
    """
    - policy(state) → logits
    - valid_mask=False 인 행동은 -1e9로 마스킹 후 softmax
    - cfg.initial_random_steps 이전에는 랜덤(유효 행동에서만) 선택
    - 반환값: (action_index, probs_tensor)   # probs_tensor: (1, num_actions)
    """
    num_actions = len(valid_mask)
    valid_mask_t = torch.from_numpy(valid_mask.astype(bool)).to(device=device)

    logits = policy(state.to(device))  # (1, num_actions)
    logits = logits.clone()
    logits[~valid_mask_t] = -1e9

    if step < cfg.initial_random_steps:
        # 유효한 행동 중 랜덤
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            # 이론상 여기 오지 않도록 학습 루프에서 처리
            a_idx = int(rng.randint(0, num_actions))
        else:
            a_idx = int(rng.choice(valid_indices))
        probs = torch.softmax(logits, dim=-1)
        return a_idx, probs

    probs = torch.softmax(logits, dim=-1)
    m = Categorical(probs=probs)
    a_idx_tensor = m.sample()
    a_idx = int(a_idx_tensor.item())
    return a_idx, probs


def train_sac_policy(
    policy: nn.Module,
    sac_cfg: SACConfig,
    scen: ScenarioConfig,
    calib_map: Dict[int, WindowCalib],
    S0_ref: NDArray,
    seed: int,
    rf_backend: str = "sklearn",
    n_estimators_eval: int = 150,
) -> nn.Module:
    """
    Discrete SAC 으로 정책 네트워크를 학습.

    인자:
      - policy      : policy_nets.build_policy() 로 만든 CNN 또는 CNN-LSTM
      - sac_cfg     : SACConfig
      - scen        : ScenarioConfig (data_gen.py)
      - calib_map   : {window_size: WindowCalib(CL, std, ...)}
      - S0_ref      : Phase I 참조 데이터 (정상군)
      - seed        : random seed
      - rf_backend  : 'sklearn' / 'lgbm' 등 RTC 분류기 backend
    반환:
      - 학습된 policy (in-place update)
    """
    rng = check_random_state(seed)
    device = sac_cfg.device

    policy.to(device)
    policy.train()

    num_actions = len(sac_cfg.action_set)
    Lmax = 15
    d = scen.d

    # --- Q 네트워크 & 타깃 네트워크 ---
    q1 = QNetwork(num_actions=num_actions, L=Lmax, d=d).to(device)
    q2 = QNetwork(num_actions=num_actions, L=Lmax, d=d).to(device)
    q1_target = QNetwork(num_actions=num_actions, L=Lmax, d=d).to(device)
    q2_target = QNetwork(num_actions=num_actions, L=Lmax, d=d).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    # --- Optimizers ---
    actor_optimizer = optim.Adam(policy.parameters(), lr=sac_cfg.actor_lr)
    critic_optimizer = optim.Adam(
        list(q1.parameters()) + list(q2.parameters()),
        lr=sac_cfg.critic_lr,
    )

    # --- Temperature α (엔트로피 계수) ---
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=sac_cfg.alpha_lr)
    if sac_cfg.target_entropy is None:
        target_entropy = -float(np.log(num_actions))
    else:
        target_entropy = sac_cfg.target_entropy

    # --- Replay Buffer ---
    buffer = ReplayBuffer(capacity=sac_cfg.buffer_size, num_actions=num_actions)

    global_step = 0

    from tqdm import tqdm
    pbar_ep = tqdm(range(sac_cfg.episodes), desc="[RL-SAC] Training", dynamic_ncols=True)

    actions = list(sac_cfg.action_set)

    for ep in pbar_ep:
        # --- 시나리오/쉬프트 크기 샘플링 (PG 코드와 동일 패턴) ---
        lam_choices_I = [np.sqrt(x) for x in [0.25, 0.5, 1, 2, 3, 5, 7, 9]]
        lam_choices_II = [np.sqrt(x) for x in [2, 3, 5, 7, 9]]

        if rng.rand() < 0.5:
            scenario = "I"
            lam = float(rng.choice(lam_choices_I))
        else:
            scenario = "II"
            lam = float(rng.choice(lam_choices_II))

        # Phase II 시계열 생성
        X, labels_ic = make_phase2_series(scen, rng, scenario, lam)  # X: (T, d)

        ep_rewards: List[float] = []

        for t in range(1, scen.T + 1):
            global_step += 1

            # --- 유효 행동 마스크(t>=w) ---
            valid = np.array([t >= w for w in actions], dtype=bool)
            if not valid.any():
                # 아직 어떤 창도 꽉 차지 못한 시점이면 상태만 policy에 넣고 통과
                w_for_state = min(max(actions), t)
                state = make_state_tensor(X[t - w_for_state:t], d, L=Lmax).to(device)
                _ = policy(state)
                continue

            # --- 상태 구성 ---
            w_for_state = min(max(actions), t)
            state = make_state_tensor(X[t - w_for_state:t], d, L=Lmax).to(device)

            # --- 유효 행동들에 대해 통계/거리 계산 ---
            ms_list: List[float] = []
            D_list: List[float] = []
            valid_indices: List[int] = []

            for a_idx_tmp, w in enumerate(actions):
                if not valid[a_idx_tmp]:
                    continue
                Sw = X[t - w:t]
                Xrf = np.vstack([S0_ref, Sw])
                yrf = np.hstack([
                    np.zeros(len(S0_ref), dtype=int),
                    np.ones(len(Sw), dtype=int),
                ])

                pS0 = compute_pS0_stat(
                    Xrf,
                    yrf,
                    np.arange(len(S0_ref)),
                    d=scen.d,
                    n_estimators=n_estimators_eval,
                    seed=rng.randint(1_000_000),
                    backend=rf_backend,
                )
                ms_list.append(pS0)
                calib = calib_map[w]
                D = (calib.CL - pS0) / max(1e-8, calib.std)
                D_list.append(float(D))
                valid_indices.append(a_idx_tmp)

            # --- 행동 선택 (SAC 정책 or 랜덤) ---
            a_idx, _ = _select_action_with_mask(
                policy=policy,
                state=state,
                valid_mask=valid,
                device=device,
                rng=rng,
                step=global_step,
                cfg=sac_cfg,
            )

            # --- 보상 계산 (기존 reward_by_algorithm1 재사용) ---
            if a_idx not in valid_indices:
                reward = -2.0  # 방어적 패널티(이론상 거의 발생 X)
            else:
                local_idx = valid_indices.index(a_idx)
                ic = bool(labels_ic[t - 1] == 1)
                reward = float(reward_by_algorithm1(local_idx, D_list, in_control=ic))
            ep_rewards.append(reward)

            # --- 다음 상태 구성 ---
            done = (t == scen.T)
            t_next = min(t + 1, scen.T)
            w_for_next_state = min(max(actions), t_next)
            next_state = make_state_tensor(X[t_next - w_for_next_state:t_next], d, L=Lmax).to(device)

            valid_next = np.array([t_next >= w for w in actions], dtype=bool)

            # --- 버퍼에 transition 추가 ---
            buffer.add(
                state=state,
                action=a_idx,
                reward=reward,
                next_state=next_state,
                done=done,
                valid_mask=valid,
                next_valid_mask=valid_next,
            )

            # --- enough data: SAC 업데이트 ---
            if len(buffer) >= sac_cfg.batch_size:
                for _ in range(sac_cfg.updates_per_step):
                    (
                        states_b,
                        actions_b,
                        rewards_b,
                        next_states_b,
                        dones_b,
                        valid_masks_b,
                        next_valid_masks_b,
                    ) = buffer.sample(sac_cfg.batch_size, device=device)

                    # -----------------------------
                    # 1) Critic 업데이트
                    # -----------------------------
                    with torch.no_grad():
                        # next-state 에서 정책 분포 계산 + 마스킹
                        next_logits = policy(next_states_b)
                        next_logits = next_logits.clone()
                        next_logits[~next_valid_masks_b] = -1e9

                        next_log_probs = torch.log_softmax(next_logits, dim=-1)
                        next_probs = torch.softmax(next_logits, dim=-1)

                        q1_next = q1_target(next_states_b)
                        q2_next = q2_target(next_states_b)
                        min_q_next = torch.min(q1_next, q2_next)

                        alpha = log_alpha.exp()
                        # V(s') = E_a[ Q(s',a) - α log π(a|s') ]
                        next_v = (next_probs * (min_q_next - alpha * next_log_probs)).sum(dim=-1)

                        target_q = rewards_b + sac_cfg.gamma * (1.0 - dones_b) * next_v

                    q1_values = q1(states_b).gather(1, actions_b.unsqueeze(-1)).squeeze(-1)
                    q2_values = q2(states_b).gather(1, actions_b.unsqueeze(-1)).squeeze(-1)

                    critic_loss = ((q1_values - target_q) ** 2 + (q2_values - target_q) ** 2).mean()

                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 1.0)
                    critic_optimizer.step()

                    # -----------------------------
                    # 2) Actor(정책) 업데이트
                    # -----------------------------
                    logits = policy(states_b)
                    logits = logits.clone()
                    logits[~valid_masks_b] = -1e9

                    log_probs = torch.log_softmax(logits, dim=-1)
                    probs = torch.softmax(logits, dim=-1)

                    q1_pi = q1(states_b)
                    q2_pi = q2(states_b)
                    min_q_pi = torch.min(q1_pi, q2_pi)

                    alpha = log_alpha.exp()
                    actor_loss = (probs * (alpha * log_probs - min_q_pi)).sum(dim=-1).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    actor_optimizer.step()

                    # -----------------------------
                    # 3) α 업데이트 (엔트로피 타깃)
                    # -----------------------------
                    entropy = -(probs * log_probs).sum(dim=-1).mean()
                    alpha_loss = -(log_alpha * (target_entropy - entropy).detach()).mean()

                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()

                    # -----------------------------
                    # 4) 타깃 네트워크 soft update
                    # -----------------------------
                    with torch.no_grad():
                        for param, target_param in zip(q1.parameters(), q1_target.parameters()):
                            target_param.data.mul_(1.0 - sac_cfg.tau)
                            target_param.data.add_(sac_cfg.tau * param.data)

                        for param, target_param in zip(q2.parameters(), q2_target.parameters()):
                            target_param.data.mul_(1.0 - sac_cfg.tau)
                            target_param.data.add_(sac_cfg.tau * param.data)

        # --- 에피소드 로그 ---
        if len(ep_rewards):
            avg_r = float(np.mean(ep_rewards))
        else:
            avg_r = 0.0
        pbar_ep.set_postfix_str(f"avg_r={avg_r:.3f}, buffer={len(buffer)}")

    return policy

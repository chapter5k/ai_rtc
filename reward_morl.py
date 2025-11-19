# ai_rtc/reward_morl.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np

# 기존 Algorithm 1 보상과 호환되도록 import
from rl_pg import reward_by_algorithm1


@dataclass
class RunningStat:
    """러닝 Z-정규화를 위한 1D 통계 추적기."""
    mean: float = 0.0
    var: float = 1.0
    count: int = 0

    def update(self, x: float):
        self.count += 1
        if self.count == 1:
            self.mean = x
            self.var = 1.0
            return
        # Welford
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += delta * (x - self.mean)

    def z(self, x: float) -> float:
        denom = max(1e-6, float(np.sqrt(max(self.var, 1e-6))))
        return float((x - self.mean) / denom)


@dataclass
class MorlStats:
    """각 보상 컴포넌트별 런닝 통계."""
    sens: RunningStat = field(default_factory=RunningStat)
    stab: RunningStat = field(default_factory=RunningStat)
    shape: RunningStat = field(default_factory=RunningStat)


@dataclass
class MorlConfig:
    """
    MORL 스칼라 보상 설정.

    w_sens  : 탐지 민감도(ARL1 감소) 쪽 가중치
    w_stab  : 안정성(ARL0 유지, false alarm 회피) 가중치(보통 음수 또는 -값에 곱)
    w_shape : Algorithm1 기반 shaping 가중치
    c_detect: 탐지 성공 시 기본 보상
    k_delay : 탐지 지연 패널티 계수
    """
    w_sens: float = 1.0
    w_stab: float = 1.0
    w_shape: float = 0.3
    c_detect: float = 1.0
    k_delay: float = 0.01


@dataclass
class MorlEpisodeState:
    """에피소드 단위로 유지하는 MORL 내부 상태."""
    detection_step: int | None = None   # 첫 '탐지' 시점 (CL을 넘은 첫 t)
    false_alarm: bool = False           # shift_time 이전에 탐지 발생 여부


def calc_reward_alg1_wrapper(
    action_global_idx: int,
    valid_indices: List[int],
    D_list: List[float],
    in_control: bool,
) -> float:
    """기존 Algorithm 1과 완전히 동일한 보상 계산 (fallback용)."""
    if action_global_idx not in valid_indices:
        return -2.0  # 방어적 패널티
    local_idx = valid_indices.index(action_global_idx)
    return float(reward_by_algorithm1(local_idx, D_list, in_control=in_control))


def calc_reward_morl(
    *,
    action_global_idx: int,
    valid_indices: List[int],
    D_list: List[float],
    in_control: bool,
    t: int,
    shift_time: int,
    selected_D: float,
    episode_state: MorlEpisodeState,
    stats: MorlStats,
    cfg: MorlConfig,
) -> Tuple[float, Dict[str, float]]:
    """
    MORL 스칼라 보상 계산.

    Parameters
    ----------
    action_global_idx : 전체 action_set에서의 인덱스 (0,1,2)
    valid_indices     : 현재 시점에서 유효한 행동들의 global index 리스트
    D_list            : 유효 행동별 D=(CL - pS0)/std 값 리스트 (valid_indices 순서)
    in_control        : 현재 시점이 IC(True)인지 OOC(False)인지
    t                 : 현재 시간(step)
    shift_time        : OOC 시작 시점 (ScenarioConfig.shift_time)
    selected_D        : 에이전트가 실제로 선택한 행동의 D 값
    episode_state     : 에피소드 전역 MORL 상태 (탐지/false alarm 기록)
    stats             : 각 보상 컴포넌트별 러닝 통계
    cfg               : MorlConfig (가중치 등)
    """

    # ---------------------------
    # 1) 기본 shaping: Algorithm1
    # ---------------------------
    r_shape = calc_reward_alg1_wrapper(
        action_global_idx=action_global_idx,
        valid_indices=valid_indices,
        D_list=D_list,
        in_control=in_control,
    )

    # ---------------------------
    # 2) 안정성 측면 (IC 구간)
    # ---------------------------
    # D = (CL - pS0) / std 이므로, D가 작을수록 CL에 근접/초과 (알람 위험 ↑)
    # IC 구간에서 D_selected가 0 근처/음수이면 false alarm 위험 → penalty
    r_stab = 0.0
    if in_control:
        # soft penalty: D_selected가 1 아래로 내려오면 점점 더 큰 penalty
        margin = max(0.0, 1.0 - selected_D)  # selected_D <= 1 -> 양수
        r_stab = -margin   # 안정성 기준으로는 마진이 클수록 나쁨(음수 보상)

    # ---------------------------
    # 3) 탐지 민감도 (OOC 구간)
    # ---------------------------
    r_sens = 0.0
    if (not in_control) and (not episode_state.false_alarm):
        # OOC 구간에서, D_selected가 0 이하(이미 CL을 넘었다고 가정)면 "탐지"로 처리
        if selected_D <= 0.0 and episode_state.detection_step is None:
            episode_state.detection_step = t
            # detection_delay = t - shift_time (0 이상)
            detection_delay = max(0, t - shift_time)
            r_sens = cfg.c_detect - cfg.k_delay * float(detection_delay)
        else:
            r_sens = 0.0

    # ---------------------------
    # 4) False Alarm 체크
    # ---------------------------
    # shift_time 이전에 D_selected <= 0 이면 False Alarm 으로 간주
    if in_control and (t < shift_time) and (selected_D <= 0.0):
        episode_state.false_alarm = True

    # ---------------------------
    # 5) 러닝 Z-정규화
    # ---------------------------
    stats.sens.update(r_sens)
    stats.stab.update(r_stab)
    stats.shape.update(r_shape)

    z_sens = stats.sens.z(r_sens)
    z_stab = stats.stab.z(r_stab)
    z_shape = stats.shape.z(r_shape)

    # ---------------------------
    # 6) 최종 스칼라 보상 (가중합)
    # ---------------------------
    reward = (
        cfg.w_sens * z_sens +
        cfg.w_stab * z_stab +
        cfg.w_shape * z_shape
    )

    components = {
        "Rsensitivity_raw": r_sens,
        "Pstability_raw": r_stab,
        "Rshape_raw": r_shape,
        "Rsensitivity_z": z_sens,
        "Pstability_z": z_stab,
        "Rshape_z": z_shape,
    }
    return float(reward), components

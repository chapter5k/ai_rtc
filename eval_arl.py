# ai_rtc/eval_arl.py
# eval_arl.py
from dataclasses import replace as _replace
from typing import List, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state
from tqdm import tqdm

import torch  # policy.forward 에서 필요하면

from data_gen import ScenarioConfig, make_phase2_series
from policy_nets import PolicyCNN          # 타입 힌트용
from cl_calib import WindowCalib
from rtc_backend import compute_pS0_stat
from utils import _np2d, _np1d
from rl_pg import make_state_tensor        # 상태 텐서 만드는 함수



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

    # [DBG] 정책 선택 히스토그램
    from collections import Counter
    action_chosen = Counter()    
    
    for t in range(1, T + 1):
        Lmax = 15
        w_for_state = min(max(actions), t)
        state = make_state_tensor(X[t - w_for_state:t], d, L=Lmax).to(device)
        with torch.no_grad():
            logits = policy(state)
            a_idx = int(torch.argmax(logits, dim=-1).item())

        w = actions[a_idx]
        
        # [DBG] 선택된 창 크기 카운트
        action_chosen[w] += 1        
        
        # 아직 창이 다 안 찼으면 알람 판단을 미룹니다.
        if t < w:
            continue
        Sw = X[t - w:t]
        Xrf = np.vstack([S0_ref, Sw])
        yrf = np.hstack([np.zeros(len(S0_ref), dtype=int), np.ones(len(Sw), dtype=int)])
        # ✅ 항상 ndarray로 통일(DF→ndarray 혼용 경고 방지)
        Xrf = _np2d(Xrf, dtype=np.float32)
        yrf = _np1d(yrf, dtype=np.int32)
        # pS0 = compute_pS0_stat(
        #     Xrf, yrf, np.arange(len(S0_ref)),
        #     d=d, n_estimators=n_estimators_eval, seed=42, backend=rf_backend
        # )        
        # 항상 ndarray로 통일 (feature names 경고 방지)
        Xrf = np.asarray(Xrf, dtype=np.float32)
        if Xrf.ndim == 1:
            Xrf = Xrf.reshape(1, -1)
        yrf = np.asarray(yrf, dtype=np.int32).ravel()
        pS0 = compute_pS0_stat(
            Xrf, yrf, np.arange(len(S0_ref)),
            d=d, n_estimators=n_estimators_eval, seed=42, backend=rf_backend
        )
        
        calib = calib_map[w]

        # [DBG] 첫 계산 가능한 시점 주변에서 pS0와 CL을 함께 출력
        if t in (w, w + 1, w + 2):
            try:
                print(f"[dbg] t={t:4d}, w={w:2d}, pS0={pS0:.4f}, CL={calib.CL:.4f}")
            except Exception:
                pass
        
        if pS0 > calib.CL:
            # [DBG] 알람 시 최종 선택 히스토그램 출력
            try:
                print(f"[dbg] action histogram (until alarm t={t}): {dict(action_chosen)}")
            except Exception:
                pass
            
            return t
    # [DBG] 알람이 안 났을 때도 히스토그램 출력
    try:
        print(f"[dbg] action histogram (no alarm, T={T}): {dict(action_chosen)}")
    except Exception:
        pass
    return T
        
def evaluate_arl0(
    scen_cfg: ScenarioConfig,
    scenario: str,
    policy: PolicyCNN,
    actions: List[int],
    calib_map: Dict[int, WindowCalib],
    S0_ref: NDArray,
    R: int,
    seed: int,
    rf_backend: str = 'sklearn',
    n_estimators_eval: int = 150,
) -> Tuple[float, float]:
    """
    ARL0(정상 상태에서의 평균 Run Length)를 시뮬레이션으로 추정.

    - Phase II 전체 구간이 in-control 이 되도록 shift_time을 T로 밀어버린 뒤,
    - lam=0.0 으로 make_phase2_series 를 여러 번 생성해 run_length_until_alarm 의 평균을 구함.
    """
    rng = check_random_state(seed)
    RLs: List[int] = []

    # shift_time=T 로 설정하여 변화가 아예 발생하지 않도록 만든다.
    scen_cfg_ic = _replace(scen_cfg, shift_time=scen_cfg.T)

    pbar = tqdm(range(R), desc="  ARL0 sim", leave=False, dynamic_ncols=True)
    for _ in pbar:
        # lam=0.0 이면 scenario I/II 상관없이 mean shift 없음
        X, labels_ic = make_phase2_series(scen_cfg_ic, rng, scenario, lam=0.0)

        rl = run_length_until_alarm(
            X=X,
            S0_ref=S0_ref,
            policy=policy,
            actions=actions,
            calib_map=calib_map,
            d=scen_cfg_ic.d,
            rf_backend=rf_backend,
            n_estimators_eval=n_estimators_eval,
        )
        RLs.append(rl)

    mean = float(np.mean(RLs))
    std = float(np.std(RLs, ddof=1))
    return mean, std

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


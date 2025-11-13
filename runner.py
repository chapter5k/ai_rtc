# ai_rtc/runner.py

from __future__ import annotations

import os
import math
import pickle
import csv

from datetime import datetime
from typing import Dict

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state
import torch
import torch.optim as optim
from tqdm import tqdm

from config import build_arg_parser, config_from_args, MainConfig
from utils import set_seed
from data_gen import ScenarioConfig, gen_reference_data
from cl_calib import estimate_CL_for_window, WindowCalib
from policy_nets import build_policy, save_policy, load_policy
from rl_pg import RLConfig, train_rl_policy
from eval_arl import evaluate_arl1
from benchmark import run_backend_benchmark  # 네가 만든 이름에 맞게 수정


def _prepare_phase1_data(cfg: MainConfig, scen: ScenarioConfig, rng) -> NDArray:
    """S0_ref (Phase I 기준 데이터)를 불러오거나 새로 생성."""
    if cfg.S0_ref_path and os.path.exists(cfg.S0_ref_path):
        print(f"[Phase I] 기존 S0_ref 로드: {cfg.S0_ref_path}")
        S0_ref = np.load(cfg.S0_ref_path)
    else:
        print("[Phase I] S0_ref 새로 생성 중...")
        S0_ref = gen_reference_data(scen, rng)
        if cfg.S0_ref_path:
            os.makedirs(os.path.dirname(cfg.S0_ref_path), exist_ok=True)
            np.save(cfg.S0_ref_path, S0_ref)
            print(f"[Phase I] S0_ref 저장: {cfg.S0_ref_path}")
    return S0_ref


def _prepare_cl_calib(
    cfg: MainConfig,
    scen: ScenarioConfig,
    S0_ref: NDArray,
) -> Dict[int, WindowCalib]:
    """윈도우별 CL 보정 수행 혹은 기존 calib_map 로드."""
    action_set = cfg.action_set

    if cfg.calib_map_path and os.path.exists(cfg.calib_map_path) and cfg.n_boot == 0:
        print(f"[CL] n_boot=0 & 기존 calib_map 사용: {cfg.calib_map_path}")
        with open(cfg.calib_map_path, "rb") as f:
            calib_map = pickle.load(f)
        return calib_map

    print(f"[CL] 부트스트랩으로 CL 추정 시작 (n_boot={cfg.n_boot})")
    calib_map: Dict[int, WindowCalib] = {}
    for w in tqdm(action_set, desc="[CL] window별 CL 추정"):
        calib = estimate_CL_for_window(
            S0_ref,
            d=scen.d,
            window=w,
            n_boot=cfg.n_boot,
            n_estimators=cfg.n_estimators_eval,
            seed=cfg.seed,
            backend=cfg.rf_backend,
        )
        calib_map[w] = calib

    if cfg.calib_map_path:
        os.makedirs(os.path.dirname(cfg.calib_map_path), exist_ok=True)
        with open(cfg.calib_map_path, "wb") as f:
            pickle.dump(calib_map, f)
        print(f"[CL] calib_map 저장: {cfg.calib_map_path}")

    return calib_map


def _train_policy(
    cfg: MainConfig,
    scen: ScenarioConfig,
    calib_map: Dict[int, WindowCalib],
    S0_ref: NDArray,
) -> torch.nn.Module:
    """정책 네트워크 생성 + RL 학습."""
    device = cfg.device
    action_set = cfg.action_set

    # 1) 정책 네트워크 생성
    policy = build_policy(cfg.policy_arch, d=scen.d, num_actions=len(action_set))
    policy.to(device)

    # 2) 기존 가중치 로드 (선택)
    if cfg.policy_in and os.path.exists(cfg.policy_in):
        print(f"[RL] 기존 정책 로드: {cfg.policy_in}")
        policy = load_policy(
            path=cfg.policy_in,
            d=scen.d,
            num_actions=len(action_set),
            device=device,
            arch=cfg.policy_arch,
        )

    # 3) RL 학습
    rl_cfg = RLConfig(
        action_set=action_set,
        episodes=cfg.episodes,
        device=device,
    )
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    print(f"[RL] Policy Gradient 학습 시작 (episodes={cfg.episodes})")
    policy = train_rl_policy(
        policy=policy,
        optimizer=optimizer,
        cfg=rl_cfg,
        scen=scen,
        calib_map=calib_map,
        S0_ref=S0_ref,
        seed=cfg.seed,
        rf_backend=cfg.rf_backend,
        n_estimators_eval=cfg.n_estimators_eval,
    )

    # 4) 학습된 정책 저장 (선택)
    if cfg.policy_out:
        os.makedirs(os.path.dirname(cfg.policy_out), exist_ok=True)
        save_policy(policy, cfg.policy_out)
        print(f"[RL] 학습된 정책 저장: {cfg.policy_out}")

    return policy


def _evaluate(
    cfg: MainConfig,
    scen: ScenarioConfig,
    calib_map: Dict[int, WindowCalib],
    S0_ref: NDArray,
    policy: torch.nn.Module,
):
    """시나리오 I/II 에 대해 ARL1 평가 + CSV 저장."""
    lam2_list = [0.25, 0.50, 1.00, 2.00, 3.00, 5.00, 7.00, 9.00]
    lam_list = [math.sqrt(x) for x in lam2_list]

    # 실행 폴더 예: outputs/run_20251113_153012/
    base_dir = os.path.dirname(cfg.S0_ref_path)  # 이미 runner에서 세팅함

    for scenario_name in ["I", "II"]:
        print(f"\n[평가] Scenario {scenario_name} (action_set={cfg.action_set})")

        arl_means, arl_stds = evaluate_arl1(
            scen_cfg=scen,
            lam_list=lam_list,
            scenario=scenario_name,
            policy=policy,
            actions=list(cfg.action_set),
            calib_map=calib_map,
            S0_ref=S0_ref,
            R=cfg.R,
            seed=cfg.seed,
            rf_backend=cfg.rf_backend,
            n_estimators_eval=cfg.n_estimators_eval,
        )

        # ---- CSV 저장 ----
        csv_path = os.path.join(base_dir, f"arl_results_scenario_{scenario_name}.csv")
        print(f"[저장] {csv_path}")

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lambda2", "lambda", "arl1_mean", "arl1_std"])
            for lam2, lam, mean, std in zip(lam2_list, lam_list, arl_means, arl_stds):
                writer.writerow([lam2, lam, mean, std])

        # ---- 콘솔 출력 ----
        for lam2, lam, mean, std in zip(lam2_list, lam_list, arl_means, arl_stds):
            print(f"  λ²={lam2:.2f} λ={lam:.4f} ARL1={mean:.2f} [{std:.2f}]")


def main():
    """전체 파이프라인 실행: Phase I → CL 보정 → RL 학습 → ARL1 평가."""
    start_time = datetime.now()

    # 1) 인자 파싱
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)

    # 🔽🔽🔽 여기부터 추가: 출력 폴더 세팅 🔽🔽🔽
    # 1) 실험 이름 결정
    if cfg.exp_name is None or cfg.exp_name == "":
        exp_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    else:
        exp_name = cfg.exp_name

    # 2) base_dir = outputs/run_...
    base_dir = os.path.join(cfg.outputs_dir, exp_name)
    os.makedirs(base_dir, exist_ok=True)
    print(f"[출력] 결과가 저장될 폴더: {base_dir}")

    # 3) S0_ref / calib_map / policy_out 이 비어 있으면 기본 경로 자동 설정
    if not cfg.S0_ref_path:
        cfg.S0_ref_path = os.path.join(base_dir, "S0_ref.npy")
    if not cfg.calib_map_path:
        cfg.calib_map_path = os.path.join(base_dir, "calib_map.pkl")
    if not cfg.policy_out:
        cfg.policy_out = os.path.join(base_dir, "policy.pt")

    # (원하면 ARL 결과, 설정값 저장용 기본 경로도 미리 잡아둘 수 있음)
    # cfg.arl_results_path = os.path.join(base_dir, "arl_results.csv")
    # cfg.config_dump_path = os.path.join(base_dir, "config.txt")


    print(f"[시작] seed={cfg.seed}, device={cfg.device}, backend={cfg.rf_backend}")
    set_seed(cfg.seed)

    rng = check_random_state(cfg.seed)

    # 2) 시나리오/데이터 설정 (논문 기본값)
    scen = ScenarioConfig(d=10, N0=1500, T=300, shift_time=100, sigma=1.0)

    # (선택) 백엔드 벤치마크
    try:
        from .benchmark import run_backend_benchmark
        run_backend_benchmark(
            d=scen.d,
            N0=scen.N0,
            backend=cfg.rf_backend,
            guess_arl1=cfg.guess_arl1,
            n_estimators=cfg.n_estimators_eval,
        )
    except Exception as e:
        print(f"[벤치마크] 실패 (무시해도 됨): {e}")

    # 3) Phase I 데이터 준비
    S0_ref = _prepare_phase1_data(cfg, scen, rng)

    # 4) CL 보정
    calib_map = _prepare_cl_calib(cfg, scen, S0_ref)

    # 5) RL 정책 학습
    policy = _train_policy(cfg, scen, calib_map, S0_ref)

    # 6) ARL1 평가
    _evaluate(cfg, scen, calib_map, S0_ref, policy)

    elapsed = datetime.now() - start_time
    print(f"\n[완료] 전체 소요 시간: {elapsed}")


if __name__ == "__main__":
    main()

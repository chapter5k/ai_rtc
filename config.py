# ai_rtc/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import argparse
import torch


RfBackend = Literal["sklearn", "cuml_cv", "lgbm"]
PolicyArch = Literal["cnn", "cnn_lstm"]
AlgoType = Literal["pg", "sac_discrete"]
RewardType = Literal["alg1", "morl"]   #

@dataclass
class MainConfig:
    """메인 실험에서 쓰는 핵심 설정값 모음."""
    seed: int = 2025
    device: str = "cuda"
    episodes: int = 30
    n_boot: int = 800
    action_set: Tuple[int, ...] = (5, 10, 15)
    R: int = 100
    rf_backend: RfBackend = "sklearn"
    guess_arl1: int = 15
    n_estimators_eval: int = 150
    policy_in: Optional[str] = None
    policy_out: Optional[str] = None
    S0_ref_path: str = ""
    calib_map_path: str = ""
    outputs_dir: str = "outputs"          # 결과 루트 폴더
    exp_name: Optional[str] = None        # 실험 이름 (없으면 timestamp로 자동 생성)
    policy_arch: Literal["cnn", "cnn_lstm"] = "cnn_lstm"
    algo: Literal["pg", "sac_discrete"] = "pg"   # 기본값 PG
    rl_lr: float = 1e-3
    reward: RewardType = "alg1"
    
def build_arg_parser() -> argparse.ArgumentParser:
    """CLI 인자 정의 (원래 ai_rtc_251103_v4.py에 있던 argparse 부분)."""
    parser = argparse.ArgumentParser(
        description="RL-RTC 논문 재현 + AI_RTC 실험 실행 스크립트"
    )

    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser.add_argument('--seed', type=int, default=MainConfig.seed)
    parser.add_argument(
        '--device',
        type=str,
        default=default_device,
        help="PyTorch CNN 연산 장치 ('cuda' 또는 'cpu')"
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=MainConfig.episodes,
        help='RL 학습 에피소드 수 (권장 100~300 이상)'
    )
    parser.add_argument(
        '--n_boot',
        type=int,
        default=MainConfig.n_boot,
        help='CL 부트스트랩 반복 (0이면 CL 보정 스킵 + 기존 파일 사용)'
    )
    parser.add_argument(
        '--action_set',
        type=str,
        default="5,10,15",
        help='윈도우 크기 옵션 (예: "5,10,15" 또는 "3,10,17")'
    )
    parser.add_argument(
        '--R',
        type=int,
        default=MainConfig.R,
        help='ARL1 평가 반복 횟수'
    )
    parser.add_argument(
        '--rf_backend',
        type=str,
        default=MainConfig.rf_backend,
        choices=['sklearn', 'cuml_cv', 'lgbm'],
        help="분류기 백엔드: 'sklearn'(CPU,OOB), 'cuml_cv'(GPU,K-fold OOB), 'lgbm'(CPU/GPU)"
    )
    parser.add_argument(
        '--guess_arl1',
        type=int,
        default=MainConfig.guess_arl1,
        help='ARL1 평균 추정치(정적 ETA 계산용, 사용 안 해도 무관)'
    )
    parser.add_argument(
        '--n_estimators_eval',
        type=int,
        default=MainConfig.n_estimators_eval,
        help='ARL1 평가 단계에서 사용할 트리 수'
    )
    parser.add_argument(
        '--policy_in',
        type=str,
        default=None,
        help='불러올 정책 가중치(.pt). 지정 시 거기서부터 학습 계속'
    )
    parser.add_argument(
        '--policy_out',
        type=str,
        default=None,
        help='학습 후 저장할 정책 경로(.pt)'
    )
    parser.add_argument(
        '--S0_ref_path',
        type=str,
        default="",
        help="기존 Phase I 기준 데이터(S0_ref .npy) 경로 (없으면 새로 생성)"
    )
    parser.add_argument(
        '--calib_map_path',
        type=str,
        default="",
        help="기존 CL 보정 맵 .pkl 경로 (없으면 새로 생성)"
    )
    parser.add_argument(
        '--policy_arch',
        type=str,
        default=MainConfig.policy_arch,
        choices=['cnn', 'cnn_lstm'],
        help="정책 네트워크 구조 선택"
    )
    parser.add_argument(
        '--outputs_dir',
        type=str,
        default=MainConfig.outputs_dir,
        help="결과물을 저장할 루트 폴더 (기본: outputs)"
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help="실험 이름 (미지정 시 run_YYYYMMDD_HHMMSS 형식으로 자동 생성)"
    )
    parser.add_argument(
        '--algo',
        type=str,
        default=MainConfig.algo,
        choices=['pg', 'sac_discrete'],
        help="강화학습 알고리즘 선택: 'pg'(기존 Policy Gradient) 또는 'sac_discrete'(이산 SAC)"
    )
    parser.add_argument(
        '--rl_lr',
        type=float,
        default=MainConfig.rl_lr,
        help="Policy Gradient(LR) 학습률 (algo='pg'일 때 사용)"
    )
    parser.add_argument(
        '--reward',
        type=str,
        default="alg1",
        choices=['alg1', 'morl'],
        help="보상 설계: 'alg1'(논문 Algorithm 1), 'morl'(ARL0/ARL1 트레이드오프용 MORL 스칼라화)"
    )


    return parser

def config_from_args(args: argparse.Namespace) -> MainConfig:
    """argparse 결과를 dataclass(MainConfig)로 변환."""
    if isinstance(args.action_set, str):
        action_tuple = tuple(int(x) for x in args.action_set.split(','))
    else:
        action_tuple = tuple(args.action_set)

    return MainConfig(
        seed=args.seed,
        device=args.device,
        episodes=args.episodes,
        n_boot=args.n_boot,
        action_set=action_tuple,
        R=args.R,
        rf_backend=args.rf_backend,
        guess_arl1=args.guess_arl1,
        n_estimators_eval=args.n_estimators_eval,
        policy_in=args.policy_in,
        policy_out=args.policy_out,
        S0_ref_path=args.S0_ref_path,
        calib_map_path=args.calib_map_path,
        policy_arch=args.policy_arch,
        outputs_dir=args.outputs_dir,
        exp_name=args.exp_name,
        algo=args.algo,
        rl_lr=args.rl_lr,
        reward=args.reward,
    )
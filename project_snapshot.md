# Project Snapshot

- Generated at: `2025-11-19 17:22:11`
- Root directory: `C:\Users\USER\project\ai_rtc`

## Directory Tree

```text
./
  __init__.py
  benchmark.py
  cl_calib.py
  config.py
  data_gen.py
  eval_arl.py
  export_project_md.py
  policy_nets.py
  project_snapshot.md
  requirements.txt
  reward_morl.py
  rl_pg.py
  rl_sac.py
  rtc_backend.py
  runner.py
  utils.py
  ~$크립트 실행.docx
  스크립트 실행.docx

scripts/
  run_ai_rtc.py
```

## Python Files

### `__init__.py`

```python

```

### `benchmark.py`

```python

# ai_rtc/benchmark.py
# benchmark.py
import os
import time
from typing import Literal  # Literal 안 쓰면 생략해도 됨

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from rtc_backend import compute_pS0_stat


def run_backend_benchmark(
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
        # ✅ LGBM 경로: fit/predict 모두 넘파이로 통일 + GPU/CPU 자동 선택
        from lightgbm import LGBMClassifier
        # 1) feature names 경고 방지: 항상 ndarray 사용
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int32)
        X_np = np.ascontiguousarray(X_np)
        y_np = np.ascontiguousarray(y_np)
        # 2) 장치 선택 (main()에서 os.environ["AI_RTC_DEVICE"]=args.device 설정 필요)
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
            device_type=("gpu" if use_gpu else "cpu"),
        )
        model.fit(X_np, y_np)
        _ = model.predict_proba(X_np)  # 예측까지 포함해 ETA가 실제에 가깝게

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return time.perf_counter() - t0

```

### `cl_calib.py`

```python
# cl_calib.py

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state
from tqdm import tqdm

from rtc_backend import compute_pS0_stat
from utils import _np2d, _np1d

@dataclass
class WindowCalib:
    """윈도우별 관리한계(CL)와 부트스트랩 표준편차를 담는 구조체."""
    CL: float   # 관리한계 (상한선)
    std: float  # 부트스트랩 std

def estimate_CL_for_window(
    S0: NDArray,
    d: int,
    window: int,
    n_boot: int,
    n_estimators: int,
    seed: int,
    backend: str = 'sklearn',
) -> WindowCalib:
    """
    주어진 window 크기에 대해, 부트스트랩으로 CL(상한)을 추정.
    반환값: WindowCalib(CL, std)
    """
    if n_boot <= 0:
        raise ValueError("estimate_CL_for_window: n_boot must be >= 1 (CL 스킵은 main에서 로드 분기를 사용).")
    
    rng = check_random_state(seed)
    alpha = 1.0 / 200.0   # ARL0 ≈ 200 을 맞추기 위한 상한 분위수
    stats = []
    N0 = len(S0)

    pbar = tqdm(range(n_boot), desc=f"  CL Boot (w={window})", leave=False, dynamic_ncols=True)
    for _ in pbar:
        start = 0 if N0 - window <= 0 else rng.randint(0, N0 - window)
        Sw = S0[start:start + window]

        # 정상(S0) vs 비정상(Sw)
        X = np.vstack([S0, Sw])
        y = np.hstack([
            np.zeros(len(S0), dtype=int),
            np.ones(len(Sw), dtype=int),
        ])

        # 항상 넘파이로 통일 (DataFrame → ndarray 혼용 방지)
        X = _np2d(X, dtype=np.float32)
        y = _np1d(y, dtype=np.int32)

        pS0 = compute_pS0_stat(
            X, y, np.arange(len(S0)),
            d=d, n_estimators=n_estimators,
            seed=rng.randint(1_000_000),
            backend=backend,
        )
        stats.append(pS0)

    stats = np.asarray(stats)
    CL = np.quantile(stats, 1 - alpha)
    std_boot = float(np.std(stats, ddof=1))

    return WindowCalib(CL=CL, std=std_boot)

```

### `config.py`

```python
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
```

### `data_gen.py`

```python
# ai_rtc/data_gen.py
from dataclasses import dataclass
from typing import Tuple
import math
import numpy as np
from numpy.typing import NDArray

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

```

### `eval_arl.py`

```python
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


```

### `policy_nets.py`

```python
# ai_rtc/policy_nets.py
import os
from typing import Literal
import torch
import torch.nn as nn
from utils import _unwrap_model

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

class PolicyCNNLSTM(nn.Module):
    """
    입력 x: (B, 1, L, d)  # make_state_tensor 결과, 기본 L=15, d=10 가정
    Conv1d(시간축) -> LSTM(128) -> FC(#actions)
    """
    def __init__(self, d: int, num_actions: int):
        super().__init__()
        if d != 10:
            raise ValueError("이 모델은 현재 d=10 가정으로 설계되었습니다.")
        conv_out = 64
        # (B,1,L,d) -> (B,L,d) -> (B,d,L) 후 Conv1d
        self.conv1 = nn.Conv1d(in_channels=d, out_channels=conv_out, kernel_size=3, padding=1)
        self.act1  = nn.ReLU()
        self.lstm  = nn.LSTM(input_size=conv_out, hidden_size=128, num_layers=1,
                             batch_first=True, bidirectional=False)
        self.fc    = nn.Linear(128, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)           # (B, L, d)
        x = x.permute(0, 2, 1)     # (B, d, L)
        z = self.act1(self.conv1(x))   # (B, 64, L)
        z = z.permute(0, 2, 1)     # (B, L, 64)  # LSTM 입력
        out, (h, c) = self.lstm(z)
        h_last = h[-1]             # (B, 128)
        logits = self.fc(h_last)   # (B, #actions)
        return logits

PolicyArch = Literal["cnn", "cnn_lstm"]


def build_policy(arch: PolicyArch, d: int, num_actions: int) -> nn.Module:
    if arch == 'cnn':
        return PolicyCNN(d=d, num_actions=num_actions)
    elif arch == 'cnn_lstm':
        return PolicyCNNLSTM(d=d, num_actions=num_actions)
    else:
        raise ValueError(f"Unknown policy_arch: {arch}")


def save_policy(policy: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base = _unwrap_model(policy)
    torch.save(base.state_dict(), path)

def load_policy(path: str, d: int, num_actions: int, device: str, arch: str = 'cnn') -> nn.Module:
    policy = build_policy(arch, d=d, num_actions=num_actions)
    state = torch.load(path, map_location=device)
    policy.load_state_dict(state)
    policy.to(device)
    policy.eval()
    return policy


```

### `reward_morl.py`

```python
# ai_rtc/reward_morl.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np

# 기존 Algorithm 1 보상(순위 기반)과 동일한 형태로 로컬 정의
def reward_by_algorithm1(action_idx: int, distances: List[float], in_control: bool) -> float:
    """
    distances: IC 구간에서 '클수록 좋은' 점수 (예: (CL - pS0) / std)
    action_idx: 전체 actions 중에서 에이전트가 선택한 행동의 인덱스 (0, 1, 2)

    - IC(in_control=True)   : rank가 높을수록(거리 큰 행동 선택) 보상 +1, 그 외 -1, -2
    - OOC(in_control=False) : 반대로 CL에 더 가까운 행동(탐지에 유리한 행동)을 선호
    """
    # 거리가 큰 순서대로 내림차순 정렬
    order = np.argsort(distances)[::-1]
    # 선택한 action_idx의 순위(0=최상위, 2=최하위)
    rank = int(np.where(order == action_idx)[0][0])

    if in_control:
        table = {0: 1.0, 1: -1.0, 2: -2.0}
    else:
        table = {0: -2.0, 1: -1.0, 2: 1.0}
    return float(table[rank])

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

```

### `rl_pg.py`

```python
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


```

### `rl_sac.py`

```python
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
from rl_pg import make_state_tensor
from reward_morl import MorlConfig, MorlStats, MorlEpisodeState, calc_reward_morl, calc_reward_alg1_wrapper, reward_by_algorithm1



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
    reward: str = "alg1"


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
    # valid_mask_t: (num_actions,) -> (1, num_actions)에 브로드캐스팅
    logits[:, ~valid_mask_t] = -1e9

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
    # 전역 MORL 통계 (에피소드 전체 공유)
    morl_cfg = MorlConfig()
    morl_stats = MorlStats()    
    
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
        morl_ep_state = MorlEpisodeState()
        
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

            # --- 보상 계산  ---
            ic = bool(labels_ic[t - 1] == 1)

            if sac_cfg.reward == "alg1":
                reward = calc_reward_alg1_wrapper(
                    action_global_idx=a_idx,
                    valid_indices=valid_indices,
                    D_list=D_list,
                    in_control=ic,
                )
            else:  # 'morl'
                if a_idx not in valid_indices:
                    selected_D = 0.0
                else:
                    local_idx = valid_indices.index(a_idx)
                    selected_D = float(D_list[local_idx])

                reward, comps = calc_reward_morl(
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
            ep_rewards.append(float(reward))


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
                        next_logits = policy(next_states_b)               # (B, num_actions)
                        next_logits = next_logits.clone()
                        next_logits = next_logits.masked_fill(~next_valid_masks_b, -1e9)


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
                    logits = policy(states_b)        # (B, num_actions)
                    logits = logits.clone()
                    logits = logits.masked_fill(~valid_masks_b, -1e9)

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

```

### `rtc_backend.py`

```python
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


```

### `runner.py`

```python
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
from eval_arl import evaluate_arl1, evaluate_arl0
from benchmark import run_backend_benchmark  

from rl_pg import train_rl_policy
from rl_sac import train_sac_policy, SACConfig


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
    """정책 네트워크 생성 + RL 학습 (PG 또는 SAC)."""
    device = cfg.device
    action_set = cfg.action_set

    # -------------------------------
    # 1) 정책 네트워크 생성
    # -------------------------------
    policy = build_policy(
        cfg.policy_arch,
        d=scen.d,
        num_actions=len(action_set)
    )
    policy.to(device)

    # -------------------------------
    # 2) 기존 가중치 로드 (선택)
    # -------------------------------
    if cfg.policy_in and os.path.exists(cfg.policy_in):
        print(f"[RL] 기존 정책 로드: {cfg.policy_in}")
        policy = load_policy(
            path=cfg.policy_in,
            d=scen.d,
            num_actions=len(action_set),
            device=device,
            arch=cfg.policy_arch,
        )

    # -------------------------------
    # 3) RL 학습 (PG 또는 SAC)
    # -------------------------------
    if cfg.algo == "pg":
        # 기존 PG 경로
        rl_cfg = RLConfig(
            action_set=action_set,
            episodes=cfg.episodes,
            device=device,
            reward=cfg.reward,
        )
        optimizer = optim.Adam(policy.parameters(), lr=cfg.rl_lr)

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

    elif cfg.algo == "sac_discrete":
        # SAC 이산 경로
        print(f"[RL] SAC(Discrete) 학습 시작 (episodes={cfg.episodes})")

        sac_cfg = SACConfig(
            action_set=tuple(cfg.action_set),
            episodes=cfg.episodes,
            device=device,
            reward=cfg.reward,
        )

        policy = train_sac_policy(
            policy=policy,
            sac_cfg=sac_cfg,
            scen=scen,
            calib_map=calib_map,
            S0_ref=S0_ref,
            seed=cfg.seed,
            rf_backend=cfg.rf_backend,
            n_estimators_eval=cfg.n_estimators_eval,
        )

    else:
        raise ValueError(f"Unknown cfg.algo '{cfg.algo}'. (지원: 'pg', 'sac_discrete')")

    # -------------------------------
    # 4) 학습된 정책 저장 (선택)
    # -------------------------------
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

        # --- ARL0 평가 (정상 상태) ---
        arl0_mean, arl0_std = evaluate_arl0(
            scen_cfg=scen,
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
            # 👇 컬럼 2개 추가
            writer.writerow(["lambda2", "lambda", "arl1_mean", "arl1_std", "arl0_mean", "arl0_std"])
            for lam2, lam, mean, std in zip(lam2_list, lam_list, arl_means, arl_stds):
                writer.writerow([lam2, lam, mean, std, arl0_mean, arl0_std])

        # ---- 콘솔 출력 ----
        print(f"  ARL0={arl0_mean:.2f} [{arl0_std:.2f}]")
        
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
    
    # 3) Phase I 데이터 준비
    S0_ref = _prepare_phase1_data(cfg, scen, rng)

    # ✅ (선택) 백엔드 벤치마크 - 정상 버전
    try:
        elapsed = run_backend_benchmark(
            S0_ref=S0_ref,
            d=scen.d,
            n_estimators=cfg.n_estimators_eval,
            seed=cfg.seed,
            backend=cfg.rf_backend,
        )
        print(f"[벤치마크] backend='{cfg.rf_backend}' 기준 1회 통계 계산 시간 ≈ {elapsed:.3f} 초")
    except Exception as e:
        print(f"[벤치마크] 실패 (무시해도 됨): {e}")

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

```

### `scripts\run_ai_rtc.py`

```python
# scripts/run_ai_rtc.py

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from runner import main   

if __name__ == "__main__":
    main()

```

### `utils.py`

```python
# ai_rtc/utils.py
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn


def _np1d(y, dtype=np.int32):
    a = np.asarray(y, dtype=dtype).ravel()
    return np.ascontiguousarray(a)

def _np2d(X, dtype=np.float32):
    A = np.asarray(X, dtype=dtype)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    return np.ascontiguousarray(A)

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sqrt_int(x: float) -> int:
    return max(1, int(round(math.sqrt(x))))

def _unwrap_model(m: nn.Module) -> nn.Module:
    # torch.compile 래핑 해제
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    # DataParallel 래핑 해제
    if isinstance(m, nn.DataParallel):
        m = m.module
    return m

```

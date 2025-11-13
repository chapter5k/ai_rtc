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

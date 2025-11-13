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


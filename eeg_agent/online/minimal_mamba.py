"""CPU mixer adapted from johnma2006/mamba-minimal (Apache-2.0).

Pinned upstream: 03de542a36d873f6e6c4057ad687278cc6ae944d, model.py.
Changes: retain only MambaBlock, remove language-model/RMSNorm components and
einops dependency, and use explicit FP32 tensors and ordinary torch operators.
Copyright 2023 Tri Dao, Albert Gu. See licenses/mamba-minimal-LICENSE.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM_COMMIT = "03de542a36d873f6e6c4057ad687278cc6ae944d"


@dataclass
class ModelArgs:
    d_model: int = 525
    d_state: int = 16
    expand: int = 4
    dt_rank: int = 33
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False

    @property
    def d_inner(self):
        return self.d_model * self.expand


class MambaBlock(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args or ModelArgs()
        a = self.args
        self.in_proj = nn.Linear(a.d_model, 2 * a.d_inner, bias=a.bias)
        self.conv1d = nn.Conv1d(a.d_inner, a.d_inner, a.d_conv,
                              groups=a.d_inner, padding=a.d_conv - 1, bias=a.conv_bias)
        self.x_proj = nn.Linear(a.d_inner, a.dt_rank + 2 * a.d_state, bias=False)
        self.dt_proj = nn.Linear(a.dt_rank, a.d_inner, bias=True)
        initial = torch.arange(1, a.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(initial.log().repeat(a.d_inner, 1))
        self.D = nn.Parameter(torch.ones(a.d_inner))
        self.out_proj = nn.Linear(a.d_inner, a.d_model, bias=a.bias)

    def forward(self, hidden):
        length = hidden.shape[1]
        x, gate = self.in_proj(hidden).chunk(2, dim=-1)
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :length].transpose(1, 2))
        dt, B, C = self.x_proj(x).split(
            [self.args.dt_rank, self.args.d_state, self.args.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        y = self.selective_scan(x, delta, -self.A_log.float().exp(), B, C, self.D.float())
        return self.out_proj(y * F.silu(gate))

    @staticmethod
    def selective_scan(u, delta, A, B, C, D):
        # All 80 steps are already inside a complete five-second EEG window.
        # Each invocation starts with zero SSM/causal-convolution state.
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)
        state = torch.zeros((u.shape[0], u.shape[2], A.shape[1]),
                            dtype=u.dtype, device=u.device)
        outputs = []
        for index in range(u.shape[1]):
            state = deltaA[:, index] * state + deltaB_u[:, index]
            outputs.append(torch.einsum('bdn,bn->bd', state, C[:, index]))
        return torch.stack(outputs, dim=1) + u * D

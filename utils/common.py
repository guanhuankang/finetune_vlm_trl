import torch
import torch.nn as nn
import numpy as np
from typing import Type


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PositionalEmbedding2D(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.pe = nn.Parameter(torch.randn(2, d_model))

    def device(self):
        return self.pe.device

    def forward(self, x):
        assert len(x.shape) == 3

        _, L, _ = x.shape
        h, w = int(np.sqrt(L)), int(np.sqrt(L))
        g = L - h * w

        x_coord = torch.linspace(-1.0, 1.0, w).to(self.device())
        y_coord = torch.linspace(-1.0, 1.0, h).to(self.device())

        x_coord = x_coord[None, :].expand(h, w)
        y_coord = y_coord[:, None].expand(h, w)

        pe = torch.stack([x_coord, y_coord], dim=-1) @ self.pe
        pe = pe.reshape(1, h * w, -1)

        ## add positional embedding
        x[:, g:, :] = x[:, g:, :] + pe

        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mlp_dim: int,
        out_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, out_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return
        
        seq_len = x.shape[0]
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(
            x.device
        )  # THETA = 10,000^(-2*i/d) or 1/10,000^(2i/d)

        seq_idx = (
            torch.arange(seq_len, device=x.device).float().to(x.device)
        )  # Position Index -> [0,1,2...seq-1]

        idx_theta = torch.einsum(
            "n,d->nd", seq_idx, theta
        )  # Calculates m*(THETA) = [ [0, 0...], [THETA_1, THETA_2...THETA_d/2], ... [seq-1*(THETA_1), seq-1*(THETA_2)...] ]

        idx_theta2 = torch.cat(
            [idx_theta, idx_theta], dim=1
        )  # [THETA_1, THETA_2...THETA_d/2] -> [THETA_1, THETA_2...THETA_d]

        self.cos_cached = idx_theta2.cos()[
            :, None, None, :
        ]  # Cache [cosTHETA_1, cosTHETA_2...cosTHETA_d]
        self.sin_cached = idx_theta2.sin()[
            :, None, None, :
        ]  # cache [sinTHETA_1, sinTHETA_2...sinTHETA_d]

    def _neg_half(self, x: torch.Tensor):

        d_2 = self.d // 2  #

        return torch.cat(
            [-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1
        )  # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]

    def forward(self, x: torch.Tensor):
        self._build_cache(x)

        neg_half_x = self._neg_half(x)

        x_rope = (x * self.cos_cached[: x.shape[0]]) + (
            neg_half_x * self.sin_cached[: x.shape[0]]
        )  # [x_1*cosTHETA_1 - x_d/2*sinTHETA_d/2, ....]

        return x_rope

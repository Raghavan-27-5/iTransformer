"""
NHiTS-Inspired Multi-Scale Forecasting Head for iTransformer.

Replaces the flat nn.Linear(d_model, pred_len) projection with a hierarchical
multi-scale decomposition operating entirely in latent (d_model) space.

Design principles borrowed from N-HiTS (Challu et al., 2023):
  1. Multi-scale compression: coarse → medium → fine stacks
  2. Interpolation-based upsampling from compressed theta to pred_len
  3. Doubly residual stacking: each block subtracts its backcast from the residual

Key adaptation vs original N-HiTS:
  - Input is d_model latent embedding (B, N, D), NOT raw time series (B, T)
  - MaxPooling downsampling is removed (not applicable in latent space)
  - N-BEATS basis functions are removed (overkill for a head module)
  - Backcast is produced in D-dim latent space for residual subtraction

Dimensions:
  Input:  h  ∈ R^(B × N × D)   variate tokens from iTransformer encoder
  Output: Ŷ  ∈ R^(B × N × S)   forecasts per variate (before permute)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class NHiTSBlock(nn.Module):
    """
    Single N-HiTS inspired block.

    Each block operates at one temporal resolution (theta_ratio controls compression):
      - Compresses D-dim input by theta_ratio
      - Produces forecast_theta of size (pred_len // theta_ratio)
      - Upsamples forecast_theta to pred_len via linear interpolation
      - Produces backcast in D-dim space for doubly-residual update

    Args:
        d_model:     Dimension of input variate tokens (D).
        pred_len:    Forecast horizon (S).
        theta_ratio: Compression factor. theta_size = max(pred_len // theta_ratio, 1).
        dropout:     Dropout rate applied after compression.
    """

    def __init__(
        self,
        d_model: int,
        pred_len: int,
        theta_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pred_len: int = pred_len
        self.theta_size: int = max(pred_len // theta_ratio, 1)
        self.d_compressed: int = max(d_model // theta_ratio, 1)

        # Compression: D → D//theta_ratio
        self.compress = nn.Sequential(
            nn.Linear(d_model, self.d_compressed),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Forecast branch: compressed_dim → theta_size (coarse forecast coefficients)
        self.forecast_fc = nn.Linear(self.d_compressed, self.theta_size)

        # Backcast branch: compressed_dim → D (for doubly-residual subtraction)
        self.backcast_fc = nn.Linear(self.d_compressed, d_model)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, N, D) variate token residual.

        Returns:
            forecast: (B, N, S)  interpolated forecast contribution.
            backcast: (B, N, D)  for doubly-residual update h_next = h - backcast.
        """
        B, N, D = h.shape

        compressed = self.compress(h)                     # (B, N, d_compressed)

        # Forecast theta: (B, N, theta_size)
        theta_f = self.forecast_fc(compressed)            # (B, N, theta_size)

        # Interpolate theta_size → pred_len
        # F.interpolate requires (B, C, L): reshape B*N as batch, 1 as channel
        theta_f_3d = theta_f.reshape(B * N, 1, self.theta_size)
        if self.theta_size == self.pred_len:
            # No interpolation needed for fine stack (theta_ratio=1)
            forecast_3d = theta_f_3d
        else:
            forecast_3d = F.interpolate(
                theta_f_3d,
                size=self.pred_len,
                mode='linear',
                align_corners=False,
            )                                              # (B*N, 1, pred_len)
        forecast = forecast_3d.reshape(B, N, self.pred_len)  # (B, N, S)

        # Backcast: (B, N, D) — subtracted from residual in parent module
        backcast = self.backcast_fc(compressed)           # (B, N, D)

        return forecast, backcast


class NHiTSHead(nn.Module):
    """
    N-HiTS inspired hierarchical multi-scale head.

    Stacks multiple NHiTSBlock modules from coarse to fine:
      Stack 0 (coarse):  theta_ratio=4, theta_size=S//4, d_compressed=D//4
      Stack 1 (medium):  theta_ratio=2, theta_size=S//2, d_compressed=D//2
      Stack 2 (fine):    theta_ratio=1, theta_size=S,    d_compressed=D

    Total forecast = Σ interpolated_forecast_i (doubly residual sum)

    Parameter count vs Linear baseline:
      Linear:    D × S  (e.g. 512×96  = 49,152)
      NHiTSHead: ~ D×(D//4) + (D//4)×(S//4) + (D//4)×D   [stack0]
                 + D×(D//2) + (D//2)×(S//2) + (D//2)×D   [stack1]
                 + D×D      + D×S           + D×D         [stack2]
      For D=512, S=96: ~1.1M params vs 49K — richer capacity, higher expressivity.

    Args:
        d_model:   Input embedding dimension (D).
        pred_len:  Forecast horizon (S).
        n_stacks:  Number of hierarchical stacks (default 3).
        dropout:   Dropout rate (default 0.1).
    """

    # Theta ratios from coarsest to finest
    _THETA_RATIOS = [4, 2, 1]

    def __init__(
        self,
        d_model: int,
        pred_len: int,
        n_stacks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_stacks > len(self._THETA_RATIOS):
            raise ValueError(
                f"n_stacks={n_stacks} exceeds max supported={len(self._THETA_RATIOS)}"
            )
        self.pred_len = pred_len
        self.n_stacks = n_stacks
        ratios = self._THETA_RATIOS[:n_stacks]

        self.blocks = nn.ModuleList([
            NHiTSBlock(
                d_model=d_model,
                pred_len=pred_len,
                theta_ratio=r,
                dropout=dropout,
            )
            for r in ratios
        ])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, N, D) variate tokens from iTransformer encoder.

        Returns:
            forecast: (B, N, S) summed hierarchical forecasts.
        """
        B, N, D = h.shape
        total_forecast = torch.zeros(
            B, N, self.pred_len,
            device=h.device,
            dtype=h.dtype,
        )
        residual = h                       # doubly-residual chain starts with h

        for block in self.blocks:
            forecast, backcast = block(residual)
            total_forecast = total_forecast + forecast
            residual = residual - backcast  # doubly-residual update

        return total_forecast               # (B, N, S)
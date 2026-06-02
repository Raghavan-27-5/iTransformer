"""
iTransformerNHiTS: iTransformer backbone + NHiTS-inspired hierarchical head.

Only change from the original iTransformer (iTransformer.py):
  self.projector = nn.Linear(d_model, pred_len)
  →
  self.projector = NHiTSHead(d_model, pred_len, n_stacks, dropout)

Everything else — embedding, encoder, normalization, de-normalization — is
IDENTICAL to the original paper implementation.

The forward pass shape contract is preserved:
  enc_out:            (B, N, D)
  self.projector(...) (B, N, S)   ← NHiTSHead outputs same shape as Linear
  .permute(0, 2, 1)   (B, S, N)
  [:, :, :N]          (B, S, N)   filter covariates

Args added to configs (via run.py):
  configs.nhits_n_stacks  (int, default 3)
  configs.nhits_dropout   (float, default 0.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from model.heads.nhits_head import NHiTSHead
import numpy as np


class Model(nn.Module):
    """
    iTransformer backbone with NHiTS-inspired multi-scale forecasting head.

    Paper: https://arxiv.org/abs/2310.06625
    Head:  NHiTS-inspired hierarchical interpolation (Challu et al., 2023)
    """

    def __init__(self, configs) -> None:
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # ── Embedding (identical to original) ──────────────────────────────
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.class_strategy = configs.class_strategy

        # ── Encoder (identical to original) ────────────────────────────────
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # ── NHiTS Head (replaces nn.Linear) ────────────────────────────────
        nhits_n_stacks: int = getattr(configs, 'nhits_n_stacks', 3)
        nhits_dropout: float = getattr(configs, 'nhits_dropout', 0.1)

        self.projector = NHiTSHead(
            d_model=configs.d_model,
            pred_len=configs.pred_len,
            n_stacks=nhits_n_stacks,
            dropout=nhits_dropout,
        )

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
    ):
        if self.use_norm:
            # Normalization (Non-stationary Transformer style) — identical to original
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N

        # Embedding: B L N → B N E
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encoder: B N E → B N E
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # NHiTS Head: B N E → B N S → permute → B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            # De-normalization — identical to original
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )

        return dec_out, attns

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        mask=None,
    ) -> torch.Tensor:
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # (B, S, N)
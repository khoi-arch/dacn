#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_pipeline/embedding.py

Embedding layer only.

Inputs:
    tokens: LongTensor [B, F], values in [0, K]
    z_values: optional FloatTensor [B, F], continuous preprocessed z in [0,1]

Output:
    cell_embeddings: FloatTensor [B, F, value_dim + feature_dim]

Design:
    V(cell) = V(value) || V(feature)

    V(value) = [z_continuous] || learnable_value_bin_embedding
        - first coordinate preserves continuous numeric signal when z_values
          is provided by build_token.py as X_*_z
        - if z_values is not provided, falls back to token / K
        - remaining coordinates are learnable but looked up through coarse bins
          so K does not force one rare embedding vector per token

    V(feature) = learnable feature identity embedding
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    import config as CFG
except Exception:
    CFG = None


class ValueFeatureConcatEmbedding(nn.Module):
    """
    Token + feature embedding with an explicit numeric value coordinate.

    tokens: [B, F]
    z_values: optional [B, F]
    output: [B, F, cell_dim]
    """

    def __init__(
        self,
        *,
        K: Optional[int] = None,
        n_features: int,
        value_dim: Optional[int] = None,
        feature_dim: Optional[int] = None,
        value_random_std: Optional[float] = None,
        value_num_bins: Optional[int] = None,
    ) -> None:
        super().__init__()

        if K is None:
            K = int(getattr(CFG, "TOKEN_K", 20000))
        if value_dim is None:
            value_dim = int(getattr(CFG, "VALUE_EMBED_DIM", 32))
        if feature_dim is None:
            feature_dim = int(getattr(CFG, "FEATURE_EMBED_DIM", 32))
        if value_random_std is None:
            value_random_std = float(getattr(CFG, "VALUE_RANDOM_STD", 0.02))
        if value_num_bins is None:
            value_num_bins = int(getattr(CFG, "VALUE_NUM_BINS", 128))

        if K <= 0:
            raise ValueError("K must be positive.")
        if n_features <= 0:
            raise ValueError("n_features must be positive.")
        if value_dim < 1:
            raise ValueError("value_dim must be >= 1 because the first coordinate is numeric z.")
        if feature_dim < 1:
            raise ValueError("feature_dim must be >= 1.")
        if value_num_bins is None or int(value_num_bins) <= 0:
            value_num_bins = int(K) + 1

        self.K = int(K)
        self.n_features = int(n_features)
        self.value_dim = int(value_dim)
        self.feature_dim = int(feature_dim)
        self.value_random_dim = int(value_dim - 1)
        self.cell_dim = int(value_dim + feature_dim)
        self.value_num_bins = int(min(max(int(value_num_bins), 1), self.K + 1))

        if self.value_random_dim > 0:
            self.value_random_embedding = nn.Embedding(self.value_num_bins, self.value_random_dim)
            nn.init.normal_(self.value_random_embedding.weight, mean=0.0, std=value_random_std)
        else:
            self.value_random_embedding = None

        self.feature_embedding = nn.Embedding(self.n_features, self.feature_dim)
        nn.init.normal_(self.feature_embedding.weight, mean=0.0, std=value_random_std)

        self.register_buffer(
            "default_feature_ids",
            torch.arange(self.n_features, dtype=torch.long),
            persistent=False,
        )

    def _numeric_z(self, tok: torch.Tensor, z_values: Optional[torch.Tensor]) -> torch.Tensor:
        if z_values is None:
            z = tok.float() / float(self.K)
        else:
            if z_values.ndim != 2:
                raise ValueError(f"z_values must have shape [B, F], got {tuple(z_values.shape)}")
            if tuple(z_values.shape) != tuple(tok.shape):
                raise ValueError(f"z_values shape must match tokens, got {tuple(z_values.shape)} vs {tuple(tok.shape)}")
            z = z_values.to(device=tok.device, dtype=torch.float32)
        return z.clamp(0.0, 1.0)

    def forward(
        self,
        tokens: torch.Tensor,
        z_values: Optional[torch.Tensor] = None,
        feature_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        tokens:
            LongTensor [B, F]

        z_values:
            optional FloatTensor [B, F]. If provided, this is used as the
            first monotonic coordinate and as the source for coarse bin lookup.
            This preserves pre-rounding numeric detail from X_*_z.

        feature_ids:
            optional LongTensor [F] or [B, F]. If None, use [0, 1, ..., F-1].
        """
        if tokens.ndim != 2:
            raise ValueError(f"tokens must have shape [B, F], got {tuple(tokens.shape)}")

        B, F = tokens.shape
        if F != self.n_features:
            raise ValueError(f"Expected F={self.n_features}, got F={F}")

        tok = tokens.long().clamp(0, self.K)
        z_scalar = self._numeric_z(tok, z_values)  # [B, F]
        z = z_scalar.unsqueeze(-1)  # [B, F, 1]

        if self.value_random_embedding is None:
            value_emb = z
        else:
            # If z_values is None and bins == K+1, this exactly matches the old
            # token-id lookup. Otherwise, multiple numeric positions share a
            # learnable bin while z still preserves continuous magnitude.
            if z_values is None and self.value_num_bins == self.K + 1:
                value_bin = tok
            else:
                value_bin = torch.round(z_scalar * float(self.value_num_bins - 1)).long()
                value_bin = value_bin.clamp(0, self.value_num_bins - 1)
            value_random = self.value_random_embedding(value_bin)
            value_emb = torch.cat([z, value_random], dim=-1)

        if feature_ids is None:
            fid = self.default_feature_ids.unsqueeze(0).expand(B, F)
        else:
            fid = feature_ids.long()
            if fid.ndim == 1:
                if fid.numel() != F:
                    raise ValueError(f"feature_ids length must be {F}, got {fid.numel()}")
                fid = fid.unsqueeze(0).expand(B, F)
            elif fid.ndim == 2:
                if tuple(fid.shape) != (B, F):
                    raise ValueError(f"feature_ids shape must be [B,F]={B,F}, got {tuple(fid.shape)}")
            else:
                raise ValueError("feature_ids must be None, [F], or [B,F].")
            fid = fid.clamp(0, self.n_features - 1).to(tok.device)

        feature_emb = self.feature_embedding(fid)
        return torch.cat([value_emb, feature_emb], dim=-1)

    def extra_repr(self) -> str:
        return (
            f"K={self.K}, n_features={self.n_features}, "
            f"value_dim={self.value_dim}, feature_dim={self.feature_dim}, "
            f"value_num_bins={self.value_num_bins}, cell_dim={self.cell_dim}"
        )


def smoke_test() -> None:
    emb = ValueFeatureConcatEmbedding(K=500, n_features=4, value_dim=8, feature_dim=8, value_num_bins=128)
    x = torch.tensor([[0, 62, 250, 500], [5, 62, 251, 499]], dtype=torch.long)
    z_cont = torch.tensor([[0.0, 0.12341, 0.5001, 1.0], [0.01, 0.12395, 0.5022, 0.998]], dtype=torch.float32)
    y_tok = emb(x)
    y_z = emb(x, z_values=z_cont)
    print("input tokens:", tuple(x.shape))
    print("output token-only:", tuple(y_tok.shape))
    print("output with z:", tuple(y_z.shape))
    print("cell_dim:", emb.cell_dim)
    print("value_num_bins:", emb.value_num_bins)
    print("token-only numeric coords sample0:", y_tok[0, :, 0].detach().cpu().tolist())
    print("continuous numeric coords sample0:", y_z[0, :, 0].detach().cpu().tolist())


if __name__ == "__main__":
    smoke_test()

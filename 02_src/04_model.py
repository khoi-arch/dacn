#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_pipeline/model.py

Tabular attention classifier.

Design:
    tokens [B, F] + optional z_values [B, F]
      -> embedding.py: ValueFeatureConcatEmbedding
      -> cell_emb [B, F, cell_dim]
      -> input projection cell_dim -> hidden_dim
      -> add learnable CLS token
      -> PyTorch TransformerEncoder across feature tokens
      -> take CLS output
      -> classifier head
      -> logits [B, num_classes]

Notes:
    - Attention is across features inside each sample, not across samples.
    - Uses PyTorch nn.TransformerEncoderLayer / nn.TransformerEncoder.
    - Does not apply softmax in forward. Training should use CrossEntropyLoss on logits.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    import config as CFG
except Exception:
    CFG = None

import importlib.util as _importlib_util
from pathlib import Path as _Path

_embedding_path = _Path(__file__).resolve().with_name("03_embedding.py")
_embedding_spec = _importlib_util.spec_from_file_location("_dacn_03_embedding", _embedding_path)
_embedding_mod = _importlib_util.module_from_spec(_embedding_spec)
assert _embedding_spec is not None and _embedding_spec.loader is not None
_embedding_spec.loader.exec_module(_embedding_mod)
ValueFeatureConcatEmbedding = _embedding_mod.ValueFeatureConcatEmbedding


def _cfg(name: str, default):
    if CFG is None:
        return default
    return getattr(CFG, name, default)


class TabularTransformerClassifier(nn.Module):
    """
    Tabular Transformer with feature-level self-attention and CLS pooling.

    Input:
        tokens: LongTensor [B, F]
        z_values: optional FloatTensor [B, F]

    Output:
        logits: FloatTensor [B, num_classes]

    Optional:
        return_info=True returns debug info with embeddings/CLS representation shapes.
    """

    def __init__(
        self,
        *,
        K: Optional[int] = None,
        n_features: int,
        num_classes: int,
        value_dim: Optional[int] = None,
        feature_dim: Optional[int] = None,
        value_num_bins: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
        dropout: Optional[float] = None,
        classifier_hidden_dim: Optional[int] = None,
        classifier_dropout: Optional[float] = None,
        norm_first: Optional[bool] = None,
        activation: Optional[str] = None,
    ) -> None:
        super().__init__()

        if K is None:
            K = int(_cfg("TOKEN_K", 20000))
        if value_dim is None:
            value_dim = int(_cfg("VALUE_EMBED_DIM", 32))
        if feature_dim is None:
            feature_dim = int(_cfg("FEATURE_EMBED_DIM", 32))
        if value_num_bins is None:
            value_num_bins = int(_cfg("VALUE_NUM_BINS", 512))
        if hidden_dim is None:
            hidden_dim = int(_cfg("MODEL_HIDDEN_DIM", 128))
        if num_layers is None:
            num_layers = int(_cfg("MODEL_NUM_LAYERS", 3))
        if num_heads is None:
            num_heads = int(_cfg("MODEL_NUM_HEADS", 4))
        if dropout is None:
            dropout = float(_cfg("MODEL_DROPOUT", 0.1))
        if classifier_hidden_dim is None:
            classifier_hidden_dim = int(_cfg("CLASSIFIER_HIDDEN_DIM", hidden_dim))
        if classifier_dropout is None:
            classifier_dropout = float(_cfg("CLASSIFIER_DROPOUT", dropout))
        if norm_first is None:
            norm_first = bool(_cfg("TRANSFORMER_NORM_FIRST", True))
        if activation is None:
            activation = str(_cfg("MODEL_ACTIVATION", "gelu"))

        if n_features <= 0:
            raise ValueError("n_features must be positive.")
        if num_classes <= 1:
            raise ValueError("num_classes must be > 1.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads, got "
                f"hidden_dim={hidden_dim}, num_heads={num_heads}"
            )
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.K = int(K)
        self.n_features = int(n_features)
        self.num_classes = int(num_classes)
        self.value_dim = int(value_dim)
        self.feature_dim = int(feature_dim)
        if value_num_bins is None or int(value_num_bins) <= 0:
            value_num_bins = int(K) + 1
        self.value_num_bins = int(min(max(int(value_num_bins), 1), int(K) + 1))
        self.cell_dim = int(value_dim + feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout_p = float(dropout)
        self.classifier_hidden_dim = int(classifier_hidden_dim)
        self.norm_first = bool(norm_first)
        self.activation = activation

        self.embedding = ValueFeatureConcatEmbedding(
            K=self.K,
            n_features=self.n_features,
            value_dim=self.value_dim,
            feature_dim=self.feature_dim,
            value_num_bins=self.value_num_bins,
        )

        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.cell_dim),
            nn.Linear(self.cell_dim, self.hidden_dim),
        )

        # Learnable CLS token in hidden space.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout_p,
            activation=self.activation,
            batch_first=True,
            norm_first=self.norm_first,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.hidden_dim),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(classifier_dropout)),
            nn.Linear(self.classifier_hidden_dim, self.num_classes),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        z_values: Optional[torch.Tensor] = None,
        *,
        return_info: bool = False,
    ):
        """
        tokens:
            LongTensor [B, F]

        z_values:
            optional FloatTensor [B, F] containing continuous preprocessed z.
            If provided, embedding uses this instead of token/K for the numeric
            coordinate and coarse value-bin lookup.

        return:
            logits [B, C]
            or (logits, info) if return_info=True
        """
        if tokens.ndim != 2:
            raise ValueError(f"tokens must have shape [B, F], got {tuple(tokens.shape)}")

        B, F = tokens.shape
        if F != self.n_features:
            raise ValueError(f"Expected F={self.n_features}, got F={F}")

        # [B, F, cell_dim]
        cell_emb = self.embedding(tokens, z_values=z_values)

        # [B, F, hidden_dim]
        x = self.input_proj(cell_emb)

        # [B, 1, hidden_dim]
        cls = self.cls_token.expand(B, 1, self.hidden_dim)

        # [B, F+1, hidden_dim]
        x = torch.cat([cls, x], dim=1)

        # Self-attention across CLS + feature tokens.
        encoded = self.encoder(x)

        # [B, hidden_dim]
        cls_out = encoded[:, 0, :]

        # [B, num_classes]
        logits = self.classifier(cls_out)

        if return_info:
            info: Dict[str, object] = {
                "cell_emb_shape": tuple(cell_emb.shape),
                "encoded_shape": tuple(encoded.shape),
                "cls_out_shape": tuple(cls_out.shape),
                "cell_dim": self.cell_dim,
                "value_num_bins": self.value_num_bins,
                "uses_continuous_z": bool(z_values is not None),
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "norm_first": self.norm_first,
            }
            return logits, info

        return logits

    def extra_repr(self) -> str:
        return (
            f"K={self.K}, n_features={self.n_features}, num_classes={self.num_classes}, "
            f"cell_dim={self.cell_dim}, value_num_bins={self.value_num_bins}, hidden_dim={self.hidden_dim}, "
            f"layers={self.num_layers}, heads={self.num_heads}, norm_first={self.norm_first}"
        )


def build_model_from_metadata(
    *,
    metadata: Dict[str, object],
    K: Optional[int] = None,
    **overrides,
) -> TabularTransformerClassifier:
    """
    Helper for train.py.

    metadata should be loaded from token_metadata_K{K}.json.
    It must contain:
        - n_features
        - label_mapping
    """
    n_features = int(metadata["n_features"])
    label_mapping = metadata["label_mapping"]
    num_classes = int(len(label_mapping))

    return TabularTransformerClassifier(
        K=K if K is not None else int(metadata.get("K", _cfg("TOKEN_K", 20000))),
        n_features=n_features,
        num_classes=num_classes,
        **overrides,
    )


def smoke_test() -> None:
    torch.manual_seed(7)

    B = 8
    F = 55
    K = 20000
    C = 16

    model = TabularTransformerClassifier(
        K=K,
        n_features=F,
        num_classes=C,
        value_dim=32,
        feature_dim=32,
        value_num_bins=512,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        norm_first=True,
    )

    x = torch.randint(low=0, high=K + 1, size=(B, F), dtype=torch.long)
    z = torch.clamp(x.float() / float(K) + 0.0001 * torch.randn(B, F), 0.0, 1.0)
    logits, info = model(x, z_values=z, return_info=True)

    print("input tokens:", tuple(x.shape))
    print("input z:", tuple(z.shape))
    print("logits:", tuple(logits.shape))
    print("info:", info)
    print("model:", model)


if __name__ == "__main__":
    smoke_test()
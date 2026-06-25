#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_train_absolute_distance_phase1.py

Phase 1: Absolute numerical distance_utilization

Runs supported:
  C2: mixed_bin + offset + raw_scaled, concat injection
  C3: mixed_bin + offset + z_preprocessed, projected+gated injection
  C4: mixed_bin + offset + raw_scaled, projected+gated injection

Fixed:
  - mixed bin artifact
  - offset artifact
  - Transformer backbone
  - train/eval helpers from 05_train.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import importlib.util
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import config as CFG


_train_path = Path(__file__).resolve().with_name("05_train.py")
_train_spec = importlib.util.spec_from_file_location("_dacn_05_train_helpers", _train_path)
_train_mod = importlib.util.module_from_spec(_train_spec)
assert _train_spec is not None and _train_spec.loader is not None
_train_spec.loader.exec_module(_train_mod)


def cfg(name: str, default):
    return getattr(CFG, name, default)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train phase 1 absolute numerical distance ablations.")

    p.add_argument("--run-id", choices=["C2", "C3", "C4"], required=True)
    p.add_argument("--K", type=int, default=int(cfg("TOKEN_K", 1000)))
    p.add_argument("--num-bins", type=int, default=int(cfg("VALUE_NUM_BINS", 128)))
    p.add_argument("--dataset-npz", default="")
    p.add_argument("--metadata-json", default="")
    p.add_argument("--out-root", default=str(cfg("OUTPUT_ROOT", Path("03_outputs")) / "train_runs_absolute_distance_phase1"))
    p.add_argument("--run-name", default="")

    p.add_argument("--train-raw", default="")
    p.add_argument("--val-raw", default="")
    p.add_argument("--raw-scale", choices=["minmax"], default="minmax")

    p.add_argument("--seed", type=int, default=int(cfg("TRAIN_SEED", 42)))
    p.add_argument("--device", default=str(cfg("TRAIN_DEVICE", "auto")))

    p.add_argument("--epochs", type=int, default=int(cfg("TRAIN_EPOCHS", 80)))
    p.add_argument("--batch-size", type=int, default=int(cfg("TRAIN_BATCH_SIZE", 256)))
    p.add_argument("--lr", type=float, default=float(cfg("TRAIN_LR", 1e-3)))
    p.add_argument("--weight-decay", type=float, default=float(cfg("TRAIN_WEIGHT_DECAY", 1e-4)))
    p.add_argument("--scheduler", choices=["none", "warmup_cosine"], default=str(cfg("TRAIN_SCHEDULER", "warmup_cosine")))
    p.add_argument("--warmup-epochs", type=int, default=int(cfg("TRAIN_WARMUP_EPOCHS", 8)))
    p.add_argument("--min-lr-ratio", type=float, default=float(cfg("TRAIN_MIN_LR_RATIO", 0.05)))
    p.add_argument("--patience", type=int, default=int(cfg("TRAIN_PATIENCE", 12)))
    p.add_argument("--min-delta", type=float, default=float(cfg("TRAIN_MIN_DELTA", 1e-4)))
    p.add_argument("--num-workers", type=int, default=int(cfg("TRAIN_NUM_WORKERS", 0)))
    p.add_argument("--grad-clip-norm", type=float, default=float(cfg("TRAIN_GRAD_CLIP_NORM", 1.0)))
    p.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=bool(cfg("USE_CLASS_WEIGHTS", True)))

    p.add_argument("--value-dim", type=int, default=int(cfg("VALUE_EMBED_DIM", 32)))
    p.add_argument("--feature-dim", type=int, default=int(cfg("FEATURE_EMBED_DIM", 32)))
    p.add_argument("--hidden-dim", type=int, default=int(cfg("MODEL_HIDDEN_DIM", 128)))
    p.add_argument("--num-layers", type=int, default=int(cfg("MODEL_NUM_LAYERS", 3)))
    p.add_argument("--num-heads", type=int, default=int(cfg("MODEL_NUM_HEADS", 4)))
    p.add_argument("--dropout", type=float, default=float(cfg("MODEL_DROPOUT", 0.1)))
    p.add_argument("--classifier-hidden-dim", type=int, default=int(cfg("CLASSIFIER_HIDDEN_DIM", 128)))
    p.add_argument("--classifier-dropout", type=float, default=float(cfg("CLASSIFIER_DROPOUT", 0.1)))
    p.add_argument("--norm-first", action=argparse.BooleanOptionalAction, default=bool(cfg("TRANSFORMER_NORM_FIRST", True)))

    # Gated branch only. Keep conservative init: gate starts near 0.5.
    p.add_argument("--gate-init", type=float, default=0.0)
    return p.parse_args()


def run_spec(run_id: str) -> Dict[str, str]:
    specs = {
        "C2": {"continuous_source": "raw_scaled", "injection": "concat"},
        "C3": {"continuous_source": "z_preprocessed", "injection": "feature_project_gate"},
        "C4": {"continuous_source": "raw_scaled", "injection": "feature_project_gate"},
    }
    return specs[run_id]


def default_dataset_path(K: int, B: int) -> Path:
    return cfg("OUTPUT_ROOT", Path("03_outputs")) / "build_mixed_quantile_offset" / f"K{K}_B{B}" / "mixed_quantile_offset_dataset.npz"


def default_metadata_path(K: int, B: int) -> Path:
    return cfg("OUTPUT_ROOT", Path("03_outputs")) / "build_mixed_quantile_offset" / f"K{K}_B{B}" / "mixed_quantile_offset_metadata.json"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path_from_meta: str | Path, fallback_relative: str) -> Path:
    p = Path(path_from_meta)
    if p.exists():
        return p

    root = repo_root()
    fallback = root / fallback_relative
    if fallback.exists():
        return fallback

    parts = list(p.parts)
    if "03_outputs" in parts:
        idx = parts.index("03_outputs")
        rel = Path(*parts[idx:])
        candidate = root / rel
        if candidate.exists():
            return candidate

    if "01_split" in parts:
        idx = parts.index("01_split")
        rel = Path(*parts[idx:])
        candidate = root / rel
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not resolve path. metadata_path={p}, fallback={fallback}")


def load_dataset(dataset_path: Path, metadata_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    if not dataset_path.exists():
        raise FileNotFoundError(str(dataset_path))
    if not metadata_path.exists():
        raise FileNotFoundError(str(metadata_path))

    data = dict(np.load(dataset_path, allow_pickle=True))
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))

    required = ["X_train_bin", "X_train_offset", "y_train", "X_val_bin", "X_val_offset", "y_val"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"dataset missing arrays: {missing}")

    return data, meta


def load_z_preprocessed(meta: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    feature_names = [str(x) for x in meta["feature_names"]]

    train_path = resolve_repo_path(
        meta["input"]["train_preprocessed"],
        "03_outputs/preprocessing/train_preprocessed_K1000.csv",
    )
    val_path = resolve_repo_path(
        meta["input"]["val_preprocessed"],
        "03_outputs/preprocessing/val_preprocessed_K1000.csv",
    )

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    missing_train = [f for f in feature_names if f not in train.columns]
    missing_val = [f for f in feature_names if f not in val.columns]
    if missing_train:
        raise ValueError(f"train_preprocessed missing features: {missing_train[:10]}")
    if missing_val:
        raise ValueError(f"val_preprocessed missing features: {missing_val[:10]}")

    X_train = train.loc[:, feature_names].to_numpy(dtype=np.float32)
    X_val = val.loc[:, feature_names].to_numpy(dtype=np.float32)

    X_train = np.clip(X_train, 0.0, 1.0).astype(np.float32)
    X_val = np.clip(X_val, 0.0, 1.0).astype(np.float32)

    info = {
        "source": "z_preprocessed",
        "train_path": str(train_path),
        "val_path": str(val_path),
        "scale": "already_preprocessed_to_0_1",
        "train_min": float(X_train.min()),
        "train_max": float(X_train.max()),
        "val_min": float(X_val.min()),
        "val_max": float(X_val.max()),
    }
    return X_train, X_val, info


def resolve_raw_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.train_raw:
        train_path = Path(args.train_raw)
    else:
        train_path = repo_root() / "01_split" / "train_raw.csv"

    if args.val_raw:
        val_path = Path(args.val_raw)
    else:
        val_path = repo_root() / "01_split" / "val_raw.csv"

    if not train_path.exists():
        train_path = resolve_repo_path(train_path, "01_split/train_raw.csv")
    if not val_path.exists():
        val_path = resolve_repo_path(val_path, "01_split/val_raw.csv")

    return train_path, val_path


def load_raw_scaled(meta: Dict[str, object], args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    feature_names = [str(x) for x in meta["feature_names"]]
    train_path, val_path = resolve_raw_paths(args)

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    missing_train = [f for f in feature_names if f not in train.columns]
    missing_val = [f for f in feature_names if f not in val.columns]
    if missing_train:
        raise ValueError(f"train_raw missing features: {missing_train[:10]}")
    if missing_val:
        raise ValueError(f"val_raw missing features: {missing_val[:10]}")

    X_train_raw = train.loc[:, feature_names].to_numpy(dtype=np.float64)
    X_val_raw = val.loc[:, feature_names].to_numpy(dtype=np.float64)

    if np.isnan(X_train_raw).any() or np.isinf(X_train_raw).any():
        raise ValueError("train_raw contains NaN/Inf in selected features")
    if np.isnan(X_val_raw).any() or np.isinf(X_val_raw).any():
        raise ValueError("val_raw contains NaN/Inf in selected features")

    mn = X_train_raw.min(axis=0)
    mx = X_train_raw.max(axis=0)
    denom = mx - mn
    constant = np.isclose(denom, 0.0)

    denom_safe = denom.copy()
    denom_safe[constant] = 1.0

    X_train = (X_train_raw - mn) / denom_safe
    X_val = (X_val_raw - mn) / denom_safe

    X_train[:, constant] = 0.5
    X_val[:, constant] = 0.5

    X_train = np.clip(X_train, 0.0, 1.0).astype(np.float32)
    X_val = np.clip(X_val, 0.0, 1.0).astype(np.float32)

    info = {
        "source": "raw_scaled",
        "train_path": str(train_path),
        "val_path": str(val_path),
        "scale": "train_only_minmax_linear_clip_val",
        "n_constant_features": int(constant.sum()),
        "constant_features": [feature_names[i] for i, c in enumerate(constant) if bool(c)],
        "train_min": float(X_train.min()),
        "train_max": float(X_train.max()),
        "val_min": float(X_val.min()),
        "val_max": float(X_val.max()),
    }
    return X_train, X_val, info


def load_continuous(meta: Dict[str, object], args: argparse.Namespace, source: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    if source == "z_preprocessed":
        return load_z_preprocessed(meta)
    if source == "raw_scaled":
        return load_raw_scaled(meta, args)
    raise ValueError(f"unknown continuous source: {source}")


class Phase1Dataset(Dataset):
    """
    Returns:
      X_bin: [F]
      values: [F, 2], values[...,0] = offset, values[...,1] = continuous
      y
    """

    def __init__(self, X_bin: np.ndarray, X_offset: np.ndarray, X_cont: np.ndarray, y: np.ndarray) -> None:
        if X_bin.ndim != 2:
            raise ValueError(f"X_bin must be [N,F], got {X_bin.shape}")
        if X_offset.ndim != 2 or X_cont.ndim != 2:
            raise ValueError(f"X_offset/X_cont must be [N,F], got {X_offset.shape}/{X_cont.shape}")
        if X_bin.shape != X_offset.shape or X_bin.shape != X_cont.shape:
            raise ValueError(f"shape mismatch: bin={X_bin.shape}, offset={X_offset.shape}, cont={X_cont.shape}")
        if y.ndim != 1 or y.shape[0] != X_bin.shape[0]:
            raise ValueError(f"y mismatch: {y.shape} vs {X_bin.shape}")

        pair = np.stack([X_offset.astype(np.float32), X_cont.astype(np.float32)], axis=-1)

        self.X = torch.as_tensor(X_bin, dtype=torch.long)
        self.V = torch.as_tensor(pair, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.V[idx], self.y[idx]


class ConcatValueEmbedding(nn.Module):
    """
    C2 style:
      value_emb = [offset, continuous] || Emb(bin_id)
      cell_emb  = value_emb || Emb(feature_id)
    """

    def __init__(self, *, num_bins: int, n_features: int, value_dim: int, feature_dim: int, init_std: float = 0.02):
        super().__init__()
        if value_dim < 3:
            raise ValueError("value_dim must be >= 3 for [offset, continuous] + bin emb")
        self.num_bins = int(num_bins)
        self.n_features = int(n_features)
        self.value_dim = int(value_dim)
        self.feature_dim = int(feature_dim)
        self.bin_emb_dim = int(value_dim - 2)
        self.cell_dim = int(value_dim + feature_dim)

        self.bin_embedding = nn.Embedding(self.num_bins, self.bin_emb_dim)
        self.feature_embedding = nn.Embedding(self.n_features, self.feature_dim)
        nn.init.normal_(self.bin_embedding.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.feature_embedding.weight, mean=0.0, std=init_std)

        self.register_buffer("default_feature_ids", torch.arange(self.n_features, dtype=torch.long), persistent=False)

    def forward(self, bin_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != 2:
            raise ValueError(f"values must be [B,F,2], got {tuple(values.shape)}")

        B, F = bin_ids.shape
        b = bin_ids.long().clamp(0, self.num_bins - 1)
        numeric = values.to(device=b.device, dtype=torch.float32).clamp(0.0, 1.0)

        bin_emb = self.bin_embedding(b)
        value_emb = torch.cat([numeric, bin_emb], dim=-1)

        fid = self.default_feature_ids.unsqueeze(0).expand(B, F)
        feature_emb = self.feature_embedding(fid)
        return torch.cat([value_emb, feature_emb], dim=-1)


class ProjectGateValueEmbedding(nn.Module):
    """
    C3/C4 style:
      bin_emb      = Emb(bin_id)
      offset_part  = [offset] kept as scalar, to keep C0/R3 offset semantics stable
      cont_base    = Linear([continuous]) -> bin_emb_dim
      cont_gate_f  = sigmoid(learnable gate per feature)
      value_emb    = [offset] || (bin_emb + gate_f * cont_base)
      cell_emb     = value_emb || Emb(feature_id)

    Note:
      Offset branch is intentionally not redesigned in Phase 1.
      Phase 1 only tests continuous source and continuous injection.
    """

    def __init__(
        self,
        *,
        num_bins: int,
        n_features: int,
        value_dim: int,
        feature_dim: int,
        gate_init: float = 0.0,
        init_std: float = 0.02,
    ):
        super().__init__()
        if value_dim < 3:
            raise ValueError("value_dim must be >= 3")
        self.num_bins = int(num_bins)
        self.n_features = int(n_features)
        self.value_dim = int(value_dim)
        self.feature_dim = int(feature_dim)
        self.bin_emb_dim = int(value_dim - 1)  # one scalar offset + bin/continuous semantic space
        self.cell_dim = int(value_dim + feature_dim)

        self.bin_embedding = nn.Embedding(self.num_bins, self.bin_emb_dim)
        self.feature_embedding = nn.Embedding(self.n_features, self.feature_dim)
        self.continuous_proj = nn.Linear(1, self.bin_emb_dim)

        # Per-feature gate. sigmoid(0)=0.5 by default.
        self.continuous_gate_logit = nn.Parameter(torch.full((self.n_features, 1), float(gate_init)))

        nn.init.normal_(self.bin_embedding.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.feature_embedding.weight, mean=0.0, std=init_std)
        nn.init.xavier_uniform_(self.continuous_proj.weight)
        nn.init.zeros_(self.continuous_proj.bias)

        self.register_buffer("default_feature_ids", torch.arange(self.n_features, dtype=torch.long), persistent=False)

    def forward(self, bin_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != 2:
            raise ValueError(f"values must be [B,F,2], got {tuple(values.shape)}")

        B, F = bin_ids.shape
        b = bin_ids.long().clamp(0, self.num_bins - 1)

        vals = values.to(device=b.device, dtype=torch.float32).clamp(0.0, 1.0)
        offset = vals[..., 0:1]
        cont = vals[..., 1:2]

        bin_emb = self.bin_embedding(b)
        cont_emb = self.continuous_proj(cont)

        gate = torch.sigmoid(self.continuous_gate_logit).to(device=b.device)  # [F,1]
        gate = gate.unsqueeze(0).expand(B, F, 1)

        semantic_value = bin_emb + gate * cont_emb
        value_emb = torch.cat([offset, semantic_value], dim=-1)

        fid = self.default_feature_ids.unsqueeze(0).expand(B, F)
        feature_emb = self.feature_embedding(fid)
        return torch.cat([value_emb, feature_emb], dim=-1)

    def gate_summary(self) -> Dict[str, float]:
        with torch.no_grad():
            g = torch.sigmoid(self.continuous_gate_logit.detach().cpu()).numpy()
        return {
            "gate_min": float(g.min()),
            "gate_max": float(g.max()),
            "gate_mean": float(g.mean()),
            "gate_std": float(g.std()),
        }


class Phase1TransformerClassifier(nn.Module):
    def __init__(
        self,
        *,
        injection: str,
        num_bins: int,
        n_features: int,
        num_classes: int,
        value_dim: int,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        classifier_hidden_dim: int,
        classifier_dropout: float,
        norm_first: bool,
        gate_init: float,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim must be divisible by num_heads: {hidden_dim}/{num_heads}")

        self.injection = str(injection)
        self.num_bins = int(num_bins)
        self.n_features = int(n_features)
        self.num_classes = int(num_classes)
        self.value_dim = int(value_dim)
        self.feature_dim = int(feature_dim)
        self.cell_dim = int(value_dim + feature_dim)
        self.hidden_dim = int(hidden_dim)

        if self.injection == "concat":
            self.embedding = ConcatValueEmbedding(
                num_bins=num_bins,
                n_features=n_features,
                value_dim=value_dim,
                feature_dim=feature_dim,
            )
        elif self.injection == "feature_project_gate":
            self.embedding = ProjectGateValueEmbedding(
                num_bins=num_bins,
                n_features=n_features,
                value_dim=value_dim,
                feature_dim=feature_dim,
                gate_init=gate_init,
            )
        else:
            raise ValueError(f"unknown injection: {self.injection}")

        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.cell_dim),
            nn.Linear(self.cell_dim, self.hidden_dim),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.hidden_dim),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, int(classifier_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(classifier_dropout)),
            nn.Linear(int(classifier_hidden_dim), int(num_classes)),
        )

    def forward(self, tokens: torch.Tensor, z_values: Optional[torch.Tensor] = None, *, return_info: bool = False):
        if z_values is None:
            raise ValueError("z_values must contain [offset, continuous].")

        cell_emb = self.embedding(tokens, z_values)
        x = self.input_proj(cell_emb)

        B = x.shape[0]
        cls = self.cls_token.expand(B, 1, self.hidden_dim)
        x = torch.cat([cls, x], dim=1)

        encoded = self.encoder(x)
        cls_out = encoded[:, 0, :]
        logits = self.classifier(cls_out)

        if return_info:
            return logits, {
                "cell_emb_shape": tuple(cell_emb.shape),
                "encoded_shape": tuple(encoded.shape),
                "cls_out_shape": tuple(cls_out.shape),
                "cell_dim": self.cell_dim,
                "num_bins": self.num_bins,
                "uses_offset": True,
                "uses_continuous": True,
                "injection": self.injection,
            }
        return logits

    def embedding_extra_summary(self) -> Dict[str, object]:
        if hasattr(self.embedding, "gate_summary"):
            return self.embedding.gate_summary()
        return {}


def strip_eval(report: Dict[str, object]) -> Dict[str, object]:
    skip = {"y_true", "y_pred", "confidence", "probs"}
    return {k: v for k, v in report.items() if k not in skip}


def malware_avg_f1(report: Dict[str, object]) -> float:
    vals = []
    for label, m in report["per_class"].items():
        if str(label).strip().lower() != "benign":
            vals.append(float(m["f1"]))
    return float(np.mean(vals)) if vals else 0.0


def make_diagnosis_summary(
    *,
    run_id: str,
    run_spec_obj: Dict[str, str],
    best_epoch: int,
    best_train: Dict[str, object],
    best_val: Dict[str, object],
    metadata: Dict[str, object],
    continuous_info: Dict[str, object],
    model: Phase1TransformerClassifier,
    args: argparse.Namespace,
) -> Dict[str, object]:
    train_macro = float(best_train["macro_f1"])
    val_macro = float(best_val["macro_f1"])

    worst_classes = sorted(
        [
            {
                "label": str(label).strip(),
                "f1": float(m["f1"]),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "support": int(m["support"]),
            }
            for label, m in best_val["per_class"].items()
        ],
        key=lambda x: (x["f1"], -x["support"]),
    )[:10]

    return {
        "phase": "Absolute numerical distance_utilization Phase 1",
        "run_id": run_id,
        "best_epoch": int(best_epoch),
        "representation": f"mixed_bin_offset_{run_spec_obj['continuous_source']}_{run_spec_obj['injection']}",
        "continuous_source": run_spec_obj["continuous_source"],
        "injection": run_spec_obj["injection"],
        "train": {
            "loss": float(best_train["loss"]),
            "accuracy": float(best_train["accuracy"]),
            "macro_f1": train_macro,
            "weighted_f1": float(best_train["weighted_f1"]),
            "malware_only_avg_f1": malware_avg_f1(best_train),
        },
        "val": {
            "loss": float(best_val["loss"]),
            "accuracy": float(best_val["accuracy"]),
            "macro_f1": val_macro,
            "weighted_f1": float(best_val["weighted_f1"]),
            "malware_only_avg_f1": malware_avg_f1(best_val),
        },
        "generalization_gap_macro_f1": float(train_macro - val_macro),
        "worst_val_classes_by_f1": worst_classes,
        "model_config": {
            "value_dim": int(args.value_dim),
            "feature_dim": int(args.feature_dim),
            "cell_dim": int(args.value_dim + args.feature_dim),
            "num_bins": int(args.num_bins),
            "effective_token_budget": int(args.num_bins),
            "uses_bin_id": True,
            "uses_offset": True,
            "uses_continuous": True,
            "continuous_source": run_spec_obj["continuous_source"],
            "injection": run_spec_obj["injection"],
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "dropout": float(args.dropout),
            "classifier_hidden_dim": int(args.classifier_hidden_dim),
            "classifier_dropout": float(args.classifier_dropout),
            "norm_first": bool(args.norm_first),
            "use_class_weights": bool(args.use_class_weights),
            "scheduler": str(args.scheduler),
            "gate_init": float(args.gate_init),
        },
        "continuous_info": continuous_info,
        "embedding_extra_summary": model.embedding_extra_summary(),
        "strategy_counts": metadata.get("strategy_counts", {}),
        "baseline_to_compare": {
            "C0_mixed_quantile_offset_no_continuous": {
                "train_macro_f1": 0.9321426589148848,
                "val_macro_f1": 0.8133554660345561,
                "gap": 0.11878719288032868,
            },
            "C1_mixed_quantile_offset_z_concat": {
                "train_macro_f1": 0.9076760392371152,
                "val_macro_f1": 0.8056568664958441,
                "gap": 0.10201917274127104,
            },
            "R0_uniform_bin_global_z_original_baseline": {
                "train_macro_f1": 0.8891,
                "val_macro_f1": 0.7952,
                "gap": 0.0939,
            },
        },
    }


def main() -> None:
    args = parse_args()
    _train_mod.set_seed(int(args.seed))

    spec = run_spec(args.run_id)
    K = int(args.K)
    B = int(args.num_bins)

    dataset_path = Path(args.dataset_npz) if args.dataset_npz else default_dataset_path(K, B)
    metadata_path = Path(args.metadata_json) if args.metadata_json else default_metadata_path(K, B)

    data, meta = load_dataset(dataset_path, metadata_path)
    X_train_cont, X_val_cont, continuous_info = load_continuous(meta, args, spec["continuous_source"])

    X_train = data["X_train_bin"].astype(np.int64)
    O_train = data["X_train_offset"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)

    X_val = data["X_val_bin"].astype(np.int64)
    O_val = data["X_val_offset"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)

    if X_train.shape != X_train_cont.shape:
        raise ValueError(f"train shape mismatch: bin={X_train.shape}, continuous={X_train_cont.shape}")
    if X_val.shape != X_val_cont.shape:
        raise ValueError(f"val shape mismatch: bin={X_val.shape}, continuous={X_val_cont.shape}")

    label_mapping = meta["label_mapping"]
    label_names = [label for label, _ in sorted(label_mapping.items(), key=lambda kv: kv[1])]
    num_classes = int(len(label_names))
    n_features = int(meta["n_features"])

    run_name = args.run_name
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_{args.run_id}_{spec['continuous_source']}_{spec['injection']}_Keff{B}_seed{args.seed}"

    out_dir = Path(args.out_root) / f"Keff{B}" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _train_mod.pick_device(str(args.device))

    config_obj = {
        "stage": "train_absolute_distance_phase1",
        "phase": "Absolute numerical distance_utilization Phase 1",
        "run_id": args.run_id,
        "run_spec": spec,
        "K_artifact": K,
        "effective_token_budget": B,
        "num_bins": B,
        "dataset_npz": str(dataset_path),
        "metadata_json": str(metadata_path),
        "out_dir": str(out_dir),
        "continuous_info": continuous_info,
        "seed": int(args.seed),
        "device": str(device),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "scheduler": str(args.scheduler),
        "warmup_epochs": int(args.warmup_epochs),
        "min_lr_ratio": float(args.min_lr_ratio),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "grad_clip_norm": float(args.grad_clip_norm),
        "use_class_weights": bool(args.use_class_weights),
        "n_features": n_features,
        "num_classes": num_classes,
        "label_names": label_names,
        "representation": {
            "name": f"mixed_bin_offset_{spec['continuous_source']}_{spec['injection']}",
            "uses_bin_id": True,
            "uses_offset": True,
            "uses_continuous": True,
            "continuous_source": spec["continuous_source"],
            "injection": spec["injection"],
            "numeric_channels": ["offset", spec["continuous_source"]],
            "strategy_counts": meta.get("strategy_counts", {}),
        },
        "model": {
            "value_dim": int(args.value_dim),
            "feature_dim": int(args.feature_dim),
            "cell_dim": int(args.value_dim + args.feature_dim),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "dropout": float(args.dropout),
            "classifier_hidden_dim": int(args.classifier_hidden_dim),
            "classifier_dropout": float(args.classifier_dropout),
            "norm_first": bool(args.norm_first),
            "gate_init": float(args.gate_init),
        },
        "data_shapes": {
            "X_train_bin": list(X_train.shape),
            "X_train_offset": list(O_train.shape),
            "X_train_continuous": list(X_train_cont.shape),
            "y_train": list(y_train.shape),
            "X_val_bin": list(X_val.shape),
            "X_val_offset": list(O_val.shape),
            "X_val_continuous": list(X_val_cont.shape),
            "y_val": list(y_val.shape),
        },
        "torch_version": torch.__version__,
    }
    _train_mod.save_json(out_dir / "config.json", config_obj)

    train_ds = Phase1Dataset(X_train, O_train, X_train_cont, y_train)
    val_ds = Phase1Dataset(X_val, O_val, X_val_cont, y_val)

    generator = torch.Generator()
    generator.manual_seed(int(args.seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = Phase1TransformerClassifier(
        injection=spec["injection"],
        num_bins=B,
        n_features=n_features,
        num_classes=num_classes,
        value_dim=int(args.value_dim),
        feature_dim=int(args.feature_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        classifier_hidden_dim=int(args.classifier_hidden_dim),
        classifier_dropout=float(args.classifier_dropout),
        norm_first=bool(args.norm_first),
        gate_init=float(args.gate_init),
        activation=str(cfg("MODEL_ACTIVATION", "gelu")),
    ).to(device)

    if args.use_class_weights:
        weights = _train_mod.compute_class_weights(y_train, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        class_weights_log = weights.detach().cpu().numpy().tolist()
    else:
        criterion = nn.CrossEntropyLoss()
        class_weights_log = None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    _train_mod.save_json(out_dir / "class_weights.json", {
        "use_class_weights": bool(args.use_class_weights),
        "class_weights": class_weights_log,
        "label_names": label_names,
        "train_counts": np.bincount(y_train, minlength=num_classes).astype(int).tolist(),
        "val_counts": np.bincount(y_val, minlength=num_classes).astype(int).tolist(),
    })

    print("===== absolute distance phase1 training start =====")
    print(f"run_id: {args.run_id}")
    print(f"spec: {spec}")
    print(f"run_dir: {out_dir}")
    print(f"device: {device}")
    print(f"dataset: {dataset_path}")
    print(f"continuous_info: {continuous_info}")
    print(f"train bin/offset/continuous shape: {X_train.shape}/{O_train.shape}/{X_train_cont.shape}")
    print(f"val bin/offset/continuous shape:   {X_val.shape}/{O_val.shape}/{X_val_cont.shape}")
    print(f"classes: {num_classes} {label_names}")
    print(f"strategy_counts: {meta.get('strategy_counts', {})}")

    history: List[Dict[str, object]] = []
    best_metric = -math.inf
    best_epoch = -1
    best_train_eval = None
    best_val_eval = None
    bad_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()

        lr_epoch = _train_mod.compute_epoch_lr(
            base_lr=float(args.lr),
            epoch=epoch,
            total_epochs=int(args.epochs),
            scheduler_name=str(args.scheduler),
            warmup_epochs=int(args.warmup_epochs),
            min_lr_ratio=float(args.min_lr_ratio),
        )
        _train_mod.set_optimizer_lr(optimizer, lr_epoch)

        train_step_loss = _train_mod.train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=float(args.grad_clip_norm),
        )

        train_eval = _train_mod.evaluate(model, train_eval_loader, criterion, device, num_classes, label_names)
        val_eval = _train_mod.evaluate(model, val_loader, criterion, device, num_classes, label_names)

        row = {
            "epoch": epoch,
            "lr": float(lr_epoch),
            "epoch_seconds": round(time.time() - t0, 3),
            "train_step_loss": float(train_step_loss),
            "train_loss": float(train_eval["loss"]),
            "train_acc": float(train_eval["accuracy"]),
            "train_macro_f1": float(train_eval["macro_f1"]),
            "train_weighted_f1": float(train_eval["weighted_f1"]),
            "train_malware_avg_f1": malware_avg_f1(train_eval),
            "val_loss": float(val_eval["loss"]),
            "val_acc": float(val_eval["accuracy"]),
            "val_macro_f1": float(val_eval["macro_f1"]),
            "val_weighted_f1": float(val_eval["weighted_f1"]),
            "val_malware_avg_f1": malware_avg_f1(val_eval),
            "macro_f1_gap_train_minus_val": float(train_eval["macro_f1"] - val_eval["macro_f1"]),
        }
        row.update(model.embedding_extra_summary())
        history.append(row)
        _train_mod.write_history_csv(out_dir / "history.csv", history)

        metric = float(val_eval["macro_f1"])
        improved = metric > best_metric + float(args.min_delta)

        if improved:
            best_metric = metric
            best_epoch = epoch
            best_train_eval = train_eval
            best_val_eval = val_eval
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_macro_f1": best_metric,
                    "config": config_obj,
                    "metadata": meta,
                    "continuous_info": continuous_info,
                },
                out_dir / "best_model.pt",
            )
        else:
            bad_epochs += 1

        print(
            f"[{args.run_id} epoch {epoch:03d}] "
            f"lr={lr_epoch:.6g} "
            f"train_macro={row['train_macro_f1']:.4f} "
            f"train_malware={row['train_malware_avg_f1']:.4f} "
            f"val_macro={row['val_macro_f1']:.4f} "
            f"val_malware={row['val_malware_avg_f1']:.4f} "
            f"gap={row['macro_f1_gap_train_minus_val']:.4f} "
            f"best={best_metric:.4f}@{best_epoch}"
        )

        if bad_epochs >= int(args.patience):
            print(f"early stop: bad_epochs={bad_epochs}, patience={args.patience}")
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": int(history[-1]["epoch"]) if history else -1,
            "config": config_obj,
            "metadata": meta,
            "continuous_info": continuous_info,
        },
        out_dir / "last_model.pt",
    )

    if best_train_eval is None or best_val_eval is None:
        raise RuntimeError("No best eval was recorded.")

    train_report = strip_eval(best_train_eval)
    val_report = strip_eval(best_val_eval)

    _train_mod.save_json(out_dir / "train_classification_report_best.json", train_report)
    _train_mod.save_json(out_dir / "val_classification_report_best.json", val_report)

    _train_mod.write_confusion_outputs(out_dir, "train", best_train_eval, label_names)
    _train_mod.write_confusion_outputs(out_dir, "val", best_val_eval, label_names)

    _train_mod.write_predictions_csv(
        out_dir / "val_predictions_best.csv",
        best_val_eval["y_true"],
        best_val_eval["y_pred"],
        best_val_eval["confidence"],
        label_names,
    )

    diagnosis = make_diagnosis_summary(
        run_id=args.run_id,
        run_spec_obj=spec,
        best_epoch=best_epoch,
        best_train=best_train_eval,
        best_val=best_val_eval,
        metadata=meta,
        continuous_info=continuous_info,
        model=model,
        args=args,
    )
    _train_mod.save_json(out_dir / "diagnosis_summary.json", diagnosis)

    print("===== absolute distance phase1 training done =====")
    print(f"run_id:                {args.run_id}")
    print(f"representation:        {diagnosis['representation']}")
    print(f"best_epoch:            {best_epoch}")
    print(f"train_macro_f1:        {diagnosis['train']['macro_f1']:.6f}")
    print(f"train_malware_avg_f1: {diagnosis['train']['malware_only_avg_f1']:.6f}")
    print(f"val_macro_f1:          {diagnosis['val']['macro_f1']:.6f}")
    print(f"val_malware_avg_f1:   {diagnosis['val']['malware_only_avg_f1']:.6f}")
    print(f"gap:                   {diagnosis['generalization_gap_macro_f1']:.6f}")
    print(f"diagnosis:             {out_dir / 'diagnosis_summary.json'}")


if __name__ == "__main__":
    main()

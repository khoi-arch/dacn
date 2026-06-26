#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_train_fusion_ablation_D0_D7.py

Overnight ablation at fixed K_effective=128.

Runs:
  D0: offset interpolation only
  D1: offset interpolation + raw scalar concat
  D2: offset interpolation + raw projected gated concat-fusion
  D3: offset interpolation + raw FiLM/multiply fusion
  D4: offset interpolation + z projected gated concat-fusion
  D5: offset interpolation + raw selective wide/tail projected gated concat-fusion
  D6: bin embedding + projected offset add, no continuous
  D7: bin embedding + projected offset concat-fusion, no continuous

Purpose:
  This is a fixed-setting representation ablation. It does NOT claim robustness
  across K/token-utilization settings. It selects candidates at K_effective=128
  for later K/preprocess interaction checks.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
    p = argparse.ArgumentParser(description="Train D0-D7 fusion ablations + E1/E2 smooth numeric embedding tests.")

    p.add_argument("--run-id", choices=["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "E1", "E2"], required=True)
    p.add_argument("--K", type=int, default=int(cfg("TOKEN_K", 1000)))
    p.add_argument("--num-bins", type=int, default=int(cfg("VALUE_NUM_BINS", 128)))
    p.add_argument("--dataset-npz", default="")
    p.add_argument("--metadata-json", default="")
    p.add_argument("--out-root", default=str(cfg("OUTPUT_ROOT", Path("03_outputs")) / "train_runs_fusion_ablation_D0_D7"))
    p.add_argument("--run-name", default="")

    p.add_argument("--train-raw", default="")
    p.add_argument("--val-raw", default="")

    # D5 selective continuous parameters; distribution-only, no labels.
    p.add_argument("--tail-frac", type=float, default=0.10)
    p.add_argument("--wide-quantile", type=float, default=0.90)

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

    # sigmoid(0)=0.5. Conservative default; model can move it up/down per feature.
    p.add_argument("--gate-init", type=float, default=0.0)
    p.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Build data/model diagnostics and exit before training. Useful for checking boundary collapse and scale diagnostics.",
    )

    return p.parse_args()


RUN_SPECS: Dict[str, Dict[str, str]] = {
    "D0": {
        "local": "offset_interpolation",
        "continuous_source": "none",
        "fusion": "none",
        "description": "offset interpolation only",
    },
    "D1": {
        "local": "offset_interpolation",
        "continuous_source": "raw_scaled",
        "fusion": "raw_scalar_concat",
        "description": "offset interpolation + raw scalar concat",
    },
    "D2": {
        "local": "offset_interpolation",
        "continuous_source": "raw_scaled",
        "fusion": "raw_project_gate_concat",
        "description": "offset interpolation + raw projected gated concat-fusion",
    },
    "D3": {
        "local": "offset_interpolation",
        "continuous_source": "raw_scaled",
        "fusion": "raw_film",
        "description": "offset interpolation + raw FiLM/multiply fusion",
    },
    "D4": {
        "local": "offset_interpolation",
        "continuous_source": "z_preprocessed",
        "fusion": "z_project_gate_concat",
        "description": "offset interpolation + z projected gated concat-fusion",
    },
    "D5": {
        "local": "offset_interpolation",
        "continuous_source": "raw_scaled",
        "fusion": "raw_selective_project_gate_concat",
        "description": "offset interpolation + raw selective wide/tail projected gated concat-fusion",
    },
    "D6": {
        "local": "projected_offset_add",
        "continuous_source": "none",
        "fusion": "none",
        "description": "bin embedding + projected offset add",
    },
    "D7": {
        "local": "projected_offset_concat",
        "continuous_source": "none",
        "fusion": "none",
        "description": "bin embedding + projected offset concat-fusion",
    },
    "E1": {
        "local": "smooth_position_mlp",
        "continuous_source": "none",
        "fusion": "none",
        "description": "smooth ordinal position MLP from (bin + offset) / num_bins",
    },
    "E2": {
        "local": "smooth_position_mlp",
        "continuous_source": "raw_scaled",
        "fusion": "raw_film",
        "description": "smooth ordinal position MLP + raw_scaled FiLM",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_dataset_path(K: int, B: int) -> Path:
    return cfg("OUTPUT_ROOT", Path("03_outputs")) / "build_mixed_quantile_offset" / f"K{K}_B{B}" / "mixed_quantile_offset_dataset.npz"


def default_metadata_path(K: int, B: int) -> Path:
    return cfg("OUTPUT_ROOT", Path("03_outputs")) / "build_mixed_quantile_offset" / f"K{K}_B{B}" / "mixed_quantile_offset_metadata.json"


def resolve_repo_path(path_from_meta: str | Path, fallback_relative: str) -> Path:
    p = Path(path_from_meta)
    if p.exists():
        return p

    root = repo_root()
    fallback = root / fallback_relative
    if fallback.exists():
        return fallback

    parts = list(p.parts)
    for anchor in ["03_outputs", "01_split", "00_raw_dataset"]:
        if anchor in parts:
            idx = parts.index(anchor)
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
    train_path = Path(args.train_raw) if args.train_raw else repo_root() / "01_split" / "train_raw.csv"
    val_path = Path(args.val_raw) if args.val_raw else repo_root() / "01_split" / "val_raw.csv"

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


def load_continuous_for_run(
    spec: Dict[str, str],
    meta: Dict[str, object],
    args: argparse.Namespace,
    train_shape: Tuple[int, int],
    val_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    source = spec["continuous_source"]
    if source == "none":
        return (
            np.zeros(train_shape, dtype=np.float32),
            np.zeros(val_shape, dtype=np.float32),
            {"source": "none", "scale": "none"},
        )
    if source == "raw_scaled":
        return load_raw_scaled(meta, args)
    if source == "z_preprocessed":
        return load_z_preprocessed(meta)
    raise ValueError(f"unknown continuous source: {source}")


def build_selective_mask(
    X_train_bin: np.ndarray,
    X_val_bin: np.ndarray,
    X_train_cont: np.ndarray,
    *,
    num_bins: int,
    tail_frac: float,
    wide_quantile: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Distribution-only selective mask.
    Active if:
      - bin index is in lower/upper tail by bin index, OR
      - empirical train width of continuous values inside feature/bin is high.

    No labels are used.
    """
    N, F = X_train_bin.shape
    tail_bins = max(1, int(round(num_bins * float(tail_frac))))

    active_table = np.zeros((F, num_bins), dtype=bool)
    width_table = np.zeros((F, num_bins), dtype=np.float32)

    tail_bin_mask = np.zeros(num_bins, dtype=bool)
    tail_bin_mask[:tail_bins] = True
    tail_bin_mask[num_bins - tail_bins:] = True

    wide_counts = []
    thresholds = []

    for j in range(F):
        bcol = np.clip(X_train_bin[:, j].astype(np.int64), 0, num_bins - 1)
        xcol = X_train_cont[:, j].astype(np.float32)

        widths = np.zeros(num_bins, dtype=np.float32)
        for b in np.unique(bcol):
            vals = xcol[bcol == b]
            if vals.size >= 2:
                widths[b] = float(vals.max() - vals.min())

        width_table[j] = widths
        nonzero = widths[widths > 0.0]
        if nonzero.size:
            thr = float(np.quantile(nonzero, float(wide_quantile)))
            wide_mask = widths >= thr
        else:
            thr = None
            wide_mask = np.zeros(num_bins, dtype=bool)

        active_table[j] = tail_bin_mask | wide_mask
        wide_counts.append(int(wide_mask.sum()))
        thresholds.append(thr)

    def sample_mask(X_bin: np.ndarray) -> np.ndarray:
        Xb = np.clip(X_bin.astype(np.int64), 0, num_bins - 1)
        out = np.zeros(Xb.shape, dtype=np.float32)
        for j in range(F):
            out[:, j] = active_table[j, Xb[:, j]].astype(np.float32)
        return out

    train_mask = sample_mask(X_train_bin)
    val_mask = sample_mask(X_val_bin)

    info = {
        "type": "wide_or_tail_distribution_only",
        "tail_frac": float(tail_frac),
        "tail_bins_each_side": int(tail_bins),
        "wide_quantile": float(wide_quantile),
        "train_active_ratio": float(train_mask.mean()),
        "val_active_ratio": float(val_mask.mean()),
        "active_bins_mean_per_feature": float(active_table.sum(axis=1).mean()),
        "active_bins_min_per_feature": int(active_table.sum(axis=1).min()),
        "active_bins_max_per_feature": int(active_table.sum(axis=1).max()),
        "wide_bins_mean_per_feature": float(np.mean(wide_counts)),
        "wide_bins_min_per_feature": int(np.min(wide_counts)),
        "wide_bins_max_per_feature": int(np.max(wide_counts)),
        "thresholds_available": int(sum(t is not None for t in thresholds)),
    }
    return train_mask, val_mask, info


class FusionAblationDataset(Dataset):
    """
    Returns:
      X_bin: [F]
      values: [F, 3]
        values[...,0] = offset
        values[...,1] = continuous
        values[...,2] = continuous_mask
      y
    """
    def __init__(self, X_bin: np.ndarray, X_offset: np.ndarray, X_cont: np.ndarray, X_mask: np.ndarray, y: np.ndarray):
        if X_bin.ndim != 2:
            raise ValueError(f"X_bin must be [N,F], got {X_bin.shape}")
        if X_offset.shape != X_bin.shape or X_cont.shape != X_bin.shape or X_mask.shape != X_bin.shape:
            raise ValueError(f"shape mismatch: bin={X_bin.shape}, offset={X_offset.shape}, cont={X_cont.shape}, mask={X_mask.shape}")
        if y.ndim != 1 or y.shape[0] != X_bin.shape[0]:
            raise ValueError(f"y mismatch: {y.shape} vs {X_bin.shape}")

        vals = np.stack([X_offset.astype(np.float32), X_cont.astype(np.float32), X_mask.astype(np.float32)], axis=-1)

        self.X = torch.as_tensor(X_bin, dtype=torch.long)
        self.V = torch.as_tensor(vals, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.V[idx], self.y[idx]


class BaseValueEmbedding(nn.Module):
    def __init__(self, *, num_bins: int, n_features: int, value_dim: int, feature_dim: int, init_std: float = 0.02):
        super().__init__()
        self.num_bins = int(num_bins)
        self.n_features = int(n_features)
        self.value_dim = int(value_dim)
        self.feature_dim = int(feature_dim)
        self.cell_dim = int(value_dim + feature_dim)
        self.init_std = float(init_std)

        self.feature_embedding = nn.Embedding(self.n_features, self.feature_dim)
        nn.init.normal_(self.feature_embedding.weight, mean=0.0, std=self.init_std)
        self.register_buffer("default_feature_ids", torch.arange(self.n_features, dtype=torch.long), persistent=False)

    def add_feature_embedding(self, value_emb: torch.Tensor) -> torch.Tensor:
        B, F, _ = value_emb.shape
        fid = self.default_feature_ids.unsqueeze(0).expand(B, F)
        feature_emb = self.feature_embedding(fid)
        return torch.cat([value_emb, feature_emb], dim=-1)

    def gate_summary(self) -> Dict[str, float]:
        return {}


class D0OffsetInterpolationEmbedding(BaseValueEmbedding):
    def __init__(self, *, num_bins: int, n_features: int, value_dim: int, feature_dim: int):
        super().__init__(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)
        self.bin_embedding = nn.Embedding(self.num_bins + 1, self.value_dim)
        nn.init.normal_(self.bin_embedding.weight, mean=0.0, std=self.init_std)

    def local_interp(self, bin_ids: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        b0 = bin_ids.long().clamp(0, self.num_bins - 1)
        b1 = b0 + 1
        e0 = self.bin_embedding(b0)
        e1 = self.bin_embedding(b1)
        return (1.0 - offset) * e0 + offset * e1

    def forward(self, bin_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        offset = values[..., 0:1].to(dtype=torch.float32, device=bin_ids.device).clamp(0.0, 1.0)
        value_emb = self.local_interp(bin_ids, offset)
        return self.add_feature_embedding(value_emb)


class D1InterpRawScalarConcatEmbedding(BaseValueEmbedding):
    def __init__(self, *, num_bins: int, n_features: int, value_dim: int, feature_dim: int):
        super().__init__(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)
        if self.value_dim < 2:
            raise ValueError("value_dim must be >= 2")
        self.local_dim = self.value_dim - 1
        self.bin_embedding = nn.Embedding(self.num_bins + 1, self.local_dim)
        nn.init.normal_(self.bin_embedding.weight, mean=0.0, std=self.init_std)

    def local_interp(self, bin_ids: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        b0 = bin_ids.long().clamp(0, self.num_bins - 1)
        b1 = b0 + 1
        e0 = self.bin_embedding(b0)
        e1 = self.bin_embedding(b1)
        return (1.0 - offset) * e0 + offset * e1

    def forward(self, bin_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        vals = values.to(dtype=torch.float32, device=bin_ids.device)
        offset = vals[..., 0:1].clamp(0.0, 1.0)
        cont = vals[..., 1:2].clamp(0.0, 1.0)
        local = self.local_interp(bin_ids, offset)
        value_emb = torch.cat([local, cont], dim=-1)
        return self.add_feature_embedding(value_emb)


class D2D4D5InterpProjectGateConcatEmbedding(BaseValueEmbedding):
    """
    Proper concat-fusion:
      local_emb  = offset interpolation in value_dim
      cont_emb   = Project(continuous) in value_dim
      cont_emb   = sample_mask * sigmoid(gate_feature) * cont_emb
      value_emb  = Linear(concat([local_emb, cont_emb])) -> value_dim

    This keeps local bin geometry and global continuous as separate branches before fusion.
    """
    def __init__(self, *, num_bins: int, n_features: int, value_dim: int, feature_dim: int, gate_init: float):
        super().__init__(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)

        self.bin_embedding = nn.Embedding(self.num_bins + 1, self.value_dim)
        self.cont_proj = nn.Sequential(
            nn.Linear(1, self.value_dim),
            nn.GELU(),
            nn.Linear(self.value_dim, self.value_dim),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(self.value_dim * 2),
            nn.Linear(self.value_dim * 2, self.value_dim),
        )
        self.cont_gate_logit = nn.Parameter(torch.full((self.n_features, 1), float(gate_init)))

        nn.init.normal_(self.bin_embedding.weight, mean=0.0, std=self.init_std)
        for m in self.cont_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.fuse[1].weight)
        nn.init.zeros_(self.fuse[1].bias)

    def local_interp(self, bin_ids: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        b0 = bin_ids.long().clamp(0, self.num_bins - 1)
        b1 = b0 + 1
        e0 = self.bin_embedding(b0)
        e1 = self.bin_embedding(b1)
        return (1.0 - offset) * e0 + offset * e1

    def forward(self, bin_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        vals = values.to(dtype=torch.float32, device=bin_ids.device)
        offset = vals[..., 0:1].clamp(0.0, 1.0)
        cont = vals[..., 1:2].clamp(0.0, 1.0)
        mask = vals[..., 2:3].clamp(0.0, 1.0)

        B, F = bin_ids.shape
        local = self.local_interp(bin_ids, offset)
        cont_emb = self.cont_proj(cont)

        gate = torch.sigmoid(self.cont_gate_logit).to(device=bin_ids.device).unsqueeze(0).expand(B, F, 1)
        cont_emb = mask * gate * cont_emb

        value_emb = self.fuse(torch.cat([local, cont_emb], dim=-1))
        return self.add_feature_embedding(value_emb)

    def gate_summary(self) -> Dict[str, float]:
        with torch.no_grad():
            g = torch.sigmoid(self.cont_gate_logit.detach().cpu()).numpy()
        return {
            "cont_gate_min": float(g.min()),
            "cont_gate_max": float(g.max()),
            "cont_gate_mean": float(g.mean()),
            "cont_gate_std": float(g.std()),
        }


class D3InterpRawFiLMEmbedding(BaseValueEmbedding):
    """
    FiLM/multiply fusion:
      local_emb = offset interpolation
      gamma,beta = Project(continuous)
      value_emb = local_emb * (1 + mask * gate * tanh(gamma)) + mask * gate * beta
    """
    def __init__(self, *, num_bins: int, n_features: int, value_dim: int, feature_dim: int, gate_init: float):
        super().__init__(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)

        self.bin_embedding = nn.Embedding(self.num_bins + 1, self.value_dim)
        self.gamma_proj = nn.Sequential(nn.Linear(1, self.value_dim), nn.GELU(), nn.Linear(self.value_dim, self.value_dim))
        self.beta_proj = nn.Sequential(nn.Linear(1, self.value_dim), nn.GELU(), nn.Linear(self.value_dim, self.value_dim))
        self.cont_gate_logit = nn.Parameter(torch.full((self.n_features, 1), float(gate_init)))

        nn.init.normal_(self.bin_embedding.weight, mean=0.0, std=self.init_std)
        for net in [self.gamma_proj, self.beta_proj]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def local_interp(self, bin_ids: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        b0 = bin_ids.long().clamp(0, self.num_bins - 1)
        b1 = b0 + 1
        e0 = self.bin_embedding(b0)
        e1 = self.bin_embedding(b1)
        return (1.0 - offset) * e0 + offset * e1

    def forward(self, bin_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        vals = values.to(dtype=torch.float32, device=bin_ids.device)
        offset = vals[..., 0:1].clamp(0.0, 1.0)
        cont = vals[..., 1:2].clamp(0.0, 1.0)
        mask = vals[..., 2:3].clamp(0.0, 1.0)

        B, F = bin_ids.shape
        local = self.local_interp(bin_ids, offset)

        gamma = torch.tanh(self.gamma_proj(cont))
        beta = self.beta_proj(cont)

        gate = torch.sigmoid(self.cont_gate_logit).to(device=bin_ids.device).unsqueeze(0).expand(B, F, 1)
        g = mask * gate

        value_emb = local * (1.0 + g * gamma) + g * beta
        return self.add_feature_embedding(value_emb)

    def gate_summary(self) -> Dict[str, float]:
        with torch.no_grad():
            g = torch.sigmoid(self.cont_gate_logit.detach().cpu()).numpy()
        return {
            "cont_gate_min": float(g.min()),
            "cont_gate_max": float(g.max()),
            "cont_gate_mean": float(g.mean()),
            "cont_gate_std": float(g.std()),
        }


class D6ProjectedOffsetAddEmbedding(BaseValueEmbedding):
    def __init__(self, *, num_bins: int, n_features: int, value_dim: int, feature_dim: int):
        super().__init__(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)
        self.bin_embedding = nn.Embedding(self.num_bins, self.value_dim)
        self.offset_proj = nn.Sequential(nn.Linear(1, self.value_dim), nn.GELU(), nn.Linear(self.value_dim, self.value_dim))

        nn.init.normal_(self.bin_embedding.weight, mean=0.0, std=self.init_std)
        for m in self.offset_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, bin_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        offset = values[..., 0:1].to(dtype=torch.float32, device=bin_ids.device).clamp(0.0, 1.0)
        b = bin_ids.long().clamp(0, self.num_bins - 1)
        value_emb = self.bin_embedding(b) + self.offset_proj(offset)
        return self.add_feature_embedding(value_emb)


class D7ProjectedOffsetConcatEmbedding(BaseValueEmbedding):
    def __init__(self, *, num_bins: int, n_features: int, value_dim: int, feature_dim: int):
        super().__init__(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)
        self.bin_embedding = nn.Embedding(self.num_bins, self.value_dim)
        self.offset_proj = nn.Sequential(nn.Linear(1, self.value_dim), nn.GELU(), nn.Linear(self.value_dim, self.value_dim))
        self.fuse = nn.Sequential(nn.LayerNorm(self.value_dim * 2), nn.Linear(self.value_dim * 2, self.value_dim))

        nn.init.normal_(self.bin_embedding.weight, mean=0.0, std=self.init_std)
        for m in self.offset_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.fuse[1].weight)
        nn.init.zeros_(self.fuse[1].bias)

    def forward(self, bin_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        offset = values[..., 0:1].to(dtype=torch.float32, device=bin_ids.device).clamp(0.0, 1.0)
        b = bin_ids.long().clamp(0, self.num_bins - 1)
        bin_emb = self.bin_embedding(b)
        offset_emb = self.offset_proj(offset)
        value_emb = self.fuse(torch.cat([bin_emb, offset_emb], dim=-1))
        return self.add_feature_embedding(value_emb)


class E1E2SmoothPositionEmbedding(BaseValueEmbedding):
    """
    E1:
      local value embedding = smooth MLP(pos)

    E2:
      local value embedding = smooth MLP(pos) + raw_scaled FiLM

    pos = (bin_id + offset) / num_bins
    This keeps tokenization/binning unchanged, but replaces independent bin lookup
    interpolation with a smooth ordinal numeric embedding.
    """
    def __init__(
        self,
        *,
        num_bins: int,
        n_features: int,
        value_dim: int,
        feature_dim: int,
        use_raw_film: bool,
        gate_init: float = 0.0,
        n_freq: int = 8,
    ):
        super().__init__(
            num_bins=num_bins,
            n_features=n_features,
            value_dim=value_dim,
            feature_dim=feature_dim,
        )

        self.use_raw_film = bool(use_raw_film)
        self.n_freq = int(n_freq)

        freqs = torch.tensor([2.0 ** i for i in range(self.n_freq)], dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)

        pos_dim = 1 + 2 * self.n_freq

        self.pos_proj = nn.Sequential(
            nn.Linear(pos_dim, self.value_dim),
            nn.GELU(),
            nn.Linear(self.value_dim, self.value_dim),
            nn.LayerNorm(self.value_dim),
        )

        if self.use_raw_film:
            self.gamma_proj = nn.Sequential(
                nn.Linear(1, self.value_dim),
                nn.GELU(),
                nn.Linear(self.value_dim, self.value_dim),
            )
            self.beta_proj = nn.Sequential(
                nn.Linear(1, self.value_dim),
                nn.GELU(),
                nn.Linear(self.value_dim, self.value_dim),
            )
            self.cont_gate_logit = nn.Parameter(
                torch.full((self.n_features, 1), float(gate_init))
            )
        else:
            self.gamma_proj = None
            self.beta_proj = None
            self.cont_gate_logit = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def position_features(self, pos: torch.Tensor) -> torch.Tensor:
        pos = pos.clamp(0.0, 1.0)
        angles = 2.0 * math.pi * pos * self.freqs.view(1, 1, -1)
        return torch.cat([pos, torch.sin(angles), torch.cos(angles)], dim=-1)

    def local_interp(self, bin_ids: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        b = bin_ids.long().clamp(0, self.num_bins - 1).to(dtype=torch.float32)
        off = offset.to(dtype=torch.float32, device=bin_ids.device).clamp(0.0, 1.0).squeeze(-1)

        pos = ((b + off) / max(float(self.num_bins), 1.0)).unsqueeze(-1)
        pos_feat = self.position_features(pos)
        return self.pos_proj(pos_feat)

    def forward(self, bin_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        vals = values.to(dtype=torch.float32, device=bin_ids.device)
        offset = vals[..., 0:1].clamp(0.0, 1.0)

        local = self.local_interp(bin_ids, offset)

        if not self.use_raw_film:
            return self.add_feature_embedding(local)

        cont = vals[..., 1:2].clamp(0.0, 1.0)
        mask = vals[..., 2:3].clamp(0.0, 1.0)

        gamma = torch.tanh(self.gamma_proj(cont))
        beta = self.beta_proj(cont)

        B, F = bin_ids.shape
        gate = torch.sigmoid(self.cont_gate_logit).to(device=bin_ids.device)
        gate = gate.unsqueeze(0).expand(B, F, 1)

        g = mask * gate
        value_emb = local * (1.0 + g * gamma) + g * beta

        return self.add_feature_embedding(value_emb)

    def gate_summary(self) -> Dict[str, float]:
        out = {
            "smooth_position_n_freq": float(self.n_freq),
            "uses_raw_film": float(1.0 if self.use_raw_film else 0.0),
        }

        if self.use_raw_film and self.cont_gate_logit is not None:
            with torch.no_grad():
                g = torch.sigmoid(self.cont_gate_logit.detach().cpu()).numpy()
            out.update({
                "cont_gate_min": float(g.min()),
                "cont_gate_max": float(g.max()),
                "cont_gate_mean": float(g.mean()),
                "cont_gate_std": float(g.std()),
            })

        return out


class FusionAblationTransformer(nn.Module):
    def __init__(
        self,
        *,
        run_id: str,
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

        self.run_id = str(run_id)
        self.spec = RUN_SPECS[self.run_id]
        self.num_bins = int(num_bins)
        self.n_features = int(n_features)
        self.num_classes = int(num_classes)
        self.value_dim = int(value_dim)
        self.feature_dim = int(feature_dim)
        self.cell_dim = int(value_dim + feature_dim)
        self.hidden_dim = int(hidden_dim)

        if run_id == "D0":
            self.embedding = D0OffsetInterpolationEmbedding(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)
        elif run_id == "D1":
            self.embedding = D1InterpRawScalarConcatEmbedding(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)
        elif run_id in {"D2", "D4", "D5"}:
            self.embedding = D2D4D5InterpProjectGateConcatEmbedding(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim, gate_init=gate_init)
        elif run_id == "D3":
            self.embedding = D3InterpRawFiLMEmbedding(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim, gate_init=gate_init)
        elif run_id == "D6":
            self.embedding = D6ProjectedOffsetAddEmbedding(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)
        elif run_id == "D7":
            self.embedding = D7ProjectedOffsetConcatEmbedding(num_bins=num_bins, n_features=n_features, value_dim=value_dim, feature_dim=feature_dim)
        elif run_id == "E1":
            self.embedding = E1E2SmoothPositionEmbedding(
                num_bins=num_bins,
                n_features=n_features,
                value_dim=value_dim,
                feature_dim=feature_dim,
                use_raw_film=False,
                gate_init=gate_init,
            )
        elif run_id == "E2":
            self.embedding = E1E2SmoothPositionEmbedding(
                num_bins=num_bins,
                n_features=n_features,
                value_dim=value_dim,
                feature_dim=feature_dim,
                use_raw_film=True,
                gate_init=gate_init,
            )
        else:
            raise ValueError(f"unknown run_id: {run_id}")

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

    def forward(self, tokens: torch.Tensor, z_values: torch.Tensor | None = None, *, return_info: bool = False):
        if z_values is None:
            raise ValueError("z_values must contain [offset, continuous, mask].")

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
                "run_id": self.run_id,
                "spec": self.spec,
            }
        return logits

    def embedding_extra_summary(self) -> Dict[str, float]:
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




def make_boundary_bin_diagnostics(
    X_bin: np.ndarray,
    X_offset: np.ndarray,
    *,
    num_bins: int,
    feature_names: List[str],
    split_name: str,
) -> Dict[str, object]:
    """Measure how much data falls into the last bin affected by old boundary collapse."""
    Xb = np.asarray(X_bin, dtype=np.int64)
    Off = np.asarray(X_offset, dtype=np.float32)
    if Xb.shape != Off.shape:
        raise ValueError(f"boundary diag shape mismatch: X_bin={Xb.shape}, X_offset={Off.shape}")

    last_bin = int(num_bins) - 1
    mask = Xb == last_bin
    n_rows, n_features = Xb.shape
    total_cells = int(mask.size)
    last_cells = int(mask.sum())

    per_feature = []
    for j, name in enumerate(feature_names):
        mj = mask[:, j]
        cnt = int(mj.sum())
        if cnt:
            offsets = Off[mj, j]
            offset_min = float(offsets.min())
            offset_max = float(offsets.max())
            offset_mean = float(offsets.mean())
            offset_std = float(offsets.std())
        else:
            offset_min = offset_max = offset_mean = offset_std = None
        per_feature.append({
            "feature": str(name),
            "last_bin_count": cnt,
            "last_bin_ratio": float(cnt / max(n_rows, 1)),
            "last_bin_offset_min": offset_min,
            "last_bin_offset_max": offset_max,
            "last_bin_offset_mean": offset_mean,
            "last_bin_offset_std": offset_std,
        })

    top = sorted(per_feature, key=lambda r: r["last_bin_ratio"], reverse=True)[:15]
    return {
        "split": split_name,
        "num_bins": int(num_bins),
        "last_bin_id": int(last_bin),
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "total_cells": total_cells,
        "last_bin_cells": last_cells,
        "last_bin_cell_ratio": float(last_cells / max(total_cells, 1)),
        "features_with_last_bin": int(sum(1 for r in per_feature if r["last_bin_count"] > 0)),
        "max_feature_last_bin_ratio": float(max((r["last_bin_ratio"] for r in per_feature), default=0.0)),
        "mean_feature_last_bin_ratio": float(np.mean([r["last_bin_ratio"] for r in per_feature])) if per_feature else 0.0,
        "top_features_by_last_bin_ratio": top,
    }


def make_embedding_runtime_diagnostics(
    model: "FusionAblationTransformer",
    *,
    X_bin: np.ndarray,
    X_offset: np.ndarray,
    X_cont: np.ndarray,
    X_mask: np.ndarray,
    device: torch.device,
    max_rows: int = 2048,
) -> Dict[str, object]:
    """Lightweight forward-scale checks before training.

    This verifies that endpoint interpolation is not collapsed at the final bin,
    and records activation scales for local embedding vs continuous branch.
    """
    emb = model.embedding
    n = int(min(max_rows, X_bin.shape[0]))
    X = torch.as_tensor(X_bin[:n], dtype=torch.long, device=device)
    V = torch.as_tensor(
        np.stack([
            X_offset[:n].astype(np.float32),
            X_cont[:n].astype(np.float32),
            X_mask[:n].astype(np.float32),
        ], axis=-1),
        dtype=torch.float32,
        device=device,
    )
    out: Dict[str, object] = {
        "n_rows_checked": n,
        "has_local_interp": bool(hasattr(emb, "local_interp")),
        "embedding_class": emb.__class__.__name__,
    }

    with torch.no_grad():
        vals = V.to(dtype=torch.float32, device=device)
        offset = vals[..., 0:1].clamp(0.0, 1.0)
        cont = vals[..., 1:2].clamp(0.0, 1.0)
        mask = vals[..., 2:3].clamp(0.0, 1.0)
        out["offset_mean"] = float(offset.mean().detach().cpu())
        out["offset_std"] = float(offset.std(unbiased=False).detach().cpu())
        out["continuous_mean"] = float(cont.mean().detach().cpu())
        out["continuous_std"] = float(cont.std(unbiased=False).detach().cpu())
        out["continuous_mask_mean"] = float(mask.mean().detach().cpu())

        if hasattr(emb, "bin_embedding"):
            out["bin_embedding_num_embeddings"] = int(emb.bin_embedding.num_embeddings)
            out["bin_embedding_dim"] = int(emb.bin_embedding.embedding_dim)
            out["bin_embedding_weight_std"] = float(emb.bin_embedding.weight.detach().std(unbiased=False).cpu())

        if hasattr(emb, "local_interp"):
            local = emb.local_interp(X, offset)
            out["local_abs_mean"] = float(local.abs().mean().detach().cpu())
            out["local_std"] = float(local.std(unbiased=False).detach().cpu())
            out["local_l2_mean_per_cell"] = float(torch.linalg.vector_norm(local, dim=-1).mean().detach().cpu())

            # Direct boundary sensitivity check: old implementation gives zero here.
            b_last = torch.full((1, 1), int(emb.num_bins) - 1, dtype=torch.long, device=device)
            off0 = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
            off1 = torch.ones((1, 1, 1), dtype=torch.float32, device=device)
            y0 = emb.local_interp(b_last, off0)
            y1 = emb.local_interp(b_last, off1)
            diff = torch.linalg.vector_norm(y1 - y0, dim=-1)
            out["last_bin_offset_sensitivity_l2"] = float(diff.item())
            out["last_bin_boundary_collapse_fixed"] = bool(float(diff.item()) > 1e-12)

            # Scale-ratio check most relevant for D1 raw_scalar_concat.
            local_abs = float(local.abs().mean().detach().cpu())
            cont_abs = float(cont.abs().mean().detach().cpu())
            out["continuous_abs_mean_over_local_abs_mean"] = float(cont_abs / max(local_abs, 1e-12))

        if hasattr(emb, "gamma_proj") and hasattr(emb, "beta_proj"):
            gamma = torch.tanh(emb.gamma_proj(cont))
            beta = emb.beta_proj(cont)
            if hasattr(emb, "cont_gate_logit"):
                Bsz, F = X.shape
                gate = torch.sigmoid(emb.cont_gate_logit).to(device=device).unsqueeze(0).expand(Bsz, F, 1)
                g = mask * gate
            else:
                g = mask
            out["film_gamma_abs_mean"] = float(gamma.abs().mean().detach().cpu())
            out["film_beta_abs_mean"] = float(beta.abs().mean().detach().cpu())
            out["film_gate_mean"] = float(g.mean().detach().cpu())
            if hasattr(emb, "local_interp"):
                delta = (local * (g * gamma)) + (g * beta)
                out["film_delta_l2_mean_per_cell"] = float(torch.linalg.vector_norm(delta, dim=-1).mean().detach().cpu())
                out["film_delta_over_local_l2_mean"] = float(
                    out["film_delta_l2_mean_per_cell"] / max(out.get("local_l2_mean_per_cell", 0.0), 1e-12)
                )

    return out

def make_diagnosis_summary(
    *,
    args: argparse.Namespace,
    spec: Dict[str, str],
    best_epoch: int,
    best_train: Dict[str, object],
    best_val: Dict[str, object],
    metadata: Dict[str, object],
    continuous_info: Dict[str, object],
    selective_info: Dict[str, object],
    boundary_data_diagnostics: Dict[str, object],
    embedding_runtime_diagnostics: Dict[str, object],
    model: FusionAblationTransformer,
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
        "phase": "Fusion ablation D0-D7 fixed K_effective",
        "run_id": args.run_id,
        "best_epoch": int(best_epoch),
        "representation": f"{args.run_id}_{spec['description']}",
        "local": spec["local"],
        "continuous_source": spec["continuous_source"],
        "fusion": spec["fusion"],
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
        "selective_info": selective_info,
        "boundary_data_diagnostics": boundary_data_diagnostics,
        "embedding_runtime_diagnostics": embedding_runtime_diagnostics,
        "embedding_extra_summary": model.embedding_extra_summary(),
        "strategy_counts": metadata.get("strategy_counts", {}),
        "baseline_to_compare": {
            "C0_mixed_bin_scalar_offset_no_continuous": {
                "train_macro_f1": 0.9321426589148848,
                "val_macro_f1": 0.8133554660345561,
                "gap": 0.11878719288032868,
            },
            "C2_scalar_offset_raw_scalar_concat": {
                "train_macro_f1": 0.9265573540092596,
                "val_macro_f1": 0.809009713024029,
                "gap": 0.11754764098523063,
            },
            "C3_scalar_offset_z_project_gate_add": {
                "train_macro_f1": 0.890405910135817,
                "val_macro_f1": 0.8087819336025909,
                "gap": 0.08162397653322617,
            },
            "C4_scalar_offset_raw_project_gate_add": {
                "train_macro_f1": 0.9110189147125328,
                "val_macro_f1": 0.807190454742599,
                "gap": 0.10382845996993384,
            },
        },
    }


def main() -> None:
    args = parse_args()
    _train_mod.set_seed(int(args.seed))

    spec = RUN_SPECS[args.run_id]
    K_artifact = int(args.K)
    B = int(args.num_bins)

    dataset_path = Path(args.dataset_npz) if args.dataset_npz else default_dataset_path(K_artifact, B)
    metadata_path = Path(args.metadata_json) if args.metadata_json else default_metadata_path(K_artifact, B)

    data, meta = load_dataset(dataset_path, metadata_path)

    X_train = data["X_train_bin"].astype(np.int64)
    O_train = data["X_train_offset"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)

    X_val = data["X_val_bin"].astype(np.int64)
    O_val = data["X_val_offset"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)

    X_train_cont, X_val_cont, continuous_info = load_continuous_for_run(
        spec=spec,
        meta=meta,
        args=args,
        train_shape=X_train.shape,
        val_shape=X_val.shape,
    )

    if X_train_cont.shape != X_train.shape:
        raise ValueError(f"train continuous shape mismatch: {X_train_cont.shape} vs {X_train.shape}")
    if X_val_cont.shape != X_val.shape:
        raise ValueError(f"val continuous shape mismatch: {X_val_cont.shape} vs {X_val.shape}")

    selective_info: Dict[str, object] = {"type": "none"}
    if args.run_id == "D5":
        M_train, M_val, selective_info = build_selective_mask(
            X_train_bin=X_train,
            X_val_bin=X_val,
            X_train_cont=X_train_cont,
            num_bins=B,
            tail_frac=float(args.tail_frac),
            wide_quantile=float(args.wide_quantile),
        )
    else:
        M_train = np.ones_like(X_train, dtype=np.float32)
        M_val = np.ones_like(X_val, dtype=np.float32)

    feature_names_for_diag = [str(x) for x in meta["feature_names"]]
    boundary_data_diagnostics = {
        "train": make_boundary_bin_diagnostics(
            X_train, O_train, num_bins=B, feature_names=feature_names_for_diag, split_name="train"
        ),
        "val": make_boundary_bin_diagnostics(
            X_val, O_val, num_bins=B, feature_names=feature_names_for_diag, split_name="val"
        ),
    }

    label_mapping = meta["label_mapping"]
    label_names = [label for label, _ in sorted(label_mapping.items(), key=lambda kv: kv[1])]
    num_classes = int(len(label_names))
    n_features = int(meta["n_features"])

    run_name = args.run_name
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_{args.run_id}_{spec['local']}_{spec['fusion']}_Keff{B}_seed{args.seed}"

    out_dir = Path(args.out_root) / f"Keff{B}" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _train_mod.pick_device(str(args.device))

    config_obj = {
        "stage": "train_fusion_ablation_D0_D7",
        "phase": "fixed_K_representation_ablation",
        "run_id": args.run_id,
        "run_spec": spec,
        "K_artifact": K_artifact,
        "effective_token_budget": B,
        "num_bins": B,
        "dataset_npz": str(dataset_path),
        "metadata_json": str(metadata_path),
        "out_dir": str(out_dir),
        "continuous_info": continuous_info,
        "selective_info": selective_info,
        "seed": int(args.seed),
        "device": str(device),
        "epochs": int(args.epochs),
        "diagnostic_only": bool(args.diagnostic_only),
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
            "local": spec["local"],
            "continuous_source": spec["continuous_source"],
            "fusion": spec["fusion"],
            "description": spec["description"],
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
        "boundary_data_diagnostics": boundary_data_diagnostics,
        "data_shapes": {
            "X_train_bin": list(X_train.shape),
            "X_train_offset": list(O_train.shape),
            "X_train_continuous": list(X_train_cont.shape),
            "X_train_mask": list(M_train.shape),
            "y_train": list(y_train.shape),
            "X_val_bin": list(X_val.shape),
            "X_val_offset": list(O_val.shape),
            "X_val_continuous": list(X_val_cont.shape),
            "X_val_mask": list(M_val.shape),
            "y_val": list(y_val.shape),
        },
        "torch_version": torch.__version__,
    }
    _train_mod.save_json(out_dir / "config.json", config_obj)

    train_ds = FusionAblationDataset(X_train, O_train, X_train_cont, M_train, y_train)
    val_ds = FusionAblationDataset(X_val, O_val, X_val_cont, M_val, y_val)

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

    model = FusionAblationTransformer(
        run_id=args.run_id,
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

    embedding_runtime_diagnostics = make_embedding_runtime_diagnostics(
        model,
        X_bin=X_train,
        X_offset=O_train,
        X_cont=X_train_cont,
        X_mask=M_train,
        device=device,
    )
    config_obj["embedding_runtime_diagnostics"] = embedding_runtime_diagnostics
    _train_mod.save_json(out_dir / "config.json", config_obj)

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

    print("===== fusion ablation training start =====")
    print(f"run_id: {args.run_id}")
    print(f"spec: {spec}")
    print(f"run_dir: {out_dir}")
    print(f"device: {device}")
    print(f"dataset: {dataset_path}")
    print(f"continuous_info: {continuous_info}")
    print(f"selective_info: {selective_info}")
    print(f"train bin/offset/continuous/mask shape: {X_train.shape}/{O_train.shape}/{X_train_cont.shape}/{M_train.shape}")
    print(f"val bin/offset/continuous/mask shape:   {X_val.shape}/{O_val.shape}/{X_val_cont.shape}/{M_val.shape}")
    print(f"classes: {num_classes} {label_names}")
    print(f"strategy_counts: {meta.get('strategy_counts', {})}")
    print(f"boundary last-bin train cell ratio: {boundary_data_diagnostics['train']['last_bin_cell_ratio']:.6f}")
    print(f"boundary last-bin val cell ratio:   {boundary_data_diagnostics['val']['last_bin_cell_ratio']:.6f}")
    print(f"last-bin offset sensitivity L2:     {embedding_runtime_diagnostics.get('last_bin_offset_sensitivity_l2')}")
    print(f"boundary collapse fixed:            {embedding_runtime_diagnostics.get('last_bin_boundary_collapse_fixed')}")

    if bool(args.diagnostic_only):
        diagnostic_only_obj = {
            "run_id": args.run_id,
            "spec": spec,
            "dataset_npz": str(dataset_path),
            "metadata_json": str(metadata_path),
            "boundary_data_diagnostics": boundary_data_diagnostics,
            "embedding_runtime_diagnostics": embedding_runtime_diagnostics,
            "continuous_info": continuous_info,
            "selective_info": selective_info,
            "data_shapes": config_obj["data_shapes"],
            "model_config": config_obj["model"],
            "note": "Diagnostic-only run exited before training.",
        }
        _train_mod.save_json(out_dir / "diagnostics_only.json", diagnostic_only_obj)
        print("diagnostic-only mode: exiting before training")
        print(f"diagnostics_only:       {out_dir / 'diagnostics_only.json'}")
        return

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
                    "selective_info": selective_info,
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
            "selective_info": selective_info,
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
        args=args,
        spec=spec,
        best_epoch=best_epoch,
        best_train=best_train_eval,
        best_val=best_val_eval,
        metadata=meta,
        continuous_info=continuous_info,
        selective_info=selective_info,
        boundary_data_diagnostics=boundary_data_diagnostics,
        embedding_runtime_diagnostics=embedding_runtime_diagnostics,
        model=model,
    )
    _train_mod.save_json(out_dir / "diagnosis_summary.json", diagnosis)

    print("===== fusion ablation training done =====")
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

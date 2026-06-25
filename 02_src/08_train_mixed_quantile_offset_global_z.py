#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Transformer on mixed bin + offset + global_z representation.

Ablation purpose:
  Previous run:
    mixed_bin + offset
    uses_continuous_z = False

  This run:
    mixed_bin + offset + global_z
    uses_continuous_z = True

This script does not modify baseline files.
It reuses the existing mixed_quantile_offset_dataset.npz for bin/offset,
and reads global_z from the same train_preprocessed/val_preprocessed CSVs
recorded in mixed_quantile_offset_metadata.json.
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
    p = argparse.ArgumentParser(description="Train mixed bin + offset + global_z Transformer.")

    p.add_argument("--K", type=int, default=int(cfg("TOKEN_K", 1000)))
    p.add_argument("--num-bins", type=int, default=int(cfg("VALUE_NUM_BINS", 128)))
    p.add_argument("--dataset-npz", default="")
    p.add_argument("--metadata-json", default="")
    p.add_argument("--out-root", default=str(cfg("OUTPUT_ROOT", Path("03_outputs")) / "train_runs_mixed_quantile_offset_global_z"))
    p.add_argument("--run-name", default="")

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
    return p.parse_args()


def default_dataset_path(K: int, B: int) -> Path:
    return cfg("OUTPUT_ROOT", Path("03_outputs")) / "build_mixed_quantile_offset" / f"K{K}_B{B}" / "mixed_quantile_offset_dataset.npz"


def default_metadata_path(K: int, B: int) -> Path:
    return cfg("OUTPUT_ROOT", Path("03_outputs")) / "build_mixed_quantile_offset" / f"K{K}_B{B}" / "mixed_quantile_offset_metadata.json"


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


def resolve_repo_path(path_from_meta: str, fallback_relative: str) -> Path:
    """
    Metadata may contain absolute local paths such as:
      /home/pak/Documents/dacn/03_outputs/...

    On Kaggle, the repo is cloned to:
      /kaggle/working/dacn

    So if the metadata path does not exist, fall back to the same relative
    path inside the current repository.
    """
    p = Path(path_from_meta)
    if p.exists():
        return p

    repo_root = Path(__file__).resolve().parents[1]
    fallback = repo_root / fallback_relative
    if fallback.exists():
        return fallback

    # Last fallback: keep only the path after "03_outputs" if present.
    parts = list(p.parts)
    if "03_outputs" in parts:
        idx = parts.index("03_outputs")
        rel = Path(*parts[idx:])
        candidate = repo_root / rel
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve path. metadata_path={p}, fallback={fallback}"
    )


def load_global_z_from_preprocessed(meta: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    feature_names = [str(x) for x in meta["feature_names"]]

    train_path = resolve_repo_path(
        meta["input"]["train_preprocessed"],
        "03_outputs/preprocessing/train_preprocessed_K1000.csv",
    )
    val_path = resolve_repo_path(
        meta["input"]["val_preprocessed"],
        "03_outputs/preprocessing/val_preprocessed_K1000.csv",
    )

    print(f"[global_z] train_preprocessed: {train_path}")
    print(f"[global_z] val_preprocessed:   {val_path}")

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    missing_train = [f for f in feature_names if f not in train.columns]
    missing_val = [f for f in feature_names if f not in val.columns]
    if missing_train:
        raise ValueError(f"train_preprocessed missing features: {missing_train[:10]}")
    if missing_val:
        raise ValueError(f"val_preprocessed missing features: {missing_val[:10]}")

    X_train_z = train.loc[:, feature_names].to_numpy(dtype=np.float32)
    X_val_z = val.loc[:, feature_names].to_numpy(dtype=np.float32)

    X_train_z = np.clip(X_train_z, 0.0, 1.0).astype(np.float32)
    X_val_z = np.clip(X_val_z, 0.0, 1.0).astype(np.float32)

    return X_train_z, X_val_z


class BinOffsetGlobalDataset(Dataset):
    """
    Returns:
      X_bin: [F]
      V: [F, 2], where V[...,0] = offset, V[...,1] = global_z
      y
    """

    def __init__(self, X_bin: np.ndarray, X_offset: np.ndarray, X_global_z: np.ndarray, y: np.ndarray) -> None:
        if X_bin.ndim != 2:
            raise ValueError(f"X_bin must be [N,F], got {X_bin.shape}")
        if X_offset.ndim != 2:
            raise ValueError(f"X_offset must be [N,F], got {X_offset.shape}")
        if X_global_z.ndim != 2:
            raise ValueError(f"X_global_z must be [N,F], got {X_global_z.shape}")
        if X_bin.shape != X_offset.shape or X_bin.shape != X_global_z.shape:
            raise ValueError(f"shape mismatch: bin={X_bin.shape}, offset={X_offset.shape}, z={X_global_z.shape}")
        if y.ndim != 1 or y.shape[0] != X_bin.shape[0]:
            raise ValueError(f"y mismatch: {y.shape} vs {X_bin.shape}")

        pair = np.stack([X_offset.astype(np.float32), X_global_z.astype(np.float32)], axis=-1)

        self.X = torch.as_tensor(X_bin, dtype=torch.long)
        self.V = torch.as_tensor(pair, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.V[idx], self.y[idx]


class ValueFeatureBinOffsetGlobalEmbedding(nn.Module):
    """
    value_emb = [offset, global_z] || Emb(bin_id)
    cell_emb  = value_emb || Emb(feature_id)

    Here:
      offset   = local position inside bin
      global_z = absolute/global continuous position after preprocessing
    """

    def __init__(
        self,
        *,
        num_bins: int,
        n_features: int,
        value_dim: int = 32,
        feature_dim: int = 32,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()

        if num_bins <= 1:
            raise ValueError("num_bins must be > 1")
        if n_features <= 0:
            raise ValueError("n_features must be > 0")
        if value_dim < 3:
            raise ValueError("value_dim must be >=3 because two numeric coords + bin embedding are used.")
        if feature_dim < 1:
            raise ValueError("feature_dim must be >=1")

        self.num_bins = int(num_bins)
        self.n_features = int(n_features)
        self.value_dim = int(value_dim)
        self.feature_dim = int(feature_dim)
        self.bin_emb_dim = int(value_dim - 2)
        self.cell_dim = int(value_dim + feature_dim)

        self.bin_embedding = nn.Embedding(self.num_bins, self.bin_emb_dim)
        nn.init.normal_(self.bin_embedding.weight, mean=0.0, std=init_std)

        self.feature_embedding = nn.Embedding(self.n_features, self.feature_dim)
        nn.init.normal_(self.feature_embedding.weight, mean=0.0, std=init_std)

        self.register_buffer(
            "default_feature_ids",
            torch.arange(self.n_features, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        bin_ids: torch.Tensor,
        values: torch.Tensor,
        feature_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bin_ids.ndim != 2:
            raise ValueError(f"bin_ids must be [B,F], got {tuple(bin_ids.shape)}")
        if values.ndim != 3 or values.shape[-1] != 2:
            raise ValueError(f"values must be [B,F,2] = offset/global_z, got {tuple(values.shape)}")

        B, F = bin_ids.shape
        if tuple(values.shape[:2]) != (B, F):
            raise ValueError(f"bin_ids/values shape mismatch: {tuple(bin_ids.shape)} vs {tuple(values.shape)}")
        if F != self.n_features:
            raise ValueError(f"Expected F={self.n_features}, got {F}")

        b = bin_ids.long().clamp(0, self.num_bins - 1)
        numeric = values.to(device=b.device, dtype=torch.float32).clamp(0.0, 1.0)

        bin_emb = self.bin_embedding(b)
        value_emb = torch.cat([numeric, bin_emb], dim=-1)

        if feature_ids is None:
            fid = self.default_feature_ids.unsqueeze(0).expand(B, F)
        else:
            fid = feature_ids.long()
            if fid.ndim == 1:
                fid = fid.unsqueeze(0).expand(B, F)
            fid = fid.to(b.device).clamp(0, self.n_features - 1)

        feature_emb = self.feature_embedding(fid)
        return torch.cat([value_emb, feature_emb], dim=-1)


class BinOffsetGlobalTransformerClassifier(nn.Module):
    def __init__(
        self,
        *,
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
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim must be divisible by num_heads: {hidden_dim}/{num_heads}")

        self.num_bins = int(num_bins)
        self.n_features = int(n_features)
        self.num_classes = int(num_classes)
        self.value_dim = int(value_dim)
        self.feature_dim = int(feature_dim)
        self.cell_dim = int(value_dim + feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.norm_first = bool(norm_first)

        self.embedding = ValueFeatureBinOffsetGlobalEmbedding(
            num_bins=self.num_bins,
            n_features=self.n_features,
            value_dim=self.value_dim,
            feature_dim=self.feature_dim,
        )

        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.cell_dim),
            nn.Linear(self.cell_dim, self.hidden_dim),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            activation=activation,
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
            nn.Linear(self.hidden_dim, int(classifier_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(classifier_dropout)),
            nn.Linear(int(classifier_hidden_dim), self.num_classes),
        )

    def forward(self, tokens: torch.Tensor, z_values: Optional[torch.Tensor] = None, *, return_info: bool = False):
        # Compatibility with helpers from 05_train.py:
        # tokens   = bin_ids
        # z_values = [offset, global_z]
        if z_values is None:
            raise ValueError("z_values must contain [offset, global_z].")

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
                "uses_continuous_z": True,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "norm_first": self.norm_first,
            }
        return logits

    def extra_repr(self) -> str:
        return (
            f"num_bins={self.num_bins}, n_features={self.n_features}, num_classes={self.num_classes}, "
            f"cell_dim={self.cell_dim}, hidden_dim={self.hidden_dim}, layers={self.num_layers}, "
            f"heads={self.num_heads}, norm_first={self.norm_first}"
        )


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
    best_epoch: int,
    best_train: Dict[str, object],
    best_val: Dict[str, object],
    metadata: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    train_macro = float(best_train["macro_f1"])
    val_macro = float(best_val["macro_f1"])
    gap = train_macro - val_macro

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
        "best_epoch": int(best_epoch),
        "representation": "mixed_quantile_offset_global_z",
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
        "generalization_gap_macro_f1": float(gap),
        "worst_val_classes_by_f1": worst_classes,
        "model_config": {
            "value_dim": int(args.value_dim),
            "feature_dim": int(args.feature_dim),
            "cell_dim": int(args.value_dim + args.feature_dim),
            "num_bins": int(args.num_bins),
            "uses_offset": True,
            "uses_continuous_z": True,
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "dropout": float(args.dropout),
            "classifier_hidden_dim": int(args.classifier_hidden_dim),
            "classifier_dropout": float(args.classifier_dropout),
            "norm_first": bool(args.norm_first),
            "use_class_weights": bool(args.use_class_weights),
            "scheduler": str(args.scheduler),
        },
        "strategy_counts": metadata.get("strategy_counts", {}),
        "baseline_to_compare": {
            "current_z_continuous_uniform_128bin": {
                "train_macro_f1": 0.8891,
                "val_macro_f1": 0.7952,
                "gap": 0.0939,
            },
            "mixed_quantile_offset_no_global_z": {
                "train_macro_f1": 0.9321426589148848,
                "val_macro_f1": 0.8133554660345561,
                "gap": 0.11878719288032868,
            },
        },
    }


def main() -> None:
    args = parse_args()
    _train_mod.set_seed(int(args.seed))

    K = int(args.K)
    B = int(args.num_bins)

    dataset_path = Path(args.dataset_npz) if args.dataset_npz else default_dataset_path(K, B)
    metadata_path = Path(args.metadata_json) if args.metadata_json else default_metadata_path(K, B)

    data, meta = load_dataset(dataset_path, metadata_path)
    X_train_z, X_val_z = load_global_z_from_preprocessed(meta)

    X_train = data["X_train_bin"].astype(np.int64)
    O_train = data["X_train_offset"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)

    X_val = data["X_val_bin"].astype(np.int64)
    O_val = data["X_val_offset"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)

    if X_train.shape != X_train_z.shape:
        raise ValueError(f"train shape mismatch: bin={X_train.shape}, global_z={X_train_z.shape}")
    if X_val.shape != X_val_z.shape:
        raise ValueError(f"val shape mismatch: bin={X_val.shape}, global_z={X_val_z.shape}")

    label_mapping = meta["label_mapping"]
    label_names = [label for label, _ in sorted(label_mapping.items(), key=lambda kv: kv[1])]
    num_classes = int(len(label_names))
    n_features = int(meta["n_features"])

    run_name = args.run_name
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_mixed_quantile_offset_global_z_K{K}_B{B}_seed{args.seed}"

    out_dir = Path(args.out_root) / f"K{K}_B{B}" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _train_mod.pick_device(str(args.device))

    config_obj = {
        "stage": "train_mixed_quantile_offset_global_z",
        "run_name": run_name,
        "K": K,
        "num_bins": B,
        "dataset_npz": str(dataset_path),
        "metadata_json": str(metadata_path),
        "out_dir": str(out_dir),
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
            "name": "mixed_quantile_offset_global_z",
            "uses_bin_id": True,
            "uses_offset": True,
            "uses_continuous_z": True,
            "numeric_channels": ["offset", "global_z"],
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
        },
        "data_shapes": {
            "X_train_bin": list(X_train.shape),
            "X_train_offset": list(O_train.shape),
            "X_train_global_z": list(X_train_z.shape),
            "y_train": list(y_train.shape),
            "X_val_bin": list(X_val.shape),
            "X_val_offset": list(O_val.shape),
            "X_val_global_z": list(X_val_z.shape),
            "y_val": list(y_val.shape),
        },
        "torch_version": torch.__version__,
    }
    _train_mod.save_json(out_dir / "config.json", config_obj)

    train_ds = BinOffsetGlobalDataset(X_train, O_train, X_train_z, y_train)
    val_ds = BinOffsetGlobalDataset(X_val, O_val, X_val_z, y_val)

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

    model = BinOffsetGlobalTransformerClassifier(
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

    print("===== mixed quantile offset + global_z training start =====")
    print(f"run_dir: {out_dir}")
    print(f"device: {device}")
    print(f"dataset: {dataset_path}")
    print(f"train bin/offset/global_z shape: {X_train.shape}/{O_train.shape}/{X_train_z.shape}")
    print(f"val bin/offset/global_z shape:   {X_val.shape}/{O_val.shape}/{X_val_z.shape}")
    print(f"classes: {num_classes} {label_names}")
    print(f"strategy_counts: {meta.get('strategy_counts', {})}")
    print(f"model: {model}")

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
                },
                out_dir / "best_model.pt",
            )
        else:
            bad_epochs += 1

        print(
            f"[epoch {epoch:03d}] "
            f"lr={lr_epoch:.6g} "
            f"train_loss={row['train_loss']:.4f} "
            f"train_macro={row['train_macro_f1']:.4f} "
            f"train_malware={row['train_malware_avg_f1']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
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
        best_epoch=best_epoch,
        best_train=best_train_eval,
        best_val=best_val_eval,
        metadata=meta,
        args=args,
    )
    _train_mod.save_json(out_dir / "diagnosis_summary.json", diagnosis)

    print("===== mixed quantile offset + global_z training done =====")
    print(f"best_epoch: {best_epoch}")
    print(f"train_macro_f1:        {diagnosis['train']['macro_f1']:.6f}")
    print(f"train_malware_avg_f1: {diagnosis['train']['malware_only_avg_f1']:.6f}")
    print(f"val_macro_f1:          {diagnosis['val']['macro_f1']:.6f}")
    print(f"val_malware_avg_f1:   {diagnosis['val']['malware_only_avg_f1']:.6f}")
    print(f"gap:                   {diagnosis['generalization_gap_macro_f1']:.6f}")
    print(f"diagnosis:             {out_dir / 'diagnosis_summary.json'}")


if __name__ == "__main__":
    main()

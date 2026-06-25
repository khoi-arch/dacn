#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_pipeline/train.py

Train TabularTransformerClassifier with detailed diagnostic logging.

Inputs:
    final_pipeline/outputs/build_token/K{K}/token_dataset_K{K}.npz
    final_pipeline/outputs/build_token/K{K}/token_metadata_K{K}.json

Outputs under:
    final_pipeline/outputs/train_runs/K{K}/{run_name}/

Logs:
    - config.json
    - history.csv
    - best_model.pt
    - last_model.pt
    - val_classification_report_best.json
    - train_classification_report_best.json
    - val_confusion_matrix_best.csv/json
    - train_confusion_matrix_best.csv/json
    - val_predictions_best.csv
    - diagnosis_summary.json

Purpose:
    Make failures diagnosable:
    - train vs val gap
    - underfit vs overfit hints
    - per-class weak points
    - class distribution / prediction distribution
    - exact config, seed, dataset paths
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import config as CFG
import importlib.util as _importlib_util

_model_path = Path(__file__).resolve().with_name("04_model.py")
_model_spec = _importlib_util.spec_from_file_location("_dacn_04_model", _model_path)
_model_mod = _importlib_util.module_from_spec(_model_spec)
assert _model_spec is not None and _model_spec.loader is not None
_model_spec.loader.exec_module(_model_mod)
TabularTransformerClassifier = _model_mod.TabularTransformerClassifier


def cfg(name: str, default):
    return getattr(CFG, name, default)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train tabular transformer on token dataset.")

    p.add_argument("--K", type=int, default=int(cfg("TOKEN_K", 20000)))
    p.add_argument("--dataset-npz", default="")
    p.add_argument("--metadata-json", default="")
    p.add_argument("--out-root", default=str(cfg("TRAIN_RUN_DIR", cfg("OUTPUT_ROOT", Path("outputs")) / "train_runs")))
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
    p.add_argument("--value-num-bins", type=int, default=int(cfg("VALUE_NUM_BINS", 512)))
    p.add_argument("--hidden-dim", type=int, default=int(cfg("MODEL_HIDDEN_DIM", 128)))
    p.add_argument("--num-layers", type=int, default=int(cfg("MODEL_NUM_LAYERS", 3)))
    p.add_argument("--num-heads", type=int, default=int(cfg("MODEL_NUM_HEADS", 4)))
    p.add_argument("--dropout", type=float, default=float(cfg("MODEL_DROPOUT", 0.1)))
    p.add_argument("--classifier-hidden-dim", type=int, default=int(cfg("CLASSIFIER_HIDDEN_DIM", 128)))
    p.add_argument("--classifier-dropout", type=float, default=float(cfg("CLASSIFIER_DROPOUT", 0.1)))
    p.add_argument("--norm-first", action=argparse.BooleanOptionalAction, default=bool(cfg("TRANSFORMER_NORM_FIRST", True)))

    return p.parse_args()


def default_dataset_path(K: int) -> Path:
    if hasattr(CFG, "token_dataset_npz_path"):
        return CFG.token_dataset_npz_path(K)
    return cfg("BUILD_TOKEN_DIR", cfg("OUTPUT_ROOT", Path("outputs")) / "build_token") / f"K{K}" / f"token_dataset_K{K}.npz"


def default_metadata_path(K: int) -> Path:
    if hasattr(CFG, "token_metadata_json_path"):
        return CFG.token_metadata_json_path(K)
    return cfg("BUILD_TOKEN_DIR", cfg("OUTPUT_ROOT", Path("outputs")) / "build_token") / f"K{K}" / f"token_metadata_K{K}.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproducible enough for debugging.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


class TokenDataset(Dataset):
    def __init__(self, X_tokens: np.ndarray, X_z: np.ndarray, y: np.ndarray) -> None:
        if X_tokens.ndim != 2:
            raise ValueError(f"X_tokens must be [N,F], got {X_tokens.shape}")
        if X_z.ndim != 2:
            raise ValueError(f"X_z must be [N,F], got {X_z.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be [N], got {y.shape}")
        if X_tokens.shape != X_z.shape:
            raise ValueError(f"X_tokens/X_z shape mismatch: {X_tokens.shape} vs {X_z.shape}")
        if X_tokens.shape[0] != y.shape[0]:
            raise ValueError(f"X/y row mismatch: {X_tokens.shape[0]} vs {y.shape[0]}")
        self.X = torch.as_tensor(X_tokens, dtype=torch.long)
        self.Z = torch.as_tensor(X_z, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.Z[idx], self.y[idx]

def load_token_dataset(dataset_path: Path, metadata_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"token dataset not found: {dataset_path}\n"
            f"Run: python -u final_pipeline/build_token.py"
        )
    if not metadata_path.exists():
        raise FileNotFoundError(f"token metadata not found: {metadata_path}")

    data = dict(np.load(dataset_path, allow_pickle=True))
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))

    required = ["X_train_tokens", "y_train", "X_val_tokens", "y_val"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"dataset npz missing arrays: {missing}")

    return data, meta


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train.astype(int), minlength=num_classes).astype(np.float64)
    total = float(counts.sum())
    weights = np.zeros(num_classes, dtype=np.float32)
    for i, c in enumerate(counts):
        if c <= 0:
            weights[i] = 0.0
        else:
            weights[i] = total / (num_classes * c)
    return torch.as_tensor(weights, dtype=torch.float32)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    label_names: List[str],
) -> Dict[str, object]:
    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    total = int(cm.sum())
    correct = int(np.trace(cm))
    acc = float(correct / total) if total else 0.0

    per_class: Dict[str, Dict[str, float]] = {}
    f1s = []
    weights = []

    for i in range(num_classes):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        support = int(cm[i, :].sum())

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        name = label_names[i] if i < len(label_names) else str(i)
        per_class[name] = {
            "class_id": int(i),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

        f1s.append(f1)
        weights.append(support)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    weighted_f1 = float(np.average(f1s, weights=weights)) if sum(weights) > 0 else 0.0

    true_counts = {label_names[i] if i < len(label_names) else str(i): int(cm[i, :].sum()) for i in range(num_classes)}
    pred_counts = {label_names[i] if i < len(label_names) else str(i): int(cm[:, i].sum()) for i in range(num_classes)}

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "true_counts": true_counts,
        "pred_counts": pred_counts,
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    label_names: List[str],
    collect_probs: bool = False,
) -> Dict[str, object]:
    model.eval()

    losses = []
    all_true = []
    all_pred = []
    all_conf = []
    all_prob = []

    for X, Z, y in loader:
        X = X.to(device, non_blocking=True)
        Z = Z.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(X, z_values=Z)
        loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

        losses.append(float(loss.item()) * int(X.size(0)))
        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        all_conf.append(conf.detach().cpu().numpy())
        if collect_probs:
            all_prob.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.asarray([], dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.asarray([], dtype=np.int64)
    conf = np.concatenate(all_conf) if all_conf else np.asarray([], dtype=np.float32)

    metrics = metrics_from_predictions(y_true, y_pred, num_classes, label_names)
    avg_loss = float(sum(losses) / max(len(y_true), 1))

    out = {
        "loss": avg_loss,
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": conf,
        **metrics,
    }
    if collect_probs:
        out["probs"] = np.concatenate(all_prob) if all_prob else np.zeros((0, num_classes), dtype=np.float32)
    return out


def compute_epoch_lr(
    *,
    base_lr: float,
    epoch: int,
    total_epochs: int,
    scheduler_name: str,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> float:
    """
    Epoch-level LR schedule.

    none:
        lr = base_lr

    warmup_cosine:
        - warmup: linearly increase LR from base_lr/warmup_epochs to base_lr
        - cosine: decay from base_lr to base_lr * min_lr_ratio

    epoch is 1-indexed.
    """
    if scheduler_name == "none":
        return float(base_lr)

    if scheduler_name != "warmup_cosine":
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    total_epochs = max(int(total_epochs), 1)
    warmup_epochs = max(int(warmup_epochs), 0)
    min_lr_ratio = float(min(max(min_lr_ratio, 0.0), 1.0))

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return float(base_lr) * float(epoch) / float(warmup_epochs)

    if total_epochs <= warmup_epochs:
        return float(base_lr)

    progress = float(epoch - warmup_epochs) / float(max(total_epochs - warmup_epochs, 1))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr_ratio = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return float(base_lr) * float(lr_ratio)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for X, Z, y in loader:
        X = X.to(device, non_blocking=True)
        Z = Z.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X, z_values=Z)
        loss = criterion(logits, y)
        loss.backward()

        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))

        optimizer.step()

        n = int(X.size(0))
        total_loss += float(loss.item()) * n
        total_n += n

    return float(total_loss / max(total_n, 1))


def write_history_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_outputs(
    out_dir: Path,
    prefix: str,
    report: Dict[str, object],
    label_names: List[str],
) -> None:
    cm = np.asarray(report["confusion_matrix"], dtype=np.int64)

    json_path = out_dir / f"{prefix}_confusion_matrix_best.json"
    csv_path = out_dir / f"{prefix}_confusion_matrix_best.csv"

    json_path.write_text(json.dumps({
        "labels": label_names,
        "matrix": cm.tolist(),
        "note": "rows=true labels, columns=predicted labels",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + label_names)
        for i, row in enumerate(cm.tolist()):
            writer.writerow([label_names[i] if i < len(label_names) else str(i)] + row)


def write_predictions_csv(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    label_names: List[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "true_id", "true_label", "pred_id", "pred_label", "correct", "confidence"])
        for idx, (t, p, c) in enumerate(zip(y_true.astype(int), y_pred.astype(int), confidence.astype(float))):
            true_label = label_names[t] if 0 <= t < len(label_names) else str(t)
            pred_label = label_names[p] if 0 <= p < len(label_names) else str(p)
            writer.writerow([idx, int(t), true_label, int(p), pred_label, int(t == p), float(c)])


def make_diagnosis_summary(
    *,
    best_epoch: int,
    best_train: Dict[str, object],
    best_val: Dict[str, object],
    label_names: List[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    train_macro = float(best_train["macro_f1"])
    val_macro = float(best_val["macro_f1"])
    train_loss = float(best_train["loss"])
    val_loss = float(best_val["loss"])
    gap = train_macro - val_macro

    per_class = best_val["per_class"]
    worst_classes = sorted(
        [
            {
                "label": label,
                "f1": float(m["f1"]),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "support": int(m["support"]),
            }
            for label, m in per_class.items()
        ],
        key=lambda x: (x["f1"], -x["support"]),
    )[:10]

    hints = []

    if train_macro < 0.50 and val_macro < 0.50:
        hints.append({
            "pattern": "train_low_val_low",
            "meaning": "Could be underfit, optimization issue, or data/preprocess signal issue.",
            "next_checks": [
                "Increase epochs or check if loss keeps decreasing.",
                "Try hidden_dim 256 or num_layers 4 while watching train F1.",
                "Compare against known old/hybrid results without rerunning if already available.",
                "Inspect preprocess_token_compare for whether representation improved but class signal did not.",
            ],
        })

    if train_macro >= 0.70 and gap > 0.15:
        hints.append({
            "pattern": "train_high_val_low",
            "meaning": "Likely overfit or train/val distribution mismatch.",
            "next_checks": [
                "Increase dropout/weight decay.",
                "Reduce hidden_dim/layers.",
                "Check per-class train vs val support.",
                "Check preprocessing applied train-fitted mapping to val correctly.",
            ],
        })

    if abs(gap) <= 0.05 and val_macro < 0.60:
        hints.append({
            "pattern": "small_gap_but_low_score",
            "meaning": "Model may be consistently limited; not obvious overfit.",
            "next_checks": [
                "Try capacity sweep: hidden_dim 64/128/256.",
                "Try depth sweep: layers 1/2/3/4.",
                "Check whether class weighting helps/hurts macro-F1.",
            ],
        })

    if val_loss > train_loss * 1.5 and gap > 0.10:
        hints.append({
            "pattern": "loss_gap",
            "meaning": "Validation loss much higher than train loss.",
            "next_checks": [
                "Potential overfit.",
                "Check label imbalance and rare-class behavior.",
                "Try stronger regularization.",
            ],
        })

    if not hints:
        hints.append({
            "pattern": "no_strong_single_pattern",
            "meaning": "No single failure mode is obvious from aggregate metrics alone.",
            "next_checks": [
                "Use per-class report and confusion matrix.",
                "Run one-variable ablations only: hidden_dim, layers, heads, norm_first.",
            ],
        })

    return {
        "best_epoch": int(best_epoch),
        "train": {
            "loss": train_loss,
            "accuracy": float(best_train["accuracy"]),
            "macro_f1": train_macro,
            "weighted_f1": float(best_train["weighted_f1"]),
        },
        "val": {
            "loss": val_loss,
            "accuracy": float(best_val["accuracy"]),
            "macro_f1": val_macro,
            "weighted_f1": float(best_val["weighted_f1"]),
        },
        "generalization_gap_macro_f1": float(gap),
        "worst_val_classes_by_f1": worst_classes,
        "diagnosis_hints_not_final_conclusion": hints,
        "ablation_order_recommended": [
            "Keep data fixed; run hidden_dim 64/128/256.",
            "Keep best hidden_dim; run num_layers 1/2/3/4.",
            "Keep best depth; run num_heads 2/4/8, ensuring hidden_dim % heads == 0.",
            "Compare --norm-first and --no-norm-first.",
            "Compare --use-class-weights and --no-use-class-weights.",
        ],
        "model_config": {
            "value_dim": args.value_dim,
            "feature_dim": args.feature_dim,
            "cell_dim": args.value_dim + args.feature_dim,
            "value_num_bins": args.value_num_bins,
            "uses_continuous_z": True,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "classifier_hidden_dim": args.classifier_hidden_dim,
            "classifier_dropout": args.classifier_dropout,
            "norm_first": args.norm_first,
            "use_class_weights": args.use_class_weights,
            "scheduler": args.scheduler,
            "warmup_epochs": args.warmup_epochs,
            "min_lr_ratio": args.min_lr_ratio,
        },
    }


def save_json(path: Path, obj: Dict[str, object]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    K = int(args.K)
    dataset_path = Path(args.dataset_npz) if args.dataset_npz else default_dataset_path(K)
    metadata_path = Path(args.metadata_json) if args.metadata_json else default_metadata_path(K)

    data, meta = load_token_dataset(dataset_path, metadata_path)

    X_train = data["X_train_tokens"].astype(np.int64)
    y_train = data["y_train"].astype(np.int64)
    X_val = data["X_val_tokens"].astype(np.int64)
    y_val = data["y_val"].astype(np.int64)

    # build_token.py stores continuous preprocessed z before rounding.
    # Use it so K controls coarse/discrete token features, while the model still
    # receives fine numeric signal. If an old NPZ is loaded, fall back to token/K.
    X_train_z = data.get("X_train_z")
    X_val_z = data.get("X_val_z")
    if X_train_z is None:
        X_train_z = X_train.astype(np.float32) / float(max(K, 1))
    else:
        X_train_z = X_train_z.astype(np.float32)
    if X_val_z is None:
        X_val_z = X_val.astype(np.float32) / float(max(K, 1))
    else:
        X_val_z = X_val_z.astype(np.float32)
    X_train_z = np.clip(X_train_z, 0.0, 1.0).astype(np.float32)
    X_val_z = np.clip(X_val_z, 0.0, 1.0).astype(np.float32)

    label_mapping = meta["label_mapping"]
    label_names = [label for label, _ in sorted(label_mapping.items(), key=lambda kv: kv[1])]
    num_classes = int(len(label_names))
    n_features = int(meta["n_features"])

    run_name = args.run_name
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = (
            f"{timestamp}_K{K}_h{args.hidden_dim}_L{args.num_layers}_"
            f"H{args.num_heads}_seed{args.seed}"
        )

    out_dir = Path(args.out_root) / f"K{K}" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)

    config_obj = {
        "stage": "train",
        "run_name": run_name,
        "K": K,
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
        "model": {
            "value_dim": int(args.value_dim),
            "feature_dim": int(args.feature_dim),
            "cell_dim": int(args.value_dim + args.feature_dim),
            "value_num_bins": int(args.value_num_bins),
            "uses_continuous_z": True,
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "dropout": float(args.dropout),
            "classifier_hidden_dim": int(args.classifier_hidden_dim),
            "classifier_dropout": float(args.classifier_dropout),
            "norm_first": bool(args.norm_first),
        },
        "data_shapes": {
            "X_train": list(X_train.shape),
            "X_train_z": list(X_train_z.shape),
            "y_train": list(y_train.shape),
            "X_val": list(X_val.shape),
            "X_val_z": list(X_val_z.shape),
            "y_val": list(y_val.shape),
        },
        "torch_version": torch.__version__,
    }
    save_json(out_dir / "config.json", config_obj)

    train_ds = TokenDataset(X_train, X_train_z, y_train)
    val_ds = TokenDataset(X_val, X_val_z, y_val)

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

    model = TabularTransformerClassifier(
        K=K,
        n_features=n_features,
        num_classes=num_classes,
        value_dim=int(args.value_dim),
        feature_dim=int(args.feature_dim),
        value_num_bins=int(args.value_num_bins),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        classifier_hidden_dim=int(args.classifier_hidden_dim),
        classifier_dropout=float(args.classifier_dropout),
        norm_first=bool(args.norm_first),
    ).to(device)

    if args.use_class_weights:
        weights = compute_class_weights(y_train, num_classes).to(device)
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

    save_json(out_dir / "class_weights.json", {
        "use_class_weights": bool(args.use_class_weights),
        "class_weights": class_weights_log,
        "label_names": label_names,
        "train_counts": np.bincount(y_train, minlength=num_classes).astype(int).tolist(),
        "val_counts": np.bincount(y_val, minlength=num_classes).astype(int).tolist(),
    })

    print("===== training start =====")
    print(f"run_dir: {out_dir}")
    print(f"device: {device}")
    print(f"dataset: {dataset_path}")
    print(f"train tokens/z shape: {X_train.shape}/{X_train_z.shape}, val tokens/z shape: {X_val.shape}/{X_val_z.shape}")
    print(f"classes: {num_classes}")
    print(f"model: {model}")
    print(f"use_class_weights: {args.use_class_weights}")
    print(f"scheduler: {args.scheduler}, warmup_epochs={args.warmup_epochs}, min_lr_ratio={args.min_lr_ratio}")

    history: List[Dict[str, object]] = []
    best_metric = -math.inf
    best_epoch = -1
    best_train_eval = None
    best_val_eval = None
    bad_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()

        lr_epoch = compute_epoch_lr(
            base_lr=float(args.lr),
            epoch=epoch,
            total_epochs=int(args.epochs),
            scheduler_name=str(args.scheduler),
            warmup_epochs=int(args.warmup_epochs),
            min_lr_ratio=float(args.min_lr_ratio),
        )
        set_optimizer_lr(optimizer, lr_epoch)

        train_step_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=float(args.grad_clip_norm),
        )

        train_eval = evaluate(model, train_eval_loader, criterion, device, num_classes, label_names)
        val_eval = evaluate(model, val_loader, criterion, device, num_classes, label_names)

        epoch_seconds = time.time() - t0
        lr_now = float(lr_epoch)

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "epoch_seconds": round(epoch_seconds, 3),
            "train_step_loss": float(train_step_loss),
            "train_loss": float(train_eval["loss"]),
            "train_acc": float(train_eval["accuracy"]),
            "train_macro_f1": float(train_eval["macro_f1"]),
            "train_weighted_f1": float(train_eval["weighted_f1"]),
            "val_loss": float(val_eval["loss"]),
            "val_acc": float(val_eval["accuracy"]),
            "val_macro_f1": float(val_eval["macro_f1"]),
            "val_weighted_f1": float(val_eval["weighted_f1"]),
            "macro_f1_gap_train_minus_val": float(train_eval["macro_f1"] - val_eval["macro_f1"]),
        }
        history.append(row)
        write_history_csv(out_dir / "history.csv", history)

        metric = float(val_eval["macro_f1"])
        improved = metric > best_metric + float(args.min_delta)

        if improved:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0
            best_train_eval = train_eval
            best_val_eval = val_eval

            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config_obj,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_metric,
                "label_names": label_names,
            }, out_dir / "best_model.pt")
        else:
            bad_epochs += 1

        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config_obj,
            "epoch": epoch,
            "val_macro_f1": metric,
            "label_names": label_names,
        }, out_dir / "last_model.pt")

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={row['train_loss']:.4f} "
            f"train_macroF1={row['train_macro_f1']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_macroF1={row['val_macro_f1']:.4f} "
            f"val_acc={row['val_acc']:.4f} "
            f"gap={row['macro_f1_gap_train_minus_val']:.4f} "
            f"best={best_metric:.4f}@{best_epoch} "
            f"bad_epochs={bad_epochs}"
        )

        if bad_epochs >= int(args.patience):
            print(f"early stopping at epoch {epoch}; best epoch={best_epoch}, best val macro-F1={best_metric:.4f}")
            break

    if best_epoch < 0:
        raise RuntimeError("No best epoch was recorded.")

    # Reload best checkpoint and compute final detailed reports.
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    train_best = evaluate(model, train_eval_loader, criterion, device, num_classes, label_names, collect_probs=True)
    val_best = evaluate(model, val_loader, criterion, device, num_classes, label_names, collect_probs=True)

    save_json(out_dir / "train_classification_report_best.json", {
        "loss": train_best["loss"],
        "accuracy": train_best["accuracy"],
        "macro_f1": train_best["macro_f1"],
        "weighted_f1": train_best["weighted_f1"],
        "per_class": train_best["per_class"],
        "true_counts": train_best["true_counts"],
        "pred_counts": train_best["pred_counts"],
    })
    save_json(out_dir / "val_classification_report_best.json", {
        "loss": val_best["loss"],
        "accuracy": val_best["accuracy"],
        "macro_f1": val_best["macro_f1"],
        "weighted_f1": val_best["weighted_f1"],
        "per_class": val_best["per_class"],
        "true_counts": val_best["true_counts"],
        "pred_counts": val_best["pred_counts"],
    })

    write_confusion_outputs(out_dir, "train", train_best, label_names)
    write_confusion_outputs(out_dir, "val", val_best, label_names)

    write_predictions_csv(
        out_dir / "val_predictions_best.csv",
        y_true=val_best["y_true"],
        y_pred=val_best["y_pred"],
        confidence=val_best["confidence"],
        label_names=label_names,
    )

    diagnosis = make_diagnosis_summary(
        best_epoch=best_epoch,
        best_train=train_best,
        best_val=val_best,
        label_names=label_names,
        args=args,
    )
    save_json(out_dir / "diagnosis_summary.json", diagnosis)

    print("===== training done =====")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_macro_f1: {float(val_best['macro_f1']):.6f}")
    print(f"best_val_weighted_f1: {float(val_best['weighted_f1']):.6f}")
    print(f"best_val_acc: {float(val_best['accuracy']):.6f}")
    print(f"run_dir: {out_dir}")
    print("Key files:")
    print(f"  {out_dir / 'history.csv'}")
    print(f"  {out_dir / 'diagnosis_summary.json'}")
    print(f"  {out_dir / 'val_classification_report_best.json'}")
    print(f"  {out_dir / 'val_confusion_matrix_best.csv'}")
    print(f"  {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
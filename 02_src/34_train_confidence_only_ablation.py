#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
34_train_confidence_only_ablation.py

Test 1: confidence-only ablation for C2 D3.

This script intentionally keeps fixed:
  - C2 K512 tokenization artifact
  - D3 shared-bin embedding + offset interpolation + raw FiLM
  - Transformer CLS backbone
  - class weights
  - optimizer/lr/dropout/model size

It changes only label smoothing strength. A local control run with label_smoothing=0.0
is included so the effect can be separated from this runner's training loop.

Outputs per run:
  - best_model.pt / last_model.pt
  - history.csv
  - diagnosis_summary.json
  - train/val classification reports
  - train/val confusion matrices
  - train/val predictions

Then it optionally launches:
  - overfit root-cause audit for every run
  - attention/gradient/occlusion audit for the control and best LS run
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
except Exception as e:  # pragma: no cover
    raise RuntimeError("sklearn is required for metrics") from e


def repo_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "02_src":
        return p.parents[1]
    return Path.cwd()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def auto_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def import_train_module(root: Path):
    src = root / "02_src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    p = src / "10_train_fusion_ablation_D0_D7.py"
    if not p.exists():
        raise FileNotFoundError(f"Missing training module: {p}")
    spec = importlib.util.spec_from_file_location("_dacn_fusion", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def labels_from_meta(meta: Dict[str, Any]) -> List[str]:
    lm = meta.get("label_mapping", {})
    if isinstance(lm, dict) and lm:
        inv = {int(v): str(k) for k, v in lm.items()}
        return [inv[i] for i in sorted(inv)]
    return ["Benign", "Ransomware", "Spyware", "Trojan"]


def feature_names_from_meta(meta: Dict[str, Any], n_features: int) -> List[str]:
    for k in ["feature_names", "features", "selected_features", "columns"]:
        v = meta.get(k)
        if isinstance(v, list) and len(v) == n_features:
            return [str(x) for x in v]
    return [f"f{i}" for i in range(n_features)]


def load_raw_scaled(root: Path, meta: Dict[str, Any], X_train_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    n_features = int(X_train_bin.shape[1])
    features = feature_names_from_meta(meta, n_features)
    train_path = root / "01_split/train_raw.csv"
    val_path = root / "01_split/val_raw.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing raw split CSVs: {train_path}, {val_path}")
    tr = pd.read_csv(train_path)
    va = pd.read_csv(val_path)
    missing = [f for f in features if f not in tr.columns or f not in va.columns]
    if missing:
        raise ValueError(f"Raw CSV missing feature columns: {missing[:20]}")
    Xtr = tr.loc[:, features].to_numpy(np.float64)
    Xva = va.loc[:, features].to_numpy(np.float64)
    mn = Xtr.min(axis=0)
    mx = Xtr.max(axis=0)
    den = mx - mn
    const = np.isclose(den, 0.0)
    den[const] = 1.0
    Str = (Xtr - mn) / den
    Sva = (Xva - mn) / den
    Str[:, const] = 0.5
    Sva[:, const] = 0.5
    Str = np.clip(Str, 0.0, 1.0).astype(np.float32)
    Sva = np.clip(Sva, 0.0, 1.0).astype(np.float32)
    info = {
        "scale": "train_minmax_clip_val",
        "train_raw": str(train_path),
        "val_raw": str(val_path),
        "n_constant_features": int(const.sum()),
    }
    return Str, Sva, info


def build_values(X_offset: np.ndarray, X_cont: np.ndarray) -> np.ndarray:
    mask = np.ones_like(X_offset, dtype=np.float32)
    return np.stack([X_offset.astype(np.float32), X_cont.astype(np.float32), mask], axis=-1)


def model_config_from_baseline(root: Path, meta: Dict[str, Any], n_features: int, num_classes: int) -> Dict[str, Any]:
    base_diag = root / "03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/diagnosis_summary.json"
    mcfg: Dict[str, Any] = {}
    if base_diag.exists():
        try:
            mcfg = read_json(base_diag).get("model_config", {}) or {}
        except Exception:
            mcfg = {}
    def get(k: str, default: Any) -> Any:
        return mcfg.get(k, default)
    return {
        "run_id": "D3",
        "num_bins": int(get("num_bins", 512)),
        "n_features": int(n_features),
        "num_classes": int(num_classes),
        "value_dim": int(get("value_dim", 32)),
        "feature_dim": int(get("feature_dim", 32)),
        "hidden_dim": int(get("hidden_dim", 128)),
        "num_layers": int(get("num_layers", 3)),
        "num_heads": int(get("num_heads", 4)),
        "dropout": float(get("dropout", 0.1)),
        "classifier_hidden_dim": int(get("classifier_hidden_dim", 128)),
        "classifier_dropout": float(get("classifier_dropout", 0.1)),
        "norm_first": bool(get("norm_first", True)),
        "gate_init": float(get("gate_init", 0.0)),
    }


def make_model(train_mod, cfg: Dict[str, Any]) -> nn.Module:
    return train_mod.FusionAblationTransformer(**cfg)


def class_weights_balanced(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=num_classes).astype(np.float64)
    weights = len(y) / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def macro_malware_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> float:
    # Malware classes are all non-Benign if Benign is present at index 0.
    idx = [i for i, name in enumerate(labels) if str(name).lower() != "benign"]
    if not idx:
        return float("nan")
    per = f1_score(y_true, y_pred, labels=idx, average=None, zero_division=0)
    return float(np.mean(per))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, labels: List[str], class_weights: torch.Tensor, label_smoothing: float) -> Dict[str, Any]:
    model.eval()
    all_y, all_pred, all_probs, losses = [], [], [], []
    weight = class_weights.to(device)
    with torch.no_grad():
        for xb, vv, yb in loader:
            xb = xb.to(device, non_blocking=True)
            vv = vv.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb, vv)
            loss = F.cross_entropy(logits, yb, weight=weight, label_smoothing=float(label_smoothing))
            probs = F.softmax(logits, dim=-1)
            losses.append(float(loss.item()) * int(yb.numel()))
            all_y.append(yb.detach().cpu().numpy())
            all_pred.append(probs.argmax(dim=1).detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
    y = np.concatenate(all_y)
    pred = np.concatenate(all_pred)
    probs = np.concatenate(all_probs)
    report = classification_report(y, pred, target_names=labels, labels=list(range(len(labels))), output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(range(len(labels))))
    correct = pred == y
    out = {
        "loss": float(np.sum(losses) / max(1, len(y))),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "malware_only_avg_f1": macro_malware_f1(y, pred, labels),
        "classification_report": report,
        "confusion_matrix": cm,
        "y_true": y,
        "y_pred": pred,
        "probs": probs,
        "confidence": probs.max(axis=1),
        "wrong_rate": float(1.0 - accuracy_score(y, pred)),
        "correct_confidence_mean": float(probs.max(axis=1)[correct].mean()) if correct.any() else float("nan"),
        "wrong_confidence_mean": float(probs.max(axis=1)[~correct].mean()) if (~correct).any() else float("nan"),
    }
    return out


def save_eval_files(run_dir: Path, split: str, metrics: Dict[str, Any], labels: List[str]) -> None:
    write_json(run_dir / f"{split}_classification_report_best.json", metrics["classification_report"])
    cm = metrics["confusion_matrix"]
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(run_dir / f"{split}_confusion_matrix_best.csv")
    write_json(run_dir / f"{split}_confusion_matrix_best.json", {"labels": labels, "matrix": cm.tolist()})
    pred_df = pd.DataFrame({
        "sample_idx": np.arange(len(metrics["y_true"])),
        "y_true": metrics["y_true"].astype(int),
        "y_pred": metrics["y_pred"].astype(int),
        "true_class": [labels[int(i)] for i in metrics["y_true"]],
        "pred_class": [labels[int(i)] for i in metrics["y_pred"]],
        "correct": metrics["y_true"] == metrics["y_pred"],
        "confidence": metrics["confidence"],
    })
    for i, name in enumerate(labels):
        pred_df[f"prob_{name}"] = metrics["probs"][:, i]
    pred_df.to_csv(run_dir / f"{split}_predictions_best.csv", index=False)


def pair_error_table(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> pd.DataFrame:
    rows = []
    for ti, tn in enumerate(labels):
        m = y_true == ti
        denom = int(m.sum())
        for pi, pn in enumerate(labels):
            if ti == pi:
                continue
            c = int(((y_true == ti) & (y_pred == pi)).sum())
            rows.append({
                "true_class": tn,
                "pred_class": pn,
                "count": c,
                "rate_within_true": float(c / denom) if denom else 0.0,
                "support_true": denom,
            })
    return pd.DataFrame(rows).sort_values(["count"], ascending=False)


def train_one_run(
    *, root: Path, train_mod, data: Dict[str, np.ndarray], meta: Dict[str, Any], labels: List[str],
    run_name: str, label_smoothing: float, args: argparse.Namespace, device: torch.device,
) -> Dict[str, Any]:
    seed_all(int(args.seed))
    n_features = data["X_train_bin"].shape[1]
    cfg = model_config_from_baseline(root, meta, n_features, len(labels))
    model = make_model(train_mod, cfg).to(device)

    Str, Sva, scale_info = load_raw_scaled(root, meta, data["X_train_bin"])
    Vtr = build_values(data["X_train_offset"], Str)
    Vva = build_values(data["X_val_offset"], Sva)

    Xtr = data["X_train_bin"].astype(np.int64)
    Xva = data["X_val_bin"].astype(np.int64)
    ytr = data["y_train"].astype(np.int64)
    yva = data["y_val"].astype(np.int64)

    class_w = class_weights_balanced(ytr, len(labels))
    train_ds = TensorDataset(
        torch.tensor(Xtr, dtype=torch.long),
        torch.tensor(Vtr, dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(Xva, dtype=torch.long),
        torch.tensor(Vva, dtype=torch.float32),
        torch.tensor(yva, dtype=torch.long),
    )
    g = torch.Generator()
    g.manual_seed(int(args.seed))
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(), generator=g)
    train_eval_loader = DataLoader(train_ds, batch_size=int(args.eval_batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=int(args.eval_batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available())

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    total_steps = max(1, int(args.epochs) * len(train_loader))
    warmup_steps = max(1, int(args.warmup_epochs) * len(train_loader))
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    out_root = root / args.out_root
    run_dir = ensure_dir(out_root / "Keff512" / run_name)
    write_json(run_dir / "config.json", {
        "run_name": run_name,
        "test": "confidence_only_label_smoothing",
        "label_smoothing": float(label_smoothing),
        "seed": int(args.seed),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "model_config": cfg,
        "class_weights": class_w.tolist(),
        "scale_info": scale_info,
        "fixed_components": ["C2 tokenization", "D3 architecture", "class weights", "dropout/model size", "optimizer schedule"],
        "changed_component": "label_smoothing_only",
    })
    write_json(run_dir / "class_weights.json", {labels[i]: float(class_w[i]) for i in range(len(labels))})

    best_val = -1.0
    best_epoch = -1
    best_state = None
    bad = 0
    history = []
    global_step = 0
    weight_dev = class_w.to(device)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, vv, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            vv = vv.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, vv)
            loss = F.cross_entropy(logits, yb, weight=weight_dev, label_smoothing=float(label_smoothing))
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            scheduler.step()
            global_step += 1
            running_loss += float(loss.item()) * int(yb.numel())
            seen += int(yb.numel())
        train_metrics = evaluate(model, train_eval_loader, device, labels, class_w, label_smoothing)
        val_metrics = evaluate(model, val_loader, device, labels, class_w, label_smoothing)
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss_batch_avg": float(running_loss / max(1, seen)),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_acc": val_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "train_malware_only_f1": train_metrics["malware_only_avg_f1"],
            "val_malware_only_f1": val_metrics["malware_only_avg_f1"],
            "train_wrong_confidence_mean": train_metrics["wrong_confidence_mean"],
            "val_wrong_confidence_mean": val_metrics["wrong_confidence_mean"],
            "train_correct_confidence_mean": train_metrics["correct_confidence_mean"],
            "val_correct_confidence_mean": val_metrics["correct_confidence_mean"],
        }
        history.append(row)
        print(f"[{run_name}] epoch={epoch:03d} train_macro={row['train_macro_f1']:.6f} val_macro={row['val_macro_f1']:.6f} val_loss={row['val_loss']:.6f} val_wrong_conf={row['val_wrong_confidence_mean']:.6f}", flush=True)
        if val_metrics["macro_f1"] > best_val:
            best_val = float(val_metrics["macro_f1"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
            torch.save({
                "model_state_dict": best_state,
                "config": cfg,
                "label_names": labels,
                "label_smoothing": float(label_smoothing),
                "best_epoch": best_epoch,
            }, run_dir / "best_model.pt")
        else:
            bad += 1
        if bad >= int(args.patience):
            print(f"[{run_name}] early stop at epoch {epoch}; best_epoch={best_epoch}; best_val_macro={best_val:.6f}", flush=True)
            break

    torch.save({
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": cfg,
        "label_names": labels,
        "label_smoothing": float(label_smoothing),
        "best_epoch": best_epoch,
    }, run_dir / "last_model.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    # Reload best for final reports.
    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    train_metrics = evaluate(model, train_eval_loader, device, labels, class_w, label_smoothing)
    val_metrics = evaluate(model, val_loader, device, labels, class_w, label_smoothing)
    save_eval_files(run_dir, "train", train_metrics, labels)
    save_eval_files(run_dir, "val", val_metrics, labels)
    pair_error_table(train_metrics["y_true"], train_metrics["y_pred"], labels).to_csv(run_dir / "train_pair_errors_best.csv", index=False)
    pair_error_table(val_metrics["y_true"], val_metrics["y_pred"], labels).to_csv(run_dir / "val_pair_errors_best.csv", index=False)

    diag = {
        "run_name": run_name,
        "test": "confidence_only_label_smoothing",
        "label_smoothing": float(label_smoothing),
        "best_epoch": int(best_epoch),
        "model_config": cfg,
        "train": {k: train_metrics[k] for k in ["loss", "accuracy", "macro_f1", "weighted_f1", "malware_only_avg_f1", "wrong_rate", "correct_confidence_mean", "wrong_confidence_mean"]},
        "val": {k: val_metrics[k] for k in ["loss", "accuracy", "macro_f1", "weighted_f1", "malware_only_avg_f1", "wrong_rate", "correct_confidence_mean", "wrong_confidence_mean"]},
        "generalization_gap_macro_f1": float(train_metrics["macro_f1"] - val_metrics["macro_f1"]),
        "generalization_gap_acc": float(train_metrics["accuracy"] - val_metrics["accuracy"]),
        "wrong_confidence_gap_val_minus_train": float(val_metrics["wrong_confidence_mean"] - train_metrics["wrong_confidence_mean"]),
        "class_weights": class_w.tolist(),
    }
    write_json(run_dir / "diagnosis_summary.json", diag)
    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "label_smoothing": float(label_smoothing),
        "best_epoch": int(best_epoch),
        "train_macro_f1": diag["train"]["macro_f1"],
        "val_macro_f1": diag["val"]["macro_f1"],
        "gap_macro_f1": diag["generalization_gap_macro_f1"],
        "val_malware_only_f1": diag["val"]["malware_only_avg_f1"],
        "train_wrong_confidence": diag["train"]["wrong_confidence_mean"],
        "val_wrong_confidence": diag["val"]["wrong_confidence_mean"],
        "wrong_confidence_gap_val_minus_train": diag["wrong_confidence_gap_val_minus_train"],
        "val_loss": diag["val"]["loss"],
    }


def run_subprocess(cmd: List[str], cwd: Path) -> None:
    print("\n$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="", flush=True)
    code = p.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def zip_items(zip_path: Path, root: Path, items: List[Path]) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for item in items:
            if not item.exists():
                print(f"[zip skip missing] {item}", flush=True)
                continue
            if item.is_file():
                z.write(item, item.relative_to(root).as_posix())
            else:
                for p in item.rglob("*"):
                    if p.is_file():
                        z.write(p, p.relative_to(root).as_posix())
    print(f"[ZIP] {zip_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-npz", default="03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_dataset.npz")
    ap.add_argument("--metadata-json", default="03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_metadata.json")
    ap.add_argument("--out-root", default="03_outputs/train_runs_test1_confidence_only")
    ap.add_argument("--audit-root", default="03_outputs/audit_test1_confidence_only")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=8)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--skip-overfit-audit", action="store_true")
    ap.add_argument("--skip-attn-grad-audit", action="store_true")
    ap.add_argument("--max-train-knn", type=int, default=30000)
    ap.add_argument("--audit-batch-size", type=int, default=512)
    ap.add_argument("--attn-max-samples-per-subset", type=int, default=220)
    ap.add_argument("--attn-max-samples-per-pair", type=int, default=50)
    args = ap.parse_args()

    root = repo_root()
    device = auto_device(args.device)
    print(f"[test1] root={root}", flush=True)
    print(f"[test1] device={device}", flush=True)
    train_mod = import_train_module(root)
    dataset_path = root / args.dataset_npz if not Path(args.dataset_npz).is_absolute() else Path(args.dataset_npz)
    metadata_path = root / args.metadata_json if not Path(args.metadata_json).is_absolute() else Path(args.metadata_json)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    data = dict(np.load(dataset_path, allow_pickle=True))
    meta = read_json(metadata_path)
    labels = labels_from_meta(meta)
    print(f"[test1] labels={labels}", flush=True)

    specs = [
        ("T1CTRL_C2_D3_CE_LS000", 0.00, "local control: same runner, no label smoothing"),
        ("T1A_C2_D3_LS003", 0.03, "confidence-only: label smoothing 0.03"),
        ("T1B_C2_D3_LS005", 0.05, "confidence-only: label smoothing 0.05"),
    ]
    rows = []
    for run_name, ls, note in specs:
        print(f"\n========== RUN {run_name} label_smoothing={ls} ==========" , flush=True)
        row = train_one_run(root=root, train_mod=train_mod, data=data, meta=meta, labels=labels, run_name=run_name, label_smoothing=ls, args=args, device=device)
        row["note"] = note
        rows.append(row)

    summary_dir = ensure_dir(root / args.audit_root / "00_test_result_summary")
    pd.DataFrame(rows).to_csv(summary_dir / "test1_result_summary.csv", index=False)
    write_json(summary_dir / "test1_result_summary.json", rows)

    # Compare each LS run to local control and old C2 if available.
    comp = []
    ctrl = next(r for r in rows if r["label_smoothing"] == 0.0)
    old_diag_path = root / "03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/diagnosis_summary.json"
    old = None
    if old_diag_path.exists():
        d = read_json(old_diag_path)
        old = {
            "run_name": "OLD_C2_D3_BASELINE",
            "val_macro_f1": d.get("val", {}).get("macro_f1"),
            "train_macro_f1": d.get("train", {}).get("macro_f1"),
            "gap_macro_f1": d.get("generalization_gap_macro_f1"),
            "val_malware_only_f1": d.get("val", {}).get("malware_only_avg_f1"),
            "val_wrong_confidence": d.get("val", {}).get("wrong_confidence_mean"),
            "train_wrong_confidence": d.get("train", {}).get("wrong_confidence_mean"),
            "val_loss": d.get("val", {}).get("loss"),
        }
    for r in rows:
        c = {"run_name": r["run_name"], "label_smoothing": r["label_smoothing"]}
        for k in ["val_macro_f1", "gap_macro_f1", "val_malware_only_f1", "val_wrong_confidence", "wrong_confidence_gap_val_minus_train", "val_loss"]:
            c[k] = r.get(k)
            c[f"delta_{k}_vs_local_control"] = None if ctrl.get(k) is None or r.get(k) is None else float(r[k] - ctrl[k])
            if old and old.get(k) is not None and r.get(k) is not None:
                c[f"delta_{k}_vs_old_C2"] = float(r[k] - old[k])
        comp.append(c)
    pd.DataFrame(comp).to_csv(summary_dir / "test1_delta_vs_control.csv", index=False)

    # Launch audits.
    audit_root = root / args.audit_root
    overfit_script = root / "02_src/32_audit_overfit_rootcause.py"
    attn_script = root / "02_src/33_audit_attention_gradient_rootcause.py"
    if not args.skip_overfit_audit:
        if not overfit_script.exists():
            print(f"[WARN] missing overfit audit script: {overfit_script}; skip", flush=True)
        else:
            for r in rows:
                run_dir = Path(r["run_dir"])
                out_dir = audit_root / "01_overfit_rootcause_by_run" / r["run_name"]
                cmd = [
                    sys.executable, "-u", str(overfit_script.relative_to(root)),
                    "--dataset-npz", str(dataset_path.relative_to(root)),
                    "--metadata-json", str(metadata_path.relative_to(root)),
                    "--run-dir", str(run_dir.relative_to(root)),
                    "--checkpoint", str((run_dir / "best_model.pt").relative_to(root)),
                    "--out-dir", str(out_dir.relative_to(root)),
                    "--device", str(device),
                    "--batch-size", str(args.audit_batch_size),
                    "--rare-threshold", "5",
                    "--knn-k", "25",
                    "--max-train-knn", str(args.max_train_knn),
                    "--seed", str(args.seed),
                ]
                run_subprocess(cmd, cwd=root)

    if not args.skip_attn_grad_audit:
        if not attn_script.exists():
            print(f"[WARN] missing attention/gradient audit script: {attn_script}; skip", flush=True)
        else:
            # Run logit/occlusion audit for local control and best non-control LS run.
            ls_runs = [r for r in rows if r["label_smoothing"] > 0]
            best_ls = sorted(ls_runs, key=lambda x: x["val_macro_f1"], reverse=True)[0] if ls_runs else None
            selected = [ctrl] + ([best_ls] if best_ls is not None else [])
            seen = set()
            for r in selected:
                if r["run_name"] in seen:
                    continue
                seen.add(r["run_name"])
                run_dir = Path(r["run_dir"])
                overfit_dir = audit_root / "01_overfit_rootcause_by_run" / r["run_name"]
                if not overfit_dir.exists():
                    print(f"[WARN] overfit audit missing for {r['run_name']}; skip attn/grad", flush=True)
                    continue
                out_dir = audit_root / "02_attention_gradient_by_run" / r["run_name"]
                cmd = [
                    sys.executable, "-u", str(attn_script.relative_to(root)),
                    "--dataset-npz", str(dataset_path.relative_to(root)),
                    "--metadata-json", str(metadata_path.relative_to(root)),
                    "--run-dir", str(run_dir.relative_to(root)),
                    "--checkpoint", str((run_dir / "best_model.pt").relative_to(root)),
                    "--overfit-audit-dir", str(overfit_dir.relative_to(root)),
                    "--out-dir", str(out_dir.relative_to(root)),
                    "--device", str(device),
                    "--batch-size", "128",
                    "--max-samples-per-subset", str(args.attn_max_samples_per_subset),
                    "--max-samples-per-pair", str(args.attn_max_samples_per_pair),
                ]
                run_subprocess(cmd, cwd=root)

    # Minimal markdown summary. Deep interpretation will be done after upload.
    lines = [
        "# Test 1 confidence-only ablation",
        "",
        "Fixed: C2 tokenization, D3 architecture, class weights, optimizer/dropout/model size.",
        "Changed only: label smoothing strength.",
        "",
        "## Result summary",
        "",
    ]
    try:
        lines.append(pd.DataFrame(rows).to_markdown(index=False))
    except Exception:
        lines.append(pd.DataFrame(rows).to_string(index=False))
    lines += ["", "## Interpretation checklist", "", "- Did val wrong confidence decrease?", "- Did val macro-F1 or malware-only F1 improve?", "- Did CLS amplification/mixed→pred decrease in overfit audit?", "- Did keep_current wrong-margin decrease in occlusion audit?", "- Did any pair simply trade errors to another malware class?"]
    (summary_dir / "test1_readme_for_analysis.md").write_text("\n".join(lines), encoding="utf-8")

    zip_path = Path("/kaggle/working/test1_confidence_only_outputs.zip") if Path("/kaggle/working").exists() else root / "test1_confidence_only_outputs.zip"
    items = [root / args.out_root, root / args.audit_root]
    # Also include old C2 reference files if present.
    old_run = root / "03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact"
    for rel in ["diagnosis_summary.json", "history.csv", "val_classification_report_best.json", "val_confusion_matrix_best.csv", "train_classification_report_best.json", "train_confusion_matrix_best.csv"]:
        p = old_run / rel
        if p.exists():
            items.append(p)
    zip_items(zip_path, root, items)
    print(f"[DONE] {zip_path}", flush=True)


if __name__ == "__main__":
    main()

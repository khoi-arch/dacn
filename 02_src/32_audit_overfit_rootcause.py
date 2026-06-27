#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
32_audit_overfit_rootcause.py

Root-cause audit for C2 D3 overfit.
This script DOES NOT train and DOES NOT change preprocessing/model.

It classifies the current error source into:
  - train/val generalization gap vs local-underfit
  - raw/token/CLS feature-space overlap
  - OOD / train-val distribution shift
  - model-boundary failure
  - group-wise overfit / harmful feature group dependence
  - rare-token cause vs symptom

Outputs:
  03_outputs/audit_overfit_rootcause/
    01_train_val_error_gap/
    02_knn_raw_token_cls/
    03_group_masking/
    04_cls_geometry/
    05_rare_causal/
    audit_overfit_rootcause_summary.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    classification_report = None
    confusion_matrix = None
    f1_score = None
    accuracy_score = None
    NearestNeighbors = None
    StandardScaler = None
    _SKLEARN_IMPORT_ERROR = e
else:
    _SKLEARN_IMPORT_ERROR = None


# ----------------------------- misc utils -----------------------------

def repo_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "02_src":
        return p.parents[1]
    return Path.cwd()


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def find_first(root: Path, patterns: List[str], must_contain: Optional[List[str]] = None) -> Optional[Path]:
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(root.rglob(pat))
    if must_contain:
        candidates = [c for c in candidates if all(x in str(c) for x in must_contain)]
    candidates = sorted(set(candidates), key=lambda x: (len(str(x)), str(x)))
    return candidates[0] if candidates else None


def auto_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def to_markdown_safe(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or len(df) == 0:
        return "(empty)"
    d = df.head(max_rows).copy()
    try:
        return d.to_markdown(index=False)
    except Exception:
        return d.to_string(index=False)


def safe_torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# ----------------------------- paths/data -----------------------------

def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    root = repo_root()

    def rel(p: Optional[str]) -> Optional[Path]:
        if not p:
            return None
        q = Path(p)
        return q if q.is_absolute() else root / q

    dataset = rel(args.dataset_npz)
    metadata = rel(args.metadata_json)
    run_dir = rel(args.run_dir)
    checkpoint = rel(args.checkpoint)
    audit_dir = rel(args.c2_audit_dir)

    if dataset is None or not dataset.exists():
        dataset = find_first(root, ["mixed_quantile_offset_dataset.npz"], ["K512_B512_C2_selective_rank_discrete_compact"])
    if metadata is None or not metadata.exists():
        metadata = find_first(root, ["mixed_quantile_offset_metadata.json"], ["K512_B512_C2_selective_rank_discrete_compact"])
    if run_dir is None or not run_dir.exists():
        p = find_first(root, ["diagnosis_summary.json"], ["D3_P1_K512_B512_C2_selective_rank_discrete_compact"])
        run_dir = p.parent if p else None
    if checkpoint is None or not checkpoint.exists():
        if run_dir is not None and (run_dir / "best_model.pt").exists():
            checkpoint = run_dir / "best_model.pt"
        else:
            checkpoint = find_first(root, ["best_model.pt"], ["D3_P1_K512_B512_C2_selective_rank_discrete_compact"])
    if audit_dir is None or not audit_dir.exists():
        cand = root / "03_outputs" / "audit_c2_best"
        audit_dir = cand if cand.exists() else None

    missing = []
    if dataset is None or not dataset.exists():
        missing.append("C2 dataset npz")
    if metadata is None or not metadata.exists():
        missing.append("C2 metadata json")
    if run_dir is None or not run_dir.exists():
        missing.append("C2 D3 run dir")
    if checkpoint is None or not checkpoint.exists():
        missing.append("C2 best_model.pt")
    if missing:
        raise FileNotFoundError("Missing required inputs:\n- " + "\n- ".join(missing))

    return {
        "root": root,
        "dataset": dataset,
        "metadata": metadata,
        "run_dir": run_dir,
        "checkpoint": checkpoint,
        "audit_dir": audit_dir if audit_dir else root / "__missing_audit_dir__",
    }


def load_dataset(dataset_path: Path, metadata_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    arr = np.load(dataset_path, allow_pickle=True)
    data = {k: arr[k] for k in arr.files}
    meta = read_json(metadata_path)
    required = ["X_train_bin", "X_train_offset", "y_train", "X_val_bin", "X_val_offset", "y_val"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"dataset missing keys: {missing}; available={list(data.keys())}")
    return data, meta


def get_features(meta: Dict[str, Any], n_features: int) -> List[str]:
    for k in ["features", "feature_names", "selected_features", "columns"]:
        v = meta.get(k)
        if isinstance(v, list) and len(v) == n_features:
            return [str(x) for x in v]
    # Some metadata stores feature entries as dict.
    fs = meta.get("feature_strategies")
    if isinstance(fs, dict) and len(fs) == n_features:
        return list(fs.keys())
    return [f"f{i}" for i in range(n_features)]


def labels_from_meta(meta: Dict[str, Any]) -> List[str]:
    lm = meta.get("label_mapping", {})
    if isinstance(lm, dict) and lm:
        inv = {int(v): str(k) for k, v in lm.items()}
        return [inv[i] for i in sorted(inv)]
    return ["Benign", "Ransomware", "Spyware", "Trojan"]


def feature_strategies(meta: Dict[str, Any], features: List[str]) -> Dict[str, str]:
    fs = meta.get("feature_strategies", {})
    fm_all = meta.get("feature_meta", {})
    out = {}
    for f in features:
        v = fs.get(f) if isinstance(fs, dict) else None
        if isinstance(v, dict):
            out[f] = str(v.get("strategy", v.get("selected_strategy", "unknown")))
        elif v is not None:
            out[f] = str(v)
        elif isinstance(fm_all, dict) and isinstance(fm_all.get(f), dict):
            fm = fm_all[f]
            out[f] = str(fm.get("strategy", fm.get("selected_strategy", "unknown")))
        else:
            out[f] = "unknown"
    return out


def raw_scaled_from_train(root: Path, features: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    train_p = root / "01_split" / "train_raw.csv"
    val_p = root / "01_split" / "val_raw.csv"
    if not train_p.exists() or not val_p.exists():
        return None, None, {"available": False, "reason": "missing 01_split/train_raw.csv or val_raw.csv"}
    tr = pd.read_csv(train_p)
    va = pd.read_csv(val_p)
    missing = [f for f in features if f not in tr.columns or f not in va.columns]
    if missing:
        return None, None, {"available": False, "reason": f"raw csv missing features: {missing[:10]}"}
    Xtr = tr.loc[:, features].to_numpy(dtype=np.float64)
    Xva = va.loc[:, features].to_numpy(dtype=np.float64)
    mn = Xtr.min(axis=0)
    mx = Xtr.max(axis=0)
    denom = mx - mn
    constant = np.isclose(denom, 0.0)
    denom[constant] = 1.0
    Rtr = np.clip((Xtr - mn) / denom, 0.0, 1.0).astype(np.float32)
    Rva = np.clip((Xva - mn) / denom, 0.0, 1.0).astype(np.float32)
    Rtr[:, constant] = 0.5
    Rva[:, constant] = 0.5
    return Rtr, Rva, {"available": True, "constant_features": [features[i] for i, c in enumerate(constant) if bool(c)]}


# ----------------------------- model -----------------------------

def import_train_module(root: Path):
    candidates = [
        root / "02_src" / "10_train_fusion_ablation_D0_D7.py",
        root / "02_src" / "10_train_fusion_ablation_D0_D7_boundary_fixed.py",
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("_dacn_fusion_train", p)
            mod = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            sys.path.insert(0, str(root / "02_src"))
            sys.path.insert(0, str(root))
            spec.loader.exec_module(mod)
            return mod, p
    raise FileNotFoundError("Cannot find 02_src/10_train_fusion_ablation_D0_D7.py")


def build_model(train_mod, ckpt: Dict[str, Any], num_bins: int, n_features: int, num_classes: int, device: torch.device):
    cfg = ckpt.get("config", {}).get("model_config", {}) if isinstance(ckpt, dict) else {}
    def get(name, default):
        return cfg.get(name, default)
    model = train_mod.FusionAblationTransformer(
        run_id="D3",
        num_bins=int(get("num_bins", num_bins)),
        n_features=int(n_features),
        num_classes=int(num_classes),
        value_dim=int(get("value_dim", 32)),
        feature_dim=int(get("feature_dim", 32)),
        hidden_dim=int(get("hidden_dim", 128)),
        num_layers=int(get("num_layers", 3)),
        num_heads=int(get("num_heads", 4)),
        dropout=float(get("dropout", 0.1)),
        classifier_hidden_dim=int(get("classifier_hidden_dim", 128)),
        classifier_dropout=float(get("classifier_dropout", 0.1)),
        norm_first=bool(get("norm_first", True)),
        gate_init=float(get("gate_init", 0.0)),
    ).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_model(model, X_bin: np.ndarray, X_offset: np.ndarray, X_cont: np.ndarray, batch_size: int, device: torch.device):
    mask = np.ones_like(X_bin, dtype=np.float32)
    vals = np.stack([X_offset.astype(np.float32), X_cont.astype(np.float32), mask], axis=-1)
    ds = TensorDataset(torch.as_tensor(X_bin, dtype=torch.long), torch.as_tensor(vals, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_all = []
    for xb, vb in dl:
        logits = model(xb.to(device), vb.to(device))
        logits_all.append(logits.detach().cpu())
    logits = torch.cat(logits_all, dim=0)
    probs = torch.softmax(logits, dim=-1).numpy()
    return logits.numpy(), probs, probs.argmax(axis=1).astype(np.int64), probs.max(axis=1).astype(np.float32)


@torch.no_grad()
def capture_classifier_input_embeddings(model, X_bin: np.ndarray, X_offset: np.ndarray, X_cont: np.ndarray, batch_size: int, device: torch.device) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Capture input to model.classifier as CLS-like embedding via pre-hook."""
    if not hasattr(model, "classifier"):
        return None, {"available": False, "reason": "model has no attribute classifier"}
    captured: List[torch.Tensor] = []

    def pre_hook(module, args):
        if args and torch.is_tensor(args[0]):
            x = args[0].detach().cpu()
            if x.ndim >= 2:
                captured.append(x.reshape(x.shape[0], -1))

    handle = model.classifier.register_forward_pre_hook(pre_hook)
    try:
        _ = predict_model(model, X_bin, X_offset, X_cont, batch_size, device)
    finally:
        handle.remove()
    if not captured:
        return None, {"available": False, "reason": "classifier pre-hook captured nothing"}
    emb = torch.cat(captured, dim=0).numpy().astype(np.float32)
    if emb.shape[0] != X_bin.shape[0]:
        return None, {"available": False, "reason": f"captured rows {emb.shape[0]} != samples {X_bin.shape[0]}"}
    return emb, {"available": True, "embedding_name": "classifier_pre_hook_input", "shape": list(emb.shape)}


# ----------------------------- metrics -----------------------------

def macro_f1_np(y: np.ndarray, pred: np.ndarray, labels: List[int]) -> float:
    if f1_score is None:
        return float("nan")
    return float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))


def acc_np(y: np.ndarray, pred: np.ndarray) -> float:
    if accuracy_score is None:
        return float(np.mean(y == pred))
    return float(accuracy_score(y, pred))


def per_class_report_df(y: np.ndarray, pred: np.ndarray, class_names: List[str], split: str) -> pd.DataFrame:
    labels = list(range(len(class_names)))
    if classification_report is not None:
        rep = classification_report(y, pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
        rows = []
        for c in class_names:
            m = rep[c]
            rows.append({"split": split, "class": c, "precision": m["precision"], "recall": m["recall"], "f1": m["f1-score"], "support": m["support"]})
        return pd.DataFrame(rows)
    rows = []
    for i, c in enumerate(class_names):
        m = (y == i)
        rows.append({"split": split, "class": c, "precision": np.nan, "recall": float((pred[m] == i).mean()) if m.any() else np.nan, "f1": np.nan, "support": int(m.sum())})
    return pd.DataFrame(rows)


def confusion_long(y: np.ndarray, pred: np.ndarray, class_names: List[str], split: str) -> pd.DataFrame:
    rows = []
    for ti, tn in enumerate(class_names):
        denom = int((y == ti).sum())
        for pi, pn in enumerate(class_names):
            n = int(((y == ti) & (pred == pi)).sum())
            if n == 0 and ti != pi:
                continue
            rows.append({"split": split, "true_class": tn, "pred_class": pn, "correct_pair": bool(ti == pi), "n": n, "rate_within_true": n / max(denom, 1)})
    return pd.DataFrame(rows)


# ----------------------------- rare masks -----------------------------

def train_token_counts(X_bin_train: np.ndarray, num_bins: int) -> List[Dict[int, int]]:
    out = []
    for j in range(X_bin_train.shape[1]):
        bc = np.bincount(np.clip(X_bin_train[:, j].astype(np.int64), 0, num_bins - 1), minlength=num_bins)
        out.append({int(k): int(v) for k, v in enumerate(bc) if v > 0})
    return out


def rare_masks(X_bin: np.ndarray, ref_counts: List[Dict[int, int]], rare_threshold: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, F = X_bin.shape
    rare = np.zeros((N, F), dtype=bool)
    unseen = np.zeros((N, F), dtype=bool)
    ref_freq = np.zeros((N, F), dtype=np.int32)
    for j in range(F):
        ref = ref_counts[j]
        b = X_bin[:, j].astype(np.int64)
        freqs = np.array([ref.get(int(x), 0) for x in b], dtype=np.int32)
        ref_freq[:, j] = freqs
        rare[:, j] = freqs <= rare_threshold
        unseen[:, j] = freqs == 0
    return rare, unseen, ref_freq


# ----------------------------- root cause: train/val error -----------------------------

def save_train_val_error_gap(out_dir: Path, ytr, ptr, ctr, yva, pva, cva, class_names: List[str]) -> None:
    od = out_dir / "01_train_val_error_gap"
    od.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    summary = pd.DataFrame([
        {"split": "train", "n": len(ytr), "accuracy": acc_np(ytr, ptr), "macro_f1": macro_f1_np(ytr, ptr, labels), "wrong_n": int((ytr != ptr).sum()), "wrong_rate": float((ytr != ptr).mean()), "confidence_correct_mean": float(ctr[ytr == ptr].mean()), "confidence_wrong_mean": float(ctr[ytr != ptr].mean()) if (ytr != ptr).any() else np.nan},
        {"split": "val", "n": len(yva), "accuracy": acc_np(yva, pva), "macro_f1": macro_f1_np(yva, pva, labels), "wrong_n": int((yva != pva).sum()), "wrong_rate": float((yva != pva).mean()), "confidence_correct_mean": float(cva[yva == pva].mean()), "confidence_wrong_mean": float(cva[yva != pva].mean()) if (yva != pva).any() else np.nan},
    ])
    summary.to_csv(od / "split_summary.csv", index=False)
    pd.concat([per_class_report_df(ytr, ptr, class_names, "train"), per_class_report_df(yva, pva, class_names, "val")], ignore_index=True).to_csv(od / "per_class_train_val.csv", index=False)

    cl = pd.concat([confusion_long(ytr, ptr, class_names, "train"), confusion_long(yva, pva, class_names, "val")], ignore_index=True)
    cl.to_csv(od / "confusion_pair_train_val_long.csv", index=False)

    # Pair gap: train error rate vs val error rate for same true->pred.
    train_pairs = cl[cl["split"] == "train"].set_index(["true_class", "pred_class"])
    val_pairs = cl[cl["split"] == "val"].set_index(["true_class", "pred_class"])
    all_idx = train_pairs.index.union(val_pairs.index)
    rows = []
    for idx in all_idx:
        tn, pn = idx
        if tn == pn:
            continue
        tr_n = int(train_pairs.loc[idx, "n"]) if idx in train_pairs.index else 0
        va_n = int(val_pairs.loc[idx, "n"]) if idx in val_pairs.index else 0
        tr_rate = float(train_pairs.loc[idx, "rate_within_true"]) if idx in train_pairs.index else 0.0
        va_rate = float(val_pairs.loc[idx, "rate_within_true"]) if idx in val_pairs.index else 0.0
        rows.append({
            "true_class": tn,
            "pred_class": pn,
            "train_n": tr_n,
            "val_n": va_n,
            "train_rate_within_true": tr_rate,
            "val_rate_within_true": va_rate,
            "val_minus_train_rate": va_rate - tr_rate,
            "rootcause_hint": "generalization_gap_pair" if va_rate > tr_rate + 0.03 else ("local_underfit_or_overlap_pair" if tr_rate > 0.03 and va_rate > 0.03 else "minor_pair"),
        })
    pd.DataFrame(rows).sort_values(["val_n", "val_minus_train_rate"], ascending=[False, False]).to_csv(od / "pair_error_gap_train_vs_val.csv", index=False)


# ----------------------------- kNN root cause -----------------------------

def standardize_fit_transform(Xtr: np.ndarray, Xva: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if StandardScaler is None:
        raise RuntimeError(f"sklearn unavailable: {_SKLEARN_IMPORT_ERROR}")
    sc = StandardScaler(with_mean=True, with_std=True)
    Xtr2 = sc.fit_transform(Xtr.astype(np.float32))
    Xva2 = sc.transform(Xva.astype(np.float32))
    return Xtr2.astype(np.float32), Xva2.astype(np.float32)


def maybe_subsample_train(X: np.ndarray, y: np.ndarray, max_n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    if max_n and max_n > 0 and len(y) > max_n:
        rng = np.random.default_rng(seed)
        keep = []
        # stratified approximate subsample
        for c in np.unique(y):
            ids = idx[y == c]
            take = max(1, int(round(max_n * len(ids) / len(y))))
            take = min(take, len(ids))
            keep.append(rng.choice(ids, size=take, replace=False))
        sub = np.concatenate(keep)
        if len(sub) > max_n:
            sub = rng.choice(sub, size=max_n, replace=False)
        sub = np.sort(sub)
        return X[sub], y[sub], sub
    return X, y, idx


def run_knn_space(
    name: str,
    Xtr: np.ndarray,
    Xva: np.ndarray,
    ytr: np.ndarray,
    yva: np.ndarray,
    ptr: np.ndarray,
    pva: np.ndarray,
    conf_va: np.ndarray,
    class_names: List[str],
    out_dir: Path,
    k: int,
    max_train_knn: int,
    seed: int,
    rare_count_val: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if NearestNeighbors is None:
        raise RuntimeError(f"sklearn unavailable: {_SKLEARN_IMPORT_ERROR}")
    od = out_dir / "02_knn_raw_token_cls" / name
    od.mkdir(parents=True, exist_ok=True)

    Xtr_s, Xva_s = standardize_fit_transform(Xtr, Xva)
    Xfit, yfit, fit_idx = maybe_subsample_train(Xtr_s, ytr, max_train_knn, seed)

    kk_val = min(k, len(yfit))
    nn = NearestNeighbors(n_neighbors=kk_val, metric="euclidean", algorithm="auto")
    nn.fit(Xfit)
    d_val, i_val = nn.kneighbors(Xva_s, return_distance=True)
    neigh_y = yfit[i_val]

    # train reference distances excluding self if full fit; if subsampled, still use fit-to-fit for distribution threshold.
    kk_train = min(k + 1, len(yfit))
    nn_ref = NearestNeighbors(n_neighbors=kk_train, metric="euclidean", algorithm="auto")
    nn_ref.fit(Xfit)
    d_ref, i_ref = nn_ref.kneighbors(Xfit, return_distance=True)
    if kk_train > 1:
        # first is usually self for fit-to-fit.
        ref_mean_dist = d_ref[:, 1:].mean(axis=1)
    else:
        ref_mean_dist = d_ref.mean(axis=1)
    q95 = float(np.quantile(ref_mean_dist, 0.95))
    q99 = float(np.quantile(ref_mean_dist, 0.99))

    rows = []
    for i in range(len(yva)):
        counts = np.bincount(neigh_y[i], minlength=len(class_names)).astype(np.int64)
        majority = int(np.argmax(counts))
        true_i = int(yva[i])
        pred_i = int(pva[i])
        true_frac = float(counts[true_i] / kk_val)
        pred_frac = float(counts[pred_i] / kk_val)
        maj_frac = float(counts[majority] / kk_val)
        mean_dist = float(d_val[i].mean())
        min_dist = float(d_val[i].min())
        is_ood95 = bool(mean_dist > q95)
        is_ood99 = bool(mean_dist > q99)
        if is_ood95:
            rc = "OOD_or_distribution_shift"
        elif true_i != pred_i and pred_frac >= 0.50:
            rc = "feature_space_overlap_with_pred_class"
        elif true_i != pred_i and true_frac >= 0.50:
            rc = "model_boundary_failure_knn_true_neighbors"
        elif true_i != pred_i:
            rc = "mixed_neighbors_ambiguous"
        elif true_i == pred_i and true_frac >= 0.50:
            rc = "correct_and_knn_consistent"
        else:
            rc = "correct_but_neighbors_mixed"
        row = {
            "space": name,
            "sample_idx": i,
            "true_class": class_names[true_i],
            "pred_class": class_names[pred_i],
            "correct": bool(true_i == pred_i),
            "confidence": float(conf_va[i]),
            "knn_k": kk_val,
            "knn_majority_class": class_names[majority],
            "knn_majority_frac": maj_frac,
            "knn_true_frac": true_frac,
            "knn_pred_frac": pred_frac,
            "knn_mean_dist": mean_dist,
            "knn_min_dist": min_dist,
            "train_ref_mean_dist_q95": q95,
            "train_ref_mean_dist_q99": q99,
            "is_ood95": is_ood95,
            "is_ood99": is_ood99,
            "rootcause_category": rc,
        }
        for ci, cn in enumerate(class_names):
            row[f"knn_frac_{cn}"] = float(counts[ci] / kk_val)
        if rare_count_val is not None:
            row["rare_cell_count"] = int(rare_count_val[i])
        rows.append(row)
    sample_df = pd.DataFrame(rows)
    sample_df.to_csv(od / f"val_knn_sample_rootcause_{name}.csv", index=False)

    summary_rows = []
    for keys, g in sample_df.groupby(["correct", "rootcause_category"], dropna=False):
        correct, cat = keys
        summary_rows.append({"space": name, "correct": bool(correct), "rootcause_category": cat, "n": int(len(g)), "rate_all_val": float(len(g) / len(sample_df)), "confidence_mean": float(g["confidence"].mean()), "knn_true_frac_mean": float(g["knn_true_frac"].mean()), "knn_pred_frac_mean": float(g["knn_pred_frac"].mean()), "ood95_rate": float(g["is_ood95"].mean())})
    summary = pd.DataFrame(summary_rows).sort_values(["correct", "n"], ascending=[True, False])
    summary.to_csv(od / f"val_knn_rootcause_summary_{name}.csv", index=False)

    pair_summary = sample_df[~sample_df["correct"]].groupby(["true_class", "pred_class", "rootcause_category"], dropna=False).agg(
        n=("sample_idx", "count"),
        confidence_mean=("confidence", "mean"),
        knn_true_frac_mean=("knn_true_frac", "mean"),
        knn_pred_frac_mean=("knn_pred_frac", "mean"),
        ood95_rate=("is_ood95", "mean"),
    ).reset_index().sort_values(["true_class", "pred_class", "n"], ascending=[True, True, False])
    pair_summary.to_csv(od / f"val_wrong_pair_knn_rootcause_{name}.csv", index=False)

    meta = {"space": name, "k": kk_val, "train_fit_n": int(len(yfit)), "train_full_n": int(len(ytr)), "train_ref_mean_dist_q95": q95, "train_ref_mean_dist_q99": q99}
    write_json(od / f"knn_meta_{name}.json", meta)
    return sample_df, summary, meta


# ----------------------------- group masking -----------------------------

def neutral_values(X_bin_train: np.ndarray, X_off_train: np.ndarray, X_cont_train: np.ndarray, num_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    F = X_bin_train.shape[1]
    nb = np.zeros(F, dtype=np.int64)
    no = np.zeros(F, dtype=np.float32)
    nc = np.zeros(F, dtype=np.float32)
    for j in range(F):
        b = np.clip(X_bin_train[:, j].astype(np.int64), 0, num_bins - 1)
        bc = np.bincount(b, minlength=num_bins)
        mode_bin = int(np.argmax(bc))
        nb[j] = mode_bin
        m = b == mode_bin
        no[j] = float(np.median(X_off_train[m, j])) if m.any() else float(np.median(X_off_train[:, j]))
        nc[j] = float(np.median(X_cont_train[:, j]))
    return nb, no, nc


def apply_neutralization(Xb, Xo, Xc, feat_idx: np.ndarray, mode: str, nb, no, nc):
    Xb2 = Xb.copy()
    Xo2 = Xo.copy()
    Xc2 = Xc.copy()
    if mode in ["all", "token_only"]:
        Xb2[:, feat_idx] = nb[feat_idx][None, :]
        Xo2[:, feat_idx] = no[feat_idx][None, :]
    if mode in ["all", "raw_only"]:
        Xc2[:, feat_idx] = nc[feat_idx][None, :]
    return Xb2, Xo2, Xc2


def group_masking_audit(
    out_dir: Path,
    model,
    Xtr_b, Xtr_o, Xtr_c, ytr,
    Xva_b, Xva_o, Xva_c, yva,
    base_ptr, base_pva,
    features: List[str], strategies: Dict[str, str], class_names: List[str],
    num_bins: int, batch_size: int, device: torch.device
):
    od = out_dir / "03_group_masking"
    od.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    nb, no, nc = neutral_values(Xtr_b, Xtr_o, Xtr_c, num_bins)
    strat_to_idx: Dict[str, List[int]] = {}
    for j, f in enumerate(features):
        strat_to_idx.setdefault(strategies.get(f, "unknown"), []).append(j)
    # add all malware-relevant non-constant group if helpful
    rows = []
    pair_rows = []
    base = {
        "train_macro_f1": macro_f1_np(ytr, base_ptr, labels),
        "val_macro_f1": macro_f1_np(yva, base_pva, labels),
        "train_acc": acc_np(ytr, base_ptr),
        "val_acc": acc_np(yva, base_pva),
    }
    for strategy, idxs in sorted(strat_to_idx.items()):
        if not idxs:
            continue
        feat_idx = np.array(idxs, dtype=np.int64)
        for mode in ["all", "token_only", "raw_only"]:
            Xtb, Xto, Xtc = apply_neutralization(Xtr_b, Xtr_o, Xtr_c, feat_idx, mode, nb, no, nc)
            Xvb, Xvo, Xvc = apply_neutralization(Xva_b, Xva_o, Xva_c, feat_idx, mode, nb, no, nc)
            _, _, ptr, ctr = predict_model(model, Xtb, Xto, Xtc, batch_size, device)
            _, _, pva, cva = predict_model(model, Xvb, Xvo, Xvc, batch_size, device)
            tr_f1 = macro_f1_np(ytr, ptr, labels)
            va_f1 = macro_f1_np(yva, pva, labels)
            rows.append({
                "strategy": strategy,
                "mode": mode,
                "n_features": int(len(feat_idx)),
                "train_macro_f1": tr_f1,
                "val_macro_f1": va_f1,
                "train_macro_f1_drop_vs_base": base["train_macro_f1"] - tr_f1,
                "val_macro_f1_drop_vs_base": base["val_macro_f1"] - va_f1,
                "drop_gap_train_minus_val": (base["train_macro_f1"] - tr_f1) - (base["val_macro_f1"] - va_f1),
                "train_acc": acc_np(ytr, ptr),
                "val_acc": acc_np(yva, pva),
                "val_changed_pred_rate": float(np.mean(pva != base_pva)),
                "val_wrong_to_correct_n": int(np.sum((base_pva != yva) & (pva == yva))),
                "val_correct_to_wrong_n": int(np.sum((base_pva == yva) & (pva != yva))),
                "val_wrong_to_correct_minus_correct_to_wrong": int(np.sum((base_pva != yva) & (pva == yva)) - np.sum((base_pva == yva) & (pva != yva))),
            })
            # pair changes for major pairs
            for ti, tn in enumerate(class_names):
                for pi, pn in enumerate(class_names):
                    if ti == pi:
                        continue
                    before = int(np.sum((yva == ti) & (base_pva == pi)))
                    after = int(np.sum((yva == ti) & (pva == pi)))
                    if before >= 20 or after >= 20:
                        pair_rows.append({
                            "strategy": strategy,
                            "mode": mode,
                            "true_class": tn,
                            "pred_class": pn,
                            "before_n": before,
                            "after_n": after,
                            "after_minus_before": after - before,
                        })
    df = pd.DataFrame(rows).sort_values(["val_macro_f1_drop_vs_base", "drop_gap_train_minus_val"], ascending=[False, False])
    df.to_csv(od / "group_neutralization_summary.csv", index=False)
    pd.DataFrame(pair_rows).sort_values(["strategy", "mode", "before_n"], ascending=[True, True, False]).to_csv(od / "group_neutralization_pair_changes.csv", index=False)
    write_json(od / "baseline_metrics.json", base)
    return df


# ----------------------------- CLS geometry -----------------------------

def centroid_geometry(name: str, Xtr: np.ndarray, Xva: np.ndarray, ytr, yva, pva, class_names: List[str], out_dir: Path) -> pd.DataFrame:
    od = out_dir / "04_cls_geometry"
    od.mkdir(parents=True, exist_ok=True)
    Xtr_s, Xva_s = standardize_fit_transform(Xtr, Xva)
    C = len(class_names)
    cent = np.zeros((C, Xtr_s.shape[1]), dtype=np.float32)
    for c in range(C):
        m = ytr == c
        cent[c] = Xtr_s[m].mean(axis=0) if m.any() else 0
    rows = []
    for i in range(len(yva)):
        d = np.linalg.norm(cent - Xva_s[i][None, :], axis=1)
        ti = int(yva[i]); pi = int(pva[i])
        nearest = int(np.argmin(d))
        rows.append({
            "space": name,
            "sample_idx": i,
            "true_class": class_names[ti],
            "pred_class": class_names[pi],
            "correct": bool(ti == pi),
            "nearest_train_centroid_class": class_names[nearest],
            "dist_to_true_centroid": float(d[ti]),
            "dist_to_pred_centroid": float(d[pi]),
            "pred_closer_than_true_centroid": bool(d[pi] < d[ti]) if ti != pi else False,
            "centroid_margin_true_minus_pred": float(d[ti] - d[pi]) if ti != pi else 0.0,
        })
    sdf = pd.DataFrame(rows)
    sdf.to_csv(od / f"val_centroid_distance_{name}.csv", index=False)

    # class centroid train-val shift and separability
    grow = []
    for c, cname in enumerate(class_names):
        mt = ytr == c
        mv = yva == c
        train_cent = Xtr_s[mt].mean(axis=0)
        val_cent = Xva_s[mv].mean(axis=0)
        shift = float(np.linalg.norm(val_cent - train_cent))
        within_train = float(np.linalg.norm(Xtr_s[mt] - train_cent[None, :], axis=1).mean())
        within_val = float(np.linalg.norm(Xva_s[mv] - train_cent[None, :], axis=1).mean())
        grow.append({"space": name, "class": cname, "train_val_centroid_shift": shift, "within_train_mean_dist_to_train_centroid": within_train, "within_val_mean_dist_to_train_centroid": within_val, "shift_over_within_train": shift / (within_train + 1e-12)})
    gdf = pd.DataFrame(grow)
    gdf.to_csv(od / f"class_centroid_shift_{name}.csv", index=False)

    ps = sdf[~sdf["correct"]].groupby(["true_class", "pred_class"], dropna=False).agg(
        n=("sample_idx", "count"),
        pred_closer_than_true_centroid_rate=("pred_closer_than_true_centroid", "mean"),
        dist_to_true_centroid_mean=("dist_to_true_centroid", "mean"),
        dist_to_pred_centroid_mean=("dist_to_pred_centroid", "mean"),
        centroid_margin_true_minus_pred_mean=("centroid_margin_true_minus_pred", "mean"),
    ).reset_index().sort_values("n", ascending=False)
    ps.to_csv(od / f"wrong_pair_centroid_summary_{name}.csv", index=False)
    return sdf


# ----------------------------- rare causal -----------------------------

def rare_causal_audit(out_dir: Path, knn_sample_dfs: Dict[str, pd.DataFrame], rare_val: np.ndarray, unseen_val: np.ndarray, yva, pva, class_names: List[str]) -> pd.DataFrame:
    od = out_dir / "05_rare_causal"
    od.mkdir(parents=True, exist_ok=True)
    rare_count = rare_val.sum(axis=1)
    unseen_count = unseen_val.sum(axis=1)
    base_rows = []
    for split_group, mask in [("correct", yva == pva), ("wrong", yva != pva)]:
        base_rows.append({
            "group": split_group,
            "n": int(mask.sum()),
            "rare_count_mean": float(rare_count[mask].mean()) if mask.any() else np.nan,
            "rare_count_median": float(np.median(rare_count[mask])) if mask.any() else np.nan,
            "rare_count_p95": float(np.quantile(rare_count[mask], 0.95)) if mask.any() else np.nan,
            "unseen_count_mean": float(unseen_count[mask].mean()) if mask.any() else np.nan,
        })
    pd.DataFrame(base_rows).to_csv(od / "rare_count_correct_vs_wrong.csv", index=False)

    all_rows = []
    for space, df in knn_sample_dfs.items():
        d = df.copy()
        d["rare_count_bin"] = pd.cut(d["rare_cell_count"], bins=[-1, 0, 1, 2, 4, 8, 1000], labels=["0", "1", "2", "3-4", "5-8", "9+"])
        for keys, g in d[~d["correct"]].groupby(["rare_count_bin", "rootcause_category"], dropna=False):
            rb, cat = keys
            all_rows.append({"space": space, "rare_count_bin": str(rb), "rootcause_category": cat, "wrong_n": int(len(g)), "wrong_rate_in_space": float(len(g) / max((~d["correct"]).sum(), 1)), "confidence_mean": float(g["confidence"].mean()), "knn_true_frac_mean": float(g["knn_true_frac"].mean()), "knn_pred_frac_mean": float(g["knn_pred_frac"].mean()), "ood95_rate": float(g["is_ood95"].mean())})
    out = pd.DataFrame(all_rows).sort_values(["space", "rare_count_bin", "wrong_n"], ascending=[True, True, False])
    out.to_csv(od / "rare_by_knn_rootcause.csv", index=False)

    pair_rows = []
    for space, df in knn_sample_dfs.items():
        d = df[~df["correct"]].copy()
        if len(d) == 0:
            continue
        d["high_rare"] = d["rare_cell_count"] >= max(2, int(np.quantile(rare_count, 0.90)))
        for keys, g in d.groupby(["true_class", "pred_class", "high_rare", "rootcause_category"], dropna=False):
            tn, pn, high, cat = keys
            pair_rows.append({"space": space, "true_class": tn, "pred_class": pn, "high_rare": bool(high), "rootcause_category": cat, "n": int(len(g)), "rare_count_mean": float(g["rare_cell_count"].mean()), "knn_true_frac_mean": float(g["knn_true_frac"].mean()), "knn_pred_frac_mean": float(g["knn_pred_frac"].mean())})
    pdf = pd.DataFrame(pair_rows).sort_values(["space", "true_class", "pred_class", "high_rare", "n"], ascending=[True, True, True, False, False])
    pdf.to_csv(od / "wrong_pair_high_rare_rootcause.csv", index=False)
    return out


# ----------------------------- summary -----------------------------

def build_summary(out_dir: Path, paths: Dict[str, Path], split_summary: pd.DataFrame, pair_gap: pd.DataFrame, knn_summaries: Dict[str, pd.DataFrame], group_df: Optional[pd.DataFrame], cls_info: Dict[str, Any]) -> None:
    md = []
    md.append("# Overfit root-cause audit summary")
    md.append("")
    md.append("## Inputs")
    md.append(f"- dataset: `{paths['dataset']}`")
    md.append(f"- metadata: `{paths['metadata']}`")
    md.append(f"- checkpoint: `{paths['checkpoint']}`")
    md.append(f"- run_dir: `{paths['run_dir']}`")
    md.append("")
    md.append("## 1. Train/val gap")
    md.append(to_markdown_safe(split_summary, 10))
    md.append("")
    md.append("## 2. Largest pair gaps: train vs val")
    md.append("Interpretation: large `val_minus_train_rate` means this pair is mainly a validation-generalization gap; high train and high val rates suggest local-underfit or true overlap.")
    md.append(to_markdown_safe(pair_gap.sort_values(["val_n", "val_minus_train_rate"], ascending=[False, False]).head(12), 12))
    md.append("")
    md.append("## 3. kNN root-cause summaries")
    md.append("Categories:")
    md.append("- `feature_space_overlap_with_pred_class`: val wrong sample is closer to train samples of predicted class.")
    md.append("- `model_boundary_failure_knn_true_neighbors`: val wrong sample is closer to train samples of true class but model predicts other class.")
    md.append("- `OOD_or_distribution_shift`: val sample is farther than train reference q95.")
    md.append("- `mixed_neighbors_ambiguous`: no dominant nearby class.")
    for space, df in knn_summaries.items():
        md.append(f"\n### {space}")
        md.append(to_markdown_safe(df, 20))
    md.append("")
    md.append("## 4. Group neutralization")
    if group_df is not None and len(group_df):
        md.append("Large train drop but small val drop suggests group overfit. Negative val drop means masking/neutralizing group improves val, so group may be harmful for val boundary.")
        cols = [c for c in ["strategy", "mode", "n_features", "train_macro_f1_drop_vs_base", "val_macro_f1_drop_vs_base", "drop_gap_train_minus_val", "val_wrong_to_correct_n", "val_correct_to_wrong_n"] if c in group_df.columns]
        md.append(to_markdown_safe(group_df[cols].sort_values("drop_gap_train_minus_val", ascending=False), 30))
    else:
        md.append("Group neutralization not available.")
    md.append("")
    md.append("## 5. CLS/embedding availability")
    md.append("```json")
    md.append(json.dumps(cls_info, indent=2, ensure_ascii=False))
    md.append("```")
    md.append("")
    md.append("## How to decide after reading outputs")
    md.append("- Wrong samples close to predicted class in raw/token space ⇒ feature overlap/ambiguity, not pure model overfit.")
    md.append("- Wrong samples close to true class in raw/token but close to predicted class in CLS ⇒ learned representation/boundary failure.")
    md.append("- Wrong samples OOD in raw/token ⇒ train-val distribution shift.")
    md.append("- Same pair wrong on train and val ⇒ local underfit or inherent class overlap, not merely overfit.")
    md.append("- Group whose train masking drop ≫ val masking drop ⇒ model overuses that group on train.")
    write_text(out_dir / "audit_overfit_rootcause_summary.md", "\n".join(md))


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-npz", default="")
    ap.add_argument("--metadata-json", default="")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--c2-audit-dir", default="")
    ap.add_argument("--out-dir", default="03_outputs/audit_overfit_rootcause")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--rare-threshold", type=int, default=5)
    ap.add_argument("--knn-k", type=int, default=25)
    ap.add_argument("--max-train-knn", type=int, default=0, help="0=use all train samples; set e.g. 30000 if too slow")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-group-masking", action="store_true")
    ap.add_argument("--skip-cls", action="store_true")
    args = ap.parse_args()

    if _SKLEARN_IMPORT_ERROR is not None:
        raise ImportError(f"scikit-learn is required for this audit: {_SKLEARN_IMPORT_ERROR}")

    root = repo_root()
    out_dir = (root / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rootcause] root: {root}")
    print(f"[rootcause] out: {out_dir}")

    paths = resolve_paths(args)
    for k, v in paths.items():
        print(f"[rootcause] {k}: {v}")

    data, meta = load_dataset(paths["dataset"], paths["metadata"])
    Xtr_b = data["X_train_bin"].astype(np.int64)
    Xtr_o = data["X_train_offset"].astype(np.float32)
    ytr = data["y_train"].astype(np.int64)
    Xva_b = data["X_val_bin"].astype(np.int64)
    Xva_o = data["X_val_offset"].astype(np.float32)
    yva = data["y_val"].astype(np.int64)
    n_features = Xtr_b.shape[1]
    features = get_features(meta, n_features)
    class_names = labels_from_meta(meta)
    strategies = feature_strategies(meta, features)
    num_bins = int(meta.get("num_bins", meta.get("K", max(int(Xtr_b.max()), int(Xva_b.max())) + 1)))
    print(f"[rootcause] Ntrain={len(ytr)} Nval={len(yva)} F={n_features} num_bins={num_bins} classes={class_names}")

    Xtr_c, Xva_c, raw_info = raw_scaled_from_train(root, features)
    write_json(out_dir / "raw_scaled_info.json", raw_info)
    if Xtr_c is None or Xva_c is None:
        raise FileNotFoundError(f"Need raw scaled continuous branch. raw_info={raw_info}")

    device = auto_device(args.device)
    train_mod, train_mod_path = import_train_module(root)
    print(f"[rootcause] imported train module: {train_mod_path}")
    ckpt = safe_torch_load(paths["checkpoint"], device)
    model = build_model(train_mod, ckpt, num_bins, n_features, len(class_names), device)

    print("[rootcause] predicting C2 train/val...")
    logits_tr, probs_tr, pred_tr, conf_tr = predict_model(model, Xtr_b, Xtr_o, Xtr_c, args.batch_size, device)
    logits_va, probs_va, pred_va, conf_va = predict_model(model, Xva_b, Xva_o, Xva_c, args.batch_size, device)

    # save predictions for traceability
    pdir = out_dir / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_idx": np.arange(len(ytr)), "split": "train", "y_true": ytr, "y_pred": pred_tr, "true_class": [class_names[i] for i in ytr], "pred_class": [class_names[i] for i in pred_tr], "correct": ytr == pred_tr, "confidence": conf_tr}).to_csv(pdir / "train_predictions_recomputed.csv", index=False)
    pd.DataFrame({"sample_idx": np.arange(len(yva)), "split": "val", "y_true": yva, "y_pred": pred_va, "true_class": [class_names[i] for i in yva], "pred_class": [class_names[i] for i in pred_va], "correct": yva == pred_va, "confidence": conf_va}).to_csv(pdir / "val_predictions_recomputed.csv", index=False)

    # rare exposure
    ref_counts = train_token_counts(Xtr_b, num_bins)
    rare_tr, unseen_tr, freq_tr = rare_masks(Xtr_b, ref_counts, args.rare_threshold)
    rare_va, unseen_va, freq_va = rare_masks(Xva_b, ref_counts, args.rare_threshold)
    np.savez_compressed(out_dir / "rare_masks_summary_arrays.npz", rare_count_train=rare_tr.sum(axis=1), rare_count_val=rare_va.sum(axis=1), unseen_count_train=unseen_tr.sum(axis=1), unseen_count_val=unseen_va.sum(axis=1))

    # 1) train-val error gap
    print("[rootcause] train-val error gap audit...")
    save_train_val_error_gap(out_dir, ytr, pred_tr, conf_tr, yva, pred_va, conf_va, class_names)
    split_summary = pd.read_csv(out_dir / "01_train_val_error_gap" / "split_summary.csv")
    pair_gap = pd.read_csv(out_dir / "01_train_val_error_gap" / "pair_error_gap_train_vs_val.csv")

    # 2) kNN spaces
    print("[rootcause] kNN raw_scaled...")
    knn_dfs: Dict[str, pd.DataFrame] = {}
    knn_summaries: Dict[str, pd.DataFrame] = {}
    df_raw, sum_raw, meta_raw = run_knn_space("raw_scaled", Xtr_c, Xva_c, ytr, yva, pred_tr, pred_va, conf_va, class_names, out_dir, args.knn_k, args.max_train_knn, args.seed, rare_count_val=rare_va.sum(axis=1))
    knn_dfs["raw_scaled"] = df_raw; knn_summaries["raw_scaled"] = sum_raw

    print("[rootcause] kNN token_bin_offset...")
    # Scale bin to [0,1] and concatenate with offset; this uses token geometry without model.
    Xtr_tok = np.concatenate([Xtr_b.astype(np.float32) / max(num_bins - 1, 1), Xtr_o.astype(np.float32)], axis=1)
    Xva_tok = np.concatenate([Xva_b.astype(np.float32) / max(num_bins - 1, 1), Xva_o.astype(np.float32)], axis=1)
    df_tok, sum_tok, meta_tok = run_knn_space("token_bin_offset", Xtr_tok, Xva_tok, ytr, yva, pred_tr, pred_va, conf_va, class_names, out_dir, args.knn_k, args.max_train_knn, args.seed, rare_count_val=rare_va.sum(axis=1))
    knn_dfs["token_bin_offset"] = df_tok; knn_summaries["token_bin_offset"] = sum_tok

    cls_info: Dict[str, Any] = {"available": False, "skipped": bool(args.skip_cls)}
    Xtr_cls = Xva_cls = None
    if not args.skip_cls:
        print("[rootcause] capturing CLS/classifier input embeddings...")
        Xtr_cls, cls_info_tr = capture_classifier_input_embeddings(model, Xtr_b, Xtr_o, Xtr_c, args.batch_size, device)
        Xva_cls, cls_info_va = capture_classifier_input_embeddings(model, Xva_b, Xva_o, Xva_c, args.batch_size, device)
        cls_info = {"train": cls_info_tr, "val": cls_info_va}
        write_json(out_dir / "cls_capture_info.json", cls_info)
        if Xtr_cls is not None and Xva_cls is not None:
            np.savez_compressed(out_dir / "cls_embeddings.npz", X_train_cls=Xtr_cls, X_val_cls=Xva_cls, y_train=ytr, y_val=yva, pred_train=pred_tr, pred_val=pred_va)
            print("[rootcause] kNN CLS/classifier input...")
            df_cls, sum_cls, meta_cls = run_knn_space("cls_classifier_input", Xtr_cls, Xva_cls, ytr, yva, pred_tr, pred_va, conf_va, class_names, out_dir, args.knn_k, args.max_train_knn, args.seed, rare_count_val=rare_va.sum(axis=1))
            knn_dfs["cls_classifier_input"] = df_cls; knn_summaries["cls_classifier_input"] = sum_cls
            print("[rootcause] centroid geometry raw/token/CLS...")
            centroid_geometry("raw_scaled", Xtr_c, Xva_c, ytr, yva, pred_va, class_names, out_dir)
            centroid_geometry("token_bin_offset", Xtr_tok, Xva_tok, ytr, yva, pred_va, class_names, out_dir)
            centroid_geometry("cls_classifier_input", Xtr_cls, Xva_cls, ytr, yva, pred_va, class_names, out_dir)
        else:
            print(f"[rootcause][WARN] CLS unavailable: {cls_info}")
            centroid_geometry("raw_scaled", Xtr_c, Xva_c, ytr, yva, pred_va, class_names, out_dir)
            centroid_geometry("token_bin_offset", Xtr_tok, Xva_tok, ytr, yva, pred_va, class_names, out_dir)
    else:
        centroid_geometry("raw_scaled", Xtr_c, Xva_c, ytr, yva, pred_va, class_names, out_dir)
        centroid_geometry("token_bin_offset", Xtr_tok, Xva_tok, ytr, yva, pred_va, class_names, out_dir)

    # 3) group masking / neutralization
    group_df = None
    if not args.skip_group_masking:
        print("[rootcause] group neutralization audit...")
        group_df = group_masking_audit(out_dir, model, Xtr_b, Xtr_o, Xtr_c, ytr, Xva_b, Xva_o, Xva_c, yva, pred_tr, pred_va, features, strategies, class_names, num_bins, args.batch_size, device)
    else:
        print("[rootcause] skip group neutralization")

    # 4) rare causal with kNN categories
    print("[rootcause] rare causal audit...")
    rare_causal_audit(out_dir, knn_dfs, rare_va, unseen_va, yva, pred_va, class_names)

    # Combined cross-space root-cause table for wrong val samples.
    print("[rootcause] combining cross-space categories...")
    comb = pd.DataFrame({"sample_idx": np.arange(len(yva)), "true_class": [class_names[i] for i in yva], "pred_class": [class_names[i] for i in pred_va], "correct": yva == pred_va, "confidence": conf_va, "rare_cell_count": rare_va.sum(axis=1), "unseen_cell_count": unseen_va.sum(axis=1)})
    for space, df in knn_dfs.items():
        small = df[["sample_idx", "rootcause_category", "knn_true_frac", "knn_pred_frac", "is_ood95", "knn_mean_dist"]].rename(columns={
            "rootcause_category": f"{space}_category",
            "knn_true_frac": f"{space}_knn_true_frac",
            "knn_pred_frac": f"{space}_knn_pred_frac",
            "is_ood95": f"{space}_is_ood95",
            "knn_mean_dist": f"{space}_knn_mean_dist",
        })
        comb = comb.merge(small, on="sample_idx", how="left")
    cod = out_dir / "02_knn_raw_token_cls"
    comb.to_csv(cod / "val_cross_space_rootcause_per_sample.csv", index=False)
    # pair aggregate cross-space categories
    agg_rows = []
    wrong = comb[~comb["correct"]]
    for (tn, pn), g in wrong.groupby(["true_class", "pred_class"], dropna=False):
        row = {"true_class": tn, "pred_class": pn, "n": int(len(g)), "confidence_mean": float(g["confidence"].mean()), "rare_cell_count_mean": float(g["rare_cell_count"].mean())}
        for space in knn_dfs.keys():
            cat_col = f"{space}_category"
            for cat in ["feature_space_overlap_with_pred_class", "model_boundary_failure_knn_true_neighbors", "OOD_or_distribution_shift", "mixed_neighbors_ambiguous"]:
                row[f"{space}_{cat}_rate"] = float((g[cat_col] == cat).mean()) if cat_col in g.columns else np.nan
            row[f"{space}_true_frac_mean"] = float(g[f"{space}_knn_true_frac"].mean()) if f"{space}_knn_true_frac" in g.columns else np.nan
            row[f"{space}_pred_frac_mean"] = float(g[f"{space}_knn_pred_frac"].mean()) if f"{space}_knn_pred_frac" in g.columns else np.nan
            row[f"{space}_ood95_rate"] = float(g[f"{space}_is_ood95"].mean()) if f"{space}_is_ood95" in g.columns else np.nan
        agg_rows.append(row)
    pd.DataFrame(agg_rows).sort_values("n", ascending=False).to_csv(cod / "wrong_pair_cross_space_rootcause_summary.csv", index=False)

    build_summary(out_dir, paths, split_summary, pair_gap, knn_summaries, group_df, cls_info)
    print("[rootcause] DONE")
    print(f"[rootcause] summary: {out_dir / 'audit_overfit_rootcause_summary.md'}")


if __name__ == "__main__":
    main()

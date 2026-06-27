#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
28_audit_c2_best.py

Audit best C2 K512 D3 run:
  1) result audit: train/val/gap/confusion/history
  2) token audit: bins/compression/rare/dominant/entropy per feature and per strategy
  3) embedding audit: D3 offset interpolation + raw FiLM + shared bin embedding stats
  4) error-conditioned audit: correct vs wrong, especially Ransomware/Spyware/Trojan pairs

The key hypothesis under test:
  "moderate overfit is driven by rare tokens / sparse tokenization"

This script does NOT change model or preprocessing. It reads the existing C2 artifact + trained checkpoint,
then writes audit CSV/JSON/MD into out-dir.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.metrics import classification_report, confusion_matrix
except Exception as e:
    classification_report = None
    confusion_matrix = None


def repo_root() -> Path:
    # script in 02_src normally
    p = Path(__file__).resolve()
    if p.parent.name == "02_src":
        return p.parents[1]
    return Path.cwd()


def auto_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def find_first(root: Path, patterns: List[str], must_contain: Optional[List[str]] = None) -> Optional[Path]:
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(root.rglob(pat))
    if must_contain:
        filtered = []
        for c in candidates:
            s = str(c)
            if all(x in s for x in must_contain):
                filtered.append(c)
        candidates = filtered
    if not candidates:
        return None
    # Prefer paths not inside unrelated copied package dirs, and shortest path.
    candidates = sorted(set(candidates), key=lambda x: (len(str(x)), str(x)))
    return candidates[0]


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    root = repo_root()

    dataset = Path(args.dataset_npz) if args.dataset_npz else None
    metadata = Path(args.metadata_json) if args.metadata_json else None
    run_dir = Path(args.run_dir) if args.run_dir else None
    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    if dataset and not dataset.is_absolute():
        dataset = root / dataset
    if metadata and not metadata.is_absolute():
        metadata = root / metadata
    if run_dir and not run_dir.is_absolute():
        run_dir = root / run_dir
    if checkpoint and not checkpoint.is_absolute():
        checkpoint = root / checkpoint

    if dataset is None or not dataset.exists():
        dataset = find_first(root, ["mixed_quantile_offset_dataset.npz"], ["K512_B512_C2_selective_rank_discrete_compact"])
    if metadata is None or not metadata.exists():
        metadata = find_first(root, ["mixed_quantile_offset_metadata.json"], ["K512_B512_C2_selective_rank_discrete_compact"])
    if run_dir is None or not run_dir.exists():
        cand = find_first(root, ["diagnosis_summary.json"], ["D3_P1_K512_B512_C2_selective_rank_discrete_compact"])
        run_dir = cand.parent if cand else None
    if checkpoint is None or not checkpoint.exists():
        if run_dir is not None:
            cp = run_dir / "best_model.pt"
            if cp.exists():
                checkpoint = cp
        if checkpoint is None or not checkpoint.exists():
            checkpoint = find_first(root, ["best_model.pt"], ["D3_P1_K512_B512_C2_selective_rank_discrete_compact"])

    missing = []
    if dataset is None or not dataset.exists():
        missing.append("C2 dataset npz: 03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_dataset.npz")
    if metadata is None or not metadata.exists():
        missing.append("C2 metadata json: 03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_metadata.json")
    if run_dir is None or not run_dir.exists():
        missing.append("C2 D3 run dir with diagnosis_summary/history/confusion")
    if checkpoint is None or not checkpoint.exists():
        missing.append("C2 D3 best_model.pt checkpoint")

    if missing:
        raise FileNotFoundError("Missing required audit inputs:\n- " + "\n- ".join(missing))

    return {"root": root, "dataset": dataset, "metadata": metadata, "run_dir": run_dir, "checkpoint": checkpoint}


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
            # Ensure imports like config.py work
            sys.path.insert(0, str(root / "02_src"))
            sys.path.insert(0, str(root))
            spec.loader.exec_module(mod)
            return mod, p
    raise FileNotFoundError("Cannot find 02_src/10_train_fusion_ablation_D0_D7.py")


def load_dataset(dataset_path: Path, metadata_path: Path):
    arr = np.load(dataset_path, allow_pickle=True)
    data = {k: arr[k] for k in arr.files}
    meta = read_json(metadata_path)
    for k in ["X_train_bin", "X_train_offset", "y_train", "X_val_bin", "X_val_offset", "y_val"]:
        if k not in data:
            raise ValueError(f"dataset missing {k}")
    return data, meta


def labels_from_meta(meta: Dict[str, Any]) -> List[str]:
    lm = meta.get("label_mapping", {})
    if lm:
        inv = {int(v): str(k).strip() for k, v in lm.items()}
        return [inv[i] for i in sorted(inv)]
    return ["Benign", "Ransomware", "Spyware", "Trojan"]


def feature_strategies(meta: Dict[str, Any], features: List[str]) -> Dict[str, str]:
    fs = meta.get("feature_strategies", {})
    out = {}
    for f in features:
        v = fs.get(f)
        if isinstance(v, dict):
            out[f] = str(v.get("strategy", v.get("selected_strategy", "unknown")))
        elif v is None:
            # fallback from feature_meta
            fm = meta.get("feature_meta", {}).get(f, {}) if isinstance(meta.get("feature_meta", {}), dict) else {}
            out[f] = str(fm.get("strategy", fm.get("selected_strategy", "unknown")))
        else:
            out[f] = str(v)
    return out


def entropy_norm_from_counts(counts: np.ndarray) -> float:
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    h = -np.sum(p * np.log(p + 1e-12))
    denom = math.log(max(2, len(counts)))
    return float(h / denom)


def gini_from_counts(counts: np.ndarray) -> float:
    counts = counts.astype(np.float64)
    if counts.sum() <= 0 or len(counts) <= 1:
        return 0.0
    x = np.sort(counts)
    n = len(x)
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def compute_token_feature_audit(
    X_bin: np.ndarray,
    X_offset: np.ndarray,
    y: np.ndarray,
    *,
    num_bins: int,
    features: List[str],
    strategies: Dict[str, str],
    class_names: List[str],
    split: str,
    raw_values: Optional[np.ndarray] = None,
    rare_threshold: int = 5,
    rare_reference_counts: Optional[List[Dict[int, int]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[int, int]]]:
    rows = []
    ref_counts_out: List[Dict[int, int]] = []
    N, F = X_bin.shape
    for j, f in enumerate(features):
        bins = X_bin[:, j].astype(np.int64)
        off = X_offset[:, j].astype(np.float32)
        binc = np.bincount(np.clip(bins, 0, num_bins - 1), minlength=num_bins)
        used_counts = binc[binc > 0]
        used_bins = int((binc > 0).sum())
        ref_counts_out.append({int(k): int(v) for k, v in enumerate(binc) if v > 0})
        dominant_count = int(used_counts.max()) if used_counts.size else 0
        raw_unique = int(pd.Series(raw_values[:, j]).nunique()) if raw_values is not None else np.nan
        possible_unique = min(raw_unique, num_bins) if raw_values is not None else np.nan
        compression_factor = float(raw_unique / max(used_bins, 1)) if raw_values is not None else np.nan
        preserve_ratio = float(used_bins / max(possible_unique, 1)) if raw_values is not None else np.nan
        rare_bins = int(((binc > 0) & (binc <= rare_threshold)).sum())
        rare_used_ratio = float(rare_bins / max(used_bins, 1))
        rare_cell_ratio_own = float(binc[(binc > 0) & (binc <= rare_threshold)].sum() / max(N, 1))

        # rare according to train-reference token counts, useful for val OOV/rare exposure.
        if rare_reference_counts is not None:
            ref = rare_reference_counts[j]
            is_rare_ref = np.array([ref.get(int(b), 0) <= rare_threshold for b in bins], dtype=bool)
            is_unseen_ref = np.array([ref.get(int(b), 0) == 0 for b in bins], dtype=bool)
            rare_cell_ratio_ref = float(is_rare_ref.mean())
            unseen_cell_ratio_ref = float(is_unseen_ref.mean())
        else:
            rare_cell_ratio_ref = rare_cell_ratio_own
            unseen_cell_ratio_ref = 0.0

        row = {
            "split": split,
            "feature_idx": j,
            "feature": f,
            "strategy": strategies.get(f, "unknown"),
            "raw_unique": raw_unique,
            "bins_used": used_bins,
            "dead_bins": int(num_bins - used_bins),
            "used_bin_ratio": float(used_bins / num_bins),
            "dominant_bin": int(np.argmax(binc)) if used_bins else -1,
            "dominant_bin_count": dominant_count,
            "dominant_bin_ratio": float(dominant_count / max(N, 1)),
            "rare_used_bins_le5": rare_bins,
            "rare_used_bin_ratio_le5": rare_used_ratio,
            "rare_cell_ratio_own_le5": rare_cell_ratio_own,
            "rare_cell_ratio_trainref_le5": rare_cell_ratio_ref,
            "unseen_cell_ratio_trainref": unseen_cell_ratio_ref,
            "entropy_norm": entropy_norm_from_counts(binc),
            "gini_used_counts": gini_from_counts(used_counts) if used_counts.size else 0.0,
            "compression_factor": compression_factor,
            "possible_unique": possible_unique,
            "preserve_ratio": preserve_ratio,
            "offset_nonzero_ratio": float((np.abs(off) > 1e-8).mean()),
            "offset_mean": float(off.mean()),
            "offset_std": float(off.std()),
            "offset_unique_approx": int(pd.Series(np.round(off, 6)).nunique()),
        }
        for ci, cname in enumerate(class_names):
            m = (y == ci)
            if m.any():
                bb = bins[m]
                cc = np.bincount(np.clip(bb, 0, num_bins - 1), minlength=num_bins)
                used = int((cc > 0).sum())
                dom = int(cc.max())
                row[f"class_{cname}_bins_used"] = used
                row[f"class_{cname}_dominant_ratio"] = float(dom / m.sum())
                if rare_reference_counts is not None:
                    ref = rare_reference_counts[j]
                    row[f"class_{cname}_rare_trainref_ratio_le5"] = float(np.mean([ref.get(int(b), 0) <= rare_threshold for b in bb]))
                else:
                    row[f"class_{cname}_rare_own_ratio_le5"] = float(cc[(cc > 0) & (cc <= rare_threshold)].sum() / m.sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    summary = {
        "split": split,
        "n_rows": int(N),
        "n_features": int(F),
        "num_bins": int(num_bins),
        "mean_bins_used": float(df["bins_used"].mean()),
        "mean_rare_used_bin_ratio_le5": float(df["rare_used_bin_ratio_le5"].mean()),
        "mean_rare_cell_ratio_trainref_le5": float(df["rare_cell_ratio_trainref_le5"].mean()),
        "mean_unseen_cell_ratio_trainref": float(df["unseen_cell_ratio_trainref"].mean()),
        "mean_entropy_norm": float(df["entropy_norm"].mean()),
        "mean_dominant_bin_ratio": float(df["dominant_bin_ratio"].mean()),
    }
    return df, summary, ref_counts_out


def strategy_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "bins_used", "used_bin_ratio", "dominant_bin_ratio", "rare_used_bin_ratio_le5",
        "rare_cell_ratio_trainref_le5", "unseen_cell_ratio_trainref", "entropy_norm", "compression_factor",
        "preserve_ratio", "offset_nonzero_ratio",
    ]
    return df.groupby("strategy", dropna=False)[cols].agg(["count", "mean", "median", "max"]).reset_index()


def load_raw_matrix(root: Path, features: List[str], split: str) -> Optional[np.ndarray]:
    p = root / "01_split" / f"{split}_raw.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"[WARN] raw {split} missing features, skip raw_unique: {missing[:5]}")
        return None
    return df.loc[:, features].to_numpy()


def raw_scaled_from_train(root: Path, features: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    train_p = root / "01_split" / "train_raw.csv"
    val_p = root / "01_split" / "val_raw.csv"
    if not train_p.exists() or not val_p.exists():
        return None, None, {"available": False, "reason": "missing train_raw.csv/val_raw.csv"}
    tr = pd.read_csv(train_p)
    va = pd.read_csv(val_p)
    if any(f not in tr.columns for f in features) or any(f not in va.columns for f in features):
        return None, None, {"available": False, "reason": "raw files missing selected features"}
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
    return Rtr, Rva, {"available": True, "constant_features": [features[i] for i,c in enumerate(constant) if bool(c)]}


def build_model(train_mod, ckpt: Dict[str, Any], meta: Dict[str, Any], num_bins: int, n_features: int, num_classes: int, device: torch.device):
    cfg = ckpt.get("config", {}).get("model_config", {})
    def get(name, default): return cfg.get(name, default)
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
    N, F = X_bin.shape
    mask = np.ones_like(X_bin, dtype=np.float32)
    vals = np.stack([X_offset.astype(np.float32), X_cont.astype(np.float32), mask], axis=-1)
    ds = TensorDataset(torch.as_tensor(X_bin, dtype=torch.long), torch.as_tensor(vals, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_all = []
    for xb, vb in dl:
        xb = xb.to(device)
        vb = vb.to(device)
        logits = model(xb, vb)
        logits_all.append(logits.detach().cpu())
    logits = torch.cat(logits_all, dim=0)
    probs = torch.softmax(logits, dim=-1).numpy()
    pred = probs.argmax(axis=1).astype(np.int64)
    conf = probs.max(axis=1).astype(np.float32)
    return logits.numpy(), probs, pred, conf


@torch.no_grad()
def embedding_feature_audit(model, X_bin: np.ndarray, X_offset: np.ndarray, X_cont: np.ndarray, features: List[str], strategies: Dict[str,str], batch_size: int, device: torch.device) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    emb = model.embedding
    N, F = X_bin.shape
    local_sum = np.zeros(F, dtype=np.float64)
    delta_sum = np.zeros(F, dtype=np.float64)
    value_sum = np.zeros(F, dtype=np.float64)
    local_sq_sum = np.zeros(F, dtype=np.float64)
    delta_sq_sum = np.zeros(F, dtype=np.float64)
    nobs = 0
    ds = TensorDataset(torch.as_tensor(X_bin, dtype=torch.long), torch.as_tensor(X_offset, dtype=torch.float32), torch.as_tensor(X_cont, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    for xb, ob, cb in dl:
        xb = xb.to(device)
        off = ob.to(device).unsqueeze(-1).clamp(0, 1)
        cont = cb.to(device).unsqueeze(-1).clamp(0, 1)
        local = emb.local_interp(xb, off)
        gamma = torch.tanh(emb.gamma_proj(cont))
        beta = emb.beta_proj(cont)
        gate = torch.sigmoid(emb.cont_gate_logit).to(device).unsqueeze(0).expand(xb.shape[0], F, 1)
        value = local * (1.0 + gate * gamma) + gate * beta
        delta = value - local
        ln = torch.linalg.vector_norm(local, dim=-1).detach().cpu().numpy()
        dn = torch.linalg.vector_norm(delta, dim=-1).detach().cpu().numpy()
        vn = torch.linalg.vector_norm(value, dim=-1).detach().cpu().numpy()
        local_sum += ln.sum(axis=0)
        delta_sum += dn.sum(axis=0)
        value_sum += vn.sum(axis=0)
        local_sq_sum += (ln ** 2).sum(axis=0)
        delta_sq_sum += (dn ** 2).sum(axis=0)
        nobs += xb.shape[0]

    gate = torch.sigmoid(emb.cont_gate_logit.detach().cpu()).numpy().reshape(-1)
    feat_norm = torch.linalg.vector_norm(emb.feature_embedding.weight.detach().cpu(), dim=1).numpy()
    bin_w = emb.bin_embedding.weight.detach().cpu()
    bin_norm = torch.linalg.vector_norm(bin_w, dim=1).numpy()
    step_norm = torch.linalg.vector_norm(bin_w[1:] - bin_w[:-1], dim=1).numpy()

    rows = []
    for j, f in enumerate(features):
        local_mean = local_sum[j] / max(nobs, 1)
        delta_mean = delta_sum[j] / max(nobs, 1)
        rows.append({
            "feature_idx": j,
            "feature": f,
            "strategy": strategies.get(f, "unknown"),
            "feature_embedding_norm": float(feat_norm[j]),
            "cont_gate": float(gate[j]),
            "local_norm_mean": float(local_mean),
            "film_delta_norm_mean": float(delta_mean),
            "film_delta_over_local_mean": float(delta_mean / (local_mean + 1e-12)),
            "value_norm_mean": float(value_sum[j] / max(nobs, 1)),
            "local_norm_rms": float(math.sqrt(local_sq_sum[j] / max(nobs, 1))),
            "film_delta_norm_rms": float(math.sqrt(delta_sq_sum[j] / max(nobs, 1))),
            "raw_cont_mean": float(X_cont[:, j].mean()),
            "raw_cont_std": float(X_cont[:, j].std()),
            "offset_mean": float(X_offset[:, j].mean()),
            "offset_std": float(X_offset[:, j].std()),
        })
    edf = pd.DataFrame(rows)

    bdf = pd.DataFrame({
        "bin_id": np.arange(len(bin_norm)),
        "bin_embedding_norm": bin_norm,
        "step_norm_to_next": np.r_[step_norm, np.nan],
    })
    summary = {
        "shared_bin_embedding_num_embeddings": int(len(bin_norm)),
        "shared_bin_embedding_dim": int(bin_w.shape[1]),
        "bin_embedding_norm_mean": float(np.mean(bin_norm)),
        "bin_embedding_norm_std": float(np.std(bin_norm)),
        "bin_step_norm_mean": float(np.nanmean(np.r_[step_norm, np.nan])),
        "bin_step_norm_std": float(np.nanstd(np.r_[step_norm, np.nan])),
        "cont_gate_min": float(gate.min()),
        "cont_gate_max": float(gate.max()),
        "cont_gate_mean": float(gate.mean()),
        "cont_gate_std": float(gate.std()),
        "feature_embedding_norm_mean": float(feat_norm.mean()),
        "feature_embedding_norm_std": float(feat_norm.std()),
    }
    return edf, bdf, summary


def save_result_audit(run_dir: Path, out_dir: Path, y_train, y_val, pred_train, pred_val, conf_train, conf_val, class_names: List[str]):
    result_dir = out_dir / "01_result_audit"
    result_dir.mkdir(parents=True, exist_ok=True)
    diag_p = run_dir / "diagnosis_summary.json"
    if diag_p.exists():
        diag = read_json(diag_p)
        write_json(result_dir / "diagnosis_summary_copy.json", diag)
    else:
        diag = {}
    hist_p = run_dir / "history.csv"
    if hist_p.exists():
        hist = pd.read_csv(hist_p)
        hist.to_csv(result_dir / "history.csv", index=False)
        # best/final compact
        best_idx = hist["val_macro_f1"].idxmax() if "val_macro_f1" in hist.columns else len(hist)-1
        compact = pd.DataFrame([
            {"row": "best_by_val_macro", **hist.loc[best_idx].to_dict()},
            {"row": "final", **hist.iloc[-1].to_dict()},
        ])
        compact.to_csv(result_dir / "history_best_vs_final.csv", index=False)
    labels = list(range(len(class_names)))
    if classification_report is not None:
        for split, yt, yp, cf in [("train", y_train, pred_train, conf_train), ("val", y_val, pred_val, conf_val)]:
            rep = classification_report(yt, yp, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
            write_json(result_dir / f"{split}_classification_report_recomputed.json", rep)
            rows = []
            for cname in class_names:
                m = rep[cname]
                rows.append({"split": split, "class": cname, "precision": m["precision"], "recall": m["recall"], "f1": m["f1-score"], "support": m["support"]})
            pd.DataFrame(rows).to_csv(result_dir / f"{split}_per_class.csv", index=False)
    if confusion_matrix is not None:
        for split, yt, yp in [("train", y_train, pred_train), ("val", y_val, pred_val)]:
            cm = confusion_matrix(yt, yp, labels=labels)
            pd.DataFrame(cm, index=[f"true_{c}" for c in class_names], columns=[f"pred_{c}" for c in class_names]).to_csv(result_dir / f"{split}_confusion_matrix_recomputed.csv")
    # confidence by correct/wrong
    for split, yt, yp, cf in [("train", y_train, pred_train, conf_train), ("val", y_val, pred_val, conf_val)]:
        rows=[]
        for ci,cname in enumerate(class_names):
            m=(yt==ci)
            for corr_name, cmask in [("correct", yp==yt), ("wrong", yp!=yt)]:
                mm=m & cmask
                rows.append({"split":split,"class":cname,"group":corr_name,"n":int(mm.sum()),"confidence_mean":float(cf[mm].mean()) if mm.any() else np.nan,"confidence_median":float(np.median(cf[mm])) if mm.any() else np.nan})
        pd.DataFrame(rows).to_csv(result_dir / f"{split}_confidence_correct_wrong_by_class.csv", index=False)


def rare_exposure_by_sample(X_bin: np.ndarray, train_ref_counts: List[Dict[int,int]], rare_threshold: int) -> Tuple[np.ndarray, np.ndarray]:
    N,F = X_bin.shape
    rare = np.zeros((N,F), dtype=bool)
    unseen = np.zeros((N,F), dtype=bool)
    for j in range(F):
        ref = train_ref_counts[j]
        bj = X_bin[:,j]
        rare[:,j] = np.array([ref.get(int(b),0) <= rare_threshold for b in bj], dtype=bool)
        unseen[:,j] = np.array([ref.get(int(b),0) == 0 for b in bj], dtype=bool)
    return rare, unseen


def error_conditioned_audit(out_dir: Path, X_bin, X_offset, X_cont, y, pred, conf, rare_mask, unseen_mask, features, strategies, class_names):
    edir = out_dir / "04_error_conditioned_audit"
    edir.mkdir(parents=True, exist_ok=True)
    correct = (y == pred)
    # sample-level true/pred pair summary
    rows=[]
    for ti,tname in enumerate(class_names):
        for pi,pname in enumerate(class_names):
            m=(y==ti)&(pred==pi)
            if not m.any():
                continue
            rows.append({
                "true_class": tname,
                "pred_class": pname,
                "correct": bool(ti==pi),
                "n": int(m.sum()),
                "rare_cell_ratio_mean": float(rare_mask[m].mean()),
                "rare_cell_count_mean": float(rare_mask[m].sum(axis=1).mean()),
                "unseen_cell_ratio_mean": float(unseen_mask[m].mean()),
                "confidence_mean": float(conf[m].mean()),
                "confidence_median": float(np.median(conf[m])),
            })
    pair_df=pd.DataFrame(rows).sort_values(["correct","n"], ascending=[True,False])
    pair_df.to_csv(edir / "val_true_pred_pair_sample_summary.csv", index=False)

    # class-level correct vs wrong sample rare exposure
    rows=[]
    for ci,cname in enumerate(class_names):
        for gname, gmask in [("correct", correct), ("wrong", ~correct)]:
            m=(y==ci)&gmask
            rows.append({
                "true_class": cname,
                "group": gname,
                "n": int(m.sum()),
                "rare_cell_ratio_mean": float(rare_mask[m].mean()) if m.any() else np.nan,
                "rare_cell_count_mean": float(rare_mask[m].sum(axis=1).mean()) if m.any() else np.nan,
                "unseen_cell_ratio_mean": float(unseen_mask[m].mean()) if m.any() else np.nan,
                "confidence_mean": float(conf[m].mean()) if m.any() else np.nan,
            })
    pd.DataFrame(rows).to_csv(edir / "val_correct_vs_wrong_rare_by_true_class.csv", index=False)

    # per-feature wrong-vs-correct rare rate by true class
    rows=[]
    for ci,cname in enumerate(class_names):
        mc=(y==ci)&correct
        mw=(y==ci)&(~correct)
        for j,f in enumerate(features):
            rc=float(rare_mask[mc,j].mean()) if mc.any() else np.nan
            rw=float(rare_mask[mw,j].mean()) if mw.any() else np.nan
            uc=float(unseen_mask[mc,j].mean()) if mc.any() else np.nan
            uw=float(unseen_mask[mw,j].mean()) if mw.any() else np.nan
            rows.append({
                "true_class": cname,
                "feature_idx": j,
                "feature": f,
                "strategy": strategies.get(f,"unknown"),
                "correct_n": int(mc.sum()),
                "wrong_n": int(mw.sum()),
                "rare_rate_correct": rc,
                "rare_rate_wrong": rw,
                "rare_wrong_minus_correct": rw-rc if np.isfinite(rw) and np.isfinite(rc) else np.nan,
                "unseen_rate_correct": uc,
                "unseen_rate_wrong": uw,
                "unseen_wrong_minus_correct": uw-uc if np.isfinite(uw) and np.isfinite(uc) else np.nan,
                "offset_mean_correct": float(X_offset[mc,j].mean()) if mc.any() else np.nan,
                "offset_mean_wrong": float(X_offset[mw,j].mean()) if mw.any() else np.nan,
                "raw_cont_mean_correct": float(X_cont[mc,j].mean()) if mc.any() else np.nan,
                "raw_cont_mean_wrong": float(X_cont[mw,j].mean()) if mw.any() else np.nan,
            })
    feat_err=pd.DataFrame(rows)
    feat_err.sort_values(["true_class","rare_wrong_minus_correct"], ascending=[True,False]).to_csv(edir / "val_feature_rare_wrong_vs_correct_by_class.csv", index=False)

    # focus malware confusion pairs, compare each mistaken pair vs true-class correct samples
    focus_rows=[]
    for ti,tname in enumerate(class_names):
        if tname == "Benign":
            continue
        correct_t=(y==ti)&(pred==ti)
        for pi,pname in enumerate(class_names):
            if pi==ti:
                continue
            m=(y==ti)&(pred==pi)
            if m.sum() < 10:
                continue
            for j,f in enumerate(features):
                focus_rows.append({
                    "true_class": tname,
                    "pred_class": pname,
                    "pair_n": int(m.sum()),
                    "feature_idx": j,
                    "feature": f,
                    "strategy": strategies.get(f,"unknown"),
                    "pair_rare_rate": float(rare_mask[m,j].mean()),
                    "true_correct_rare_rate": float(rare_mask[correct_t,j].mean()) if correct_t.any() else np.nan,
                    "rare_pair_minus_true_correct": float(rare_mask[m,j].mean() - rare_mask[correct_t,j].mean()) if correct_t.any() else np.nan,
                    "pair_unseen_rate": float(unseen_mask[m,j].mean()),
                    "pair_bin_mean": float(X_bin[m,j].mean()),
                    "true_correct_bin_mean": float(X_bin[correct_t,j].mean()) if correct_t.any() else np.nan,
                    "pair_offset_mean": float(X_offset[m,j].mean()),
                    "true_correct_offset_mean": float(X_offset[correct_t,j].mean()) if correct_t.any() else np.nan,
                    "pair_raw_cont_mean": float(X_cont[m,j].mean()),
                    "true_correct_raw_cont_mean": float(X_cont[correct_t,j].mean()) if correct_t.any() else np.nan,
                })
    focus=pd.DataFrame(focus_rows)
    if not focus.empty:
        focus.sort_values(["true_class","pred_class","rare_pair_minus_true_correct"], ascending=[True,True,False]).to_csv(edir / "val_feature_audit_by_confusion_pair.csv", index=False)
        # top compact view
        focus.groupby(["true_class","pred_class"]).head(20).to_csv(edir / "val_top20_features_by_pair_rare_delta.csv", index=False)


def make_markdown_summary(out_dir: Path, paths: Dict[str,Path], result_obj: Dict[str,Any], token_summary: Dict[str,Any], emb_summary: Dict[str,Any]) -> None:
    md = []
    md.append("# C2 D3 Best Audit Summary\n")
    md.append("## Inputs\n")
    for k,v in paths.items():
        if k != "root":
            md.append(f"- {k}: `{v}`")
    md.append("\n## What this audit tests\n")
    md.append("Hypothesis: model has moderate overfit and the most likely driver is rare/sparse tokens. This audit does not change the model; it checks whether validation errors are actually enriched with rare/unseen tokens.\n")
    md.append("## Result snapshot\n")
    diag = result_obj.get("diagnosis", {})
    if diag:
        tr=diag.get("train",{}); va=diag.get("val",{})
        md.append(f"- train macro-F1: `{tr.get('macro_f1')}`")
        md.append(f"- val macro-F1: `{va.get('macro_f1')}`")
        md.append(f"- gap: `{diag.get('generalization_gap_macro_f1')}`")
    md.append("\n## Token snapshot\n")
    md.append("```json")
    md.append(json.dumps(token_summary, indent=2, ensure_ascii=False))
    md.append("```\n")
    md.append("## Embedding snapshot\n")
    md.append("```json")
    md.append(json.dumps(emb_summary, indent=2, ensure_ascii=False))
    md.append("```\n")
    md.append("## Key files to inspect first\n")
    md.append("- `01_result_audit/history_best_vs_final.csv`")
    md.append("- `02_token_audit/token_feature_audit_train.csv`")
    md.append("- `02_token_audit/token_strategy_summary_train.csv`")
    md.append("- `03_embedding_audit/embedding_feature_audit_val.csv`")
    md.append("- `04_error_conditioned_audit/val_correct_vs_wrong_rare_by_true_class.csv`")
    md.append("- `04_error_conditioned_audit/val_feature_rare_wrong_vs_correct_by_class.csv`")
    md.append("- `04_error_conditioned_audit/val_top20_features_by_pair_rare_delta.csv`")
    (out_dir / "audit_summary.md").write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-npz", default="")
    ap.add_argument("--metadata-json", default="")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--out-dir", default="03_outputs/audit_c2_best")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--rare-threshold", type=int, default=5)
    args = ap.parse_args()

    root = repo_root()
    paths = resolve_paths(args)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[audit] root:", root)
    for k,v in paths.items():
        print(f"[audit] {k}: {v}")
    print("[audit] out_dir:", out_dir)

    data, meta = load_dataset(paths["dataset"], paths["metadata"])
    features = [str(x) for x in meta["feature_names"]]
    strategies = feature_strategies(meta, features)
    class_names = labels_from_meta(meta)
    num_bins = int(meta.get("num_bins", meta.get("K", 512)))
    num_classes = len(class_names)

    Xtr_bin = data["X_train_bin"].astype(np.int64)
    Xva_bin = data["X_val_bin"].astype(np.int64)
    Xtr_off = data["X_train_offset"].astype(np.float32)
    Xva_off = data["X_val_offset"].astype(np.float32)
    ytr = data["y_train"].astype(np.int64)
    yva = data["y_val"].astype(np.int64)

    raw_tr = load_raw_matrix(root, features, "train")
    raw_va = load_raw_matrix(root, features, "val")
    Rtr, Rva, raw_cont_info = raw_scaled_from_train(root, features)
    if Rtr is None or Rva is None:
        raise FileNotFoundError("Need 01_split/train_raw.csv and val_raw.csv for D3 raw FiLM audit.")

    token_dir = out_dir / "02_token_audit"
    token_dir.mkdir(parents=True, exist_ok=True)
    train_token_df, train_token_summary, train_ref_counts = compute_token_feature_audit(
        Xtr_bin, Xtr_off, ytr, num_bins=num_bins, features=features, strategies=strategies, class_names=class_names,
        split="train", raw_values=raw_tr, rare_threshold=args.rare_threshold, rare_reference_counts=None,
    )
    val_token_df, val_token_summary, _ = compute_token_feature_audit(
        Xva_bin, Xva_off, yva, num_bins=num_bins, features=features, strategies=strategies, class_names=class_names,
        split="val", raw_values=raw_va, rare_threshold=args.rare_threshold, rare_reference_counts=train_ref_counts,
    )
    train_token_df.to_csv(token_dir / "token_feature_audit_train.csv", index=False)
    val_token_df.to_csv(token_dir / "token_feature_audit_val.csv", index=False)
    strategy_summary(train_token_df).to_csv(token_dir / "token_strategy_summary_train.csv", index=False)
    strategy_summary(val_token_df).to_csv(token_dir / "token_strategy_summary_val.csv", index=False)
    write_json(token_dir / "token_global_summary.json", {"train": train_token_summary, "val": val_token_summary, "raw_cont_info": raw_cont_info})

    # model load/predict
    train_mod, train_mod_path = import_train_module(root)
    device = auto_device(args.device)
    ckpt = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    model = build_model(train_mod, ckpt, meta, num_bins, len(features), num_classes, device)
    print("[audit] imported train module:", train_mod_path)
    print("[audit] device:", device)

    logits_tr, probs_tr, pred_tr, conf_tr = predict_model(model, Xtr_bin, Xtr_off, Rtr, args.batch_size, device)
    logits_va, probs_va, pred_va, conf_va = predict_model(model, Xva_bin, Xva_off, Rva, args.batch_size, device)

    result_obj = {"diagnosis": read_json(paths["run_dir"] / "diagnosis_summary.json") if (paths["run_dir"]/"diagnosis_summary.json").exists() else {}}
    save_result_audit(paths["run_dir"], out_dir, ytr, yva, pred_tr, pred_va, conf_tr, conf_va, class_names)

    # save predictions for downstream analysis, compact
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    pd.DataFrame({"split":"train", "y_true":ytr, "y_pred":pred_tr, "confidence":conf_tr, "true_label":[class_names[i] for i in ytr], "pred_label":[class_names[i] for i in pred_tr]}).to_csv(pred_dir / "train_predictions.csv", index=False)
    pd.DataFrame({"split":"val", "y_true":yva, "y_pred":pred_va, "confidence":conf_va, "true_label":[class_names[i] for i in yva], "pred_label":[class_names[i] for i in pred_va]}).to_csv(pred_dir / "val_predictions.csv", index=False)

    # rare exposure and error conditioned
    rare_tr, unseen_tr = rare_exposure_by_sample(Xtr_bin, train_ref_counts, args.rare_threshold)
    rare_va, unseen_va = rare_exposure_by_sample(Xva_bin, train_ref_counts, args.rare_threshold)
    # split/class rare exposure table
    rows=[]
    for split, y, pred, rare, unseen in [("train",ytr,pred_tr,rare_tr,unseen_tr),("val",yva,pred_va,rare_va,unseen_va)]:
        for ci,cname in enumerate(class_names):
            for group,mask in [("all",np.ones_like(y,dtype=bool)),("correct",y==pred),("wrong",y!=pred)]:
                m=(y==ci)&mask
                rows.append({"split":split,"class":cname,"group":group,"n":int(m.sum()),"rare_cell_ratio_mean":float(rare[m].mean()) if m.any() else np.nan,"rare_cell_count_mean":float(rare[m].sum(axis=1).mean()) if m.any() else np.nan,"unseen_cell_ratio_mean":float(unseen[m].mean()) if m.any() else np.nan})
    pd.DataFrame(rows).to_csv(token_dir / "rare_exposure_by_split_class_correctness.csv", index=False)

    error_conditioned_audit(out_dir, Xva_bin, Xva_off, Rva, yva, pred_va, conf_va, rare_va, unseen_va, features, strategies, class_names)

    # embedding audit
    emb_dir = out_dir / "03_embedding_audit"
    emb_dir.mkdir(parents=True, exist_ok=True)
    edf, bdf, emb_summary = embedding_feature_audit(model, Xva_bin, Xva_off, Rva, features, strategies, args.batch_size, device)
    # join token metrics for one richer table
    enrich = edf.merge(val_token_df[["feature_idx","bins_used","dominant_bin_ratio","rare_used_bin_ratio_le5","rare_cell_ratio_trainref_le5","entropy_norm","compression_factor","preserve_ratio"]], on="feature_idx", how="left")
    enrich.to_csv(emb_dir / "embedding_feature_audit_val.csv", index=False)
    bdf.to_csv(emb_dir / "shared_bin_embedding_audit.csv", index=False)
    write_json(emb_dir / "embedding_global_summary.json", emb_summary)

    # correlations to directly test rare-token hypothesis
    corr_rows=[]
    merged = enrich.copy()
    for x in ["rare_used_bin_ratio_le5", "rare_cell_ratio_trainref_le5", "dominant_bin_ratio", "entropy_norm", "compression_factor", "film_delta_over_local_mean", "cont_gate"]:
        for ycol in ["film_delta_over_local_mean", "cont_gate", "feature_embedding_norm"]:
            if x == ycol or x not in merged or ycol not in merged:
                continue
            a = pd.to_numeric(merged[x], errors="coerce")
            b = pd.to_numeric(merged[ycol], errors="coerce")
            m = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
            if m.sum() >= 3:
                corr_rows.append({"x":x,"y":ycol,"pearson":float(a[m].corr(b[m], method="pearson")),"spearman":float(a[m].corr(b[m], method="spearman")),"n_features":int(m.sum())})
    pd.DataFrame(corr_rows).to_csv(out_dir / "rare_token_embedding_correlations.csv", index=False)

    make_markdown_summary(out_dir, paths, result_obj, {"train":train_token_summary,"val":val_token_summary}, emb_summary)

    # zip outputs for upload back
    import zipfile
    zip_path = root.parent / "c2_best_audit_outputs.zip"
    if str(root).startswith("/kaggle/working"):
        zip_path = Path("/kaggle/working/c2_best_audit_outputs.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(out_dir.parent)))
    print("[audit] DONE")
    print("[audit] output_dir:", out_dir)
    print("[audit] zip:", zip_path)


if __name__ == "__main__":
    main()

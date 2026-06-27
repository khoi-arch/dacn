#!/usr/bin/env python3
"""
Group-wise tokenization + pair-geometry audit for the C2 D3 best run.

This script does NOT train a model. It reads existing C2 audit outputs, C2 dataset artifacts,
and optional K1024 rerun artifacts, then exports:
  1) result audit across C2/K1024 runs
  2) group-wise tokenization summaries and K deltas
  3) pair geometry audits, e.g. Trojan correct vs Trojan->Ransomware vs Ransomware correct

Expected to be run from repo root: ~/Documents/dacn
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


CLASS_DEFAULT = ["Benign", "Ransomware", "Spyware", "Trojan"]
MAJOR_PAIRS = [
    ("Trojan", "Ransomware"),
    ("Ransomware", "Spyware"),
    ("Spyware", "Ransomware"),
    ("Ransomware", "Trojan"),
    ("Spyware", "Trojan"),
    ("Trojan", "Spyware"),
]


def repo_root() -> Path:
    return Path.cwd().resolve()


def read_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_read_csv(p: Path, **kwargs) -> Optional[pd.DataFrame]:
    if not p.exists():
        return None
    return pd.read_csv(p, **kwargs)


def load_dataset(npz_path: Path, metadata_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing dataset npz: {npz_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata json: {metadata_path}")
    arr = np.load(npz_path, allow_pickle=True)
    data = {k: arr[k] for k in arr.files}
    required = ["X_train_bin", "X_train_offset", "y_train", "X_val_bin", "X_val_offset", "y_val"]
    miss = [k for k in required if k not in data]
    if miss:
        raise ValueError(f"{npz_path} missing keys: {miss}; keys={arr.files}")
    meta = read_json(metadata_path)
    return data, meta


def labels_from_meta(meta: Dict[str, Any]) -> List[str]:
    lm = meta.get("label_mapping", {})
    if isinstance(lm, dict) and lm:
        inv = {int(v): str(k).strip() for k, v in lm.items()}
        return [inv[i] for i in sorted(inv)]
    return CLASS_DEFAULT


def feature_names_from_meta(meta: Dict[str, Any], data: Dict[str, np.ndarray]) -> List[str]:
    names = meta.get("feature_names") or meta.get("features") or meta.get("feature_list")
    if names is not None:
        return [str(x) for x in names]
    f = int(data["X_train_bin"].shape[1])
    return [f"feature_{i}" for i in range(f)]


def feature_strategies_from_meta(meta: Dict[str, Any], features: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    fs = meta.get("feature_strategies", {})
    fm_all = meta.get("feature_meta", {})
    policy = meta.get("policy", {})
    for f in features:
        v = fs.get(f) if isinstance(fs, dict) else None
        if isinstance(v, dict):
            out[f] = str(v.get("strategy", v.get("selected_strategy", v.get("action", "unknown"))))
        elif isinstance(v, str):
            out[f] = v
        elif isinstance(fm_all, dict) and isinstance(fm_all.get(f), dict):
            fm = fm_all[f]
            out[f] = str(fm.get("strategy", fm.get("selected_strategy", fm.get("action", "unknown"))))
        elif isinstance(policy, dict) and isinstance(policy.get(f), dict):
            out[f] = str(policy[f].get("strategy", policy[f].get("action", "unknown")))
        else:
            out[f] = "unknown"
    return out


def entropy_norm(counts: np.ndarray) -> float:
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    h = -np.sum(p * np.log(p + 1e-12))
    return float(h / math.log(max(2, len(counts))))


def compute_token_feature_audit(
    X_bin: np.ndarray,
    X_off: np.ndarray,
    y: np.ndarray,
    *,
    num_bins: int,
    features: List[str],
    strategies: Dict[str, str],
    class_names: List[str],
    split: str,
    rare_threshold: int,
    train_ref_counts: Optional[List[np.ndarray]] = None,
) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    rows = []
    ref_counts_out: List[np.ndarray] = []
    n, f = X_bin.shape
    for j, feat in enumerate(features):
        b = np.clip(X_bin[:, j].astype(np.int64), 0, num_bins - 1)
        off = X_off[:, j].astype(np.float32)
        counts = np.bincount(b, minlength=num_bins)
        used = np.flatnonzero(counts)
        used_counts = counts[used]
        rare_used = int((used_counts <= rare_threshold).sum()) if len(used_counts) else 0
        rare_used_ratio = rare_used / max(1, len(used_counts))
        rare_cell_own = float((counts[b] <= rare_threshold).mean()) if len(b) else 0.0
        if train_ref_counts is not None:
            ref = train_ref_counts[j]
            ref_safe = np.zeros(num_bins, dtype=np.int64)
            ref_safe[: min(num_bins, len(ref))] = ref[: min(num_bins, len(ref))]
            rare_cell_trainref = float((ref_safe[b] <= rare_threshold).mean())
            unseen_cell_trainref = float((ref_safe[b] == 0).mean())
        else:
            rare_cell_trainref = rare_cell_own
            unseen_cell_trainref = 0.0
        dom_count = int(used_counts.max()) if len(used_counts) else 0
        dom_bin = int(used[np.argmax(used_counts)]) if len(used_counts) else -1
        possible_unique = min(int(pd.Series(b).nunique()), num_bins)
        raw_unique = int(pd.Series(b).nunique())
        class_part = {}
        for ci, cname in enumerate(class_names):
            m = (y == ci)
            if not m.any():
                class_part[f"class_{cname}_bins_used"] = 0
                class_part[f"class_{cname}_dominant_ratio"] = np.nan
                class_part[f"class_{cname}_rare_trainref_ratio"] = np.nan
                continue
            bc = np.bincount(b[m], minlength=num_bins)
            bu = np.flatnonzero(bc)
            buc = bc[bu]
            class_part[f"class_{cname}_bins_used"] = int(len(bu))
            class_part[f"class_{cname}_dominant_ratio"] = float(buc.max() / m.sum()) if len(buc) else np.nan
            if train_ref_counts is not None:
                class_part[f"class_{cname}_rare_trainref_ratio"] = float((ref_safe[b[m]] <= rare_threshold).mean())
            else:
                class_part[f"class_{cname}_rare_trainref_ratio"] = float((bc[b[m]] <= rare_threshold).mean())
        rows.append({
            "split": split,
            "feature_idx": j,
            "feature": feat,
            "strategy": strategies.get(feat, "unknown"),
            "num_bins": num_bins,
            "bins_used": int(len(used)),
            "dead_bins": int(num_bins - len(used)),
            "used_bin_ratio": float(len(used) / max(1, num_bins)),
            "dominant_bin": dom_bin,
            "dominant_bin_count": dom_count,
            "dominant_bin_ratio": float(dom_count / max(1, n)),
            "rare_used_bins_le5": rare_used,
            "rare_used_bin_ratio_le5": rare_used_ratio,
            "rare_cell_ratio_own_le5": rare_cell_own,
            "rare_cell_ratio_trainref_le5": rare_cell_trainref,
            "unseen_cell_ratio_trainref": unseen_cell_trainref,
            "entropy_norm": entropy_norm(counts[used]) if len(used) else 0.0,
            "compression_factor": float(n / max(1, len(used))),
            "offset_nonzero_ratio": float((np.abs(off) > 1e-12).mean()),
            "offset_mean": float(off.mean()),
            "offset_std": float(off.std()),
            "offset_unique_approx": int(pd.Series(np.round(off, 6)).nunique()),
            **class_part,
        })
        ref_counts_out.append(counts)
    return pd.DataFrame(rows), ref_counts_out


def group_summary(df: pd.DataFrame, run: str) -> pd.DataFrame:
    metrics = [
        "bins_used", "used_bin_ratio", "dominant_bin_ratio", "rare_used_bin_ratio_le5",
        "rare_cell_ratio_trainref_le5", "unseen_cell_ratio_trainref", "entropy_norm",
        "compression_factor", "offset_nonzero_ratio", "offset_std",
    ]
    rows = []
    for (split, strategy), g in df.groupby(["split", "strategy"], dropna=False):
        row = {"run": run, "split": split, "strategy": strategy, "n_features": len(g)}
        for m in metrics:
            if m in g.columns:
                row[m + "_mean"] = float(g[m].mean())
                row[m + "_median"] = float(g[m].median())
                row[m + "_max"] = float(g[m].max())
        rows.append(row)
    return pd.DataFrame(rows)


def group_class_summary(df: pd.DataFrame, run: str, class_names: List[str]) -> pd.DataFrame:
    rows = []
    for (split, strategy), g in df.groupby(["split", "strategy"], dropna=False):
        for cname in class_names:
            row = {"run": run, "split": split, "strategy": strategy, "class": cname, "n_features": len(g)}
            for metric in ["bins_used", "dominant_ratio", "rare_trainref_ratio"]:
                col = f"class_{cname}_{metric}"
                if col in g.columns:
                    row[col + "_mean"] = float(g[col].mean())
                    row[col + "_median"] = float(g[col].median())
                    row[col + "_max"] = float(g[col].max())
            rows.append(row)
    return pd.DataFrame(rows)


def read_report(run_dir: Path) -> Dict[str, Any]:
    val = read_json(run_dir / "val_classification_report_best.json") if (run_dir / "val_classification_report_best.json").exists() else {}
    train = read_json(run_dir / "train_classification_report_best.json") if (run_dir / "train_classification_report_best.json").exists() else {}
    diag = read_json(run_dir / "diagnosis_summary.json") if (run_dir / "diagnosis_summary.json").exists() else {}
    return {"val": val, "train": train, "diag": diag}


def extract_metric(report: Dict[str, Any], split: str, key: str, default=np.nan) -> float:
    diag = report.get("diag", {})
    if isinstance(diag.get(split), dict) and key in diag[split]:
        return float(diag[split][key])
    obj = report.get(split, {})
    if key in obj:
        return float(obj[key])
    if key == "macro_f1":
        for k in ["macro avg", "macro_avg", "macro"]:
            if isinstance(obj.get(k), dict) and "f1-score" in obj[k]:
                return float(obj[k]["f1-score"])
            if isinstance(obj.get(k), dict) and "f1" in obj[k]:
                return float(obj[k]["f1"])
    return default


def read_perclass(report: Dict[str, Any], split: str, class_names: List[str]) -> Dict[str, float]:
    obj = report.get(split, {})
    out = {}
    per = obj.get("per_class") if isinstance(obj, dict) else None
    for cname in class_names:
        v = None
        if isinstance(per, dict):
            for k, d in per.items():
                if str(k).strip() == cname and isinstance(d, dict):
                    v = d.get("f1", d.get("f1-score"))
        if v is None and isinstance(obj.get(cname), dict):
            v = obj[cname].get("f1", obj[cname].get("f1-score"))
        out[f"{split}_{cname}_f1"] = float(v) if v is not None else np.nan
    return out


def result_summary(run_dirs: Dict[str, Path], class_names: List[str]) -> pd.DataFrame:
    rows = []
    for run, rd in run_dirs.items():
        if not rd.exists():
            continue
        rep = read_report(rd)
        row = {"run": run, "run_dir": str(rd)}
        for split in ["train", "val"]:
            row[f"{split}_macro_f1"] = extract_metric(rep, split, "macro_f1")
            row[f"{split}_accuracy"] = extract_metric(rep, split, "accuracy")
            row[f"{split}_weighted_f1"] = extract_metric(rep, split, "weighted_f1")
            row.update(read_perclass(rep, split, class_names))
        if not np.isnan(row.get("train_macro_f1", np.nan)) and not np.isnan(row.get("val_macro_f1", np.nan)):
            row["gap_macro_f1"] = row["train_macro_f1"] - row["val_macro_f1"]
        rows.append(row)
    return pd.DataFrame(rows)


def read_confusion(run_dir: Path, split: str = "val") -> Optional[pd.DataFrame]:
    p = run_dir / f"{split}_confusion_matrix_best.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0)
    df.index = [str(x).strip() for x in df.index]
    df.columns = [str(x).strip() for x in df.columns]
    return df


def confusion_long(run_dirs: Dict[str, Path], class_names: List[str]) -> pd.DataFrame:
    rows = []
    for run, rd in run_dirs.items():
        cm = read_confusion(rd, "val")
        if cm is None:
            continue
        for t in class_names:
            for p in class_names:
                val = int(cm.loc[t, p]) if (t in cm.index and p in cm.columns) else 0
                rows.append({"run": run, "true_class": t, "pred_class": p, "count": val, "correct": t == p})
    return pd.DataFrame(rows)


def normalize_predictions(df: pd.DataFrame, class_names: List[str]) -> pd.DataFrame:
    out = df.copy()
    if "true_label" not in out.columns and "true_id" in out.columns:
        out["true_label"] = out["true_id"].map({i: c for i, c in enumerate(class_names)})
    if "pred_label" not in out.columns and "pred_id" in out.columns:
        out["pred_label"] = out["pred_id"].map({i: c for i, c in enumerate(class_names)})
    if "true_label" in out.columns:
        out["true_label"] = out["true_label"].astype(str).str.strip()
    if "pred_label" in out.columns:
        out["pred_label"] = out["pred_label"].astype(str).str.strip()
    if "correct" not in out.columns:
        out["correct"] = out["true_label"] == out["pred_label"]
    return out


def load_predictions(run: str, run_dir: Path, c2_audit_dir: Path, class_names: List[str]) -> Optional[pd.DataFrame]:
    candidates = []
    if run == "C2_K512":
        candidates.append(c2_audit_dir / "predictions" / "val_predictions.csv")
    candidates.append(run_dir / "val_predictions_best.csv")
    for p in candidates:
        if p.exists():
            return normalize_predictions(pd.read_csv(p), class_names)
    return None


def load_raw_scaled(root: Path, features: List[str]) -> Optional[np.ndarray]:
    train_p = root / "01_split" / "train_raw.csv"
    val_p = root / "01_split" / "val_raw.csv"
    if not train_p.exists() or not val_p.exists():
        return None
    tr = pd.read_csv(train_p)
    va = pd.read_csv(val_p)
    vals = []
    for f in features:
        if f not in va.columns or f not in tr.columns:
            vals.append(np.full(len(va), np.nan, dtype=np.float32))
            continue
        xtr = pd.to_numeric(tr[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        xva = pd.to_numeric(va[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = float(np.nanmedian(xtr.values)) if np.isfinite(np.nanmedian(xtr.values)) else 0.0
        xtr = xtr.fillna(med).astype(float)
        xva = xva.fillna(med).astype(float)
        mn = float(xtr.min()); mx = float(xtr.max())
        if abs(mx - mn) < 1e-12:
            vals.append(np.zeros(len(xva), dtype=np.float32))
        else:
            vals.append(((xva - mn) / (mx - mn)).clip(0, 1).astype(np.float32).values)
    return np.stack(vals, axis=1)


def pair_geometry_for_run(
    run: str,
    data: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    pred: pd.DataFrame,
    *,
    class_names: List[str],
    out_dir: Path,
    raw_scaled_val: Optional[np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    features = feature_names_from_meta(meta, data)
    strategies = feature_strategies_from_meta(meta, features)
    Xb = data["X_val_bin"].astype(np.float64)
    Xo = data["X_val_offset"].astype(np.float64)
    n = Xb.shape[0]
    if len(pred) != n:
        warnings.warn(f"{run}: predictions length {len(pred)} != X_val length {n}; truncating to min")
        m = min(len(pred), n)
        pred = pred.iloc[:m].reset_index(drop=True)
        Xb = Xb[:m]; Xo = Xo[:m]
        raw = raw_scaled_val[:m] if raw_scaled_val is not None else None
    else:
        raw = raw_scaled_val

    rows = []
    pair_counts = []
    for true_c, pred_c in MAJOR_PAIRS:
        pair_mask = (pred["true_label"] == true_c) & (pred["pred_label"] == pred_c)
        true_correct = (pred["true_label"] == true_c) & (pred["pred_label"] == true_c)
        pred_correct = (pred["true_label"] == pred_c) & (pred["pred_label"] == pred_c)
        pair_n = int(pair_mask.sum())
        pair_counts.append({"run": run, "true_class": true_c, "pred_class": pred_c, "pair_n": pair_n})
        if pair_n == 0 or true_correct.sum() == 0 or pred_correct.sum() == 0:
            continue
        for j, feat in enumerate(features):
            def mean_arr(arr, mask):
                return float(np.nanmean(arr[mask.values, j])) if mask.any() else np.nan
            pair_bin = mean_arr(Xb, pair_mask)
            tc_bin = mean_arr(Xb, true_correct)
            pc_bin = mean_arr(Xb, pred_correct)
            pair_off = mean_arr(Xo, pair_mask)
            tc_off = mean_arr(Xo, true_correct)
            pc_off = mean_arr(Xo, pred_correct)
            row = {
                "run": run,
                "true_class": true_c,
                "pred_class": pred_c,
                "pair_n": pair_n,
                "feature_idx": j,
                "feature": feat,
                "strategy": strategies.get(feat, "unknown"),
                "pair_bin_mean": pair_bin,
                "true_correct_bin_mean": tc_bin,
                "pred_correct_bin_mean": pc_bin,
                "dist_bin_to_true_correct": abs(pair_bin - tc_bin),
                "dist_bin_to_pred_correct": abs(pair_bin - pc_bin),
                "bin_closer_to_pred_correct": abs(pair_bin - pc_bin) < abs(pair_bin - tc_bin),
                "pair_offset_mean": pair_off,
                "true_correct_offset_mean": tc_off,
                "pred_correct_offset_mean": pc_off,
                "dist_offset_to_true_correct": abs(pair_off - tc_off),
                "dist_offset_to_pred_correct": abs(pair_off - pc_off),
                "offset_closer_to_pred_correct": abs(pair_off - pc_off) < abs(pair_off - tc_off),
            }
            if raw is not None:
                pair_raw = mean_arr(raw, pair_mask)
                tc_raw = mean_arr(raw, true_correct)
                pc_raw = mean_arr(raw, pred_correct)
                row.update({
                    "pair_raw_mean": pair_raw,
                    "true_correct_raw_mean": tc_raw,
                    "pred_correct_raw_mean": pc_raw,
                    "dist_raw_to_true_correct": abs(pair_raw - tc_raw),
                    "dist_raw_to_pred_correct": abs(pair_raw - pc_raw),
                    "raw_closer_to_pred_correct": abs(pair_raw - pc_raw) < abs(pair_raw - tc_raw),
                })
            rows.append(row)
    all_df = pd.DataFrame(rows)
    counts_df = pd.DataFrame(pair_counts)
    if not all_df.empty:
        out_dir.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(out_dir / f"pair_geometry_{run}.csv", index=False)
        for t, p in MAJOR_PAIRS:
            sub = all_df[(all_df["true_class"] == t) & (all_df["pred_class"] == p)]
            if len(sub):
                sub.sort_values("dist_bin_to_true_correct", ascending=False).to_csv(
                    out_dir / f"pair_geometry_{run}_{t}_to_{p}.csv", index=False
                )
    return all_df, counts_df


def summarize_pair_geometry(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for (run, true_c, pred_c, strategy), g in df.groupby(["run", "true_class", "pred_class", "strategy"], dropna=False):
        row = {
            "run": run,
            "true_class": true_c,
            "pred_class": pred_c,
            "strategy": strategy,
            "n_features": len(g),
            "pair_n": int(g["pair_n"].iloc[0]),
            "bin_closer_to_pred_count": int(g["bin_closer_to_pred_correct"].sum()),
            "bin_closer_to_pred_ratio": float(g["bin_closer_to_pred_correct"].mean()),
            "dist_bin_to_true_correct_mean": float(g["dist_bin_to_true_correct"].mean()),
            "dist_bin_to_pred_correct_mean": float(g["dist_bin_to_pred_correct"].mean()),
            "dist_offset_to_true_correct_mean": float(g["dist_offset_to_true_correct"].mean()),
            "dist_offset_to_pred_correct_mean": float(g["dist_offset_to_pred_correct"].mean()),
        }
        if "raw_closer_to_pred_correct" in g.columns:
            row["raw_closer_to_pred_count"] = int(g["raw_closer_to_pred_correct"].sum())
            row["raw_closer_to_pred_ratio"] = float(g["raw_closer_to_pred_correct"].mean())
            row["dist_raw_to_true_correct_mean"] = float(g["dist_raw_to_true_correct"].mean())
            row["dist_raw_to_pred_correct_mean"] = float(g["dist_raw_to_pred_correct"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2-dataset-npz", default="03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_dataset.npz")
    ap.add_argument("--c2-metadata-json", default="03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_metadata.json")
    ap.add_argument("--c2-run-dir", default="03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact")
    ap.add_argument("--c2-audit-dir", default="03_outputs/audit_c2_best")
    ap.add_argument("--k1024-rank-build-dir", default="03_outputs/build_mixed_quantile_offset/K1024_B1024_C2policy_rank_safe_native")
    ap.add_argument("--k1024-rank-run-dir", default="03_outputs/train_runs_k1024_fixed_c2policy/Keff1024/T1_K1024_C2POLICY_RANK_SAFE_NATIVE_D3")
    ap.add_argument("--k1024-abs-build-dir", default="03_outputs/build_mixed_quantile_offset/K1024_B1024_C2policy_abs_for_rank_control_native")
    ap.add_argument("--k1024-abs-run-dir", default="03_outputs/train_runs_k1024_fixed_c2policy/Keff1024/T2_K1024_C2POLICY_ABS_FOR_RANK_CONTROL_NATIVE_D3")
    ap.add_argument("--out-dir", default="03_outputs/audit_group_pair_geometry")
    ap.add_argument("--rare-threshold", type=int, default=5)
    ap.add_argument("--skip-k1024", action="store_true")
    args = ap.parse_args()

    root = repo_root()
    out = root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[group-audit] root:", root)
    print("[group-audit] out:", out)

    c2_data, c2_meta = load_dataset(root / args.c2_dataset_npz, root / args.c2_metadata_json)
    class_names = labels_from_meta(c2_meta)
    raw_scaled_val = load_raw_scaled(root, feature_names_from_meta(c2_meta, c2_data))

    runs: Dict[str, Dict[str, Any]] = {
        "C2_K512": {
            "data": c2_data,
            "meta": c2_meta,
            "run_dir": root / args.c2_run_dir,
            "audit_dir": root / args.c2_audit_dir,
            "num_bins": int(c2_meta.get("num_bins", c2_meta.get("K", 512))),
        }
    }

    if not args.skip_k1024:
        k_configs = [
            ("K1024_RANK_SAFE", root / args.k1024_rank_build_dir, root / args.k1024_rank_run_dir),
            ("K1024_ABS_CONTROL", root / args.k1024_abs_build_dir, root / args.k1024_abs_run_dir),
        ]
        for name, build_dir, run_dir in k_configs:
            npz = build_dir / "mixed_quantile_offset_dataset.npz"
            meta = build_dir / "mixed_quantile_offset_metadata.json"
            if npz.exists() and meta.exists() and run_dir.exists():
                data, m = load_dataset(npz, meta)
                runs[name] = {"data": data, "meta": m, "run_dir": run_dir, "audit_dir": root / args.c2_audit_dir, "num_bins": int(m.get("num_bins", m.get("K", 1024)))}
                print(f"[group-audit] found {name}")
            else:
                print(f"[group-audit] skip {name}; missing build/run artifacts")

    # 1 result summaries
    result_dir = out / "01_result_tradeoff"
    result_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = {name: info["run_dir"] for name, info in runs.items()}
    res = result_summary(run_dirs, class_names)
    res.to_csv(result_dir / "result_summary_all_runs.csv", index=False)
    if "C2_K512" in res["run"].values:
        base = res.set_index("run").loc["C2_K512"]
        delta_rows = []
        for _, row in res.iterrows():
            if row["run"] == "C2_K512":
                continue
            d = {"run": row["run"]}
            for col in res.columns:
                if col.startswith("val_") or col.startswith("train_") or col == "gap_macro_f1":
                    if pd.api.types.is_number(row[col]) and col in base.index:
                        d[col + "_delta_vs_C2"] = row[col] - base[col]
            delta_rows.append(d)
        pd.DataFrame(delta_rows).to_csv(result_dir / "result_delta_vs_C2.csv", index=False)
    conf = confusion_long(run_dirs, class_names)
    conf.to_csv(result_dir / "confusion_all_runs_long.csv", index=False)
    if not conf.empty:
        base_conf = conf[conf["run"] == "C2_K512"].set_index(["true_class", "pred_class"])["count"]
        rows = []
        for _, r in conf.iterrows():
            base_v = int(base_conf.get((r["true_class"], r["pred_class"]), 0))
            rows.append({**r.to_dict(), "delta_vs_C2": int(r["count"] - base_v)})
        pd.DataFrame(rows).to_csv(result_dir / "confusion_pair_delta_vs_C2.csv", index=False)

    # 2 token group summaries
    token_dir = out / "02_group_tokenization"
    token_dir.mkdir(parents=True, exist_ok=True)
    all_feature_dfs = []
    all_group_dfs = []
    all_group_class = []
    for run, info in runs.items():
        data = info["data"]; meta = info["meta"]
        features = feature_names_from_meta(meta, data)
        strategies = feature_strategies_from_meta(meta, features)
        nb = int(info["num_bins"])
        train_df, train_ref = compute_token_feature_audit(
            data["X_train_bin"].astype(np.int64), data["X_train_offset"].astype(np.float32), data["y_train"].astype(np.int64),
            num_bins=nb, features=features, strategies=strategies, class_names=class_names,
            split="train", rare_threshold=args.rare_threshold, train_ref_counts=None,
        )
        val_df, _ = compute_token_feature_audit(
            data["X_val_bin"].astype(np.int64), data["X_val_offset"].astype(np.float32), data["y_val"].astype(np.int64),
            num_bins=nb, features=features, strategies=strategies, class_names=class_names,
            split="val", rare_threshold=args.rare_threshold, train_ref_counts=train_ref,
        )
        fdf = pd.concat([train_df, val_df], ignore_index=True)
        fdf.insert(0, "run", run)
        fdf.to_csv(token_dir / f"token_feature_audit_{run}.csv", index=False)
        all_feature_dfs.append(fdf)
        all_group_dfs.append(group_summary(fdf, run))
        all_group_class.append(group_class_summary(fdf, run, class_names))
    all_feat = pd.concat(all_feature_dfs, ignore_index=True)
    all_feat.to_csv(token_dir / "token_feature_audit_all_runs.csv", index=False)
    group_all = pd.concat(all_group_dfs, ignore_index=True)
    group_all.to_csv(token_dir / "group_token_summary_all_runs.csv", index=False)
    group_class_all = pd.concat(all_group_class, ignore_index=True)
    group_class_all.to_csv(token_dir / "group_token_by_class_all_runs.csv", index=False)

    # feature and group deltas vs C2
    base_feat = all_feat[(all_feat["run"] == "C2_K512")].copy()
    deltas = []
    delta_cols = ["bins_used", "used_bin_ratio", "dominant_bin_ratio", "rare_used_bin_ratio_le5", "rare_cell_ratio_trainref_le5", "unseen_cell_ratio_trainref", "entropy_norm", "compression_factor", "offset_nonzero_ratio", "offset_std"]
    for run in sorted(all_feat["run"].unique()):
        if run == "C2_K512":
            continue
        cur = all_feat[all_feat["run"] == run]
        merged = cur.merge(base_feat, on=["split", "feature"], suffixes=("", "_C2"), how="inner")
        for _, row in merged.iterrows():
            outrow = {"run": run, "split": row["split"], "feature": row["feature"], "strategy_C2": row.get("strategy_C2"), "strategy": row.get("strategy")}
            for c in delta_cols:
                if c in row.index and c + "_C2" in row.index:
                    outrow[c] = row[c]
                    outrow[c + "_C2"] = row[c + "_C2"]
                    outrow[c + "_delta_vs_C2"] = row[c] - row[c + "_C2"]
            deltas.append(outrow)
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(token_dir / "token_feature_delta_vs_C2.csv", index=False)
    if not delta_df.empty:
        summary_rows = []
        for (run, split, strategy), g in delta_df.groupby(["run", "split", "strategy_C2"], dropna=False):
            row = {"run": run, "split": split, "strategy_C2": strategy, "n_features": len(g)}
            for c in delta_cols:
                dc = c + "_delta_vs_C2"
                if dc in g.columns:
                    row[dc + "_mean"] = float(g[dc].mean())
                    row[dc + "_median"] = float(g[dc].median())
                    row[dc + "_min"] = float(g[dc].min())
                    row[dc + "_max"] = float(g[dc].max())
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(token_dir / "group_token_delta_vs_C2_summary.csv", index=False)
        # top suspicious changes
        for run in sorted(delta_df["run"].unique()):
            sub = delta_df[(delta_df["run"] == run) & (delta_df["split"] == "train")].copy()
            if len(sub):
                sub.sort_values(["rare_cell_ratio_trainref_le5_delta_vs_C2", "entropy_norm_delta_vs_C2"], ascending=[False, True]).head(30).to_csv(token_dir / f"top30_token_worse_sparse_{run}_train.csv", index=False)
                sub.sort_values("compression_factor_delta_vs_C2", ascending=True).head(30).to_csv(token_dir / f"top30_compression_improved_{run}_train.csv", index=False)

    # 3 pair geometry
    pair_dir = out / "03_pair_geometry"
    pair_dir.mkdir(parents=True, exist_ok=True)
    all_pair_geom = []
    all_pair_counts = []
    for run, info in runs.items():
        pred = load_predictions(run, info["run_dir"], root / args.c2_audit_dir, class_names)
        if pred is None:
            print(f"[group-audit] skip pair geometry {run}; no val predictions")
            continue
        raw_for_run = raw_scaled_val
        geom, counts = pair_geometry_for_run(
            run, info["data"], info["meta"], pred, class_names=class_names,
            out_dir=pair_dir, raw_scaled_val=raw_for_run,
        )
        if not geom.empty:
            all_pair_geom.append(geom)
        all_pair_counts.append(counts)
    if all_pair_geom:
        pair_all = pd.concat(all_pair_geom, ignore_index=True)
        pair_all.to_csv(pair_dir / "pair_geometry_all_runs.csv", index=False)
        pair_sum = summarize_pair_geometry(pair_all)
        pair_sum.to_csv(pair_dir / "pair_geometry_summary_by_strategy.csv", index=False)
        # top features for Trojan->Ransomware by bin/raw distances in C2
        c2_tr = pair_all[(pair_all["run"] == "C2_K512") & (pair_all["true_class"] == "Trojan") & (pair_all["pred_class"] == "Ransomware")].copy()
        if len(c2_tr):
            c2_tr["bin_pred_minus_true_distance_advantage"] = c2_tr["dist_bin_to_true_correct"] - c2_tr["dist_bin_to_pred_correct"]
            c2_tr.sort_values("bin_pred_minus_true_distance_advantage", ascending=False).head(30).to_csv(pair_dir / "C2_Trojan_to_Ransomware_top30_closer_to_Ransomware_by_bin.csv", index=False)
            if "dist_raw_to_true_correct" in c2_tr.columns:
                c2_tr["raw_pred_minus_true_distance_advantage"] = c2_tr["dist_raw_to_true_correct"] - c2_tr["dist_raw_to_pred_correct"]
                c2_tr.sort_values("raw_pred_minus_true_distance_advantage", ascending=False).head(30).to_csv(pair_dir / "C2_Trojan_to_Ransomware_top30_closer_to_Ransomware_by_raw.csv", index=False)
    if all_pair_counts:
        pc = pd.concat(all_pair_counts, ignore_index=True)
        pc.to_csv(pair_dir / "major_pair_counts_all_runs.csv", index=False)
        if "C2_K512" in pc["run"].values:
            base = pc[pc["run"] == "C2_K512"].set_index(["true_class", "pred_class"])["pair_n"]
            rows = []
            for _, r in pc.iterrows():
                b = int(base.get((r["true_class"], r["pred_class"]), 0))
                rows.append({**r.to_dict(), "delta_vs_C2": int(r["pair_n"] - b)})
            pd.DataFrame(rows).to_csv(pair_dir / "major_pair_count_delta_vs_C2.csv", index=False)

    # 4 concise markdown summary stub
    md = []
    md.append("# Group + Pair Geometry Audit\n")
    md.append("This audit adds group-wise tokenization comparisons and pair geometry analysis.\n")
    md.append("## Key files\n")
    md.append("- `01_result_tradeoff/result_summary_all_runs.csv`\n")
    md.append("- `01_result_tradeoff/confusion_pair_delta_vs_C2.csv`\n")
    md.append("- `02_group_tokenization/group_token_summary_all_runs.csv`\n")
    md.append("- `02_group_tokenization/group_token_delta_vs_C2_summary.csv`\n")
    md.append("- `02_group_tokenization/group_token_by_class_all_runs.csv`\n")
    md.append("- `03_pair_geometry/pair_geometry_all_runs.csv`\n")
    md.append("- `03_pair_geometry/pair_geometry_summary_by_strategy.csv`\n")
    md.append("- `03_pair_geometry/C2_Trojan_to_Ransomware_top30_closer_to_Ransomware_by_bin.csv`\n")
    if not res.empty:
        md.append("\n## Result summary preview\n\n")
        md.append(res[[c for c in ["run", "train_macro_f1", "val_macro_f1", "gap_macro_f1"] if c in res.columns]].to_markdown(index=False))
        md.append("\n")
    (out / "audit_group_pair_summary.md").write_text("\n".join(md), encoding="utf-8")

    print("[group-audit] DONE")
    print("[group-audit] out_dir:", out)


if __name__ == "__main__":
    main()

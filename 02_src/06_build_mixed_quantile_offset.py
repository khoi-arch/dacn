#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build mixed quantile/uniform bin + offset dataset.

No training.
No model change.
No overwrite baseline token artifacts.

Input:
  - train_preprocessed_K{K}.csv
  - val_preprocessed_K{K}.csv
  - preprocess_policy_K{K}.json
  - quantile_vs_uniform_bin_diag_K{K}_B{B}.json

Output:
  03_outputs/build_mixed_quantile_offset/K{K}_B{B}/
    mixed_quantile_offset_dataset.npz
    mixed_quantile_offset_metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import config as CFG


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build mixed quantile/uniform bin + offset dataset.")
    p.add_argument("--K", type=int, default=int(CFG.TOKEN_K))
    p.add_argument("--num-bins", type=int, default=int(getattr(CFG, "VALUE_NUM_BINS", 128)))
    p.add_argument("--train-preprocessed", default="")
    p.add_argument("--val-preprocessed", default="")
    p.add_argument("--policy-json", default="")
    p.add_argument("--diag-json", default="")
    p.add_argument("--out-root", default=str(CFG.OUTPUT_ROOT / "build_mixed_quantile_offset"))
    p.add_argument("--label-col", default=str(CFG.DEFAULT_LABEL_COL))

    # Mixed-selection rule.
    p.add_argument("--min-unique-for-quantile", type=int, default=128)
    p.add_argument("--min-dominant-reduction", type=float, default=0.02)
    p.add_argument("--min-entropy-delta", type=float, default=0.05)
    return p.parse_args()


def load_feature_order(policy_path: Path) -> List[str]:
    obj = json.loads(policy_path.read_text(encoding="utf-8"))

    meta_features = obj.get("metadata", {}).get("feature_order")
    if meta_features:
        return [str(x) for x in meta_features]

    policies = obj.get("policies", [])
    if not policies:
        raise ValueError(f"No policies found in: {policy_path}")

    return [str(p["feature"]) for p in policies]


def validate_split(name: str, df: pd.DataFrame, features: Sequence[str], label_col: str) -> None:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"{name} missing features: {missing[:10]}")
    if label_col not in df.columns:
        raise ValueError(f"{name} missing label column: {label_col}")

    arr = df.loc[:, list(features)].to_numpy(dtype=np.float64)
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError(f"{name} contains NaN/Inf in features.")

    mn = float(np.min(arr)) if arr.size else 0.0
    mx = float(np.max(arr)) if arr.size else 0.0
    if mn < -1e-6 or mx > 1.0 + 1e-6:
        raise ValueError(f"{name} expected preprocessed features in [0,1], got min={mn}, max={mx}")


def build_label_mapping(train_labels: pd.Series) -> Dict[str, int]:
    labels = sorted([str(x) for x in train_labels.dropna().unique().tolist()])
    return {label: idx for idx, label in enumerate(labels)}


def encode_labels(labels: pd.Series, mapping: Dict[str, int], split_name: str) -> np.ndarray:
    out = []
    unknown = []
    for x in labels.tolist():
        key = str(x)
        if key not in mapping:
            unknown.append(key)
            out.append(-1)
        else:
            out.append(mapping[key])
    if unknown:
        raise ValueError(f"{split_name} contains unseen labels: {sorted(set(unknown))[:10]}")
    return np.asarray(out, dtype=np.int64)


def load_diag(diag_path: Path) -> Dict[str, Dict[str, object]]:
    obj = json.loads(diag_path.read_text(encoding="utf-8"))
    rows = obj.get("features", [])
    return {str(r["feature"]): r for r in rows}


def choose_strategy(
    *,
    feature: str,
    z_train: np.ndarray,
    diag_row: Dict[str, object] | None,
    min_unique_for_quantile: int,
    min_dominant_reduction: float,
    min_entropy_delta: float,
) -> Tuple[str, str]:
    unique_n = int(np.unique(z_train).size)

    if unique_n <= 1:
        return "constant", "constant_or_all_same"

    if diag_row is None:
        return "uniform_offset", "no_diag_row"

    dom_red = float(diag_row.get("dominant_ratio_reduction", 0.0))
    ent_delta = float(diag_row.get("delta_entropy_norm", 0.0))
    u_dom = float(diag_row.get("uniform_dominant_bin_ratio", 1.0))
    q_dom = float(diag_row.get("quantile_dominant_bin_ratio", 1.0))

    if unique_n < int(min_unique_for_quantile):
        return "uniform_offset", f"low_unique_{unique_n}"

    if dom_red >= float(min_dominant_reduction) and ent_delta >= float(min_entropy_delta) and q_dom < u_dom:
        return "quantile_offset", (
            f"quantile_improves_dom_{dom_red:.4f}_entropy_{ent_delta:.4f}_unique_{unique_n}"
        )

    return "uniform_offset", (
        f"quantile_not_enough_dom_{dom_red:.4f}_entropy_{ent_delta:.4f}_unique_{unique_n}"
    )


def make_uniform_edges(num_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, int(num_bins) + 1, dtype=np.float64)


def make_quantile_edges(values: np.ndarray, num_bins: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, int(num_bins) + 1)
    try:
        edges = np.quantile(v, qs, method="linear")
    except TypeError:
        edges = np.quantile(v, qs, interpolation="linear")
    edges[0] = float(np.min(v))
    edges[-1] = float(np.max(v))
    return edges.astype(np.float64)


def assign_bin_offset(values: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    num_bins = int(len(edges) - 1)

    if num_bins <= 0:
        raise ValueError("edges must have at least 2 values.")

    if np.isclose(edges[0], edges[-1]):
        return (
            np.zeros(v.shape, dtype=np.int64),
            np.full(v.shape, 0.5, dtype=np.float32),
        )

    # Clip only to learned train range for this representation.
    # For preprocessed z, normal range should be [0,1].
    v_clip = np.clip(v, edges[0], edges[-1])

    internal = edges[1:-1]
    bin_id = np.searchsorted(internal, v_clip, side="right")
    bin_id = np.clip(bin_id, 0, num_bins - 1).astype(np.int64)

    left = edges[bin_id]
    right = edges[bin_id + 1]
    width = right - left

    offset = np.zeros(v.shape, dtype=np.float32)
    good = ~np.isclose(width, 0.0)
    offset[good] = ((v_clip[good] - left[good]) / width[good]).astype(np.float32)
    offset[~good] = 0.5
    offset = np.clip(offset, 0.0, 1.0).astype(np.float32)
    return bin_id, offset


def main() -> None:
    args = parse_args()
    K = int(args.K)
    B = int(args.num_bins)

    train_path = Path(args.train_preprocessed) if args.train_preprocessed else CFG.preprocess_train_csv_path(K)
    val_path = Path(args.val_preprocessed) if args.val_preprocessed else CFG.preprocess_val_csv_path(K)
    policy_path = Path(args.policy_json) if args.policy_json else CFG.preprocess_policy_json_path(K)
    diag_path = (
        Path(args.diag_json)
        if args.diag_json
        else CFG.OUTPUT_ROOT / "bin_diag" / f"quantile_vs_uniform_bin_diag_K{K}_B{B}.json"
    )

    for p in [train_path, val_path, policy_path, diag_path]:
        if not p.exists():
            raise FileNotFoundError(str(p))

    features = load_feature_order(policy_path)
    diag = load_diag(diag_path)

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    label_col = str(args.label_col)

    validate_split("train", train, features, label_col)
    validate_split("val", val, features, label_col)

    n_train = len(train)
    n_val = len(val)
    n_feat = len(features)

    X_train_bin = np.zeros((n_train, n_feat), dtype=np.int64)
    X_train_offset = np.zeros((n_train, n_feat), dtype=np.float32)
    X_val_bin = np.zeros((n_val, n_feat), dtype=np.int64)
    X_val_offset = np.zeros((n_val, n_feat), dtype=np.float32)

    feature_meta = []
    strategy_counts: Dict[str, int] = {}

    for j, f in enumerate(features):
        z_train = np.clip(train[f].to_numpy(dtype=np.float64), 0.0, 1.0)
        z_val = np.clip(val[f].to_numpy(dtype=np.float64), 0.0, 1.0)

        strategy, reason = choose_strategy(
            feature=f,
            z_train=z_train,
            diag_row=diag.get(f),
            min_unique_for_quantile=int(args.min_unique_for_quantile),
            min_dominant_reduction=float(args.min_dominant_reduction),
            min_entropy_delta=float(args.min_entropy_delta),
        )

        if strategy == "constant":
            edges = np.asarray([0.0, 0.0], dtype=np.float64)
            tr_bin = np.zeros(n_train, dtype=np.int64)
            tr_off = np.full(n_train, 0.5, dtype=np.float32)
            va_bin = np.zeros(n_val, dtype=np.int64)
            va_off = np.full(n_val, 0.5, dtype=np.float32)
        elif strategy == "quantile_offset":
            edges = make_quantile_edges(z_train, B)
            tr_bin, tr_off = assign_bin_offset(z_train, edges)
            va_bin, va_off = assign_bin_offset(z_val, edges)
        else:
            edges = make_uniform_edges(B)
            tr_bin, tr_off = assign_bin_offset(z_train, edges)
            va_bin, va_off = assign_bin_offset(z_val, edges)

        X_train_bin[:, j] = tr_bin
        X_train_offset[:, j] = tr_off
        X_val_bin[:, j] = va_bin
        X_val_offset[:, j] = va_off

        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        drow = diag.get(f, {})
        feature_meta.append({
            "feature": f,
            "strategy": strategy,
            "reason": reason,
            "z_train_unique": int(np.unique(z_train).size),
            "train_bin_used": int(np.unique(tr_bin).size),
            "val_bin_used": int(np.unique(va_bin).size),
            "train_offset_min": float(np.min(tr_off)),
            "train_offset_max": float(np.max(tr_off)),
            "train_offset_mean": float(np.mean(tr_off)),
            "val_offset_min": float(np.min(va_off)),
            "val_offset_max": float(np.max(va_off)),
            "val_offset_mean": float(np.mean(va_off)),
            "diag_uniform_dominant_bin_ratio": drow.get("uniform_dominant_bin_ratio"),
            "diag_quantile_dominant_bin_ratio": drow.get("quantile_dominant_bin_ratio"),
            "diag_dominant_ratio_reduction": drow.get("dominant_ratio_reduction"),
            "diag_uniform_entropy_norm": drow.get("uniform_entropy_norm"),
            "diag_quantile_entropy_norm": drow.get("quantile_entropy_norm"),
            "diag_delta_entropy_norm": drow.get("delta_entropy_norm"),
            "edge_min": float(edges[0]),
            "edge_max": float(edges[-1]),
            "edge_unique_count": int(np.unique(edges).size),
            "zero_width_edge_count": int(np.sum(np.isclose(np.diff(edges), 0.0))),
            "edges": [float(x) for x in edges.tolist()],
        })

    label_mapping = build_label_mapping(train[label_col])
    y_train = encode_labels(train[label_col], label_mapping, "train")
    y_val = encode_labels(val[label_col], label_mapping, "val")

    out_dir = Path(args.out_root) / f"K{K}_B{B}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / "mixed_quantile_offset_dataset.npz"
    out_meta = out_dir / "mixed_quantile_offset_metadata.json"

    np.savez_compressed(
        out_npz,
        X_train_bin=X_train_bin,
        X_train_offset=X_train_offset,
        y_train=y_train,
        X_val_bin=X_val_bin,
        X_val_offset=X_val_offset,
        y_val=y_val,
        feature_names=np.asarray(features, dtype=object),
        label_names=np.asarray([k for k, _ in sorted(label_mapping.items(), key=lambda kv: kv[1])], dtype=object),
        K=np.asarray([K], dtype=np.int64),
        num_bins=np.asarray([B], dtype=np.int64),
    )

    metadata = {
        "stage": "build_mixed_quantile_offset",
        "K": K,
        "num_bins": B,
        "input": {
            "train_preprocessed": str(train_path),
            "val_preprocessed": str(val_path),
            "policy_json": str(policy_path),
            "diag_json": str(diag_path),
        },
        "selection_rule": {
            "min_unique_for_quantile": int(args.min_unique_for_quantile),
            "min_dominant_reduction": float(args.min_dominant_reduction),
            "min_entropy_delta": float(args.min_entropy_delta),
        },
        "label_col": label_col,
        "label_mapping": label_mapping,
        "n_features": n_feat,
        "feature_names": features,
        "strategy_counts": strategy_counts,
        "splits": {
            "train": {
                "n_rows": n_train,
                "X_bin_shape": list(X_train_bin.shape),
                "X_offset_shape": list(X_train_offset.shape),
                "bin_min": int(X_train_bin.min()),
                "bin_max": int(X_train_bin.max()),
                "offset_min": float(X_train_offset.min()),
                "offset_max": float(X_train_offset.max()),
            },
            "val": {
                "n_rows": n_val,
                "X_bin_shape": list(X_val_bin.shape),
                "X_offset_shape": list(X_val_offset.shape),
                "bin_min": int(X_val_bin.min()),
                "bin_max": int(X_val_bin.max()),
                "offset_min": float(X_val_offset.min()),
                "offset_max": float(X_val_offset.max()),
            },
        },
        "feature_meta": feature_meta,
        "outputs": {
            "dataset_npz": str(out_npz),
            "metadata_json": str(out_meta),
        },
        "note": "Mixed representation. Quantile+offset only for features where diagnostic improves; others use uniform+offset or constant.",
    }

    out_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== build mixed quantile offset done =====")
    print(f"K={K}, B={B}")
    print(f"features={n_feat}")
    print(f"strategy_counts={strategy_counts}")
    print(f"dataset={out_npz}")
    print(f"metadata={out_meta}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_quantile_bin_diag.py

Diagnostic only. No training. No model change.

Goal:
Compare current uniform bins against quantile bins on the same preprocessed z.

Current embedding uses:
    uniform_bin = round(z * (B - 1))

This script compares:
    uniform_128bin vs quantile_128bin

Metrics:
    - bins used
    - empty bins
    - dominant bin ratio
    - max bin count
    - normalized entropy
    - quantile degenerate edges / zero-width bins
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

import config as CFG


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare uniform bins vs quantile bins.")
    p.add_argument("--K", type=int, default=int(CFG.TOKEN_K))
    p.add_argument("--num-bins", type=int, default=int(getattr(CFG, "VALUE_NUM_BINS", 128)))
    p.add_argument("--train-preprocessed", default="")
    p.add_argument("--policy-json", default="")
    p.add_argument("--out-dir", default=str(CFG.OUTPUT_ROOT / "bin_diag"))
    return p.parse_args()


def load_feature_order(policy_path: Path) -> List[str]:
    if not policy_path.exists():
        raise FileNotFoundError(f"policy json not found: {policy_path}")

    obj = json.loads(policy_path.read_text(encoding="utf-8"))
    meta_features = obj.get("metadata", {}).get("feature_order")
    if meta_features:
        return [str(x) for x in meta_features]

    policies = obj.get("policies", [])
    if not policies:
        raise ValueError(f"No policies found in {policy_path}")

    return [str(p["feature"]) for p in policies]


def entropy_norm_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0 or counts.size <= 1:
        return 0.0
    p = counts[counts > 0] / total
    ent = -float(np.sum(p * np.log(p)))
    return float(ent / np.log(counts.size))


def bin_stats(bin_ids: np.ndarray, num_bins: int) -> Dict[str, object]:
    bin_ids = np.asarray(bin_ids, dtype=np.int64)
    counts = np.bincount(bin_ids, minlength=num_bins).astype(np.int64)

    used = int(np.sum(counts > 0))
    empty = int(num_bins - used)
    max_count = int(counts.max()) if counts.size else 0
    n = int(counts.sum())
    dominant_ratio = float(max_count / max(n, 1))
    mean_nonempty = float(counts[counts > 0].mean()) if used else 0.0

    return {
        "bins_used": used,
        "empty_bins": empty,
        "max_bin_count": max_count,
        "dominant_bin_ratio": dominant_ratio,
        "entropy_norm": entropy_norm_from_counts(counts),
        "mean_nonempty_bin_count": mean_nonempty,
    }


def uniform_bins(z: np.ndarray, num_bins: int) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=np.float64), 0.0, 1.0)
    return np.clip(np.rint(z * float(num_bins - 1)), 0, num_bins - 1).astype(np.int64)


def quantile_edges(values: np.ndarray, num_bins: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, num_bins + 1)
    try:
        edges = np.quantile(v, qs, method="linear")
    except TypeError:
        edges = np.quantile(v, qs, interpolation="linear")

    # Force exact boundaries from train.
    edges[0] = float(np.min(v))
    edges[-1] = float(np.max(v))
    return edges.astype(np.float64)


def quantile_bin_and_offset(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    v = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    num_bins = int(len(edges) - 1)

    # Bin assignment: 0..num_bins-1.
    # side="right" puts values equal to an internal edge into the upper bin.
    internal = edges[1:-1]
    bin_id = np.searchsorted(internal, v, side="right")
    bin_id = np.clip(bin_id, 0, num_bins - 1).astype(np.int64)

    left = edges[bin_id]
    right = edges[bin_id + 1]
    width = right - left

    zero_width = np.isclose(width, 0.0)
    offset = np.zeros_like(v, dtype=np.float32)
    offset[~zero_width] = ((v[~zero_width] - left[~zero_width]) / width[~zero_width]).astype(np.float32)
    offset[zero_width] = 0.5
    offset = np.clip(offset, 0.0, 1.0)

    duplicate_edge_count = int((len(edges) - len(np.unique(edges))))
    zero_width_bin_count = int(np.sum(np.isclose(np.diff(edges), 0.0)))
    zero_width_sample_ratio = float(np.mean(zero_width)) if v.size else 0.0

    diag = {
        "edge_count": int(len(edges)),
        "unique_edge_count": int(len(np.unique(edges))),
        "duplicate_edge_count": duplicate_edge_count,
        "zero_width_bin_count": zero_width_bin_count,
        "zero_width_sample_ratio": zero_width_sample_ratio,
        "offset_min": float(np.min(offset)) if offset.size else None,
        "offset_max": float(np.max(offset)) if offset.size else None,
        "offset_mean": float(np.mean(offset)) if offset.size else None,
    }

    return bin_id, offset, diag


def main() -> None:
    args = parse_args()
    K = int(args.K)
    B = int(args.num_bins)

    if B <= 1:
        raise ValueError("num-bins must be > 1.")

    train_pre_path = Path(args.train_preprocessed) if args.train_preprocessed else CFG.preprocess_train_csv_path(K)
    policy_path = Path(args.policy_json) if args.policy_json else CFG.preprocess_policy_json_path(K)

    if not train_pre_path.exists():
        raise FileNotFoundError(f"train_preprocessed not found: {train_pre_path}")
    if not policy_path.exists():
        raise FileNotFoundError(f"policy json not found: {policy_path}")

    features = load_feature_order(policy_path)
    df = pd.read_csv(train_pre_path)

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"train_preprocessed missing features: {missing[:10]}")

    rows: List[Dict[str, object]] = []

    for f in features:
        z = df[f].to_numpy(dtype=np.float64)
        z = np.clip(z, 0.0, 1.0)

        u_bin = uniform_bins(z, B)
        u_stats = bin_stats(u_bin, B)

        q_edges = quantile_edges(z, B)
        q_bin, q_offset, q_diag = quantile_bin_and_offset(z, q_edges)
        q_stats = bin_stats(q_bin, B)

        row = {
            "feature": f,
            "n": int(len(z)),
            "num_bins": B,
            "z_min": float(np.min(z)) if z.size else None,
            "z_max": float(np.max(z)) if z.size else None,
            "z_num_unique": int(np.unique(z).size),
            "uniform_bins_used": u_stats["bins_used"],
            "quantile_bins_used": q_stats["bins_used"],
            "delta_bins_used": int(q_stats["bins_used"] - u_stats["bins_used"]),
            "uniform_empty_bins": u_stats["empty_bins"],
            "quantile_empty_bins": q_stats["empty_bins"],
            "uniform_dominant_bin_ratio": u_stats["dominant_bin_ratio"],
            "quantile_dominant_bin_ratio": q_stats["dominant_bin_ratio"],
            "delta_dominant_bin_ratio": float(q_stats["dominant_bin_ratio"] - u_stats["dominant_bin_ratio"]),
            "dominant_ratio_reduction": float(u_stats["dominant_bin_ratio"] - q_stats["dominant_bin_ratio"]),
            "uniform_max_bin_count": u_stats["max_bin_count"],
            "quantile_max_bin_count": q_stats["max_bin_count"],
            "uniform_entropy_norm": u_stats["entropy_norm"],
            "quantile_entropy_norm": q_stats["entropy_norm"],
            "delta_entropy_norm": float(q_stats["entropy_norm"] - u_stats["entropy_norm"]),
            "quantile_duplicate_edge_count": q_diag["duplicate_edge_count"],
            "quantile_zero_width_bin_count": q_diag["zero_width_bin_count"],
            "quantile_zero_width_sample_ratio": q_diag["zero_width_sample_ratio"],
            "offset_min": q_diag["offset_min"],
            "offset_max": q_diag["offset_max"],
            "offset_mean": q_diag["offset_mean"],
        }
        rows.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"quantile_vs_uniform_bin_diag_K{K}_B{B}.csv"
    out_json = out_dir / f"quantile_vs_uniform_bin_diag_K{K}_B{B}.json"

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_csv, index=False)

    improved_dom = result_df[result_df["dominant_ratio_reduction"] > 1e-12]
    worsened_dom = result_df[result_df["dominant_ratio_reduction"] < -1e-12]
    improved_entropy = result_df[result_df["delta_entropy_norm"] > 1e-12]

    summary = {
        "stage": "quantile_vs_uniform_bin_diag",
        "K": K,
        "num_bins": B,
        "input_train_preprocessed": str(train_pre_path),
        "policy_json": str(policy_path),
        "n_features": int(len(rows)),
        "mean_uniform_dominant_bin_ratio": float(result_df["uniform_dominant_bin_ratio"].mean()),
        "mean_quantile_dominant_bin_ratio": float(result_df["quantile_dominant_bin_ratio"].mean()),
        "mean_dominant_ratio_reduction": float(result_df["dominant_ratio_reduction"].mean()),
        "mean_uniform_entropy_norm": float(result_df["uniform_entropy_norm"].mean()),
        "mean_quantile_entropy_norm": float(result_df["quantile_entropy_norm"].mean()),
        "mean_delta_entropy_norm": float(result_df["delta_entropy_norm"].mean()),
        "features_dominant_ratio_improved": int(len(improved_dom)),
        "features_dominant_ratio_worsened": int(len(worsened_dom)),
        "features_entropy_improved": int(len(improved_entropy)),
        "features_with_quantile_zero_width_bins": int((result_df["quantile_zero_width_bin_count"] > 0).sum()),
        "top_dominant_ratio_reductions": result_df.sort_values(
            "dominant_ratio_reduction", ascending=False
        ).head(20).to_dict(orient="records"),
        "top_entropy_improvements": result_df.sort_values(
            "delta_entropy_norm", ascending=False
        ).head(20).to_dict(orient="records"),
        "top_quantile_degenerate_features": result_df.sort_values(
            ["quantile_zero_width_bin_count", "quantile_zero_width_sample_ratio"],
            ascending=False,
        ).head(20).to_dict(orient="records"),
        "outputs": {
            "csv": str(out_csv),
            "json": str(out_json),
        },
        "note": (
            "This diagnostic compares density compression at bin level. "
            "It does not use unique_preserve_ratio because quantile bins are not meant "
            "to preserve every raw unique value by bin_id; offset will carry within-bin detail."
        ),
    }

    out_json.write_text(
        json.dumps({"summary": summary, "features": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("===== quantile vs uniform bin diagnostic done =====")
    print(f"K: {K}")
    print(f"num_bins: {B}")
    print(f"features: {len(rows)}")
    print(f"mean dominant ratio: uniform={summary['mean_uniform_dominant_bin_ratio']:.6f} "
          f"quantile={summary['mean_quantile_dominant_bin_ratio']:.6f} "
          f"reduction={summary['mean_dominant_ratio_reduction']:.6f}")
    print(f"mean entropy norm:    uniform={summary['mean_uniform_entropy_norm']:.6f} "
          f"quantile={summary['mean_quantile_entropy_norm']:.6f} "
          f"delta={summary['mean_delta_entropy_norm']:.6f}")
    print(f"dominant improved/worsened: {summary['features_dominant_ratio_improved']}/"
          f"{summary['features_dominant_ratio_worsened']}")
    print(f"features with zero-width quantile bins: {summary['features_with_quantile_zero_width_bins']}")
    print(f"csv:  {out_csv}")
    print(f"json: {out_json}")


if __name__ == "__main__":
    main()

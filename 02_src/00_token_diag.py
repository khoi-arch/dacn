#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_token_diag.py

Train-only token diagnostic for Phase A.

Purpose:
  Diagnose whether each raw feature is compressed under the SAME effective
  bin budget used by the model.

Important:
  - K means effective number of bins.
  - If K=128, token ids are 0..127, exactly 128 bins.
  - token = floor(K * z), clipped to [0, K-1].

Core metrics:
  1. preserve_ratio
     = bins_used / min(raw_unique, K)

  2. compression_factor
     = raw_unique / max(bins_used, 1)

  3. entropy_norm
     = normalized entropy of the used token distribution
     = -sum(p_i log p_i) / log(bins_used)

Decision:
  - drop       : constant feature, raw_unique <= 1
  - keep       : not compressed according to the three metrics
  - transform  : should be processed by later preprocessing phase

Why entropy_norm:
  dominant_token_ratio only sees the single largest token.
  top-k ratio is better, but still depends on arbitrary k.
  normalized entropy summarizes the whole token distribution.

Entropy trigger rule:
  entropy_norm only triggers transform when raw_unique > K.
  If raw_unique <= K and preserve/compression are already good, low entropy is
  usually the original data imbalance, not token compression.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

import config as CFG


def cfg(name: str, default):
    return getattr(CFG, name, default)


def csv_list(value: str | None) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    default_bins = int(cfg("VALUE_NUM_BINS", cfg("TOKEN_K", 128)))

    p = argparse.ArgumentParser(description="Train-only token compression diagnostic at effective bin budget.")
    p.add_argument("--train-csv", default=str(CFG.TRAIN_CSV))
    p.add_argument("--out-dir", default=str(CFG.TOKEN_DIAG_DIR))

    # Keep argument name --K for compatibility, but semantically it is effective bin count.
    p.add_argument("--K", type=int, default=default_bins, help="Effective number of bins. If K=128, token ids are 0..127.")

    p.add_argument("--target-cols", default=",".join(CFG.TARGET_COLS))
    p.add_argument("--drop-cols", default=",".join(CFG.DROP_COLS))

    p.add_argument("--preserve-threshold", type=float, default=float(cfg("UNIQUE_PRESERVE_THRESHOLD", 0.95)))
    p.add_argument("--compression-threshold", type=float, default=float(cfg("COMPRESSION_FACTOR_THRESHOLD", 8.0)))
    p.add_argument("--entropy-threshold", type=float, default=float(cfg("TOKEN_ENTROPY_NORM_THRESHOLD", 0.75)))

    return p.parse_args()


def detect_numeric_features(
    df: pd.DataFrame,
    target_cols: Sequence[str],
    drop_cols: Sequence[str],
) -> List[str]:
    excluded = set(target_cols) | set(drop_cols)
    excluded |= {c for c in df.columns if str(c).startswith("Unnamed:")}
    return [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def assert_finite(df: pd.DataFrame, features: Sequence[str]) -> None:
    arr = df.loc[:, list(features)].to_numpy(dtype=float)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count or inf_count:
        raise ValueError(
            f"Train numeric matrix contains non-finite values: "
            f"nan={nan_count}, inf={inf_count}"
        )


def to_train_bins(values: np.ndarray, num_bins: int) -> tuple[np.ndarray, Dict[str, float]]:
    values = np.asarray(values, dtype=float)

    train_min = float(np.min(values))
    train_max = float(np.max(values))
    train_range = float(train_max - train_min)

    if abs(train_range) <= 1e-12:
        z = np.zeros_like(values, dtype=float)
    else:
        z = (values - train_min) / train_range

    z = np.clip(z, 0.0, 1.0)

    # Exactly num_bins bins: 0..num_bins-1.
    tokens = np.floor(float(num_bins) * z).astype(np.int64)
    tokens = np.clip(tokens, 0, int(num_bins) - 1)

    return tokens, {
        "train_min": train_min,
        "train_max": train_max,
        "train_range": train_range,
    }


def q_name(q: int) -> str:
    if q == 0:
        return "min"
    if q == 100:
        return "max"
    return f"q{q}"


def quantile_token(tokens: np.ndarray, q_percent: int) -> int:
    q = float(q_percent) / 100.0
    try:
        return int(np.quantile(tokens, q, method="nearest"))
    except TypeError:
        return int(np.quantile(tokens, q, interpolation="nearest"))


def normalized_entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(counts.sum())

    if total <= 0.0:
        return 0.0

    used = int(np.sum(counts > 0.0))
    if used <= 1:
        return 1.0

    p = counts[counts > 0.0] / total
    entropy = float(-np.sum(p * np.log(p)))
    denom = float(math.log(used))

    if denom <= 0.0:
        return 1.0

    return float(entropy / denom)


def decide_feature(
    *,
    raw_unique: int,
    K: int,
    preserve_ratio: float,
    compression_factor: float,
    entropy_norm: float,
    preserve_threshold: float,
    compression_threshold: float,
    entropy_threshold: float,
) -> tuple[str, List[str]]:
    reasons: List[str] = []

    if raw_unique <= 1:
        return "drop", ["constant_feature"]

    if preserve_ratio < preserve_threshold:
        reasons.append("low_preserve_ratio")

    if compression_factor > compression_threshold:
        reasons.append("high_compression_factor")

    # Entropy is a compression trigger only when unique values exceed available bins.
    # For low-unique features, low entropy usually reflects the original value
    # distribution, not a preprocessing/tokenization failure.
    if raw_unique > int(K) and entropy_norm < entropy_threshold:
        reasons.append("low_entropy_norm_high_unique")

    if reasons:
        return "transform", reasons

    return "keep", ["metrics_within_thresholds"]


def feature_diag(
    feature: str,
    values: np.ndarray,
    num_bins: int,
    *,
    preserve_threshold: float,
    compression_threshold: float,
    entropy_threshold: float,
) -> Dict[str, object]:
    values = np.asarray(values, dtype=float)
    tokens, fit = to_train_bins(values, num_bins)

    unique_tokens, counts = np.unique(tokens, return_counts=True)
    n = int(tokens.size)

    raw_unique = int(np.unique(values).size)
    bins_used = int(unique_tokens.size)
    possible_unique = int(min(raw_unique, int(num_bins)))

    preserve_ratio = float(bins_used / max(possible_unique, 1))
    compression_factor = float(raw_unique / max(bins_used, 1))
    entropy_norm = normalized_entropy_from_counts(counts)

    decision, reasons = decide_feature(
        raw_unique=raw_unique,
        K=int(num_bins),
        preserve_ratio=preserve_ratio,
        compression_factor=compression_factor,
        entropy_norm=entropy_norm,
        preserve_threshold=preserve_threshold,
        compression_threshold=compression_threshold,
        entropy_threshold=entropy_threshold,
    )

    zero_ratio = float(np.mean(np.isclose(values, 0.0)))

    # Kept for compatibility with current preprocessing/collision-analysis code.
    quantiles = {q_name(q): quantile_token(tokens, int(q)) for q in cfg("TOKEN_QUANTILES", [0, 10, 25, 75, 90, 100])}

    return {
        "feature": feature,
        "n": n,
        "K": int(num_bins),
        "effective_num_bins": int(num_bins),
        "raw": {
            **fit,
            "num_unique": raw_unique,
            "zero_ratio": zero_ratio,
            "is_constant": bool(raw_unique <= 1 or abs(fit["train_range"]) <= 1e-12),
        },
        "token": {
            "num_tokens_used": bins_used,
            "bins_used": bins_used,
            "possible_unique": possible_unique,
            "preserve_ratio": preserve_ratio,
            "unique_preserve_ratio": preserve_ratio,
            "compression_factor": compression_factor,
            "entropy_norm": entropy_norm,
            "quantiles": quantiles,
        },
        "decision": {
            "label": decision,
            "reasons": reasons,
            "thresholds": {
                "preserve_threshold": float(preserve_threshold),
                "compression_threshold": float(compression_threshold),
                "entropy_threshold": float(entropy_threshold),
                "entropy_trigger_requires_raw_unique_gt_K": True,
            },
        },
    }


def flatten_summary_row(row: Dict[str, object]) -> Dict[str, object]:
    raw = row["raw"]
    token = row["token"]
    decision = row["decision"]

    return {
        "feature": row["feature"],
        "decision": decision["label"],
        "reasons": "|".join(decision["reasons"]),
        "raw_unique": int(raw["num_unique"]),
        "bins_used": int(token["bins_used"]),
        "possible_unique": int(token["possible_unique"]),
        "preserve_ratio": float(token["preserve_ratio"]),
        "compression_factor": float(token["compression_factor"]),
        "entropy_norm": float(token["entropy_norm"]),
    }


def build_output(
    train_csv: Path,
    num_bins: int,
    target_cols: Sequence[str],
    drop_cols: Sequence[str],
    *,
    preserve_threshold: float,
    compression_threshold: float,
    entropy_threshold: float,
) -> Dict[str, object]:
    train = pd.read_csv(train_csv)

    features = detect_numeric_features(
        train,
        target_cols=target_cols,
        drop_cols=drop_cols,
    )
    if not features:
        raise ValueError("No numeric features detected.")

    assert_finite(train, features)

    feature_rows = [
        feature_diag(
            f,
            train[f].to_numpy(dtype=float),
            num_bins,
            preserve_threshold=preserve_threshold,
            compression_threshold=compression_threshold,
            entropy_threshold=entropy_threshold,
        )
        for f in features
    ]

    summary_rows = [flatten_summary_row(r) for r in feature_rows]
    decision_counts = pd.Series([r["decision"] for r in summary_rows]).value_counts().to_dict()

    return {
        "metadata": {
            "stage": "token_diag",
            "input_split": "train_only",
            "train_csv": str(train_csv),
            "K": int(num_bins),
            "effective_num_bins": int(num_bins),
            "n_rows": int(len(train)),
            "n_features": int(len(features)),
            "excluded_target_cols": list(target_cols),
            "drop_cols": list(drop_cols),
            "token_rule": "train-only MinMax; z=(x-min)/(max-min); clipped [0,1]; token=floor(K*z); clipped [0,K-1]; exactly K bins",
            "core_metrics": {
                "preserve_ratio": "bins_used / min(raw_unique, K)",
                "compression_factor": "raw_unique / max(bins_used, 1)",
                "entropy_norm": "-sum(p_i log p_i) / log(bins_used), computed over used token bins only",
            },
            "decision_rule": {
                "drop": "raw_unique <= 1",
                "transform": "preserve_ratio < preserve_threshold OR compression_factor > compression_threshold OR (raw_unique > K AND entropy_norm < entropy_threshold)",
                "keep": "otherwise",
            },
            "thresholds": {
                "preserve_threshold": float(preserve_threshold),
                "compression_threshold": float(compression_threshold),
                "entropy_threshold": float(entropy_threshold),
                "entropy_trigger_requires_raw_unique_gt_K": True,
            },
            "decision_counts": {str(k): int(v) for k, v in decision_counts.items()},
        },
        "features": feature_rows,
        "summary_rows": summary_rows,
    }


def main() -> None:
    args = parse_args()

    train_csv = Path(args.train_csv)
    out_dir = Path(args.out_dir)
    num_bins = int(args.K)

    if num_bins <= 1:
        raise ValueError("K/effective_num_bins must be > 1.")

    if not train_csv.exists():
        raise FileNotFoundError(f"train csv not found: {train_csv}")

    target_cols = csv_list(args.target_cols)
    drop_cols = csv_list(args.drop_cols)

    out_dir.mkdir(parents=True, exist_ok=True)

    result = build_output(
        train_csv=train_csv,
        num_bins=num_bins,
        target_cols=target_cols,
        drop_cols=drop_cols,
        preserve_threshold=float(args.preserve_threshold),
        compression_threshold=float(args.compression_threshold),
        entropy_threshold=float(args.entropy_threshold),
    )

    out_json_k = out_dir / f"token_diag_train_B{num_bins}.json"
    out_json_latest = out_dir / "token_diag_train.json"
    out_csv_k = out_dir / f"token_diag_summary_B{num_bins}.csv"
    out_csv_latest = out_dir / "token_diag_summary.csv"
    out_counts = out_dir / f"token_diag_decision_counts_B{num_bins}.json"

    json_obj = {
        "metadata": result["metadata"],
        "features": result["features"],
    }
    text = json.dumps(json_obj, ensure_ascii=False, indent=2)
    out_json_k.write_text(text, encoding="utf-8")
    out_json_latest.write_text(text, encoding="utf-8")

    summary_df = pd.DataFrame(result["summary_rows"])
    summary_df.to_csv(out_csv_k, index=False)
    summary_df.to_csv(out_csv_latest, index=False)

    out_counts.write_text(
        json.dumps(result["metadata"]["decision_counts"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("===== token_diag done =====")
    print(f"input: {train_csv}")
    print(f"effective_num_bins/K: {num_bins}")
    print(f"features: {result['metadata']['n_features']}")
    print(f"decision_counts: {result['metadata']['decision_counts']}")
    print(f"json: {out_json_k}")
    print(f"csv:  {out_csv_k}")
    print(f"latest_json: {out_json_latest}")
    print(f"latest_csv:  {out_csv_latest}")


if __name__ == "__main__":
    main()
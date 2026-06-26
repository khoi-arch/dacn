#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_preprocessing.py

Phase-B preprocessing with token-level selection.

Input:
  - train_raw.csv
  - val_raw.csv
  - token_diag_train_B{K}.json from 00_token_diag.py

Core idea:
  1. Use token_diag decision directly:
       drop / keep / transform

  2. For keep:
       train_minmax

  3. For drop:
       constant_zero
       The column is preserved for pipeline compatibility, but its value is 0.

  4. For transform:
       try candidate transforms on TRAIN only, evaluate each candidate by the
       same token-level metrics used in Phase A:
         preserve_ratio
         compression_factor
         entropy_norm

       choose the best candidate by token-level score.

No labels. No F1. No model training.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import config as CFG


def cfg(name: str, default):
    return getattr(CFG, name, default)


def csv_list(value: str | None) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    return vals


def default_token_diag_path(K: int) -> Path:
    token_dir = Path(cfg("TOKEN_DIAG_DIR", Path("03_outputs") / "token_diag"))
    candidates = [
        token_dir / f"token_diag_train_B{K}.json",
        token_dir / "token_diag_train.json",
        token_dir / f"token_diag_train_K{K}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def parse_args() -> argparse.Namespace:
    default_bins = int(cfg("VALUE_NUM_BINS", cfg("TOKEN_K", 128)))

    p = argparse.ArgumentParser(description="Train-fitted preprocessing selected by token-level metrics.")
    p.add_argument("--train-csv", default=str(CFG.TRAIN_CSV))
    p.add_argument("--val-csv", default=str(CFG.VAL_CSV))
    p.add_argument("--token-diag-json", default="")
    p.add_argument("--out-dir", default=str(cfg("PREPROCESS_DIR", Path("03_outputs") / "preprocessing")))

    # Kept as --K for compatibility, but semantically it is effective number of bins.
    p.add_argument("--K", type=int, default=default_bins)

    p.add_argument("--target-cols", default=",".join(CFG.TARGET_COLS))
    p.add_argument("--drop-cols", default=",".join(CFG.DROP_COLS))

    # Same thresholds as 00_token_diag.py.
    p.add_argument("--preserve-threshold", type=float, default=float(cfg("UNIQUE_PRESERVE_THRESHOLD", 0.95)))
    p.add_argument("--compression-threshold", type=float, default=float(cfg("COMPRESSION_FACTOR_THRESHOLD", 8.0)))
    p.add_argument("--entropy-threshold", type=float, default=float(cfg("TOKEN_ENTROPY_NORM_THRESHOLD", 0.75)))

    # Candidate list for transform features.
    p.add_argument(
        "--blend-alphas",
        default=str(cfg("PREPROCESS_BLEND_ALPHAS", "0.25,0.50,0.75")),
        help="Comma-separated alphas for blended_rank candidates.",
    )
    p.add_argument(
        "--include-minmax-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For transform features, include train_minmax as a control candidate.",
    )
    p.add_argument(
        "--include-rank-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For transform features, include pure piecewise_unique_rank candidate.",
    )

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


def assert_split_has_features(split_name: str, df: pd.DataFrame, features: Sequence[str]) -> None:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"{split_name} missing {len(missing)} feature columns; first: {missing[:10]}")

    arr = df.loc[:, list(features)].to_numpy(dtype=float)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count or inf_count:
        raise ValueError(
            f"{split_name} contains non-finite numeric values: "
            f"nan={nan_count}, inf={inf_count}"
        )


def load_token_diag(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(
            f"token_diag json not found: {path}\n"
            f"Run 00_token_diag.py first, e.g.:\n"
            f"  python -u 02_src/00_token_diag.py --K 128"
        )

    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("features", [])
    if not rows:
        raise ValueError(f"No features found in token_diag json: {path}")

    return {str(r["feature"]): r for r in rows}


def minmax_fit(values: np.ndarray) -> Dict[str, object]:
    values = np.asarray(values, dtype=float)
    return {
        "method": "train_minmax",
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def minmax_scale(values: np.ndarray, transform: Dict[str, object]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mn = float(transform["min"])
    mx = float(transform["max"])
    if abs(mx - mn) <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    z = (values - mn) / (mx - mn)
    return np.clip(z, 0.0, 1.0).astype(np.float32)


def unique_rank_fit(values: np.ndarray) -> Dict[str, object]:
    u = np.unique(np.asarray(values, dtype=float))
    return {
        "method": "piecewise_unique_rank",
        "num_unique_breakpoints": int(u.size),
        "unique_raw_values": u.tolist(),
        "raw_start": float(u[0]) if u.size else None,
        "raw_end": float(u[-1]) if u.size else None,
    }


def unique_rank_scale(values: np.ndarray, transform: Dict[str, object]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    u = np.asarray(transform["unique_raw_values"], dtype=float)
    m = int(u.size)
    if m <= 1:
        return np.zeros_like(values, dtype=np.float32)

    ranks = np.arange(m, dtype=np.float64) / float(m - 1)
    z = np.interp(values, u, ranks, left=0.0, right=1.0)
    return np.clip(z, 0.0, 1.0).astype(np.float32)


def blended_rank_fit(values: np.ndarray, alpha: float) -> Dict[str, object]:
    values = np.asarray(values, dtype=float)
    mm = minmax_fit(values)
    u = np.unique(values)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return {
        "method": "blended_rank",
        "min": float(mm["min"]),
        "max": float(mm["max"]),
        "alpha": alpha,
        "num_unique_breakpoints": int(u.size),
        "unique_raw_values": u.tolist(),
        "raw_start": float(u[0]) if u.size else None,
        "raw_end": float(u[-1]) if u.size else None,
    }


def blended_rank_scale(values: np.ndarray, transform: Dict[str, object]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    alpha = float(np.clip(float(transform.get("alpha", 0.25)), 0.0, 1.0))
    z_mm = minmax_scale(values, transform).astype(np.float64)

    u = np.asarray(transform["unique_raw_values"], dtype=float)
    m = int(u.size)
    if m <= 1:
        return z_mm.astype(np.float32)

    ranks = np.arange(m, dtype=np.float64) / float(m - 1)
    z_rank = np.interp(values, u, ranks, left=0.0, right=1.0)

    z = (1.0 - alpha) * z_mm + alpha * z_rank
    return np.clip(z, 0.0, 1.0).astype(np.float32)


def apply_transform(values: np.ndarray, transform: Dict[str, object]) -> np.ndarray:
    method = str(transform.get("method", ""))

    if method == "constant_zero":
        return np.zeros_like(np.asarray(values, dtype=float), dtype=np.float32)

    if method == "train_minmax":
        return minmax_scale(values, transform)

    if method == "piecewise_unique_rank":
        return unique_rank_scale(values, transform)

    if method == "blended_rank":
        return blended_rank_scale(values, transform)

    raise ValueError(f"Unknown transform method: {method}")


def tokens_from_z(z: np.ndarray, K: int) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = np.clip(z, 0.0, 1.0)
    tokens = np.floor(float(K) * z).astype(np.int64)
    return np.clip(tokens, 0, int(K) - 1)


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


def token_metrics_from_z(z: np.ndarray, raw_unique: int, K: int) -> Dict[str, float | int]:
    tokens = tokens_from_z(z, K)
    unique_tokens, counts = np.unique(tokens, return_counts=True)

    bins_used = int(unique_tokens.size)
    possible_unique = int(min(int(raw_unique), int(K)))
    preserve_ratio = float(bins_used / max(possible_unique, 1))
    compression_factor = float(int(raw_unique) / max(bins_used, 1))
    entropy_norm = normalized_entropy_from_counts(counts)

    return {
        "bins_used": bins_used,
        "possible_unique": possible_unique,
        "preserve_ratio": preserve_ratio,
        "compression_factor": compression_factor,
        "entropy_norm": entropy_norm,
    }


def before_metrics_from_diag(diag_row: Dict[str, object], K: int, train_values: np.ndarray) -> Dict[str, float | int]:
    token = diag_row.get("token", {})
    raw = diag_row.get("raw", {})
    raw_unique = int(raw.get("num_unique", np.unique(train_values).size))

    if all(k in token for k in ["preserve_ratio", "compression_factor", "entropy_norm"]):
        return {
            "bins_used": int(token.get("bins_used", token.get("num_tokens_used", 0))),
            "possible_unique": int(token.get("possible_unique", min(raw_unique, K))),
            "preserve_ratio": float(token["preserve_ratio"]),
            "compression_factor": float(token["compression_factor"]),
            "entropy_norm": float(token["entropy_norm"]),
        }

    # Fallback for older diag format.
    z = minmax_scale(train_values, minmax_fit(train_values))
    return token_metrics_from_z(z, raw_unique=raw_unique, K=K)


def candidate_score(
    metrics: Dict[str, float | int],
    *,
    preserve_threshold: float,
    compression_threshold: float,
    entropy_threshold: float,
) -> Tuple[float, float, float, float]:
    """
    Higher is better.

    The score is intentionally token-level only:
      - preserve_ratio high
      - entropy_norm high
      - compression_factor low

    compression_score is capped relative to threshold, because when raw_unique >> K,
    compression_factor cannot be fully fixed at a fixed K. It should not dominate
    the ranking once the candidate already uses all bins.
    """
    preserve = float(metrics["preserve_ratio"])
    entropy = float(metrics["entropy_norm"])
    compression = float(metrics["compression_factor"])

    compression_score = min(1.0, float(compression_threshold) / max(compression, 1e-12))

    pass_count = 0
    if preserve >= float(preserve_threshold):
        pass_count += 1
    if compression <= float(compression_threshold):
        pass_count += 1
    if entropy >= float(entropy_threshold):
        pass_count += 1

    # Lexicographic tuple; pass_count first, then actual quality.
    return (
        float(pass_count),
        preserve,
        entropy,
        compression_score,
    )


def candidate_priority(name: str) -> int:
    """
    Tie-breaker: prefer less aggressive transforms if token metrics are equal.
    """
    order = {
        "train_minmax": 0,
        "blended_rank_alpha_0.25": 1,
        "blended_rank_alpha_0.50": 2,
        "blended_rank_alpha_0.5": 2,
        "blended_rank_alpha_0.75": 3,
        "piecewise_unique_rank": 4,
    }
    return order.get(name, 99)


def build_candidates(
    train_values: np.ndarray,
    *,
    include_minmax: bool,
    blend_alphas: Sequence[float],
    include_rank: bool,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []

    if include_minmax:
        candidates.append({
            "candidate": "train_minmax",
            "transform": minmax_fit(train_values),
        })

    for alpha in blend_alphas:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        candidates.append({
            "candidate": f"blended_rank_alpha_{alpha:.2f}",
            "transform": blended_rank_fit(train_values, alpha=alpha),
        })

    if include_rank:
        candidates.append({
            "candidate": "piecewise_unique_rank",
            "transform": unique_rank_fit(train_values),
        })

    # Remove duplicate alpha candidate names if user passes duplicates.
    seen = set()
    out = []
    for c in candidates:
        name = str(c["candidate"])
        if name in seen:
            continue
        seen.add(name)
        out.append(c)

    return out


def select_transform_for_feature(
    feature: str,
    train_values: np.ndarray,
    diag_row: Dict[str, object],
    K: int,
    *,
    preserve_threshold: float,
    compression_threshold: float,
    entropy_threshold: float,
    include_minmax_candidate: bool,
    blend_alphas: Sequence[float],
    include_rank_candidate: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, float | int]]:
    raw = diag_row.get("raw", {})
    decision_obj = diag_row.get("decision", {})
    diag_decision = str(decision_obj.get("label", "transform"))
    diag_reasons = list(decision_obj.get("reasons", []))

    raw_unique = int(raw.get("num_unique", np.unique(train_values).size))
    before = before_metrics_from_diag(diag_row, K, train_values)

    base = {
        "feature": feature,
        "K": int(K),
        "diag_decision": diag_decision,
        "diag_reasons": diag_reasons,
        "raw_num_unique": raw_unique,
        "raw_min": float(np.min(train_values)),
        "raw_max": float(np.max(train_values)),
        "before_metrics": before,
    }

    if diag_decision == "drop" or raw_unique <= 1:
        transform = {"method": "constant_zero"}
        z = apply_transform(train_values, transform)
        after = token_metrics_from_z(z, raw_unique=raw_unique, K=K)

        policy = {
            **base,
            "action": "constant_zero",
            "selected_candidate": "constant_zero",
            "reason": "diag decision is drop / raw feature is constant",
            "transform": transform,
            "after_metrics": after,
        }
        return policy, [], before

    if diag_decision == "keep":
        transform = minmax_fit(train_values)
        z = apply_transform(train_values, transform)
        after = token_metrics_from_z(z, raw_unique=raw_unique, K=K)

        policy = {
            **base,
            "action": "keep_minmax",
            "selected_candidate": "train_minmax",
            "reason": "diag decision is keep",
            "transform": transform,
            "after_metrics": after,
        }
        return policy, [], before

    # diag_decision == transform
    candidates = build_candidates(
        train_values,
        include_minmax=include_minmax_candidate,
        blend_alphas=blend_alphas,
        include_rank=include_rank_candidate,
    )

    candidate_rows: List[Dict[str, object]] = []
    best = None

    for c in candidates:
        name = str(c["candidate"])
        transform = c["transform"]

        z = apply_transform(train_values, transform)
        metrics = token_metrics_from_z(z, raw_unique=raw_unique, K=K)
        score = candidate_score(
            metrics,
            preserve_threshold=preserve_threshold,
            compression_threshold=compression_threshold,
            entropy_threshold=entropy_threshold,
        )

        row = {
            "feature": feature,
            "candidate": name,
            "raw_unique": raw_unique,
            "before_preserve_ratio": float(before["preserve_ratio"]),
            "before_compression_factor": float(before["compression_factor"]),
            "before_entropy_norm": float(before["entropy_norm"]),
            "after_bins_used": int(metrics["bins_used"]),
            "after_possible_unique": int(metrics["possible_unique"]),
            "after_preserve_ratio": float(metrics["preserve_ratio"]),
            "after_compression_factor": float(metrics["compression_factor"]),
            "after_entropy_norm": float(metrics["entropy_norm"]),
            "score_pass_count": float(score[0]),
            "score_preserve": float(score[1]),
            "score_entropy": float(score[2]),
            "score_compression": float(score[3]),
            "candidate_priority": int(candidate_priority(name)),
        }
        candidate_rows.append(row)

        # Higher score better; if tied, lower priority value better.
        sort_key = (score, -candidate_priority(name))
        if best is None or sort_key > best["sort_key"]:
            best = {
                "name": name,
                "transform": transform,
                "metrics": metrics,
                "score": score,
                "sort_key": sort_key,
            }

    assert best is not None

    policy = {
        **base,
        "action": str(best["transform"]["method"]),
        "selected_candidate": str(best["name"]),
        "reason": "diag decision is transform; selected by train token-level metrics",
        "candidate_selection_rule": {
            "metrics": ["preserve_ratio", "compression_factor", "entropy_norm"],
            "score_order": [
                "number of passed thresholds",
                "higher preserve_ratio",
                "higher entropy_norm",
                "lower compression_factor via capped compression score",
                "less aggressive transform tie-breaker",
            ],
            "thresholds": {
                "preserve_threshold": float(preserve_threshold),
                "compression_threshold": float(compression_threshold),
                "entropy_threshold": float(entropy_threshold),
            },
        },
        "transform": best["transform"],
        "after_metrics": best["metrics"],
    }
    return policy, candidate_rows, before


def apply_policy_to_df(
    df: pd.DataFrame,
    features: Sequence[str],
    policies_by_feature: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    out = df.copy()

    for f in features:
        p = policies_by_feature[f]
        transform = p["transform"]
        values = df[f].to_numpy(dtype=float)
        out[f] = apply_transform(values, transform)

    return out


def flatten_policy_summary(policy: Dict[str, object]) -> Dict[str, object]:
    before = policy["before_metrics"]
    after = policy["after_metrics"]

    return {
        "feature": policy["feature"],
        "diag_decision": policy["diag_decision"],
        "diag_reasons": "|".join(policy.get("diag_reasons", [])),
        "selected_candidate": policy["selected_candidate"],
        "action": policy["action"],
        "raw_unique": int(policy["raw_num_unique"]),

        "before_preserve_ratio": float(before["preserve_ratio"]),
        "before_compression_factor": float(before["compression_factor"]),
        "before_entropy_norm": float(before["entropy_norm"]),

        "after_bins_used": int(after["bins_used"]),
        "after_possible_unique": int(after["possible_unique"]),
        "after_preserve_ratio": float(after["preserve_ratio"]),
        "after_compression_factor": float(after["compression_factor"]),
        "after_entropy_norm": float(after["entropy_norm"]),

        "delta_preserve_ratio": float(after["preserve_ratio"]) - float(before["preserve_ratio"]),
        "delta_compression_factor": float(after["compression_factor"]) - float(before["compression_factor"]),
        "delta_entropy_norm": float(after["entropy_norm"]) - float(before["entropy_norm"]),
    }


def main() -> None:
    args = parse_args()

    K = int(args.K)
    if K <= 1:
        raise ValueError("K/effective_num_bins must be > 1.")

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    out_dir = Path(args.out_dir)

    token_diag_json = Path(args.token_diag_json) if args.token_diag_json else default_token_diag_path(K)

    if not train_csv.exists():
        raise FileNotFoundError(f"train csv not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"val csv not found: {val_csv}")

    target_cols = csv_list(args.target_cols)
    drop_cols = csv_list(args.drop_cols)
    blend_alphas = parse_float_list(args.blend_alphas)

    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)

    features = detect_numeric_features(train, target_cols=target_cols, drop_cols=drop_cols)
    if not features:
        raise ValueError("No numeric features detected.")

    assert_split_has_features("train", train, features)
    assert_split_has_features("val", val, features)

    diag_by_feature = load_token_diag(token_diag_json)
    missing_diag = [f for f in features if f not in diag_by_feature]
    if missing_diag:
        raise ValueError(f"token_diag missing {len(missing_diag)} features, first: {missing_diag[:10]}")

    policies: List[Dict[str, object]] = []
    candidate_eval_rows: List[Dict[str, object]] = []

    for f in features:
        policy, rows, _before = select_transform_for_feature(
            feature=f,
            train_values=train[f].to_numpy(dtype=float),
            diag_row=diag_by_feature[f],
            K=K,
            preserve_threshold=float(args.preserve_threshold),
            compression_threshold=float(args.compression_threshold),
            entropy_threshold=float(args.entropy_threshold),
            include_minmax_candidate=bool(args.include_minmax_candidate),
            blend_alphas=blend_alphas,
            include_rank_candidate=bool(args.include_rank_candidate),
        )
        policies.append(policy)
        candidate_eval_rows.extend(rows)

    policies_by_feature = {str(p["feature"]): p for p in policies}

    train_pre = apply_policy_to_df(train, features, policies_by_feature)
    val_pre = apply_policy_to_df(val, features, policies_by_feature)

    out_dir.mkdir(parents=True, exist_ok=True)

    out_train = out_dir / f"train_preprocessed_K{K}.csv"
    out_val = out_dir / f"val_preprocessed_K{K}.csv"
    out_policy = out_dir / f"preprocess_policy_K{K}.json"
    out_report = out_dir / f"preprocess_report_K{K}.json"
    out_summary = out_dir / f"preprocess_token_eval_summary_K{K}.csv"
    out_candidates = out_dir / f"preprocess_candidate_eval_K{K}.csv"

    train_pre.to_csv(out_train, index=False)
    val_pre.to_csv(out_val, index=False)

    summary_rows = [flatten_policy_summary(p) for p in policies]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_summary, index=False)

    candidate_df = pd.DataFrame(candidate_eval_rows)
    candidate_df.to_csv(out_candidates, index=False)

    action_counts = Counter(str(p["action"]) for p in policies)
    diag_counts = Counter(str(p["diag_decision"]) for p in policies)
    selected_counts = Counter(str(p["selected_candidate"]) for p in policies)

    policy_obj = {
        "metadata": {
            "stage": "preprocessing",
            "phase": "Phase-B token-level preprocessing selection",
            "fit_split": "train_only",
            "applied_splits": ["train", "val"],
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "token_diag_json": str(token_diag_json),
            "K": int(K),
            "effective_num_bins": int(K),
            "token_rule": "z in [0,1]; token=floor(K*z); clipped [0,K-1]",
            "thresholds": {
                "preserve_threshold": float(args.preserve_threshold),
                "compression_threshold": float(args.compression_threshold),
                "entropy_threshold": float(args.entropy_threshold),
            },
            "candidate_transforms_for_diag_transform": {
                "include_minmax_candidate": bool(args.include_minmax_candidate),
                "blend_alphas": [float(x) for x in blend_alphas],
                "include_rank_candidate": bool(args.include_rank_candidate),
            },
            "n_train_rows": int(len(train)),
            "n_val_rows": int(len(val)),
            "n_numeric_features": int(len(features)),
            "target_cols_preserved": target_cols,
            "drop_cols": drop_cols,
            "feature_order": features,
        },
        "policies": policies,
    }

    report = {
        "metadata": policy_obj["metadata"],
        "diag_decision_counts": dict(diag_counts),
        "selected_action_counts": dict(action_counts),
        "selected_candidate_counts": dict(selected_counts),
        "token_eval_summary_csv": str(out_summary),
        "candidate_eval_csv": str(out_candidates),
        "outputs": {
            "train_preprocessed_csv": str(out_train),
            "val_preprocessed_csv": str(out_val),
            "preprocess_policy_json": str(out_policy),
            "preprocess_report_json": str(out_report),
        },
        "transform_features": [
            flatten_policy_summary(p)
            for p in policies
            if str(p["diag_decision"]) == "transform"
        ],
    }

    out_policy.write_text(json.dumps(policy_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== preprocessing done =====")
    print(f"K/effective_num_bins: {K}")
    print(f"features: {len(features)}")
    print(f"diag_decision_counts: {dict(diag_counts)}")
    print(f"selected_action_counts: {dict(action_counts)}")
    print(f"selected_candidate_counts: {dict(selected_counts)}")
    print(f"train_preprocessed: {out_train}")
    print(f"val_preprocessed: {out_val}")
    print(f"policy: {out_policy}")
    print(f"report: {out_report}")
    print(f"summary_csv: {out_summary}")
    print(f"candidate_eval_csv: {out_candidates}")


if __name__ == "__main__":
    main()

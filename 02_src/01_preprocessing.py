#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_pipeline/preprocessing.py

Train-fitted preprocessing, applied to train and val.

Input:
    - data/01_split/train_raw.csv
    - data/01_split/val_raw.csv
    - token_diag_train_K{K}.json from token_diag.py

Output:
    - train_preprocessed_K{K}.csv
    - val_preprocessed_K{K}.csv
    - preprocess_policy_K{K}.json
    - preprocess_report_K{K}.json

Decision is train-only:
    possible_unique = min(raw_unique, K + 1)
    unique_preserve_ratio = num_tokens_used / possible_unique

If unique_preserve_ratio is below threshold:
    default action = blended_rank
        z = (1-alpha) * train_minmax_z + alpha * unique_rank_z
    Local piecewise and global rank are kept as explicit experimental options.
Otherwise:
    action = keep_minmax

No test split.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import config as CFG


def csv_list(value: str | None) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train-fitted preprocessing applied to train and val.")
    p.add_argument("--train-csv", default=str(CFG.TRAIN_CSV))
    p.add_argument("--val-csv", default=str(CFG.VAL_CSV))
    p.add_argument("--token-diag-json", default="")
    p.add_argument("--out-dir", default=str(CFG.PREPROCESS_DIR))
    p.add_argument("--K", type=int, default=int(CFG.TOKEN_K))
    p.add_argument("--target-cols", default=",".join(CFG.TARGET_COLS))
    p.add_argument("--drop-cols", default=",".join(CFG.DROP_COLS))
    p.add_argument("--unique-preserve-threshold", type=float, default=float(CFG.UNIQUE_PRESERVE_THRESHOLD))
    p.add_argument(
        "--compressed-action",
        choices=["blended_rank", "local_piecewise", "global_rank", "keep_minmax"],
        default=str(getattr(CFG, "COMPRESSED_FEATURE_ACTION", "blended_rank")),
        help="Action for features below unique-preserve threshold.",
    )
    p.add_argument(
        "--blend-alpha",
        type=float,
        default=float(getattr(CFG, "BLENDED_RANK_ALPHA", 0.25)),
        help="Alpha for blended_rank: z=(1-alpha)*minmax + alpha*rank.",
    )
    return p.parse_args()


def detect_numeric_features(df: pd.DataFrame, target_cols: Sequence[str], drop_cols: Sequence[str]) -> List[str]:
    excluded = set(target_cols) | set(drop_cols)
    excluded |= {c for c in df.columns if str(c).startswith("Unnamed:")}
    return [
        c for c in df.columns
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
        raise ValueError(f"{split_name} contains non-finite numeric values: nan={nan_count}, inf={inf_count}")


def is_special_delay_feature(feature: str) -> bool:
    f = feature.lower()
    return any(k.lower() in f for k in CFG.SPECIAL_DELAY_KEYWORDS)


def load_token_diag(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(
            f"token_diag json not found: {path}\n"
            f"Run token_diag first:\n"
            f"  python -u final_pipeline/token_diag.py --K {CFG.TOKEN_K}"
        )
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("features", [])
    if not rows:
        raise ValueError(f"No features found in token_diag json: {path}")
    return {str(r["feature"]): r for r in rows}


def minmax_scale(values: np.ndarray, mn: float, mx: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if abs(mx - mn) <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    z = (values - mn) / (mx - mn)
    return np.clip(z, 0.0, 1.0).astype(np.float32)


def special_delay_scale(values: np.ndarray, mx: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    denom = float(mx) + float(CFG.SPECIAL_DELAY_EPS)
    if abs(denom) <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    z = values / denom
    return np.clip(z, 0.0, 1.0).astype(np.float32)


def piecewise_unique_rank_fit(train_values: np.ndarray) -> Dict[str, object]:
    unique_vals = np.unique(np.asarray(train_values, dtype=float))
    m = int(unique_vals.size)
    return {
        "method": "piecewise_unique_rank",
        "num_unique_breakpoints": m,
        "unique_raw_values": unique_vals.tolist(),
        "raw_start": float(unique_vals[0]) if m else None,
        "raw_end": float(unique_vals[-1]) if m else None,
    }


def piecewise_unique_rank_scale(values: np.ndarray, unique_raw_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    u = np.asarray(unique_raw_values, dtype=float)
    m = int(u.size)
    if m <= 1:
        return np.zeros_like(values, dtype=np.float32)

    ranks = np.arange(m, dtype=np.float64) / float(m - 1)

    # Train exact values map exactly.
    # Val unseen values interpolate monotonically between train unique values.
    z = np.interp(values, u, ranks, left=0.0, right=1.0)
    return np.clip(z, 0.0, 1.0).astype(np.float32)




def blended_rank_fit(train_values: np.ndarray, mn: float, mx: float, alpha: float) -> Dict[str, object]:
    """
    Fit a soft rank/minmax transform.

    alpha=0.0 -> pure train minmax
    alpha=1.0 -> pure unique-rank / quantile-like mapping

    This is intentionally softer than local piecewise: it reduces dense-region
    compression without hard breakpoints and without blindly zooming a collision
    interval that may be too wide or label-pure.
    """
    unique_vals = np.unique(np.asarray(train_values, dtype=float))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return {
        "method": "blended_rank",
        "min": float(mn),
        "max": float(mx),
        "alpha": alpha,
        "num_unique_breakpoints": int(unique_vals.size),
        "unique_raw_values": unique_vals.tolist(),
        "raw_start": float(unique_vals[0]) if unique_vals.size else None,
        "raw_end": float(unique_vals[-1]) if unique_vals.size else None,
    }


def blended_rank_scale(values: np.ndarray, transform: Dict[str, object]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mn = float(transform["min"])
    mx = float(transform["max"])
    alpha = float(np.clip(float(transform.get("alpha", 0.25)), 0.0, 1.0))

    z_mm = minmax_scale(values, mn, mx).astype(np.float64)

    u = np.asarray(transform["unique_raw_values"], dtype=float)
    if u.size <= 1:
        return z_mm.astype(np.float32)

    ranks = np.arange(int(u.size), dtype=np.float64) / float(int(u.size) - 1)
    z_rank = np.interp(values, u, ranks, left=0.0, right=1.0)
    z = (1.0 - alpha) * z_mm + alpha * z_rank
    return np.clip(z, 0.0, 1.0).astype(np.float32)


def _cfg_float(name: str, default: float) -> float:
    return float(getattr(CFG, name, default))


def _cfg_int(name: str, default: int) -> int:
    return int(getattr(CFG, name, default))


def _safe_width(start: float, end: float) -> float:
    return float(max(float(end) - float(start), 0.0))


def local_piecewise_fit(
    train_values: np.ndarray,
    mn: float,
    mx: float,
    K: int,
    collision: Dict[str, object],
) -> Dict[str, object]:
    """
    Fit a local piecewise transform around the raw interval where minmax
    tokenization collapses multiple raw unique values into the same tokens.

    This is token-budget reallocation, not global rank transform:
        [mn, a] -> linear [0, new_z_start]
        [a, b]  -> local unique-rank [new_z_start, new_z_end]
        [b, mx] -> linear [new_z_end, 1]

    If a stable local interval cannot be built, fall back to the old global
    unique-rank policy for backward-safe behavior.
    """
    values = np.asarray(train_values, dtype=float)
    unique_vals = np.unique(values)
    raw_range = float(mx - mn)

    if unique_vals.size <= 1 or raw_range <= 1e-12:
        return {
            "method": "constant_zero",
            "fallback_reason": "feature has <=1 unique value or zero raw range",
        }

    interval = collision.get("collision_raw_interval", {}) if isinstance(collision, dict) else {}
    a = interval.get("raw_start")
    b = interval.get("raw_end")

    if a is None or b is None:
        fit = piecewise_unique_rank_fit(values)
        fit["fallback_reason"] = "collision interval is missing"
        return fit

    a = float(a)
    b = float(b)
    if not np.isfinite(a) or not np.isfinite(b):
        fit = piecewise_unique_rank_fit(values)
        fit["fallback_reason"] = "collision interval is not finite"
        return fit

    a = float(np.clip(a, mn, mx))
    b = float(np.clip(b, mn, mx))
    if b < a:
        a, b = b, a

    local_unique = unique_vals[(unique_vals >= a) & (unique_vals <= b)]
    min_local_unique = _cfg_int("LOCAL_PIECEWISE_MIN_UNIQUE", 3)
    if int(local_unique.size) < min_local_unique or abs(b - a) <= 1e-12:
        fit = piecewise_unique_rank_fit(values)
        fit["fallback_reason"] = (
            f"local interval has too few unique values: {int(local_unique.size)} < {min_local_unique}"
        )
        fit["collision_raw_interval"] = {"raw_start": a, "raw_end": b}
        return fit

    old_z_start = float((a - mn) / raw_range)
    old_z_end = float((b - mn) / raw_range)
    old_width = _safe_width(old_z_start, old_z_end)

    margin = _cfg_float("LOCAL_PIECEWISE_MARGIN", 1.10)
    min_width = _cfg_float("LOCAL_PIECEWISE_MIN_WIDTH", 0.03)
    max_width = _cfg_float("LOCAL_PIECEWISE_MAX_WIDTH", 0.50)
    max_width = float(min(max(max_width, min_width), 1.0))

    # To give m unique values a chance to occupy distinct rounded tokens, the
    # local interval needs roughly (m-1)/K of z-space. Margin adds slack for
    # rounding/interpolation. Cap prevents one feature region from eating the
    # entire token axis.
    m = int(local_unique.size)
    required_width = float(max(m - 1, 1) / max(int(K), 1))
    new_width = max(old_width, min_width, required_width * margin)
    new_width = float(min(new_width, max_width, 1.0))

    center = float((old_z_start + old_z_end) / 2.0)
    new_z_start = center - new_width / 2.0
    new_z_end = center + new_width / 2.0

    if new_z_start < 0.0:
        new_z_end -= new_z_start
        new_z_start = 0.0
    if new_z_end > 1.0:
        shift = new_z_end - 1.0
        new_z_start -= shift
        new_z_end = 1.0

    new_z_start = float(np.clip(new_z_start, 0.0, 1.0))
    new_z_end = float(np.clip(new_z_end, 0.0, 1.0))
    if new_z_end <= new_z_start:
        fit = piecewise_unique_rank_fit(values)
        fit["fallback_reason"] = "computed local z interval is degenerate"
        return fit

    return {
        "method": "local_piecewise_unique_rank",
        "raw_min": float(mn),
        "raw_max": float(mx),
        "local_raw_start": float(a),
        "local_raw_end": float(b),
        "old_z_start": float(old_z_start),
        "old_z_end": float(old_z_end),
        "old_width": float(old_width),
        "new_z_start": float(new_z_start),
        "new_z_end": float(new_z_end),
        "new_width": float(new_z_end - new_z_start),
        "required_width_estimate": float(required_width),
        "local_unique_count": int(m),
        "unique_raw_values_in_interval": local_unique.tolist(),
        "margin": float(margin),
        "min_width": float(min_width),
        "max_width": float(max_width),
        "note": "Only the collision interval is unique-rank mapped; outside intervals remain linear.",
    }


def _linear_map_segment(
    values: np.ndarray,
    src_start: float,
    src_end: float,
    dst_start: float,
    dst_end: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if abs(float(src_end) - float(src_start)) <= 1e-12:
        return np.full_like(values, fill_value=float(dst_start), dtype=np.float64)
    t = (values - float(src_start)) / (float(src_end) - float(src_start))
    return float(dst_start) + t * (float(dst_end) - float(dst_start))


def local_piecewise_scale(values: np.ndarray, transform: Dict[str, object]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mn = float(transform["raw_min"])
    mx = float(transform["raw_max"])
    a = float(transform["local_raw_start"])
    b = float(transform["local_raw_end"])
    za = float(transform["new_z_start"])
    zb = float(transform["new_z_end"])
    local_u = np.asarray(transform["unique_raw_values_in_interval"], dtype=float)

    if local_u.size <= 1 or abs(mx - mn) <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)

    z = np.empty_like(values, dtype=np.float64)

    left_mask = values < a
    mid_mask = (values >= a) & (values <= b)
    right_mask = values > b

    if np.any(left_mask):
        z[left_mask] = _linear_map_segment(values[left_mask], mn, a, 0.0, za)
    if np.any(mid_mask):
        local_ranks = np.linspace(za, zb, num=int(local_u.size), dtype=np.float64)
        z[mid_mask] = np.interp(values[mid_mask], local_u, local_ranks, left=za, right=zb)
    if np.any(right_mask):
        z[right_mask] = _linear_map_segment(values[right_mask], b, mx, zb, 1.0)

    return np.clip(z, 0.0, 1.0).astype(np.float32)


def tokens_from_minmax(values: np.ndarray, mn: float, mx: float, K: int) -> np.ndarray:
    z = minmax_scale(values, mn, mx)
    return np.clip(np.rint(float(K) * z), 0, int(K)).astype(np.int64)


def collision_analysis(values: np.ndarray, mn: float, mx: float, K: int, diag_row: Dict[str, object]) -> Dict[str, object]:
    values = np.asarray(values, dtype=float)
    unique_vals = np.unique(values)
    unique_tokens = tokens_from_minmax(unique_vals, mn, mx, K)

    token_to_values: Dict[int, List[float]] = defaultdict(list)
    for raw_v, tok in zip(unique_vals, unique_tokens):
        token_to_values[int(tok)].append(float(raw_v))

    collision_tokens = {tok: vals for tok, vals in token_to_values.items() if len(vals) > 1}
    total_collision_loss = int(sum(len(vals) - 1 for vals in collision_tokens.values()))

    q = diag_row["token"]["quantiles"]
    q10 = int(q["q10"])
    q90 = int(q["q90"])

    region_loss = Counter({"left": 0, "body": 0, "right": 0})
    region_values: Dict[str, List[float]] = {"left": [], "body": [], "right": []}

    for tok, vals in collision_tokens.items():
        loss = len(vals) - 1
        if tok <= q10:
            region = "left"
        elif tok >= q90:
            region = "right"
        else:
            region = "body"
        region_loss[region] += int(loss)
        region_values[region].extend(vals)

    if total_collision_loss <= 0:
        side = "none"
        dominant_fraction = 0.0
        all_vals: List[float] = []
    else:
        side0, loss0 = region_loss.most_common(1)[0]
        dominant_fraction = float(loss0 / total_collision_loss)
        if dominant_fraction < float(CFG.PIECEWISE_SIDE_DOMINANCE_RATIO):
            side = "mixed"
            all_vals = [v for vals in region_values.values() for v in vals]
        else:
            side = side0
            all_vals = region_values[side0]

    return {
        "total_collision_tokens": int(len(collision_tokens)),
        "total_collision_loss": int(total_collision_loss),
        "region_collision_loss": dict(region_loss),
        "piecewise_side": side,
        "piecewise_side_dominant_fraction": dominant_fraction,
        "collision_raw_interval": {
            "raw_start": float(min(all_vals)) if all_vals else None,
            "raw_end": float(max(all_vals)) if all_vals else None,
        },
        "example_collision_tokens": [
            {
                "token": int(tok),
                "num_raw_values": int(len(vals)),
                "raw_min": float(min(vals)),
                "raw_max": float(max(vals)),
            }
            for tok, vals in sorted(collision_tokens.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
        ],
    }


def decide_feature_policy(
    feature: str,
    train_values: np.ndarray,
    diag_row: Dict[str, object],
    K: int,
    unique_threshold: float,
    compressed_action: str,
    blend_alpha: float,
) -> Dict[str, object]:
    raw_info = diag_row["raw"]
    token_info = diag_row["token"]

    mn = float(raw_info["train_min"])
    mx = float(raw_info["train_max"])
    raw_unique = int(raw_info["num_unique"])
    num_tokens_used = int(token_info["num_tokens_used"])
    is_constant = bool(raw_info["is_constant"])

    possible_unique = int(min(raw_unique, int(K) + 1))
    unique_preserve_ratio = float(num_tokens_used / max(possible_unique, 1))

    base = {
        "feature": feature,
        "K": int(K),
        "raw_num_unique": raw_unique,
        "num_tokens_used": num_tokens_used,
        "possible_unique": possible_unique,
        "unique_preserve_ratio": unique_preserve_ratio,
        "unique_preserve_threshold": float(unique_threshold),
        "raw_min": mn,
        "raw_max": mx,
    }

    if is_constant:
        return {
            **base,
            "action": "constant_zero",
            "reason": "raw feature is constant",
            "transform": {"method": "constant_zero"},
        }

    if is_special_delay_feature(feature):
        return {
            **base,
            "action": "special_delay_scale",
            "reason": f"feature name matches special delay keywords {CFG.SPECIAL_DELAY_KEYWORDS}",
            "transform": {
                "method": "x_over_max_plus_eps",
                "max": mx,
                "eps": float(CFG.SPECIAL_DELAY_EPS),
                "discretization_multiplier": float(K / (mx + CFG.SPECIAL_DELAY_EPS)) if abs(mx + CFG.SPECIAL_DELAY_EPS) > 1e-12 else 0.0,
            },
        }

    if unique_preserve_ratio >= unique_threshold:
        return {
            **base,
            "action": "keep_minmax",
            "reason": "unique_preserve_ratio is high; tokenization preserves raw unique values well enough",
            "transform": {
                "method": "train_minmax",
                "min": mn,
                "max": mx,
            },
        }

    collision = collision_analysis(train_values, mn, mx, K, diag_row)
    compressed_action = str(compressed_action)

    if compressed_action == "keep_minmax":
        return {
            **base,
            "action": "keep_minmax",
            "reason": "feature is compressed, but compressed_action=keep_minmax",
            "collision_analysis": collision,
            "transform": {
                "method": "train_minmax",
                "min": mn,
                "max": mx,
            },
        }

    if compressed_action == "global_rank":
        rank_fit = piecewise_unique_rank_fit(train_values)
        return {
            **base,
            "action": "piecewise",
            "reason": "unique_preserve_ratio is below threshold; compressed_action=global_rank",
            "collision_analysis": collision,
            "transform": {
                **rank_fit,
                "piecewise_side": collision["piecewise_side"],
                "collision_raw_interval": collision["collision_raw_interval"],
            },
        }

    if compressed_action == "local_piecewise":
        piecewise_fit = local_piecewise_fit(train_values, mn, mx, K, collision)
        return {
            **base,
            "action": "piecewise",
            "reason": "unique_preserve_ratio is below threshold; compressed_action=local_piecewise",
            "collision_analysis": collision,
            "transform": {
                **piecewise_fit,
                "piecewise_side": collision["piecewise_side"],
                "collision_raw_interval": collision["collision_raw_interval"],
            },
        }

    if compressed_action == "blended_rank":
        blend_fit = blended_rank_fit(train_values, mn, mx, alpha=blend_alpha)
        return {
            **base,
            "action": "blended_rank",
            "reason": "unique_preserve_ratio is below threshold; soft blend of minmax and unique-rank reduces dense-region compression without hard local zoom",
            "collision_analysis": collision,
            "transform": {
                **blend_fit,
                "collision_raw_interval": collision["collision_raw_interval"],
                "piecewise_side": collision["piecewise_side"],
            },
        }

    raise ValueError(f"Unknown compressed_action: {compressed_action}")


def apply_policy_to_df(df: pd.DataFrame, features: Sequence[str], policies_by_feature: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    out = df.copy()

    for f in features:
        p = policies_by_feature[f]
        action = str(p["action"])
        transform = p["transform"]
        values = df[f].to_numpy(dtype=float)

        if action == "constant_zero":
            z = np.zeros_like(values, dtype=np.float32)

        elif action == "keep_minmax":
            z = minmax_scale(values, float(transform["min"]), float(transform["max"]))

        elif action == "special_delay_scale":
            z = special_delay_scale(values, float(transform["max"]))

        elif action == "blended_rank":
            z = blended_rank_scale(values, transform)

        elif action == "piecewise":
            method = str(transform.get("method", "piecewise_unique_rank"))
            if method == "local_piecewise_unique_rank":
                z = local_piecewise_scale(values, transform)
            elif method == "piecewise_unique_rank":
                # Backward-compatible fallback for old policy files.
                z = piecewise_unique_rank_scale(values, transform["unique_raw_values"])
            elif method == "constant_zero":
                z = np.zeros_like(values, dtype=np.float32)
            else:
                raise ValueError(f"Unknown piecewise transform method for {f}: {method}")

        else:
            raise ValueError(f"Unknown action for {f}: {action}")

        out[f] = z

    return out


def main() -> None:
    args = parse_args()
    K = int(args.K)

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    out_dir = Path(args.out_dir)
    token_diag_json = Path(args.token_diag_json) if args.token_diag_json else CFG.token_diag_json_path(K)

    if K <= 0:
        raise ValueError("K must be positive.")
    if not train_csv.exists():
        raise FileNotFoundError(f"train csv not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"val csv not found: {val_csv}")

    target_cols = csv_list(args.target_cols)
    drop_cols = csv_list(args.drop_cols)

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
    for f in features:
        policy = decide_feature_policy(
            feature=f,
            train_values=train[f].to_numpy(dtype=float),
            diag_row=diag_by_feature[f],
            K=K,
            unique_threshold=float(args.unique_preserve_threshold),
            compressed_action=str(args.compressed_action),
            blend_alpha=float(args.blend_alpha),
        )
        policies.append(policy)

    policies_by_feature = {str(p["feature"]): p for p in policies}

    train_pre = apply_policy_to_df(train, features, policies_by_feature)
    val_pre = apply_policy_to_df(val, features, policies_by_feature)

    out_dir.mkdir(parents=True, exist_ok=True)

    out_train = out_dir / f"train_preprocessed_K{K}.csv"
    out_val = out_dir / f"val_preprocessed_K{K}.csv"
    out_policy = out_dir / f"preprocess_policy_K{K}.json"
    out_report = out_dir / f"preprocess_report_K{K}.json"

    train_pre.to_csv(out_train, index=False)
    val_pre.to_csv(out_val, index=False)

    action_counts = Counter(p["action"] for p in policies)

    policy_obj = {
        "metadata": {
            "stage": "preprocessing",
            "fit_split": "train_only",
            "applied_splits": ["train", "val"],
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "token_diag_json": str(token_diag_json),
            "K": int(K),
            "unique_preserve_threshold": float(args.unique_preserve_threshold),
            "compressed_action": str(args.compressed_action),
            "blend_alpha": float(args.blend_alpha),
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
        "action_counts": dict(action_counts),
        "piecewise_features": [
            {
                "feature": p["feature"],
                "unique_preserve_ratio": p["unique_preserve_ratio"],
                "possible_unique": p["possible_unique"],
                "raw_num_unique": p["raw_num_unique"],
                "num_tokens_used": p["num_tokens_used"],
                "piecewise_side": p.get("collision_analysis", {}).get("piecewise_side"),
                "collision_raw_interval": p.get("collision_analysis", {}).get("collision_raw_interval"),
                "transform_method": p.get("transform", {}).get("method"),
                "old_z_start": p.get("transform", {}).get("old_z_start"),
                "old_z_end": p.get("transform", {}).get("old_z_end"),
                "old_width": p.get("transform", {}).get("old_width"),
                "new_z_start": p.get("transform", {}).get("new_z_start"),
                "new_z_end": p.get("transform", {}).get("new_z_end"),
                "new_width": p.get("transform", {}).get("new_width"),
                "local_unique_count": p.get("transform", {}).get("local_unique_count"),
                "fallback_reason": p.get("transform", {}).get("fallback_reason"),
            }
            for p in policies if p["action"] == "piecewise"
        ],
        "blended_rank_features": [
            {
                "feature": p["feature"],
                "unique_preserve_ratio": p["unique_preserve_ratio"],
                "possible_unique": p["possible_unique"],
                "raw_num_unique": p["raw_num_unique"],
                "num_tokens_used": p["num_tokens_used"],
                "alpha": p.get("transform", {}).get("alpha"),
                "piecewise_side": p.get("collision_analysis", {}).get("piecewise_side"),
                "collision_raw_interval": p.get("collision_analysis", {}).get("collision_raw_interval"),
            }
            for p in policies if p["action"] == "blended_rank"
        ],
        "outputs": {
            "train_preprocessed_csv": str(out_train),
            "val_preprocessed_csv": str(out_val),
            "preprocess_policy_json": str(out_policy),
            "preprocess_report_json": str(out_report),
        },
    }

    out_policy.write_text(json.dumps(policy_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== preprocessing done =====")
    print(f"K: {K}")
    print(f"features: {len(features)}")
    print(f"action_counts: {dict(action_counts)}")
    print(f"train_preprocessed: {out_train}")
    print(f"val_preprocessed:   {out_val}")
    print(f"policy:             {out_policy}")
    print(f"report:             {out_report}")


if __name__ == "__main__":
    main()
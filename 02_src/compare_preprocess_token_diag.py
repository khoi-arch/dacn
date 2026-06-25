#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_pipeline/compare_preprocess_token_diag.py

Evaluate token diagnostics before vs after preprocessing.

This is the missing checkpoint between:
    preprocessing.py
and:
    build_token.py

Important:
    - BEFORE uses token_diag_train_K{K}.json from raw train under MinMax token rule.
    - AFTER uses train_preprocessed_K{K}.csv directly as z in [0,1]:
          token = round(K * z)
      It does NOT MinMax again.
      This matters for special_delay_scale, where train max can be < 1.

Output:
    final_pipeline/outputs/preprocess_eval/preprocess_token_compare_K{K}.json
    final_pipeline/outputs/preprocess_eval/preprocess_token_compare_K{K}.csv
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

import config as CFG


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare token diagnostics before and after preprocessing.")
    p.add_argument("--K", type=int, default=int(CFG.TOKEN_K))
    p.add_argument("--before-diag-json", default="")
    p.add_argument("--train-preprocessed", default="")
    p.add_argument("--policy-json", default="")
    p.add_argument("--out-dir", default=str(getattr(CFG, "PREPROCESS_EVAL_DIR", CFG.OUTPUT_ROOT / "preprocess_eval")))
    return p.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_before_diag(path: Path) -> Dict[str, Dict[str, object]]:
    obj = load_json(path)
    rows = obj.get("features", [])
    if not rows:
        raise ValueError(f"No features found in before diag: {path}")
    return {str(r["feature"]): r for r in rows}


def load_policy(path: Path) -> tuple[List[str], Dict[str, Dict[str, object]], Dict[str, object]]:
    obj = load_json(path)
    meta = obj.get("metadata", {})
    policies = obj.get("policies", [])
    if not policies:
        raise ValueError(f"No policies found in policy json: {path}")

    feature_order = meta.get("feature_order")
    if feature_order:
        features = [str(x) for x in feature_order]
    else:
        features = [str(p["feature"]) for p in policies]

    by_feature = {str(p["feature"]): p for p in policies}
    return features, by_feature, meta


def direct_z_token_diag(z_values: np.ndarray, K: int) -> Dict[str, object]:
    z = np.asarray(z_values, dtype=float)
    z_clipped = np.clip(z, 0.0, 1.0)
    tokens = np.clip(np.rint(float(K) * z_clipped), 0, int(K)).astype(np.int64)

    unique_tokens, counts = np.unique(tokens, return_counts=True)
    dom_idx = int(np.argmax(counts))
    dominant_token = int(unique_tokens[dom_idx])
    dominant_count = int(counts[dom_idx])
    n = int(tokens.size)

    quantiles = {q_name(q): quantile_token(tokens, int(q)) for q in CFG.TOKEN_QUANTILES}

    return {
        "z_min": float(np.min(z)) if z.size else None,
        "z_max": float(np.max(z)) if z.size else None,
        "z_num_unique": int(np.unique(z).size),
        "num_tokens_used": int(unique_tokens.size),
        "dominant_token": dominant_token,
        "dominant_token_count": dominant_count,
        "dominant_token_ratio": float(dominant_count / max(n, 1)),
        "quantiles": quantiles,
        "spans": {
            "body_q10_q90": int(quantiles["q90"] - quantiles["q10"]),
            "middle_q25_q75": int(quantiles["q75"] - quantiles["q25"]),
            "left_min_q10": int(quantiles["q10"] - quantiles["min"]),
            "right_q90_max": int(quantiles["max"] - quantiles["q90"]),
        },
    }


def main() -> None:
    args = parse_args()
    K = int(args.K)

    if K <= 0:
        raise ValueError("K must be positive.")

    before_diag_path = Path(args.before_diag_json) if args.before_diag_json else CFG.token_diag_json_path(K)
    train_pre_path = Path(args.train_preprocessed) if args.train_preprocessed else CFG.preprocess_train_csv_path(K)
    policy_path = Path(args.policy_json) if args.policy_json else CFG.preprocess_policy_json_path(K)

    if not before_diag_path.exists():
        raise FileNotFoundError(
            f"before token diag not found: {before_diag_path}\n"
            f"Run: python -u final_pipeline/token_diag.py"
        )
    if not train_pre_path.exists():
        raise FileNotFoundError(
            f"train preprocessed csv not found: {train_pre_path}\n"
            f"Run: python -u final_pipeline/preprocessing.py"
        )
    if not policy_path.exists():
        raise FileNotFoundError(f"policy json not found: {policy_path}")

    before_by_feature = load_before_diag(before_diag_path)
    feature_order, policy_by_feature, policy_meta = load_policy(policy_path)

    train_pre = pd.read_csv(train_pre_path)

    missing = [f for f in feature_order if f not in train_pre.columns]
    if missing:
        raise ValueError(f"train_preprocessed missing features, first: {missing[:10]}")

    rows: List[Dict[str, object]] = []
    details: List[Dict[str, object]] = []

    threshold = float(policy_meta.get("unique_preserve_threshold", getattr(CFG, "UNIQUE_PRESERVE_THRESHOLD", 0.95)))

    for f in feature_order:
        if f not in before_by_feature:
            raise ValueError(f"before diag missing feature: {f}")

        before = before_by_feature[f]
        p = policy_by_feature[f]

        raw_unique = int(before["raw"]["num_unique"])
        possible_unique = int(min(raw_unique, K + 1))

        before_tokens_used = int(before["token"]["num_tokens_used"])
        before_ratio = float(before_tokens_used / max(possible_unique, 1))
        before_loss = int(max(possible_unique - before_tokens_used, 0))

        after = direct_z_token_diag(train_pre[f].to_numpy(dtype=float), K)
        after_tokens_used = int(after["num_tokens_used"])
        after_ratio = float(after_tokens_used / max(possible_unique, 1))
        after_loss = int(max(possible_unique - after_tokens_used, 0))

        action = str(p.get("action", "unknown"))

        row = {
            "feature": f,
            "action": action,
            "possible_unique": possible_unique,
            "raw_num_unique": raw_unique,
            "before_num_tokens_used": before_tokens_used,
            "after_num_tokens_used": after_tokens_used,
            "delta_tokens_used": int(after_tokens_used - before_tokens_used),
            "before_unique_preserve_ratio": before_ratio,
            "after_unique_preserve_ratio": after_ratio,
            "delta_unique_preserve_ratio": float(after_ratio - before_ratio),
            "before_collision_loss": before_loss,
            "after_collision_loss": after_loss,
            "delta_collision_loss": int(after_loss - before_loss),
            "before_dominant_token_ratio": float(before["token"]["dominant_token_ratio"]),
            "after_dominant_token_ratio": float(after["dominant_token_ratio"]),
            "before_below_threshold": bool(before_ratio < threshold),
            "after_below_threshold": bool(after_ratio < threshold),
        }

        if action == "piecewise":
            ca = p.get("collision_analysis", {})
            row["piecewise_side"] = ca.get("piecewise_side")
            row["collision_raw_interval"] = ca.get("collision_raw_interval")
        else:
            row["piecewise_side"] = None
            row["collision_raw_interval"] = None

        rows.append(row)

        details.append({
            **row,
            "before_token": before["token"],
            "after_token": after,
            "policy_reason": p.get("reason"),
        })

    df = pd.DataFrame(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"preprocess_token_compare_K{K}.json"
    out_csv = out_dir / f"preprocess_token_compare_K{K}.csv"

    action_counts = Counter([r["action"] for r in rows])

    piecewise_rows = [r for r in rows if r["action"] == "piecewise"]
    improved_piecewise = [r for r in piecewise_rows if r["delta_tokens_used"] > 0]
    unchanged_piecewise = [r for r in piecewise_rows if r["delta_tokens_used"] == 0]
    worsened_piecewise = [r for r in piecewise_rows if r["delta_tokens_used"] < 0]

    summary = {
        "stage": "preprocess_token_compare",
        "K": int(K),
        "before_diag_json": str(before_diag_path),
        "train_preprocessed_csv": str(train_pre_path),
        "policy_json": str(policy_path),
        "threshold": threshold,
        "n_features": int(len(rows)),
        "action_counts": dict(action_counts),
        "features_below_threshold_before": int(sum(r["before_below_threshold"] for r in rows)),
        "features_below_threshold_after": int(sum(r["after_below_threshold"] for r in rows)),
        "piecewise_count": int(len(piecewise_rows)),
        "piecewise_improved_count": int(len(improved_piecewise)),
        "piecewise_unchanged_count": int(len(unchanged_piecewise)),
        "piecewise_worsened_count": int(len(worsened_piecewise)),
        "top_token_improvements": sorted(
            [
                {
                    "feature": r["feature"],
                    "action": r["action"],
                    "delta_tokens_used": r["delta_tokens_used"],
                    "before_ratio": r["before_unique_preserve_ratio"],
                    "after_ratio": r["after_unique_preserve_ratio"],
                }
                for r in rows
            ],
            key=lambda x: x["delta_tokens_used"],
            reverse=True,
        )[:20],
        "outputs": {
            "json": str(out_json),
            "csv": str(out_csv),
        },
        "note": "After-preprocess diagnostics tokenize preprocessed z directly; no second MinMax is applied.",
    }

    result = {
        "summary": summary,
        "features": details,
    }

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)

    print("===== preprocess token compare done =====")
    print(f"K: {K}")
    print(f"features: {len(rows)}")
    print(f"action_counts: {dict(action_counts)}")
    print(f"below_threshold before: {summary['features_below_threshold_before']}")
    print(f"below_threshold after:  {summary['features_below_threshold_after']}")
    print(f"piecewise improved/unchanged/worsened: {summary['piecewise_improved_count']}/{summary['piecewise_unchanged_count']}/{summary['piecewise_worsened_count']}")
    print(f"json: {out_json}")
    print(f"csv:  {out_csv}")


if __name__ == "__main__":
    main()

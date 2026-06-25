#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_pipeline/token_diag.py

Train-only token diagnostic.

Reads train CSV only and outputs human-readable JSON:
- raw min/max/range/num_unique/zero_ratio/is_constant
- token num_tokens_used/dominant token/selected token quantiles

No validation.
No preprocessing decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

import config as CFG


def csv_list(value: str | None) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train-only token diagnostic JSON.")
    p.add_argument("--train-csv", default=str(CFG.TRAIN_CSV))
    p.add_argument("--out-dir", default=str(CFG.TOKEN_DIAG_DIR))
    p.add_argument("--K", type=int, default=int(CFG.TOKEN_K))
    p.add_argument("--target-cols", default=",".join(CFG.TARGET_COLS))
    p.add_argument("--drop-cols", default=",".join(CFG.DROP_COLS))
    return p.parse_args()


def detect_numeric_features(
    df: pd.DataFrame,
    target_cols: Sequence[str],
    drop_cols: Sequence[str],
) -> List[str]:
    excluded = set(target_cols) | set(drop_cols)
    excluded |= {c for c in df.columns if str(c).startswith("Unnamed:")}
    return [
        c for c in df.columns
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


def to_train_tokens(values: np.ndarray, K: int) -> tuple[np.ndarray, Dict[str, float]]:
    values = np.asarray(values, dtype=float)
    train_min = float(np.min(values))
    train_max = float(np.max(values))
    train_range = train_max - train_min

    if abs(train_range) <= 1e-12:
        z = np.zeros_like(values, dtype=float)
    else:
        z = (values - train_min) / train_range

    z = np.clip(z, 0.0, 1.0)
    tokens = np.rint(float(K) * z).astype(np.int64)
    tokens = np.clip(tokens, 0, int(K))

    return tokens, {
        "train_min": train_min,
        "train_max": train_max,
        "train_range": float(train_range),
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


def feature_diag(feature: str, values: np.ndarray, K: int) -> Dict[str, object]:
    values = np.asarray(values, dtype=float)
    tokens, fit = to_train_tokens(values, K)

    unique_tokens, counts = np.unique(tokens, return_counts=True)
    dominant_idx = int(np.argmax(counts))
    dominant_token = int(unique_tokens[dominant_idx])
    dominant_count = int(counts[dominant_idx])
    n = int(tokens.size)

    quantiles = {q_name(q): quantile_token(tokens, int(q)) for q in CFG.TOKEN_QUANTILES}

    raw_unique = int(np.unique(values).size)
    zero_ratio = float(np.mean(np.isclose(values, 0.0)))

    return {
        "feature": feature,
        "n": n,
        "K": int(K),
        "raw": {
            **fit,
            "num_unique": raw_unique,
            "zero_ratio": zero_ratio,
            "is_constant": bool(abs(fit["train_range"]) <= 1e-12),
        },
        "token": {
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
        },
    }


def build_output(train_csv: Path, K: int, target_cols: Sequence[str], drop_cols: Sequence[str]) -> Dict[str, object]:
    train = pd.read_csv(train_csv)
    features = detect_numeric_features(train, target_cols=target_cols, drop_cols=drop_cols)
    if not features:
        raise ValueError("No numeric features detected.")

    assert_finite(train, features)

    feature_rows = [
        feature_diag(f, train[f].to_numpy(dtype=float), K)
        for f in features
    ]

    return {
        "metadata": {
            "stage": "token_diag",
            "input_split": "train_only",
            "train_csv": str(train_csv),
            "K": int(K),
            "n_rows": int(len(train)),
            "n_features": int(len(features)),
            "excluded_target_cols": list(target_cols),
            "drop_cols": list(drop_cols),
            "token_rule": "train-only MinMax; z=(x-min)/(max-min); clipped [0,1]; token=round(K*z)",
            "quantiles": ["min", "q1", "q5"] + [f"q{i}" for i in range(10, 95, 5)] + ["q95", "q99", "max"],
            "how_to_read": {
                "num_tokens_used": "number of distinct tokens actually used by this feature",
                "raw.num_unique": "number of distinct raw values in train",
                "unique_preservation": "computed in preprocessing as num_tokens_used / min(raw.num_unique, K+1)",
                "spans": "reading aid only; spans describe distribution shape, not direct preprocessing decisions",
            },
        },
        "features": feature_rows,
    }


def main() -> None:
    args = parse_args()
    train_csv = Path(args.train_csv)
    out_dir = Path(args.out_dir)
    K = int(args.K)

    if K <= 0:
        raise ValueError("K must be positive.")
    if not train_csv.exists():
        raise FileNotFoundError(f"train csv not found: {train_csv}")

    target_cols = csv_list(args.target_cols)
    drop_cols = csv_list(args.drop_cols)

    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_output(train_csv=train_csv, K=K, target_cols=target_cols, drop_cols=drop_cols)

    out_k = out_dir / f"token_diag_train_K{K}.json"
    out_latest = out_dir / "token_diag_train.json"

    text = json.dumps(result, ensure_ascii=False, indent=2)
    out_k.write_text(text, encoding="utf-8")
    out_latest.write_text(text, encoding="utf-8")

    print("===== token_diag done =====")
    print(f"input: {train_csv}")
    print(f"K: {K}")
    print(f"features: {result['metadata']['n_features']}")
    print(f"output K-specific: {out_k}")
    print(f"output latest:     {out_latest}")


if __name__ == "__main__":
    main()

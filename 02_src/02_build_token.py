#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_pipeline/build_token.py

Build model-ready token dataset from preprocessed train/val CSVs.

Input:
    - train_preprocessed_K{K}.csv
    - val_preprocessed_K{K}.csv
    - preprocess_policy_K{K}.json for feature order

Output:
    - token_dataset_K{K}.npz
    - token_metadata_K{K}.json

This file does only:
    z in [0,1] -> token = round(K*z)

No preprocessing fit.
No embedding.
No model.
No test split.
"""

from __future__ import annotations

import argparse
import json
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
    p = argparse.ArgumentParser(description="Build token dataset from preprocessed train/val CSVs.")
    p.add_argument("--K", type=int, default=int(CFG.TOKEN_K))
    p.add_argument("--train-preprocessed", default="")
    p.add_argument("--val-preprocessed", default="")
    p.add_argument("--policy-json", default="")
    p.add_argument("--out-root", default=str(CFG.BUILD_TOKEN_DIR))
    p.add_argument("--label-col", default=str(CFG.DEFAULT_LABEL_COL))
    p.add_argument("--target-cols", default=",".join(CFG.TARGET_COLS))
    p.add_argument("--drop-cols", default=",".join(CFG.DROP_COLS))
    return p.parse_args()


def load_feature_order(policy_path: Path) -> List[str]:
    if not policy_path.exists():
        raise FileNotFoundError(f"preprocess policy not found: {policy_path}")

    obj = json.loads(policy_path.read_text(encoding="utf-8"))

    meta_features = obj.get("metadata", {}).get("feature_order")
    if meta_features:
        return [str(x) for x in meta_features]

    policies = obj.get("policies", [])
    if not policies:
        raise ValueError(f"No policies found in: {policy_path}")

    return [str(p["feature"]) for p in policies]


def validate_preprocessed_split(split_name: str, df: pd.DataFrame, features: Sequence[str], label_col: str) -> None:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"{split_name} missing {len(missing)} features; first: {missing[:10]}")

    if label_col not in df.columns:
        raise ValueError(f"{split_name} missing label column: {label_col}")

    arr = df.loc[:, list(features)].to_numpy(dtype=float)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count or inf_count:
        raise ValueError(f"{split_name} contains non-finite feature values: nan={nan_count}, inf={inf_count}")

    mn = float(np.min(arr)) if arr.size else 0.0
    mx = float(np.max(arr)) if arr.size else 0.0
    if mn < -1e-6 or mx > 1.0 + 1e-6:
        raise ValueError(
            f"{split_name} preprocessed features should be in [0,1], got min={mn}, max={mx}"
        )


def z_to_tokens(df: pd.DataFrame, features: Sequence[str], K: int) -> Tuple[np.ndarray, np.ndarray]:
    X_z = df.loc[:, list(features)].to_numpy(dtype=np.float32)
    X_z = np.clip(X_z, 0.0, 1.0)
    X_tokens = np.clip(np.rint(float(K) * X_z), 0, int(K)).astype(np.int64)
    return X_tokens, X_z


def build_label_mapping(train_labels: pd.Series) -> Dict[str, int]:
    labels = sorted([str(x) for x in train_labels.dropna().unique().tolist()])
    return {label: idx for idx, label in enumerate(labels)}


def encode_labels(labels: pd.Series, mapping: Dict[str, int], split_name: str, label_col: str) -> np.ndarray:
    encoded = []
    unknown = []
    for x in labels.tolist():
        key = str(x)
        if key not in mapping:
            unknown.append(key)
            encoded.append(-1)
        else:
            encoded.append(mapping[key])

    if unknown:
        raise ValueError(
            f"{split_name}.{label_col} contains labels not seen in train mapping: "
            f"{sorted(set(unknown))[:10]}"
        )

    return np.asarray(encoded, dtype=np.int64)


def main() -> None:
    args = parse_args()
    K = int(args.K)

    if K <= 0:
        raise ValueError("K must be positive.")

    train_pre = Path(args.train_preprocessed) if args.train_preprocessed else CFG.preprocess_train_csv_path(K)
    val_pre = Path(args.val_preprocessed) if args.val_preprocessed else CFG.preprocess_val_csv_path(K)
    policy_path = Path(args.policy_json) if args.policy_json else CFG.preprocess_policy_json_path(K)

    if not train_pre.exists():
        raise FileNotFoundError(f"train preprocessed csv not found: {train_pre}")
    if not val_pre.exists():
        raise FileNotFoundError(f"val preprocessed csv not found: {val_pre}")

    feature_names = load_feature_order(policy_path)

    train = pd.read_csv(train_pre)
    val = pd.read_csv(val_pre)

    label_col = str(args.label_col)

    validate_preprocessed_split("train", train, feature_names, label_col)
    validate_preprocessed_split("val", val, feature_names, label_col)

    X_train_tokens, X_train_z = z_to_tokens(train, feature_names, K)
    X_val_tokens, X_val_z = z_to_tokens(val, feature_names, K)

    label_mapping = build_label_mapping(train[label_col])
    y_train = encode_labels(train[label_col], label_mapping, "train", label_col)
    y_val = encode_labels(val[label_col], label_mapping, "val", label_col)

    out_dir = Path(args.out_root) / f"K{K}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / f"token_dataset_K{K}.npz"
    out_meta = out_dir / f"token_metadata_K{K}.json"

    np.savez_compressed(
        out_npz,
        X_train_tokens=X_train_tokens,
        X_train_z=X_train_z,
        y_train=y_train,
        X_val_tokens=X_val_tokens,
        X_val_z=X_val_z,
        y_val=y_val,
        feature_names=np.asarray(feature_names, dtype=object),
        label_names=np.asarray([k for k, _ in sorted(label_mapping.items(), key=lambda kv: kv[1])], dtype=object),
        K=np.asarray([K], dtype=np.int64),
    )

    metadata = {
        "stage": "build_token",
        "K": int(K),
        "input": {
            "train_preprocessed_csv": str(train_pre),
            "val_preprocessed_csv": str(val_pre),
            "preprocess_policy_json": str(policy_path),
        },
        "label_col": label_col,
        "label_mapping": label_mapping,
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "splits": {
            "train": {
                "n_rows": int(len(train)),
                "X_tokens_shape": list(X_train_tokens.shape),
                "X_z_shape": list(X_train_z.shape),
                "y_shape": list(y_train.shape),
                "token_min": int(X_train_tokens.min()) if X_train_tokens.size else None,
                "token_max": int(X_train_tokens.max()) if X_train_tokens.size else None,
            },
            "val": {
                "n_rows": int(len(val)),
                "X_tokens_shape": list(X_val_tokens.shape),
                "X_z_shape": list(X_val_z.shape),
                "y_shape": list(y_val.shape),
                "token_min": int(X_val_tokens.min()) if X_val_tokens.size else None,
                "token_max": int(X_val_tokens.max()) if X_val_tokens.size else None,
            },
        },
        "outputs": {
            "token_dataset_npz": str(out_npz),
            "token_metadata_json": str(out_meta),
        },
        "note": "This file only converts preprocessed z values to integer token IDs. Embedding/model are separate files.",
    }

    out_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== build_token done =====")
    print(f"K: {K}")
    print(f"features: {len(feature_names)}")
    print(f"train: {X_train_tokens.shape}")
    print(f"val:   {X_val_tokens.shape}")
    print(f"dataset:  {out_npz}")
    print(f"metadata: {out_meta}")


if __name__ == "__main__":
    main()

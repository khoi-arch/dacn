import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLS = {"label_L1", "label_L2", "label_L3", "Class", "Category"}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def as_str_list(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def get_feature_names(template_npz, train_df):
    if "feature_names" in template_npz.files:
        return as_str_list(template_npz["feature_names"])
    return [
        c for c in train_df.columns
        if c not in TARGET_COLS and pd.api.types.is_numeric_dtype(train_df[c])
    ]


def uniform_bin_offset(z, num_bins):
    z = np.asarray(z, dtype=np.float64)
    z = np.nan_to_num(z, nan=0.0, posinf=1.0, neginf=0.0)
    z = np.clip(z, 0.0, 1.0)

    scaled = z * float(num_bins)
    b = np.floor(scaled).astype(np.int64)
    b = np.clip(b, 0, num_bins - 1)

    off = scaled - b.astype(np.float64)

    # z == 1.0 should be final bin with offset 1.0, not bin num_bins offset 0
    final = z >= 1.0
    off[final] = 1.0

    off = np.clip(off, 0.0, 1.0).astype(np.float32)
    return b.astype(np.int64), off


def build_uniform_only(
    *,
    train_preprocessed: Path,
    val_preprocessed: Path,
    template_npz_path: Path,
    template_meta_path: Path,
    out_dir: Path,
    K: int,
    num_bins: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_preprocessed)
    val_df = pd.read_csv(val_preprocessed)

    with np.load(template_npz_path, allow_pickle=True) as data:
        template = {k: data[k] for k in data.files}
        feature_names = get_feature_names(data, train_df)

    n_train = len(train_df)
    n_val = len(val_df)
    n_features = len(feature_names)

    X_train_bin = np.zeros((n_train, n_features), dtype=np.int64)
    X_val_bin = np.zeros((n_val, n_features), dtype=np.int64)
    X_train_offset = np.zeros((n_train, n_features), dtype=np.float32)
    X_val_offset = np.zeros((n_val, n_features), dtype=np.float32)

    strategies = {}
    constant_features = []

    for j, feat in enumerate(feature_names):
        tr = train_df[feat].to_numpy(dtype=np.float64)
        va = val_df[feat].to_numpy(dtype=np.float64)

        uniq = np.unique(tr[~np.isnan(tr)]).size
        if uniq <= 1:
            strategies[feat] = "constant"
            constant_features.append(feat)
            X_train_bin[:, j] = 0
            X_val_bin[:, j] = 0
            X_train_offset[:, j] = 0.0
            X_val_offset[:, j] = 0.0
        else:
            strategies[feat] = "uniform_offset"
            bt, ot = uniform_bin_offset(tr, num_bins)
            bv, ov = uniform_bin_offset(va, num_bins)
            X_train_bin[:, j] = bt
            X_val_bin[:, j] = bv
            X_train_offset[:, j] = ot
            X_val_offset[:, j] = ov

    out_arrays = dict(template)
    out_arrays["X_train_bin"] = X_train_bin
    out_arrays["X_val_bin"] = X_val_bin
    out_arrays["X_train_offset"] = X_train_offset
    out_arrays["X_val_offset"] = X_val_offset

    np.savez_compressed(out_dir / "mixed_quantile_offset_dataset.npz", **out_arrays)

    meta = load_json(template_meta_path)
    meta["stage"] = "quantile_policy_ablation"
    meta["policy_name"] = "uniform_only"
    meta["K"] = K
    meta["num_bins"] = num_bins
    meta["source_template_npz"] = str(template_npz_path)
    meta["strategy_counts"] = {
        "uniform_offset": int(n_features - len(constant_features)),
        "quantile_offset": 0,
        "constant": int(len(constant_features)),
    }
    meta["uniform_features"] = [f for f in feature_names if strategies[f] == "uniform_offset"]
    meta["quantile_features"] = []
    meta["constant_features"] = constant_features
    meta["feature_strategies"] = strategies
    meta["splits"] = {
        "train": {
            "n_rows": int(n_train),
            "X_bin_shape": list(X_train_bin.shape),
            "X_offset_shape": list(X_train_offset.shape),
            "bin_min": int(X_train_bin.min()),
            "bin_max": int(X_train_bin.max()),
            "offset_min": float(X_train_offset.min()),
            "offset_max": float(X_train_offset.max()),
        },
        "val": {
            "n_rows": int(n_val),
            "X_bin_shape": list(X_val_bin.shape),
            "X_offset_shape": list(X_val_offset.shape),
            "bin_min": int(X_val_bin.min()),
            "bin_max": int(X_val_bin.max()),
            "offset_min": float(X_val_offset.min()),
            "offset_max": float(X_val_offset.max()),
        },
    }

    save_json(meta, out_dir / "mixed_quantile_offset_metadata.json")
    return meta


def infer_quantile_features(meta):
    for key in ["quantile_features", "selected_quantile_features"]:
        if key in meta and isinstance(meta[key], list):
            return [str(x) for x in meta[key]]

    out = []
    fs = meta.get("feature_strategies", None)
    if isinstance(fs, dict):
        for feat, strat in fs.items():
            if "quantile" in str(strat):
                out.append(str(feat))

    if not out:
        feats = meta.get("features", [])
        if isinstance(feats, list):
            for row in feats:
                if isinstance(row, dict):
                    feat = row.get("feature") or row.get("name")
                    strat = row.get("strategy") or row.get("selected_strategy")
                    if feat and strat and "quantile" in str(strat):
                        out.append(str(feat))

    return out


def build_quantile_no_offset(
    *,
    template_npz_path: Path,
    template_meta_path: Path,
    out_dir: Path,
    K: int,
    num_bins: int,
    fill_value: float = 0.5,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_json(template_meta_path)

    with np.load(template_npz_path, allow_pickle=True) as data:
        out_arrays = {k: data[k] for k in data.files}
        feature_names = as_str_list(data["feature_names"]) if "feature_names" in data.files else None

    if feature_names is None:
        fs = meta.get("feature_strategies", {})
        if isinstance(fs, dict) and fs:
            feature_names = list(fs.keys())
        else:
            raise RuntimeError("Cannot infer feature_names from npz or metadata.")

    q_features = infer_quantile_features(meta)
    q_set = set(q_features)
    q_indices = [i for i, f in enumerate(feature_names) if f in q_set]

    X_train_offset = np.array(out_arrays["X_train_offset"], copy=True)
    X_val_offset = np.array(out_arrays["X_val_offset"], copy=True)

    for j in q_indices:
        X_train_offset[:, j] = float(fill_value)
        X_val_offset[:, j] = float(fill_value)

    out_arrays["X_train_offset"] = X_train_offset.astype(np.float32)
    out_arrays["X_val_offset"] = X_val_offset.astype(np.float32)

    np.savez_compressed(out_dir / "mixed_quantile_offset_dataset.npz", **out_arrays)

    new_meta = dict(meta)
    new_meta["stage"] = "quantile_policy_ablation"
    new_meta["policy_name"] = "quantile_no_offset"
    new_meta["K"] = K
    new_meta["num_bins"] = num_bins
    new_meta["source_template_npz"] = str(template_npz_path)
    new_meta["quantile_offset_fill_value"] = float(fill_value)
    new_meta["quantile_features_offset_disabled"] = q_features
    new_meta["n_quantile_features_offset_disabled"] = len(q_indices)
    new_meta["note"] = (
        "Keeps mixed quantile/uniform bin_id unchanged, but replaces offset with "
        "a constant value for quantile features to test whether within-quantile-bin "
        "offset geometry is harmful."
    )

    save_json(new_meta, out_dir / "mixed_quantile_offset_metadata.json")
    return new_meta


def zip_outputs(paths, out_zip):
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            p = Path(p)
            if p.is_file():
                z.write(p, p.as_posix())
            elif p.is_dir():
                for fp in p.rglob("*"):
                    if fp.is_file():
                        z.write(fp, fp.as_posix())


def main():
    K = 256
    B = 256

    train_pre = Path(f"03_outputs/preprocessing/train_preprocessed_K{K}.csv")
    val_pre = Path(f"03_outputs/preprocessing/val_preprocessed_K{K}.csv")

    base_dir = Path(f"03_outputs/build_mixed_quantile_offset/K{K}_B{B}")
    template_npz = base_dir / "mixed_quantile_offset_dataset.npz"
    template_meta = base_dir / "mixed_quantile_offset_metadata.json"

    if not train_pre.exists():
        raise FileNotFoundError(train_pre)
    if not val_pre.exists():
        raise FileNotFoundError(val_pre)
    if not template_npz.exists():
        raise FileNotFoundError(template_npz)
    if not template_meta.exists():
        raise FileNotFoundError(template_meta)

    uniform_dir = Path(f"03_outputs/build_mixed_quantile_offset/K{K}_B{B}_uniform_only")
    qno_dir = Path(f"03_outputs/build_mixed_quantile_offset/K{K}_B{B}_quantile_no_offset")

    print("[1/2] Building uniform-only artifact...")
    uniform_meta = build_uniform_only(
        train_preprocessed=train_pre,
        val_preprocessed=val_pre,
        template_npz_path=template_npz,
        template_meta_path=template_meta,
        out_dir=uniform_dir,
        K=K,
        num_bins=B,
    )

    print("[2/2] Building quantile-no-offset artifact...")
    qno_meta = build_quantile_no_offset(
        template_npz_path=template_npz,
        template_meta_path=template_meta,
        out_dir=qno_dir,
        K=K,
        num_bins=B,
        fill_value=0.5,
    )

    summary = {
        "K": K,
        "num_bins": B,
        "base_artifact": str(base_dir),
        "uniform_only_artifact": str(uniform_dir),
        "quantile_no_offset_artifact": str(qno_dir),
        "uniform_only_strategy_counts": uniform_meta.get("strategy_counts"),
        "quantile_no_offset_disabled_features": qno_meta.get("quantile_features_offset_disabled", []),
        "quantile_offset_fill_value": 0.5,
    }

    summary_path = Path("03_outputs/build_mixed_quantile_offset/K256_B256_policy_ablation_summary.json")
    save_json(summary, summary_path)

    out_zip = Path("K256_quantile_policy_ablation_artifacts.zip")
    zip_outputs([uniform_dir, qno_dir, summary_path], out_zip)

    print("\nDone.")
    print("Uniform-only:", uniform_dir)
    print("Quantile-no-offset:", qno_dir)
    print("Summary:", summary_path)
    print("Zip:", out_zip.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

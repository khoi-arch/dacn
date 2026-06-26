import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLS = {"label_L1", "label_L2", "label_L3", "Class", "Category"}


def entropy_norm_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    h = -np.sum(p * np.log(p + 1e-12))
    return float(h / np.log(counts.size))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def as_str_list(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def uniform_bin_offset(z, num_bins):
    z = np.asarray(z, dtype=np.float64)
    z = np.nan_to_num(z, nan=0.0, posinf=1.0, neginf=0.0)
    z = np.clip(z, 0.0, 1.0)

    scaled = z * float(num_bins)
    b = np.floor(scaled).astype(np.int64)
    b = np.clip(b, 0, num_bins - 1)

    off = scaled - b.astype(np.float64)

    final = z >= 1.0
    off[final] = 1.0

    return b.astype(np.int64), np.clip(off, 0.0, 1.0).astype(np.float32)


def piecewise_unique_rank_fit_transform(train_values, val_values):
    train_values = np.asarray(train_values, dtype=np.float64)
    val_values = np.asarray(val_values, dtype=np.float64)

    train_values = np.nan_to_num(train_values, nan=0.0, posinf=np.nanmax(train_values[np.isfinite(train_values)]) if np.isfinite(train_values).any() else 0.0, neginf=np.nanmin(train_values[np.isfinite(train_values)]) if np.isfinite(train_values).any() else 0.0)
    val_values = np.nan_to_num(val_values, nan=0.0, posinf=np.nanmax(train_values) if train_values.size else 0.0, neginf=np.nanmin(train_values) if train_values.size else 0.0)

    uniq = np.unique(train_values)

    if uniq.size <= 1:
        return np.zeros_like(train_values, dtype=np.float64), np.zeros_like(val_values, dtype=np.float64), uniq

    ranks = np.linspace(0.0, 1.0, uniq.size, dtype=np.float64)

    # train exact mapping via searchsorted because values are from uniq
    idx = np.searchsorted(uniq, train_values, side="left")
    idx = np.clip(idx, 0, uniq.size - 1)
    z_train = ranks[idx]

    # val piecewise interpolation over train unique values
    z_val = np.interp(val_values, uniq, ranks, left=0.0, right=1.0)

    return z_train, z_val, uniq


def feature_diag(feature, raw_train, z_train, bin_ids, num_bins):
    counts = np.bincount(bin_ids, minlength=num_bins)
    used = int(np.count_nonzero(counts))
    n = int(len(raw_train))
    raw_unique = int(np.unique(raw_train).size)

    rare_1 = int(np.sum(counts == 1))
    rare_5 = int(np.sum((counts > 0) & (counts <= 5)))
    rare_10 = int(np.sum((counts > 0) & (counts <= 10)))

    return {
        "feature": feature,
        "strategy": "rank_uniform_offset",
        "n": n,
        "raw_unique": raw_unique,
        "bins_used": used,
        "empty_bins": int(num_bins - used),
        "empty_bin_ratio": float((num_bins - used) / max(num_bins, 1)),
        "rare_bins_count_eq_1": rare_1,
        "rare_bins_count_le_5": rare_5,
        "rare_bins_count_le_10": rare_10,
        "rare_used_bin_ratio_le_5": float(rare_5 / max(used, 1)),
        "rare_used_bin_ratio_le_10": float(rare_10 / max(used, 1)),
        "dominant_bin_ratio": float(counts.max() / max(n, 1)),
        "entropy_norm": entropy_norm_from_counts(counts),
        "compression_factor": float(raw_unique / max(used, 1)),
        "z_min": float(np.min(z_train)) if n else None,
        "z_max": float(np.max(z_train)) if n else None,
        "uniform_transformed_bin_width": float(1.0 / num_bins),
    }


def zip_paths(paths, out_zip):
    out_zip = Path(out_zip)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for item in paths:
            item = Path(item)
            if item.is_file():
                z.write(item, item.as_posix())
            elif item.is_dir():
                for fp in item.rglob("*"):
                    if fp.is_file():
                        z.write(fp, fp.as_posix())


def main():
    K = 512
    B = 512

    train_raw_path = Path("01_split/train_raw.csv")
    val_raw_path = Path("01_split/val_raw.csv")

    current_dir = Path(f"03_outputs/build_mixed_quantile_offset/K{K}_B{B}")
    template_npz_path = current_dir / "mixed_quantile_offset_dataset.npz"
    template_meta_path = current_dir / "mixed_quantile_offset_metadata.json"

    if not train_raw_path.exists():
        raise FileNotFoundError(train_raw_path)
    if not val_raw_path.exists():
        raise FileNotFoundError(val_raw_path)
    if not template_npz_path.exists():
        raise FileNotFoundError(template_npz_path)
    if not template_meta_path.exists():
        raise FileNotFoundError(template_meta_path)

    train_df = pd.read_csv(train_raw_path)
    val_df = pd.read_csv(val_raw_path)

    with np.load(template_npz_path, allow_pickle=True) as data:
        template = {k: data[k] for k in data.files}
        if "feature_names" in data.files:
            feature_names = as_str_list(data["feature_names"])
        else:
            feature_names = [
                c for c in train_df.columns
                if c not in TARGET_COLS and pd.api.types.is_numeric_dtype(train_df[c])
            ]

    n_train = len(train_df)
    n_val = len(val_df)
    n_features = len(feature_names)

    X_train_bin = np.zeros((n_train, n_features), dtype=np.int64)
    X_val_bin = np.zeros((n_val, n_features), dtype=np.int64)
    X_train_offset = np.zeros((n_train, n_features), dtype=np.float32)
    X_val_offset = np.zeros((n_val, n_features), dtype=np.float32)

    strategies = {}
    rows = []
    constant_features = []

    for j, feat in enumerate(feature_names):
        tr_raw = train_df[feat].to_numpy(dtype=np.float64)
        va_raw = val_df[feat].to_numpy(dtype=np.float64)

        uniq = np.unique(tr_raw[np.isfinite(tr_raw)]).size

        if uniq <= 1:
            strategies[feat] = "constant"
            constant_features.append(feat)
            X_train_bin[:, j] = 0
            X_val_bin[:, j] = 0
            X_train_offset[:, j] = 0.0
            X_val_offset[:, j] = 0.0

            rows.append({
                "feature": feat,
                "strategy": "constant",
                "n": int(n_train),
                "raw_unique": int(uniq),
                "bins_used": 1,
                "empty_bins": int(B - 1),
                "empty_bin_ratio": float((B - 1) / B),
                "rare_bins_count_eq_1": 0,
                "rare_bins_count_le_5": 0,
                "rare_bins_count_le_10": 0,
                "rare_used_bin_ratio_le_5": 0.0,
                "rare_used_bin_ratio_le_10": 0.0,
                "dominant_bin_ratio": 1.0,
                "entropy_norm": 0.0,
                "compression_factor": float(uniq),
                "z_min": 0.0,
                "z_max": 0.0,
                "uniform_transformed_bin_width": float(1.0 / B),
            })
            continue

        strategies[feat] = "rank_uniform_offset"

        z_tr, z_va, _ = piecewise_unique_rank_fit_transform(tr_raw, va_raw)

        bt, ot = uniform_bin_offset(z_tr, B)
        bv, ov = uniform_bin_offset(z_va, B)

        X_train_bin[:, j] = bt
        X_val_bin[:, j] = bv
        X_train_offset[:, j] = ot
        X_val_offset[:, j] = ov

        rows.append(feature_diag(feat, tr_raw, z_tr, bt, B))

    out_dir = Path(f"03_outputs/build_mixed_quantile_offset/K{K}_B{B}_rank_uniform_only")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_arrays = dict(template)
    out_arrays["X_train_bin"] = X_train_bin
    out_arrays["X_val_bin"] = X_val_bin
    out_arrays["X_train_offset"] = X_train_offset
    out_arrays["X_val_offset"] = X_val_offset

    np.savez_compressed(out_dir / "mixed_quantile_offset_dataset.npz", **out_arrays)

    current_meta = load_json(template_meta_path)

    meta = dict(current_meta)
    meta["stage"] = "rank_uniform_policy_ablation"
    meta["policy_name"] = "rank_uniform_only"
    meta["K"] = K
    meta["num_bins"] = B
    meta["source_raw_train"] = str(train_raw_path)
    meta["source_raw_val"] = str(val_raw_path)
    meta["source_template_npz"] = str(template_npz_path)
    meta["strategy_counts"] = {
        "rank_uniform_offset": int(n_features - len(constant_features)),
        "uniform_offset": 0,
        "quantile_offset": 0,
        "constant": int(len(constant_features)),
    }
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

    diag_dir = Path("03_outputs/bin_diag")
    diag_dir.mkdir(parents=True, exist_ok=True)

    diag_csv = diag_dir / f"rank_uniform_token_diag_K{K}_B{B}.csv"
    diag_json = diag_dir / f"rank_uniform_token_diag_K{K}_B{B}.json"

    df_diag = pd.DataFrame(rows)
    df_diag.to_csv(diag_csv, index=False)

    nonconst = df_diag[df_diag["strategy"] != "constant"]

    summary = {
        "stage": "rank_uniform_token_diag",
        "policy_name": "rank_uniform_only",
        "K": K,
        "num_bins": B,
        "n_features": int(n_features),
        "n_constant": int(len(constant_features)),
        "n_nonconstant": int(len(nonconst)),
        "mean_bins_used_nonconstant": float(nonconst["bins_used"].mean()),
        "median_bins_used_nonconstant": float(nonconst["bins_used"].median()),
        "mean_empty_bin_ratio_nonconstant": float(nonconst["empty_bin_ratio"].mean()),
        "mean_compression_nonconstant": float(nonconst["compression_factor"].mean()),
        "median_compression_nonconstant": float(nonconst["compression_factor"].median()),
        "mean_entropy_nonconstant": float(nonconst["entropy_norm"].mean()),
        "mean_dominant_bin_ratio_nonconstant": float(nonconst["dominant_bin_ratio"].mean()),
        "features_full_512_bins": int((nonconst["bins_used"] == B).sum()),
        "features_bins_used_ge_400": int((nonconst["bins_used"] >= 400).sum()),
        "features_bins_used_lt_128": int((nonconst["bins_used"] < 128).sum()),
        "features_rare_ratio_le_5_gt_0_2": int((nonconst["rare_used_bin_ratio_le_5"] > 0.2).sum()),
        "top_compression_features": nonconst.sort_values("compression_factor", ascending=False).head(20).to_dict(orient="records"),
        "top_dominant_features": nonconst.sort_values("dominant_bin_ratio", ascending=False).head(20).to_dict(orient="records"),
        "top_rare_features": nonconst.sort_values("rare_used_bin_ratio_le_5", ascending=False).head(20).to_dict(orient="records"),
    }

    save_json({"summary": summary, "features": rows}, diag_json)

    summary_path = Path(f"03_outputs/build_mixed_quantile_offset/K{K}_A_current_vs_B_rank_uniform_summary.json")
    save_json({
        "A_current_mixed_artifact": str(current_dir),
        "B_rank_uniform_artifact": str(out_dir),
        "B_rank_uniform_diag_csv": str(diag_csv),
        "B_rank_uniform_diag_json": str(diag_json),
        "B_summary": summary,
    }, summary_path)

    out_zip = Path("K512_A_current_and_B_rank_uniform_artifacts.zip")
    zip_paths([
        current_dir,
        out_dir,
        diag_csv,
        diag_json,
        summary_path,
    ], out_zip)

    print("Done.")
    print("A current:", current_dir)
    print("B rank-uniform:", out_dir)
    print("B diag:", diag_json)
    print("Zip:", out_zip.resolve())
    print(json.dumps(summary, indent=2)[:5000])


if __name__ == "__main__":
    main()

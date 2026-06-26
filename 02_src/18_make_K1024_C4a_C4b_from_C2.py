import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLS = {"label_L1", "label_L2", "label_L3", "Class", "Category"}


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


def numeric_features(df):
    return [
        c for c in df.columns
        if c not in TARGET_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def nearest_unique_index(values, uniq):
    values = np.asarray(values, dtype=np.float64)
    uniq = np.asarray(uniq, dtype=np.float64)

    idx = np.searchsorted(uniq, values, side="left")
    idx = np.clip(idx, 0, len(uniq) - 1)

    left_idx = np.clip(idx - 1, 0, len(uniq) - 1)
    right_idx = idx

    left_dist = np.abs(values - uniq[left_idx])
    right_dist = np.abs(values - uniq[right_idx])

    choose_left = left_dist < right_dist
    out = np.where(choose_left, left_idx, right_idx)
    return out.astype(np.int64)


def rank_uniform_bins(train_values, val_values, K):
    train_values = np.asarray(train_values, dtype=np.float64)
    val_values = np.asarray(val_values, dtype=np.float64)

    finite = train_values[np.isfinite(train_values)]

    if finite.size == 0:
        train_values = np.zeros_like(train_values, dtype=np.float64)
        val_values = np.zeros_like(val_values, dtype=np.float64)
        uniq = np.array([0.0], dtype=np.float64)
    else:
        fill = float(np.median(finite))
        mn = float(np.min(finite))
        mx = float(np.max(finite))
        train_values = np.nan_to_num(train_values, nan=fill, posinf=mx, neginf=mn)
        val_values = np.nan_to_num(val_values, nan=fill, posinf=mx, neginf=mn)
        uniq = np.unique(train_values)

    if uniq.size <= 1:
        ztr = np.zeros_like(train_values, dtype=np.float64)
        zva = np.zeros_like(val_values, dtype=np.float64)
    else:
        tr_idx = nearest_unique_index(train_values, uniq)
        va_idx = nearest_unique_index(val_values, uniq)
        ztr = tr_idx.astype(np.float64) / float(uniq.size - 1)
        zva = va_idx.astype(np.float64) / float(uniq.size - 1)

    tr_pos = ztr * K
    va_pos = zva * K

    tr_bin = np.floor(tr_pos).astype(np.int64)
    va_bin = np.floor(va_pos).astype(np.int64)

    tr_bin = np.clip(tr_bin, 0, K - 1)
    va_bin = np.clip(va_bin, 0, K - 1)

    tr_off = (tr_pos - tr_bin).astype(np.float32)
    va_off = (va_pos - va_bin).astype(np.float32)

    tr_off = np.clip(tr_off, 0.0, 1.0)
    va_off = np.clip(va_off, 0.0, 1.0)

    return tr_bin, tr_off, va_bin, va_off


def rebin_from_old_bin_offset(old_bin, old_offset, old_K, new_K):
    old_bin = np.asarray(old_bin, dtype=np.float64)
    old_offset = np.asarray(old_offset, dtype=np.float64)

    z = (old_bin + old_offset) / float(old_K)
    z = np.clip(z, 0.0, 1.0)

    pos = z * new_K
    new_bin = np.floor(pos).astype(np.int64)
    new_bin = np.clip(new_bin, 0, new_K - 1)

    new_offset = (pos - new_bin).astype(np.float32)
    new_offset = np.clip(new_offset, 0.0, 1.0)

    return new_bin, new_offset


def token_summary(X_bin, X_offset, features, strategies, K):
    rows = []
    for j, feat in enumerate(features):
        bins = X_bin[:, j].astype(np.int64)
        offs = X_offset[:, j].astype(np.float64)
        counts = np.bincount(bins, minlength=K)

        used = int(np.count_nonzero(counts))
        rare5 = int(np.sum((counts > 0) & (counts <= 5)))

        rows.append({
            "feature": feat,
            "strategy": strategies.get(feat, "UNKNOWN"),
            "bins_used": used,
            "empty_ratio": float((K - used) / K),
            "rare_used_bin_ratio_le5": float(rare5 / max(used, 1)),
            "dominant_bin_ratio": float(counts.max() / max(counts.sum(), 1)),
            "offset_unique_count": int(np.unique(offs).size),
            "offset_nonzero_ratio": float(np.mean(np.abs(offs) > 1e-12)),
            "offset_std": float(np.std(offs)),
            "bin_min": int(np.min(bins)),
            "bin_max": int(np.max(bins)),
        })
    return rows


def zip_paths(paths, out_zip):
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
    old_K = 512
    new_K = 1024

    train_raw_path = Path("01_split/train_raw.csv")
    val_raw_path = Path("01_split/val_raw.csv")

    C2_dir = Path("03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact")
    C2_npz_path = C2_dir / "mixed_quantile_offset_dataset.npz"
    C2_meta_path = C2_dir / "mixed_quantile_offset_metadata.json"

    for p in [train_raw_path, val_raw_path, C2_npz_path, C2_meta_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    train_df = pd.read_csv(train_raw_path)
    val_df = pd.read_csv(val_raw_path)

    with np.load(C2_npz_path, allow_pickle=True) as data:
        C2 = {k: np.array(data[k], copy=True) for k in data.files}
        if "feature_names" in data.files:
            feature_names = as_str_list(data["feature_names"])
        else:
            feature_names = numeric_features(train_df)

    C2_meta = load_json(C2_meta_path)
    C2_strategies = C2_meta.get("feature_strategies", {})
    if not isinstance(C2_strategies, dict):
        C2_strategies = {}

    C4a = {k: np.array(v, copy=True) for k, v in C2.items()}
    C4b = {k: np.array(v, copy=True) for k, v in C2.items()}

    C4a_strategies = dict(C2_strategies)
    C4b_strategies = dict(C2_strategies)

    changed_a = []
    changed_b = []

    for j, feat in enumerate(feature_names):
        strategy = C2_strategies.get(feat, "UNKNOWN")

        tr_raw = train_df[feat].to_numpy(dtype=np.float64)
        va_raw = val_df[feat].to_numpy(dtype=np.float64)

        if strategy == "rank_uniform_offset":
            tr_b, tr_o, va_b, va_o = rank_uniform_bins(tr_raw, va_raw, new_K)

            C4a["X_train_bin"][:, j] = tr_b
            C4a["X_train_offset"][:, j] = tr_o
            C4a["X_val_bin"][:, j] = va_b
            C4a["X_val_offset"][:, j] = va_o
            C4a_strategies[feat] = "rank_uniform_offset_K1024"
            changed_a.append(feat)

            C4b["X_train_bin"][:, j] = tr_b
            C4b["X_train_offset"][:, j] = tr_o
            C4b["X_val_bin"][:, j] = va_b
            C4b["X_val_offset"][:, j] = va_o
            C4b_strategies[feat] = "rank_uniform_offset_K1024"
            changed_b.append(feat)

        elif strategy == "keep_current":
            # C4a: keep exactly C2 bin/offset.
            # C4b: keep same continuous position from C2, but rebin from K512 to K1024.
            tr_b, tr_o = rebin_from_old_bin_offset(
                C2["X_train_bin"][:, j],
                C2["X_train_offset"][:, j],
                old_K,
                new_K,
            )
            va_b, va_o = rebin_from_old_bin_offset(
                C2["X_val_bin"][:, j],
                C2["X_val_offset"][:, j],
                old_K,
                new_K,
            )

            C4b["X_train_bin"][:, j] = tr_b
            C4b["X_train_offset"][:, j] = tr_o
            C4b["X_val_bin"][:, j] = va_b
            C4b["X_val_offset"][:, j] = va_o
            C4b_strategies[feat] = "keep_current_rebin_K1024"
            changed_b.append(feat)

        elif strategy == "discrete_compact_offset0":
            # Keep compact exactly as C2.
            C4a_strategies[feat] = "discrete_compact_offset0"
            C4b_strategies[feat] = "discrete_compact_offset0"

        elif strategy == "constant":
            C4a_strategies[feat] = "constant"
            C4b_strategies[feat] = "constant"

    out_a_dir = Path("03_outputs/build_mixed_quantile_offset/K1024_B1024_C4a_rank1024_rest_C2")
    out_b_dir = Path("03_outputs/build_mixed_quantile_offset/K1024_B1024_C4b_rank1024_current_rebin1024")
    out_a_dir.mkdir(parents=True, exist_ok=True)
    out_b_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_a_dir / "mixed_quantile_offset_dataset.npz", **C4a)
    np.savez_compressed(out_b_dir / "mixed_quantile_offset_dataset.npz", **C4b)

    diag_dir = Path("03_outputs/bin_diag")
    diag_dir.mkdir(parents=True, exist_ok=True)

    rows_a = token_summary(C4a["X_train_bin"], C4a["X_train_offset"], feature_names, C4a_strategies, new_K)
    rows_b = token_summary(C4b["X_train_bin"], C4b["X_train_offset"], feature_names, C4b_strategies, new_K)

    df_a = pd.DataFrame(rows_a)
    df_b = pd.DataFrame(rows_b)
    df_a["variant"] = "C4a_rank1024_rest_C2"
    df_b["variant"] = "C4b_rank1024_current_rebin1024"

    diag_csv = diag_dir / "K1024_C4a_C4b_policy_diag.csv"
    diag_json = diag_dir / "K1024_C4a_C4b_policy_diag.json"

    diag_df = pd.concat([df_a, df_b], ignore_index=True)
    diag_df.to_csv(diag_csv, index=False)

    def counts(d):
        out = {}
        for v in d.values():
            out[v] = out.get(v, 0) + 1
        return out

    summary = {
        "old_K": old_K,
        "new_K": new_K,
        "source_C2": str(C2_dir),
        "C4a": {
            "description": "C2 baseline, rank_uniform features recomputed at K1024; compact/current/constant kept as C2.",
            "output_dir": str(out_a_dir),
            "changed_features": changed_a,
            "strategy_counts": counts(C4a_strategies),
        },
        "C4b": {
            "description": "C2 baseline, rank_uniform features recomputed at K1024; keep_current rebinned from C2 continuous position to K1024; compact unchanged.",
            "output_dir": str(out_b_dir),
            "changed_features": changed_b,
            "strategy_counts": counts(C4b_strategies),
        },
    }

    save_json({"summary": summary, "features": diag_df.to_dict(orient="records")}, diag_json)

    def make_meta(policy_name, strategies, note, changed):
        meta = dict(C2_meta)
        meta["stage"] = "C4_K1024_resolution_ablation"
        meta["policy_name"] = policy_name
        meta["K"] = new_K
        meta["num_bins"] = new_K
        meta["source_C2_K512"] = str(C2_dir)
        meta["feature_strategies"] = strategies
        meta["strategy_counts"] = counts(strategies)
        meta["changed_features"] = changed
        meta["source_note"] = note
        meta["policy_diag_csv"] = str(diag_csv)
        meta["policy_diag_json"] = str(diag_json)
        return meta

    save_json(
        make_meta(
            "C4a_rank1024_rest_C2",
            C4a_strategies,
            "Only C2 rank_uniform features are recomputed at K1024. Other features keep C2 bin/offset.",
            changed_a,
        ),
        out_a_dir / "mixed_quantile_offset_metadata.json",
    )

    save_json(
        make_meta(
            "C4b_rank1024_current_rebin1024",
            C4b_strategies,
            "C2 rank_uniform features are recomputed at K1024. C2 keep_current features are rebinned from old continuous bin+offset position to K1024. Compact features unchanged.",
            changed_b,
        ),
        out_b_dir / "mixed_quantile_offset_metadata.json",
    )

    summary_path = Path("03_outputs/build_mixed_quantile_offset/K1024_C4a_C4b_summary.json")
    save_json(summary, summary_path)

    out_zip = Path("K1024_C4a_C4b_artifacts.zip")
    zip_paths([out_a_dir, out_b_dir, diag_csv, diag_json, summary_path], out_zip)

    print("Done.")
    print("C4a:", out_a_dir)
    print("C4b:", out_b_dir)
    print("Diag:", diag_json)
    print("Zip:", out_zip.resolve())
    print(json.dumps(summary, indent=2)[:5000])


if __name__ == "__main__":
    main()

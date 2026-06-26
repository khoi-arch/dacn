import json
import zipfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


TARGET_COLS = {"label_L1", "label_L2", "label_L3", "Class", "Category"}


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def as_str_list(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def entropy_norm(counts):
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    h = -np.sum(p * np.log(p + 1e-12))
    return float(h / np.log(counts.size))


def numeric_features(df):
    return [
        c for c in df.columns
        if c not in TARGET_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def token_stats(raw, bins, offsets, num_bins):
    raw = np.asarray(raw, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.float64)

    finite_mask = np.isfinite(raw)
    raw_f = raw[finite_mask]
    bin_f = bins[finite_mask]
    off_f = offsets[finite_mask]

    raw_unique = int(np.unique(raw_f).size) if raw_f.size else 0

    counts = np.bincount(bin_f, minlength=num_bins)
    used = int(np.count_nonzero(counts))
    rare_le5 = int(np.sum((counts > 0) & (counts <= 5)))
    rare_le10 = int(np.sum((counts > 0) & (counts <= 10)))

    bin_to_raws = defaultdict(set)
    for rv, b in zip(raw_f, bin_f):
        bin_to_raws[int(b)].add(float(rv))

    unique_per_used_bin = [len(v) for v in bin_to_raws.values()]
    max_unique_per_bin = int(max(unique_per_used_bin)) if unique_per_used_bin else 0
    mean_unique_per_bin = float(np.mean(unique_per_used_bin)) if unique_per_used_bin else 0.0

    return {
        "raw_unique": raw_unique,
        "bins_used": used,
        "empty_bins": int(num_bins - used),
        "empty_ratio": float((num_bins - used) / max(num_bins, 1)),
        "dominant_bin_ratio": float(counts.max() / max(counts.sum(), 1)),
        "entropy_norm": entropy_norm(counts),
        "compression_factor": float(raw_unique / max(used, 1)),
        "rare_bins_le5": rare_le5,
        "rare_bins_le10": rare_le10,
        "rare_used_bin_ratio_le5": float(rare_le5 / max(used, 1)),
        "rare_used_bin_ratio_le10": float(rare_le10 / max(used, 1)),
        "max_raw_unique_per_bin": max_unique_per_bin,
        "mean_raw_unique_per_bin": mean_unique_per_bin,
        "offset_unique_count": int(np.unique(off_f).size) if off_f.size else 0,
        "offset_nonzero_ratio": float(np.mean(np.abs(off_f) > 1e-12)) if off_f.size else 0.0,
        "offset_std": float(np.std(off_f)) if off_f.size else 0.0,
        "offset_min": float(np.min(off_f)) if off_f.size else 0.0,
        "offset_max": float(np.max(off_f)) if off_f.size else 0.0,
    }


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


def make_discrete_compact(train_values, val_values, num_bins):
    train_values = np.asarray(train_values, dtype=np.float64)
    val_values = np.asarray(val_values, dtype=np.float64)

    finite = train_values[np.isfinite(train_values)]

    if finite.size == 0:
        train_values = np.zeros_like(train_values, dtype=np.float64)
        val_values = np.zeros_like(val_values, dtype=np.float64)
    else:
        fill = float(np.median(finite))
        mn = float(np.min(finite))
        mx = float(np.max(finite))
        train_values = np.nan_to_num(train_values, nan=fill, posinf=mx, neginf=mn)
        val_values = np.nan_to_num(val_values, nan=fill, posinf=mx, neginf=mn)

    uniq = np.unique(train_values)

    if uniq.size > num_bins:
        raise ValueError(f"Cannot compact raw_unique={uniq.size} into num_bins={num_bins}")

    tr_idx = nearest_unique_index(train_values, uniq)
    va_idx = nearest_unique_index(val_values, uniq)

    tr_off = np.zeros_like(tr_idx, dtype=np.float32)
    va_off = np.zeros_like(va_idx, dtype=np.float32)

    return tr_idx.astype(np.int64), tr_off, va_idx.astype(np.int64), va_off, uniq


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
    K = 512
    B = 512

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
        C2_arrays = {k: data[k] for k in data.files}
        if "feature_names" in data.files:
            feature_names = as_str_list(data["feature_names"])
        else:
            feature_names = numeric_features(train_df)

    C2_meta = load_json(C2_meta_path)
    C2_strategies = C2_meta.get("feature_strategies", {})
    if not isinstance(C2_strategies, dict):
        C2_strategies = {}

    C3a = {k: np.array(v, copy=True) for k, v in C2_arrays.items()}
    C3b = {k: np.array(v, copy=True) for k, v in C2_arrays.items()}

    rows = []
    bad_features = []
    C3a_strategies = dict(C2_strategies)
    C3b_strategies = dict(C2_strategies)

    for j, feat in enumerate(feature_names):
        raw_train = train_df[feat].to_numpy(dtype=np.float64)
        raw_val = val_df[feat].to_numpy(dtype=np.float64)

        strategy = C2_strategies.get(feat, "UNKNOWN")

        stats_before = token_stats(
            raw_train,
            C2_arrays["X_train_bin"][:, j],
            C2_arrays["X_train_offset"][:, j],
            B,
        )

        is_bad_current = (
            strategy == "keep_current"
            and stats_before["max_raw_unique_per_bin"] <= 1
            and stats_before["rare_used_bin_ratio_le5"] >= 0.30
            and stats_before["dominant_bin_ratio"] >= 0.20
            and stats_before["raw_unique"] <= B
        )

        c3a_strategy = strategy
        c3b_strategy = strategy

        if is_bad_current:
            bad_features.append(feat)

            # C3a: remap to compact discrete token, offset=0
            tr_b, tr_o, va_b, va_o, uniq = make_discrete_compact(raw_train, raw_val, B)
            C3a["X_train_bin"][:, j] = tr_b
            C3a["X_train_offset"][:, j] = tr_o
            C3a["X_val_bin"][:, j] = va_b
            C3a["X_val_offset"][:, j] = va_o
            c3a_strategy = "bad_current_discrete_compact_offset0"

            # C3b: keep current bin, set offset=0
            C3b["X_train_offset"][:, j] = np.zeros_like(C3b["X_train_offset"][:, j], dtype=np.float32)
            C3b["X_val_offset"][:, j] = np.zeros_like(C3b["X_val_offset"][:, j], dtype=np.float32)
            c3b_strategy = "bad_current_offset_off"

        C3a_strategies[feat] = c3a_strategy
        C3b_strategies[feat] = c3b_strategy

        # Recompute after stats for changed variants
        stats_c3a = token_stats(raw_train, C3a["X_train_bin"][:, j], C3a["X_train_offset"][:, j], B)
        stats_c3b = token_stats(raw_train, C3b["X_train_bin"][:, j], C3b["X_train_offset"][:, j], B)

        row = {
            "feature": feat,
            "C2_strategy": strategy,
            "is_bad_current": bool(is_bad_current),
            "C3a_strategy": c3a_strategy,
            "C3b_strategy": c3b_strategy,

            "C2_raw_unique": stats_before["raw_unique"],
            "C2_bins_used": stats_before["bins_used"],
            "C2_max_raw_unique_per_bin": stats_before["max_raw_unique_per_bin"],
            "C2_rare_used_bin_ratio_le5": stats_before["rare_used_bin_ratio_le5"],
            "C2_dominant_bin_ratio": stats_before["dominant_bin_ratio"],
            "C2_entropy_norm": stats_before["entropy_norm"],
            "C2_offset_unique_count": stats_before["offset_unique_count"],
            "C2_offset_nonzero_ratio": stats_before["offset_nonzero_ratio"],

            "C3a_bins_used": stats_c3a["bins_used"],
            "C3a_max_raw_unique_per_bin": stats_c3a["max_raw_unique_per_bin"],
            "C3a_rare_used_bin_ratio_le5": stats_c3a["rare_used_bin_ratio_le5"],
            "C3a_dominant_bin_ratio": stats_c3a["dominant_bin_ratio"],
            "C3a_offset_unique_count": stats_c3a["offset_unique_count"],
            "C3a_offset_nonzero_ratio": stats_c3a["offset_nonzero_ratio"],

            "C3b_bins_used": stats_c3b["bins_used"],
            "C3b_max_raw_unique_per_bin": stats_c3b["max_raw_unique_per_bin"],
            "C3b_rare_used_bin_ratio_le5": stats_c3b["rare_used_bin_ratio_le5"],
            "C3b_dominant_bin_ratio": stats_c3b["dominant_bin_ratio"],
            "C3b_offset_unique_count": stats_c3b["offset_unique_count"],
            "C3b_offset_nonzero_ratio": stats_c3b["offset_nonzero_ratio"],
        }

        rows.append(row)

    out_C3a_dir = Path("03_outputs/build_mixed_quantile_offset/K512_B512_C3a_bad_current_compact")
    out_C3b_dir = Path("03_outputs/build_mixed_quantile_offset/K512_B512_C3b_bad_current_offset_off")
    out_C3a_dir.mkdir(parents=True, exist_ok=True)
    out_C3b_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_C3a_dir / "mixed_quantile_offset_dataset.npz", **C3a)
    np.savez_compressed(out_C3b_dir / "mixed_quantile_offset_dataset.npz", **C3b)

    diag_dir = Path("03_outputs/bin_diag")
    diag_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    diag_csv = diag_dir / "K512_C3a_C3b_bad_current_policy_diag.csv"
    diag_json = diag_dir / "K512_C3a_C3b_bad_current_policy_diag.json"

    df.to_csv(diag_csv, index=False)

    def counts(d):
        out = {}
        for v in d.values():
            out[v] = out.get(v, 0) + 1
        return out

    summary = {
        "K": K,
        "num_bins": B,
        "source_C2": str(C2_dir),
        "bad_current_rule": {
            "strategy": "keep_current",
            "max_raw_unique_per_bin": "<= 1",
            "rare_used_bin_ratio_le5": ">= 0.30",
            "dominant_bin_ratio": ">= 0.20",
            "raw_unique_guard": "<= num_bins",
            "logic": "all conditions are simultaneous AND",
        },
        "bad_current_features": bad_features,
        "n_bad_current_features": len(bad_features),
        "C3a_strategy_counts": counts(C3a_strategies),
        "C3b_strategy_counts": counts(C3b_strategies),
    }

    save_json({"summary": summary, "features": rows}, diag_json)

    def make_meta(policy_name, strategies, note):
        meta = dict(C2_meta)
        meta["stage"] = "C3_bad_current_ablation"
        meta["policy_name"] = policy_name
        meta["source_C2"] = str(C2_dir)
        meta["feature_strategies"] = strategies
        meta["strategy_counts"] = counts(strategies)
        meta["bad_current_rule"] = summary["bad_current_rule"]
        meta["bad_current_features"] = bad_features
        meta["policy_diag_csv"] = str(diag_csv)
        meta["policy_diag_json"] = str(diag_json)
        meta["source_note"] = note
        return meta

    C3a_meta = make_meta(
        "C3a_bad_current_discrete_compact",
        C3a_strategies,
        "Start from C2. Bad keep_current features are remapped to compact unique-token ids with offset=0.",
    )
    C3b_meta = make_meta(
        "C3b_bad_current_offset_off",
        C3b_strategies,
        "Start from C2. Bad keep_current features keep C2 bins but their offsets are set to 0.",
    )

    save_json(C3a_meta, out_C3a_dir / "mixed_quantile_offset_metadata.json")
    save_json(C3b_meta, out_C3b_dir / "mixed_quantile_offset_metadata.json")

    summary_path = Path("03_outputs/build_mixed_quantile_offset/K512_C3a_C3b_bad_current_summary.json")
    save_json({
        "C2_source": str(C2_dir),
        "C3a_bad_current_compact": str(out_C3a_dir),
        "C3b_bad_current_offset_off": str(out_C3b_dir),
        "summary": summary,
    }, summary_path)

    out_zip = Path("K512_C3a_C3b_bad_current_artifacts.zip")
    zip_paths([out_C3a_dir, out_C3b_dir, diag_csv, diag_json, summary_path], out_zip)

    print("Done.")
    print("C3a:", out_C3a_dir)
    print("C3b:", out_C3b_dir)
    print("Diag:", diag_json)
    print("Zip:", out_zip.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

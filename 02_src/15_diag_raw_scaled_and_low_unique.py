import json
import zipfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


TARGET_COLS = {"label_L1", "label_L2", "label_L3", "Class", "Category"}


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


def numeric_feature_names(df):
    return [
        c for c in df.columns
        if c not in TARGET_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def clean_with_train_stats(x, train_min, train_max, train_median):
    x = np.asarray(x, dtype=np.float64)
    return np.nan_to_num(
        x,
        nan=float(train_median),
        posinf=float(train_max),
        neginf=float(train_min),
    )


def minmax_raw_scaled(train_x, val_x):
    finite = train_x[np.isfinite(train_x)]

    if finite.size == 0:
        train_clean = np.zeros_like(train_x, dtype=np.float64)
        val_clean = np.zeros_like(val_x, dtype=np.float64)
        return train_clean, val_clean, 0.0, 0.0, 0.0, True

    mn = float(np.min(finite))
    mx = float(np.max(finite))
    med = float(np.median(finite))

    train_clean = clean_with_train_stats(train_x, mn, mx, med)
    val_clean = clean_with_train_stats(val_x, mn, mx, med)

    if mx <= mn:
        ztr = np.full_like(train_clean, 0.5, dtype=np.float64)
        zva = np.full_like(val_clean, 0.5, dtype=np.float64)
        return ztr, zva, mn, mx, med, True

    ztr = (train_clean - mn) / (mx - mn)
    zva = (val_clean - mn) / (mx - mn)

    ztr = np.clip(ztr, 0.0, 1.0)
    zva = np.clip(zva, 0.0, 1.0)

    return ztr, zva, mn, mx, med, False


def unique_count(x):
    return int(np.unique(x).size)


def raw_scaled_diag(train_df, val_df, features):
    rows = []
    f32_tiny = float(np.finfo(np.float32).tiny)
    f32_eps = float(np.finfo(np.float32).eps)

    for feat in features:
        train_raw = train_df[feat].to_numpy(dtype=np.float64)
        val_raw = val_df[feat].to_numpy(dtype=np.float64)

        finite_train = train_raw[np.isfinite(train_raw)]
        raw_unique = int(np.unique(finite_train).size) if finite_train.size else 0

        ztr64, zva64, mn, mx, med, is_constant = minmax_raw_scaled(train_raw, val_raw)

        ztr32 = ztr64.astype(np.float32)
        zva32 = zva64.astype(np.float32)

        z64_unique = unique_count(ztr64)
        z32_unique = unique_count(ztr32)

        nonmin_mask = train_raw > mn
        nonmax_mask = train_raw < mx

        nonmin_to_zero_mask = nonmin_mask & (ztr32 == np.float32(0.0))
        nonmax_to_one_mask = nonmax_mask & (ztr32 == np.float32(1.0))

        pos64 = ztr64[ztr64 > 0]
        pos32 = ztr32[ztr32 > 0]

        subnormal_pos32 = pos32[(pos32 > 0) & (pos32 < np.float32(f32_tiny))]

        collision_ratio_vs_z64 = (
            float(1.0 - (z32_unique / max(z64_unique, 1)))
            if z64_unique > 0 else 0.0
        )

        preserve_raw_unique_ratio_z32 = (
            float(z32_unique / max(raw_unique, 1))
            if raw_unique > 0 else 0.0
        )

        val_below_min = int(np.sum(val_raw < mn))
        val_above_max = int(np.sum(val_raw > mx))

        rows.append({
            "feature": feat,
            "raw_unique_train": raw_unique,
            "train_min": mn,
            "train_max": mx,
            "train_range": float(mx - mn),
            "is_constant": bool(is_constant),

            "z64_unique_train": z64_unique,
            "z32_unique_train": z32_unique,
            "float32_unique_loss_vs_z64": int(max(z64_unique - z32_unique, 0)),
            "float32_collision_ratio_vs_z64": collision_ratio_vs_z64,
            "preserve_raw_unique_ratio_z32": preserve_raw_unique_ratio_z32,

            "nonmin_to_zero_samples_train": int(np.sum(nonmin_to_zero_mask)),
            "nonmin_to_zero_unique_train": int(np.unique(train_raw[nonmin_to_zero_mask]).size) if np.any(nonmin_to_zero_mask) else 0,
            "nonmax_to_one_samples_train": int(np.sum(nonmax_to_one_mask)),
            "nonmax_to_one_unique_train": int(np.unique(train_raw[nonmax_to_one_mask]).size) if np.any(nonmax_to_one_mask) else 0,

            "min_positive_z64_train": float(np.min(pos64)) if pos64.size else None,
            "min_positive_z32_train": float(np.min(pos32)) if pos32.size else None,
            "num_positive_subnormal_z32_train": int(subnormal_pos32.size),
            "float32_tiny": f32_tiny,
            "float32_eps": f32_eps,

            "z64_q001_train": float(np.quantile(ztr64, 0.001)),
            "z64_q01_train": float(np.quantile(ztr64, 0.01)),
            "z64_q05_train": float(np.quantile(ztr64, 0.05)),
            "z64_q50_train": float(np.quantile(ztr64, 0.50)),

            "val_below_train_min_count": val_below_min,
            "val_above_train_max_count": val_above_max,

            "flag_nonmin_collapsed_to_zero": bool(np.sum(nonmin_to_zero_mask) > 0),
            "flag_nonmax_collapsed_to_one": bool(np.sum(nonmax_to_one_mask) > 0),
            "flag_float32_collision_gt_1pct": bool(collision_ratio_vs_z64 > 0.01),
            "flag_preserve_raw_unique_lt_0_99": bool(preserve_raw_unique_ratio_z32 < 0.99 and raw_unique <= 200000),
        })

    return pd.DataFrame(rows)


def load_artifact_npz(path):
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        obj = {k: data[k] for k in data.files}
        if "feature_names" in data.files:
            feature_names = as_str_list(data["feature_names"])
        else:
            feature_names = None
    return obj, feature_names


def mode_or_first(values):
    values = list(values)
    if not values:
        return None
    counts = defaultdict(int)
    for v in values:
        counts[int(v)] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def low_unique_bin_diag_for_policy(policy_name, artifact_dir, train_df, feature_names, low_unique_threshold=128):
    artifact_dir = Path(artifact_dir)
    npz_path = artifact_dir / "mixed_quantile_offset_dataset.npz"
    meta_path = artifact_dir / "mixed_quantile_offset_metadata.json"

    if not npz_path.exists():
        return [], {}

    arrays, npz_features = load_artifact_npz(npz_path)
    if npz_features is not None:
        feature_names = npz_features

    Xb = arrays["X_train_bin"]
    Xo = arrays["X_train_offset"]

    metadata = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    rows = []
    mappings = {}

    for j, feat in enumerate(feature_names):
        raw = train_df[feat].to_numpy(dtype=np.float64)
        finite_raw = raw[np.isfinite(raw)]

        if finite_raw.size == 0:
            raw_unique_vals = np.array([0.0], dtype=np.float64)
        else:
            raw_unique_vals = np.unique(finite_raw)

        raw_unique = int(raw_unique_vals.size)
        is_binary = raw_unique == 2
        is_low_unique = raw_unique <= low_unique_threshold

        if not is_low_unique:
            continue

        bins = Xb[:, j].astype(np.int64)
        offs = Xo[:, j].astype(np.float64)

        raw_to_bins = defaultdict(set)
        raw_to_offsets = defaultdict(set)
        bin_to_raws = defaultdict(set)
        bin_to_count = defaultdict(int)

        for rv, b, off in zip(raw, bins, offs):
            if not np.isfinite(rv):
                continue
            key = float(rv)
            raw_to_bins[key].add(int(b))
            raw_to_offsets[key].add(float(off))
            bin_to_raws[int(b)].add(key)
            bin_to_count[int(b)] += 1

        used_bins = sorted(bin_to_raws.keys())
        raw_unique_per_bin = [len(bin_to_raws[b]) for b in used_bins]

        max_raw_unique_per_bin = int(max(raw_unique_per_bin)) if raw_unique_per_bin else 0
        mean_raw_unique_per_bin = float(np.mean(raw_unique_per_bin)) if raw_unique_per_bin else 0.0
        bins_with_multi_raw_unique = int(sum(v > 1 for v in raw_unique_per_bin))

        representative_bins = []
        for rv in sorted(raw_to_bins.keys()):
            representative_bins.append(mode_or_first(raw_to_bins[rv]))

        representative_bins = [b for b in representative_bins if b is not None]

        if len(representative_bins) >= 2:
            gaps = np.diff(np.array(representative_bins, dtype=np.int64))
            bin_gap_min = int(np.min(gaps))
            bin_gap_median = float(np.median(gaps))
            bin_gap_max = int(np.max(gaps))
            bin_span = int(np.max(representative_bins) - np.min(representative_bins))
        else:
            bin_gap_min = None
            bin_gap_median = None
            bin_gap_max = None
            bin_span = 0

        offset_unique_count = int(np.unique(offs).size)
        offset_nonzero_ratio = float(np.mean(np.abs(offs) > 1e-12))
        offset_std = float(np.std(offs))
        offset_min = float(np.min(offs))
        offset_max = float(np.max(offs))

        raw_values_multibin = int(sum(len(v) > 1 for v in raw_to_bins.values()))
        raw_values_multioffset = int(sum(len(v) > 1 for v in raw_to_offsets.values()))

        offset_geometric_noise_suspect = bool(
            is_low_unique
            and max_raw_unique_per_bin <= 1
            and offset_unique_count > 1
        )

        rows.append({
            "policy": policy_name,
            "artifact_dir": str(artifact_dir),
            "feature": feat,
            "is_binary": bool(is_binary),
            "is_low_unique": bool(is_low_unique),
            "raw_unique": raw_unique,
            "bins_used": int(len(used_bins)),
            "max_raw_unique_per_bin": max_raw_unique_per_bin,
            "mean_raw_unique_per_bin": mean_raw_unique_per_bin,
            "bins_with_multi_raw_unique": bins_with_multi_raw_unique,
            "bins_with_multi_raw_unique_ratio": float(bins_with_multi_raw_unique / max(len(used_bins), 1)),
            "raw_values_multibin": raw_values_multibin,
            "raw_values_multioffset": raw_values_multioffset,
            "bin_span": bin_span,
            "bin_gap_min_between_sorted_raw_values": bin_gap_min,
            "bin_gap_median_between_sorted_raw_values": bin_gap_median,
            "bin_gap_max_between_sorted_raw_values": bin_gap_max,
            "offset_unique_count": offset_unique_count,
            "offset_nonzero_ratio": offset_nonzero_ratio,
            "offset_std": offset_std,
            "offset_min": offset_min,
            "offset_max": offset_max,
            "offset_geometric_noise_suspect": offset_geometric_noise_suspect,
            "strategy_from_metadata": metadata.get("feature_strategies", {}).get(feat, None)
                if isinstance(metadata.get("feature_strategies", {}), dict) else None,
        })

        if raw_unique <= 20:
            fmap = []
            for rv in sorted(raw_to_bins.keys()):
                fmap.append({
                    "raw_value": rv,
                    "bins": sorted(list(raw_to_bins[rv])),
                    "offsets": sorted(list(raw_to_offsets[rv]))[:20],
                    "n_offsets": len(raw_to_offsets[rv]),
                })
            mappings[f"{policy_name}::{feat}"] = {
                "policy": policy_name,
                "feature": feat,
                "raw_unique": raw_unique,
                "mapping": fmap,
            }

    return rows, mappings


def main():
    K = 512
    B = 512

    train_raw_path = Path("01_split/train_raw.csv")
    val_raw_path = Path("01_split/val_raw.csv")

    if not train_raw_path.exists():
        raise FileNotFoundError(train_raw_path)
    if not val_raw_path.exists():
        raise FileNotFoundError(val_raw_path)

    train_df = pd.read_csv(train_raw_path)
    val_df = pd.read_csv(val_raw_path)

    features = numeric_feature_names(train_df)

    out_dir = Path("03_outputs/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/2] Diagnosing raw_scaled float32 stability...")
    raw_diag = raw_scaled_diag(train_df, val_df, features)

    raw_csv = out_dir / "raw_scaled_float32_diag.csv"
    raw_json = out_dir / "raw_scaled_float32_diag.json"

    raw_diag.to_csv(raw_csv, index=False)

    raw_summary = {
        "n_features": int(len(raw_diag)),
        "features_nonmin_collapsed_to_zero": raw_diag[raw_diag["flag_nonmin_collapsed_to_zero"]]["feature"].tolist(),
        "features_nonmax_collapsed_to_one": raw_diag[raw_diag["flag_nonmax_collapsed_to_one"]]["feature"].tolist(),
        "features_float32_collision_gt_1pct": raw_diag[raw_diag["flag_float32_collision_gt_1pct"]]["feature"].tolist(),
        "features_preserve_raw_unique_lt_0_99": raw_diag[raw_diag["flag_preserve_raw_unique_lt_0_99"]]["feature"].tolist(),
        "top_float32_collision": raw_diag.sort_values("float32_collision_ratio_vs_z64", ascending=False).head(20).to_dict(orient="records"),
        "top_smallest_positive_z32": raw_diag[raw_diag["min_positive_z32_train"].notna()].sort_values("min_positive_z32_train", ascending=True).head(20).to_dict(orient="records"),
        "note": "D3 uses train-only minmax raw_scaled and torch float32. This diagnostic checks whether minmax-scaled values collapse after float32 cast.",
    }

    save_json({"summary": raw_summary, "features": raw_diag.to_dict(orient="records")}, raw_json)

    print("[2/2] Diagnosing binary/low-unique bin width, unique-per-bin, and offset...")
    artifact_policies = {
        "A_K512_current_mixed": "03_outputs/build_mixed_quantile_offset/K512_B512",
        "B_K512_rank_uniform_all": "03_outputs/build_mixed_quantile_offset/K512_B512_rank_uniform_only",
        "C1_K512_selective_rank_current": "03_outputs/build_mixed_quantile_offset/K512_B512_C1_selective_rank_current",
        "C2_K512_selective_rank_discrete_compact": "03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact",
        "K256_current_mixed": "03_outputs/build_mixed_quantile_offset/K256_B256",
    }

    all_rows = []
    all_mappings = {}

    for policy, path in artifact_policies.items():
        if not Path(path).exists():
            print("Skipping missing artifact:", policy, path)
            continue

        rows, mappings = low_unique_bin_diag_for_policy(
            policy,
            path,
            train_df,
            features,
            low_unique_threshold=128,
        )
        all_rows.extend(rows)
        all_mappings.update(mappings)

    low_df = pd.DataFrame(all_rows)

    low_csv = out_dir / "low_unique_bin_offset_diag.csv"
    low_json = out_dir / "low_unique_bin_offset_diag.json"
    mapping_json = out_dir / "low_unique_raw_to_bin_offset_mapping.json"

    low_df.to_csv(low_csv, index=False)

    if len(low_df):
        low_summary = {
            "n_rows": int(len(low_df)),
            "policies": sorted(low_df["policy"].unique().tolist()),
            "n_binary_rows": int(low_df["is_binary"].sum()),
            "n_offset_noise_suspect_rows": int(low_df["offset_geometric_noise_suspect"].sum()),
            "offset_noise_suspect_by_policy": {
                str(k): int(v)
                for k, v in low_df.groupby("policy")["offset_geometric_noise_suspect"].sum().to_dict().items()
            },
            "binary_summary_by_policy": low_df[low_df["is_binary"]].groupby("policy").agg({
                "feature": "count",
                "bins_used": "mean",
                "bin_span": "mean",
                "offset_unique_count": "mean",
                "offset_nonzero_ratio": "mean",
                "offset_geometric_noise_suspect": "sum",
            }).reset_index().to_dict(orient="records"),
            "low_unique_summary_by_policy": low_df.groupby("policy").agg({
                "feature": "count",
                "raw_unique": "mean",
                "bins_used": "mean",
                "max_raw_unique_per_bin": "mean",
                "bins_with_multi_raw_unique": "sum",
                "offset_unique_count": "mean",
                "offset_nonzero_ratio": "mean",
                "offset_geometric_noise_suspect": "sum",
            }).reset_index().to_dict(orient="records"),
            "top_offset_noise_suspects": low_df[low_df["offset_geometric_noise_suspect"]].sort_values(
                ["policy", "raw_unique", "bin_span"],
                ascending=[True, True, False],
            ).head(100).to_dict(orient="records"),
            "note": (
                "For low-unique features, if max_raw_unique_per_bin <= 1 then each used token corresponds "
                "to a single raw value. In that case offset may be geometric inductive-bias noise rather than "
                "within-bin numeric information."
            ),
        }
    else:
        low_summary = {"n_rows": 0, "note": "No artifact found or no low-unique features."}

    save_json({"summary": low_summary, "features": all_rows}, low_json)
    save_json(all_mappings, mapping_json)

    out_zip = Path("K512_raw_scaled_low_unique_diagnostics.zip")

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in [raw_csv, raw_json, low_csv, low_json, mapping_json]:
            if p.exists():
                z.write(p, p.as_posix())

    print("\nDone.")
    print("raw_scaled csv:", raw_csv)
    print("raw_scaled json:", raw_json)
    print("low_unique csv:", low_csv)
    print("low_unique json:", low_json)
    print("mapping json:", mapping_json)
    print("zip:", out_zip.resolve())

    print("\n=== raw_scaled summary ===")
    print(json.dumps(raw_summary, indent=2)[:4000])

    print("\n=== low_unique summary ===")
    print(json.dumps(low_summary, indent=2)[:4000])


if __name__ == "__main__":
    main()

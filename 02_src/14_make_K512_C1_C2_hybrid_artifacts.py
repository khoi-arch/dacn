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


def as_str_list(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def bin_stats(bin_ids, raw_unique, num_bins):
    counts = np.bincount(np.asarray(bin_ids, dtype=np.int64), minlength=num_bins)
    used = int(np.count_nonzero(counts))
    rare_le5 = int(np.sum((counts > 0) & (counts <= 5)))
    rare_le10 = int(np.sum((counts > 0) & (counts <= 10)))
    return {
        "bins_used": used,
        "empty_bins": int(num_bins - used),
        "empty_ratio": float((num_bins - used) / max(num_bins, 1)),
        "dominant_bin_ratio": float(counts.max() / max(counts.sum(), 1)),
        "entropy_norm": entropy_norm_from_counts(counts),
        "compression_factor": float(raw_unique / max(used, 1)),
        "rare_bins_le5": rare_le5,
        "rare_bins_le10": rare_le10,
        "rare_used_bin_ratio_le5": float(rare_le5 / max(used, 1)),
        "rare_used_bin_ratio_le10": float(rare_le10 / max(used, 1)),
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
        train_values = np.nan_to_num(train_values, nan=fill, posinf=float(finite.max()), neginf=float(finite.min()))
        val_values = np.nan_to_num(val_values, nan=fill, posinf=float(finite.max()), neginf=float(finite.min()))

    uniq = np.unique(train_values)

    if uniq.size > num_bins:
        raise ValueError(f"discrete_compact requires unique <= num_bins, got {uniq.size}")

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

    A_dir = Path("03_outputs/build_mixed_quantile_offset/K512_B512")
    B_dir = Path("03_outputs/build_mixed_quantile_offset/K512_B512_rank_uniform_only")

    A_npz_path = A_dir / "mixed_quantile_offset_dataset.npz"
    B_npz_path = B_dir / "mixed_quantile_offset_dataset.npz"
    A_meta_path = A_dir / "mixed_quantile_offset_metadata.json"
    B_meta_path = B_dir / "mixed_quantile_offset_metadata.json"

    required = [train_raw_path, val_raw_path, A_npz_path, B_npz_path, A_meta_path, B_meta_path]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    train_df = pd.read_csv(train_raw_path)
    val_df = pd.read_csv(val_raw_path)

    with np.load(A_npz_path, allow_pickle=True) as A_data, np.load(B_npz_path, allow_pickle=True) as B_data:
        A = {k: A_data[k] for k in A_data.files}
        Bdata = {k: B_data[k] for k in B_data.files}

        if "feature_names" in A_data.files:
            feature_names = as_str_list(A_data["feature_names"])
        else:
            feature_names = [
                c for c in train_df.columns
                if c not in TARGET_COLS and pd.api.types.is_numeric_dtype(train_df[c])
            ]

    n_train = A["X_train_bin"].shape[0]
    n_val = A["X_val_bin"].shape[0]
    n_features = len(feature_names)

    # Start C1/C2 from A current mixed
    C1 = {k: np.array(v, copy=True) for k, v in A.items()}
    C2 = {k: np.array(v, copy=True) for k, v in A.items()}

    rows = []

    C1_strategies = {}
    C2_strategies = {}

    for j, feat in enumerate(feature_names):
        tr_raw = train_df[feat].to_numpy(dtype=np.float64)
        va_raw = val_df[feat].to_numpy(dtype=np.float64)

        finite = tr_raw[np.isfinite(tr_raw)]
        raw_unique = int(np.unique(finite).size) if finite.size else 1

        A_stats = bin_stats(A["X_train_bin"][:, j], raw_unique, B)
        B_stats = bin_stats(Bdata["X_train_bin"][:, j], raw_unique, B)

        is_constant = raw_unique <= 1

        rank_candidate = (
            raw_unique >= 512
            and (
                A_stats["compression_factor"] >= 8.0
                or A_stats["dominant_bin_ratio"] >= 0.10
                or A_stats["entropy_norm"] < 0.75
            )
            and B_stats["bins_used"] >= 256
            and B_stats["rare_used_bin_ratio_le5"] <= 0.30
        )

        low_unique_discrete_signal = (
            raw_unique <= 128
            or A_stats["bins_used"] <= 128
            or B_stats["bins_used"] <= 128
        )

        compact_allowed = (
            raw_unique <= 128
            or (
                raw_unique <= 512
                and A_stats["bins_used"] <= 128
                and B_stats["bins_used"] <= 128
            )
        )

        # C1: selective rank, else keep current
        if is_constant:
            c1_strategy = "constant"
        elif rank_candidate:
            c1_strategy = "rank_uniform_offset"
            C1["X_train_bin"][:, j] = Bdata["X_train_bin"][:, j]
            C1["X_val_bin"][:, j] = Bdata["X_val_bin"][:, j]
            C1["X_train_offset"][:, j] = Bdata["X_train_offset"][:, j]
            C1["X_val_offset"][:, j] = Bdata["X_val_offset"][:, j]
        else:
            c1_strategy = "keep_current"

        # C2: selective rank; low/discrete compact; else keep current
        if is_constant:
            c2_strategy = "constant"
        elif rank_candidate:
            c2_strategy = "rank_uniform_offset"
            C2["X_train_bin"][:, j] = Bdata["X_train_bin"][:, j]
            C2["X_val_bin"][:, j] = Bdata["X_val_bin"][:, j]
            C2["X_train_offset"][:, j] = Bdata["X_train_offset"][:, j]
            C2["X_val_offset"][:, j] = Bdata["X_val_offset"][:, j]
        elif compact_allowed:
            c2_strategy = "discrete_compact_offset0"
            tr_b, tr_o, va_b, va_o, uniq = make_discrete_compact(tr_raw, va_raw, B)
            C2["X_train_bin"][:, j] = tr_b
            C2["X_val_bin"][:, j] = va_b
            C2["X_train_offset"][:, j] = tr_o
            C2["X_val_offset"][:, j] = va_o
        else:
            c2_strategy = "keep_current"

        C1_strategies[feat] = c1_strategy
        C2_strategies[feat] = c2_strategy

        rows.append({
            "feature": feat,
            "raw_unique": raw_unique,

            "A_bins_used": A_stats["bins_used"],
            "A_empty_ratio": A_stats["empty_ratio"],
            "A_compression_factor": A_stats["compression_factor"],
            "A_dominant_bin_ratio": A_stats["dominant_bin_ratio"],
            "A_entropy_norm": A_stats["entropy_norm"],
            "A_rare_used_bin_ratio_le5": A_stats["rare_used_bin_ratio_le5"],

            "B_bins_used": B_stats["bins_used"],
            "B_empty_ratio": B_stats["empty_ratio"],
            "B_compression_factor": B_stats["compression_factor"],
            "B_dominant_bin_ratio": B_stats["dominant_bin_ratio"],
            "B_entropy_norm": B_stats["entropy_norm"],
            "B_rare_used_bin_ratio_le5": B_stats["rare_used_bin_ratio_le5"],

            "is_constant": bool(is_constant),
            "rank_candidate": bool(rank_candidate),
            "low_unique_discrete_signal": bool(low_unique_discrete_signal),
            "compact_allowed": bool(compact_allowed),
            "C1_strategy": c1_strategy,
            "C2_strategy": c2_strategy,
        })

    out_C1_dir = Path("03_outputs/build_mixed_quantile_offset/K512_B512_C1_selective_rank_current")
    out_C2_dir = Path("03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact")
    out_C1_dir.mkdir(parents=True, exist_ok=True)
    out_C2_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_C1_dir / "mixed_quantile_offset_dataset.npz", **C1)
    np.savez_compressed(out_C2_dir / "mixed_quantile_offset_dataset.npz", **C2)

    df = pd.DataFrame(rows)

    diag_dir = Path("03_outputs/bin_diag")
    diag_dir.mkdir(parents=True, exist_ok=True)

    diag_csv = diag_dir / "K512_C1_C2_hybrid_policy_diag.csv"
    diag_json = diag_dir / "K512_C1_C2_hybrid_policy_diag.json"

    df.to_csv(diag_csv, index=False)

    def counts_for(col):
        return {str(k): int(v) for k, v in df[col].value_counts().to_dict().items()}

    summary = {
        "K": K,
        "num_bins": B,
        "n_features": int(n_features),
        "C1_strategy_counts": counts_for("C1_strategy"),
        "C2_strategy_counts": counts_for("C2_strategy"),
        "n_rank_candidates": int(df["rank_candidate"].sum()),
        "n_low_unique_discrete_signal": int(df["low_unique_discrete_signal"].sum()),
        "n_compact_allowed": int(df["compact_allowed"].sum()),
        "rank_candidate_features": df[df["rank_candidate"]]["feature"].tolist(),
        "C2_discrete_compact_features": df[df["C2_strategy"] == "discrete_compact_offset0"]["feature"].tolist(),
        "C1_keep_current_features": df[df["C1_strategy"] == "keep_current"]["feature"].tolist(),
        "C2_keep_current_features": df[df["C2_strategy"] == "keep_current"]["feature"].tolist(),
        "thresholds": {
            "constant": "raw_unique <= 1",
            "compact_allowed": "raw_unique <= 128 OR (raw_unique <= 512 AND A_bins_used <= 128 AND B_bins_used <= 128)",
            "rank_candidate": "raw_unique >= 512 AND (A_compression >= 8 OR A_dominant >= 0.10 OR A_entropy < 0.75) AND B_bins_used >= 256 AND B_rare_used_bin_ratio_le5 <= 0.30",
        },
    }

    save_json({"summary": summary, "features": rows}, diag_json)

    A_meta = load_json(A_meta_path)
    B_meta = load_json(B_meta_path)

    def make_meta(policy_name, strategies, source_note):
        strategy_counts = {}
        for s in strategies.values():
            strategy_counts[s] = strategy_counts.get(s, 0) + 1

        meta = dict(A_meta)
        meta["stage"] = "hybrid_C_policy_ablation"
        meta["policy_name"] = policy_name
        meta["K"] = K
        meta["num_bins"] = B
        meta["source_A_current_mixed"] = str(A_dir)
        meta["source_B_rank_uniform"] = str(B_dir)
        meta["source_note"] = source_note
        meta["strategy_counts"] = strategy_counts
        meta["feature_strategies"] = strategies
        meta["policy_diag_csv"] = str(diag_csv)
        meta["policy_diag_json"] = str(diag_json)
        meta["thresholds"] = summary["thresholds"]
        meta["splits"] = {
            "train": {
                "n_rows": int(n_train),
                "X_bin_shape": list(A["X_train_bin"].shape),
                "X_offset_shape": list(A["X_train_offset"].shape),
            },
            "val": {
                "n_rows": int(n_val),
                "X_bin_shape": list(A["X_val_bin"].shape),
                "X_offset_shape": list(A["X_val_offset"].shape),
            },
        }
        return meta

    C1_meta = make_meta(
        "C1_selective_rank_current",
        C1_strategies,
        "rank candidates use B rank-uniform; all other non-constant features keep A current mixed.",
    )

    C2_meta = make_meta(
        "C2_selective_rank_discrete_compact",
        C2_strategies,
        "rank candidates use B rank-uniform; compact_allowed low/discrete features use compact token ids with offset=0; rest keep A current mixed.",
    )

    save_json(C1_meta, out_C1_dir / "mixed_quantile_offset_metadata.json")
    save_json(C2_meta, out_C2_dir / "mixed_quantile_offset_metadata.json")

    comparison_path = Path("03_outputs/build_mixed_quantile_offset/K512_C1_C2_hybrid_summary.json")
    save_json({
        "A_current_mixed": str(A_dir),
        "B_rank_uniform_all": str(B_dir),
        "C1_selective_rank_current": str(out_C1_dir),
        "C2_selective_rank_discrete_compact": str(out_C2_dir),
        "summary": summary,
    }, comparison_path)

    out_zip = Path("K512_C1_C2_hybrid_artifacts.zip")
    zip_paths([out_C1_dir, out_C2_dir, diag_csv, diag_json, comparison_path], out_zip)

    print("Done.")
    print("C1:", out_C1_dir)
    print("C2:", out_C2_dir)
    print("Diag:", diag_json)
    print("Zip:", out_zip.resolve())
    print(json.dumps(summary, indent=2)[:6000])


if __name__ == "__main__":
    main()

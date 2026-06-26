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


def minmax_raw_scaled(train_x):
    x = np.asarray(train_x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=np.float64), True

    mn = float(np.min(finite))
    mx = float(np.max(finite))
    med = float(np.median(finite))

    x = np.nan_to_num(x, nan=med, posinf=mx, neginf=mn)

    if mx <= mn:
        return np.full_like(x, 0.5, dtype=np.float64), True

    z = (x - mn) / (mx - mn)
    return np.clip(z, 0.0, 1.0), False


def find_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None


def load_npz_artifact(artifact_dir):
    artifact_dir = Path(artifact_dir)
    npz_path = artifact_dir / "mixed_quantile_offset_dataset.npz"
    meta_path = artifact_dir / "mixed_quantile_offset_metadata.json"

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    with np.load(npz_path, allow_pickle=True) as data:
        arrays = {k: data[k] for k in data.files}
        features = as_str_list(data["feature_names"]) if "feature_names" in data.files else None

    meta = load_json(meta_path)
    return arrays, meta, features


def token_stats_for_feature(raw, bins, offsets, num_bins):
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
    rare_le1 = int(np.sum(counts == 1))
    rare_le5 = int(np.sum((counts > 0) & (counts <= 5)))
    rare_le10 = int(np.sum((counts > 0) & (counts <= 10)))

    bin_to_raws = defaultdict(set)
    raw_to_bins = defaultdict(set)
    raw_to_offsets = defaultdict(set)

    for rv, b, off in zip(raw_f, bin_f, off_f):
        rv = float(rv)
        b = int(b)
        bin_to_raws[b].add(rv)
        raw_to_bins[rv].add(b)
        raw_to_offsets[rv].add(float(off))

    unique_per_used_bin = [len(v) for v in bin_to_raws.values()]
    max_unique_per_bin = int(max(unique_per_used_bin)) if unique_per_used_bin else 0
    mean_unique_per_bin = float(np.mean(unique_per_used_bin)) if unique_per_used_bin else 0.0

    bins_with_multi_unique = int(sum(v > 1 for v in unique_per_used_bin))
    raw_values_multibin = int(sum(len(v) > 1 for v in raw_to_bins.values()))
    raw_values_multi_offset = int(sum(len(v) > 1 for v in raw_to_offsets.values()))

    offset_unique = int(np.unique(off_f).size) if off_f.size else 0
    offset_nonzero_ratio = float(np.mean(np.abs(off_f) > 1e-12)) if off_f.size else 0.0

    offset_noise_suspect = bool(
        raw_unique <= 128
        and max_unique_per_bin <= 1
        and offset_unique > 1
    )

    return {
        "raw_unique": raw_unique,
        "is_binary": bool(raw_unique == 2),
        "is_low_unique_128": bool(raw_unique <= 128),
        "bins_used": used,
        "empty_bins": int(num_bins - used),
        "empty_ratio": float((num_bins - used) / max(num_bins, 1)),
        "dominant_bin_ratio": float(counts.max() / max(counts.sum(), 1)),
        "entropy_norm": entropy_norm(counts),
        "compression_factor": float(raw_unique / max(used, 1)),
        "rare_bins_le1": rare_le1,
        "rare_bins_le5": rare_le5,
        "rare_bins_le10": rare_le10,
        "rare_used_bin_ratio_le5": float(rare_le5 / max(used, 1)),
        "rare_used_bin_ratio_le10": float(rare_le10 / max(used, 1)),
        "max_raw_unique_per_bin": max_unique_per_bin,
        "mean_raw_unique_per_bin": mean_unique_per_bin,
        "bins_with_multi_raw_unique": bins_with_multi_unique,
        "bins_with_multi_raw_unique_ratio": float(bins_with_multi_unique / max(used, 1)),
        "raw_values_multibin": raw_values_multibin,
        "raw_values_multi_offset": raw_values_multi_offset,
        "offset_unique_count": offset_unique,
        "offset_nonzero_ratio": offset_nonzero_ratio,
        "offset_std": float(np.std(off_f)) if off_f.size else 0.0,
        "offset_min": float(np.min(off_f)) if off_f.size else 0.0,
        "offset_max": float(np.max(off_f)) if off_f.size else 0.0,
        "offset_noise_suspect": offset_noise_suspect,
    }


def raw_scaled_stats(raw):
    z, is_const = minmax_raw_scaled(raw)
    return {
        "raw_scaled_is_constant": bool(is_const),
        "raw_scaled_unique_f64": int(np.unique(z).size),
        "raw_scaled_unique_f32": int(np.unique(z.astype(np.float32)).size),
        "raw_scaled_mean": float(np.mean(z)),
        "raw_scaled_std": float(np.std(z)),
        "raw_scaled_q001": float(np.quantile(z, 0.001)),
        "raw_scaled_q01": float(np.quantile(z, 0.01)),
        "raw_scaled_q05": float(np.quantile(z, 0.05)),
        "raw_scaled_q50": float(np.quantile(z, 0.50)),
        "raw_scaled_q95": float(np.quantile(z, 0.95)),
        "raw_scaled_q99": float(np.quantile(z, 0.99)),
        "raw_scaled_zero_ratio": float(np.mean(z == 0.0)),
        "raw_scaled_near_zero_1e_6_ratio": float(np.mean(z <= 1e-6)),
        "raw_scaled_near_zero_1e_4_ratio": float(np.mean(z <= 1e-4)),
        "raw_scaled_one_ratio": float(np.mean(z == 1.0)),
    }


def load_gate_from_checkpoint(run_dir, n_features):
    """
    Try to find cont_gate_logit in best_model.pt/checkpoint.
    Works even if we do not import the model class.
    """
    run_dir = Path(run_dir)
    ckpt_candidates = [
        run_dir / "best_model.pt",
        run_dir / "checkpoint_best.pt",
        run_dir / "model_best.pt",
    ]
    ckpt_candidates += list(run_dir.glob("*best*.pt"))

    ckpt_path = None
    for p in ckpt_candidates:
        if p.exists():
            ckpt_path = p
            break

    if ckpt_path is None:
        return None, None, "no checkpoint found"

    try:
        import torch
        obj = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        return None, str(ckpt_path), f"torch load failed: {e}"

    if isinstance(obj, dict) and "model_state_dict" in obj:
        sd = obj["model_state_dict"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        return None, str(ckpt_path), f"unknown checkpoint object type: {type(obj)}"

    gate_key = None
    for k in sd.keys():
        if "cont_gate_logit" in k:
            gate_key = k
            break

    if gate_key is None:
        return None, str(ckpt_path), "cont_gate_logit not found"

    gate_logit = sd[gate_key]
    try:
        import torch
        gate = torch.sigmoid(gate_logit.detach().float()).cpu().numpy().reshape(-1)
    except Exception:
        gate = 1 / (1 + np.exp(-np.asarray(gate_logit).reshape(-1)))

    if gate.size != n_features:
        return gate, str(ckpt_path), f"gate size {gate.size} != n_features {n_features}"

    return gate, str(ckpt_path), "ok"


def groupby_summary(df, group_col):
    numeric_cols = [
        "raw_unique", "bins_used", "empty_ratio", "dominant_bin_ratio",
        "entropy_norm", "compression_factor", "rare_used_bin_ratio_le5",
        "rare_used_bin_ratio_le10", "max_raw_unique_per_bin",
        "offset_unique_count", "offset_nonzero_ratio", "offset_std",
        "raw_scaled_std", "raw_scaled_near_zero_1e_4_ratio",
        "raw_scaled_zero_ratio", "film_gate",
    ]
    rows = []
    for group, g in df.groupby(group_col):
        row = {
            group_col: str(group),
            "n_features": int(len(g)),
            "n_binary": int(g["is_binary"].sum()),
            "n_low_unique_128": int(g["is_low_unique_128"].sum()),
            "n_offset_noise_suspect": int(g["offset_noise_suspect"].sum()),
        }
        for col in numeric_cols:
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors="coerce")
                if vals.notna().any():
                    row[f"mean_{col}"] = float(vals.mean())
                    row[f"median_{col}"] = float(vals.median())
                    row[f"max_{col}"] = float(vals.max())
        rows.append(row)
    return rows


def training_dynamics(run_dir):
    run_dir = Path(run_dir)
    hist_path = run_dir / "history.csv"
    diag_path = run_dir / "diagnosis_summary.json"

    out = {
        "history_found": hist_path.exists(),
        "diagnosis_found": diag_path.exists(),
    }

    if diag_path.exists():
        try:
            diag = load_json(diag_path)
            out["diagnosis_summary"] = diag
        except Exception as e:
            out["diagnosis_read_error"] = str(e)

    if not hist_path.exists():
        return out

    hist = pd.read_csv(hist_path)
    out["n_epochs_logged"] = int(len(hist))

    cols = list(hist.columns)
    out["history_columns"] = cols

    # Find likely columns robustly
    val_macro_col = None
    train_macro_col = None
    val_loss_col = None
    train_loss_col = None

    for c in cols:
        lc = c.lower()
        if val_macro_col is None and "val" in lc and "macro" in lc and "f1" in lc:
            val_macro_col = c
        if train_macro_col is None and "train" in lc and "macro" in lc and "f1" in lc:
            train_macro_col = c
        if val_loss_col is None and "val" in lc and "loss" in lc:
            val_loss_col = c
        if train_loss_col is None and "train" in lc and "loss" in lc:
            train_loss_col = c

    out["detected_columns"] = {
        "train_macro_f1": train_macro_col,
        "val_macro_f1": val_macro_col,
        "train_loss": train_loss_col,
        "val_loss": val_loss_col,
    }

    if val_macro_col is not None:
        best_idx = int(hist[val_macro_col].idxmax())
        out["best_val_epoch_index"] = best_idx
        out["best_val_macro_f1"] = float(hist.loc[best_idx, val_macro_col])

        if train_macro_col is not None:
            out["train_macro_at_best_val"] = float(hist.loc[best_idx, train_macro_col])
            out["max_train_macro_f1"] = float(hist[train_macro_col].max())
            max_train_idx = int(hist[train_macro_col].idxmax())
            out["max_train_epoch_index"] = max_train_idx
            out["val_macro_at_max_train"] = float(hist.loc[max_train_idx, val_macro_col])
            out["train_increase_after_best"] = float(hist[train_macro_col].max() - hist.loc[best_idx, train_macro_col])
            out["val_drop_from_best_to_max_train"] = float(hist.loc[best_idx, val_macro_col] - hist.loc[max_train_idx, val_macro_col])

        if val_loss_col is not None:
            out["val_loss_at_best_val_macro"] = float(hist.loc[best_idx, val_loss_col])
            out["min_val_loss"] = float(hist[val_loss_col].min())

    return out


def main():
    K = 512
    B = 512

    # Accept both local repo and Kaggle workdir layout.
    train_raw_path = find_existing([
        "01_split/train_raw.csv",
        "/kaggle/working/dacn/01_split/train_raw.csv",
    ])
    if train_raw_path is None:
        raise FileNotFoundError("Cannot find 01_split/train_raw.csv")

    train_df = pd.read_csv(train_raw_path)

    C2_artifact_dir = find_existing([
        "03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact",
        "/kaggle/working/dacn/03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact",
    ])
    if C2_artifact_dir is None:
        raise FileNotFoundError("Cannot find C2 artifact directory")

    C2_run_dir = find_existing([
        "03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact",
        "/kaggle/working/dacn_K512_C2_selective_rank_discrete_compact/03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact",
        "/kaggle/working/dacn/03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact",
    ])

    arrays, meta, feature_names = load_npz_artifact(C2_artifact_dir)
    if feature_names is None:
        feature_names = numeric_features(train_df)

    strategies = meta.get("feature_strategies", {})
    if not isinstance(strategies, dict):
        strategies = {}

    Xb = arrays["X_train_bin"]
    Xo = arrays["X_train_offset"]

    gate, ckpt_path, gate_status = load_gate_from_checkpoint(C2_run_dir, len(feature_names)) if C2_run_dir else (None, None, "run dir not found")

    rows = []
    for j, feat in enumerate(feature_names):
        raw = train_df[feat].to_numpy(dtype=np.float64)

        row = {
            "feature": feat,
            "strategy": strategies.get(feat, "UNKNOWN"),
            "feature_index": int(j),
        }
        row.update(token_stats_for_feature(raw, Xb[:, j], Xo[:, j], B))
        row.update(raw_scaled_stats(raw))

        if gate is not None and len(gate) == len(feature_names):
            row["film_gate"] = float(gate[j])
        else:
            row["film_gate"] = None

        rows.append(row)

    feat_df = pd.DataFrame(rows)

    out_dir = Path("03_outputs/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_csv = out_dir / "C2_feature_group_token_raw_film_diag.csv"
    feature_json = out_dir / "C2_feature_group_token_raw_film_diag.json"

    feat_df.to_csv(feature_csv, index=False)

    group_summary = groupby_summary(feat_df, "strategy")

    dynamics = training_dynamics(C2_run_dir) if C2_run_dir else {"run_dir_found": False}

    summary = {
        "K": K,
        "num_bins": B,
        "C2_artifact_dir": str(C2_artifact_dir),
        "C2_run_dir": str(C2_run_dir) if C2_run_dir else None,
        "gate_checkpoint_path": ckpt_path,
        "gate_status": gate_status,
        "strategy_counts": {str(k): int(v) for k, v in feat_df["strategy"].value_counts().to_dict().items()},
        "group_summary": group_summary,
        "top_bad_keep_current_by_rare": feat_df[feat_df["strategy"] == "keep_current"].sort_values(
            "rare_used_bin_ratio_le5", ascending=False
        ).head(20).to_dict(orient="records"),
        "top_bad_keep_current_by_dominant": feat_df[feat_df["strategy"] == "keep_current"].sort_values(
            "dominant_bin_ratio", ascending=False
        ).head(20).to_dict(orient="records"),
        "top_rank_compression": feat_df[feat_df["strategy"] == "rank_uniform_offset"].sort_values(
            "compression_factor", ascending=False
        ).head(20).to_dict(orient="records"),
        "compact_discrete_features": feat_df[feat_df["strategy"] == "discrete_compact_offset0"].sort_values(
            "raw_unique", ascending=True
        ).to_dict(orient="records"),
        "training_dynamics": dynamics,
    }

    save_json({"summary": summary, "features": rows}, feature_json)

    out_zip = Path("K512_C2_representation_diagnostics.zip")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in [feature_csv, feature_json]:
            z.write(p, p.as_posix())

    print("Done.")
    print("Wrote:", out_zip.resolve())
    print("Feature diag:", feature_csv)
    print("JSON diag:", feature_json)

    print("\n=== strategy counts ===")
    print(json.dumps(summary["strategy_counts"], indent=2))

    print("\n=== gate status ===")
    print(gate_status, ckpt_path)

    print("\n=== group summary ===")
    print(json.dumps(group_summary, indent=2)[:6000])

    print("\n=== training dynamics summary ===")
    print(json.dumps(dynamics, indent=2)[:4000])


if __name__ == "__main__":
    main()

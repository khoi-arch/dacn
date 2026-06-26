import argparse
import json
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


def bin_by_edges(z, edges):
    z = np.asarray(z, dtype=np.float64)
    b = np.searchsorted(edges, z, side="right") - 1
    b = np.clip(b, 0, len(edges) - 2)
    return b.astype(np.int64)


def offset_by_edges(z, bin_ids, edges):
    left = edges[bin_ids]
    right = edges[bin_ids + 1]
    width = right - left

    off = np.zeros_like(z, dtype=np.float64)
    ok = width > 1e-12

    off[ok] = (z[ok] - left[ok]) / width[ok]
    off[~ok] = 0.5

    return np.clip(off, 0.0, 1.0)


def diag_one_feature(z, feature, num_bins):
    z = np.asarray(z, dtype=np.float64)
    z = np.clip(z, 0.0, 1.0)

    n = int(z.shape[0])
    z_unique = int(np.unique(z).size)

    uniform_edges = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)
    uniform_bins = bin_by_edges(z, uniform_edges)
    uniform_counts = np.bincount(uniform_bins, minlength=num_bins)

    q = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)
    quantile_edges = np.quantile(z, q)
    quantile_edges[0] = 0.0
    quantile_edges[-1] = 1.0
    quantile_edges = np.clip(quantile_edges, 0.0, 1.0)

    quantile_bins = bin_by_edges(z, quantile_edges)
    quantile_counts = np.bincount(quantile_bins, minlength=num_bins)

    uniform_used = int(np.count_nonzero(uniform_counts))
    quantile_used = int(np.count_nonzero(quantile_counts))

    uniform_dom = float(uniform_counts.max() / max(n, 1))
    quantile_dom = float(quantile_counts.max() / max(n, 1))

    uniform_entropy = entropy_norm_from_counts(uniform_counts)
    quantile_entropy = entropy_norm_from_counts(quantile_counts)

    q_width = np.diff(quantile_edges)
    q_zero = int(np.sum(q_width <= 1e-12))
    q_dup = q_zero

    off = offset_by_edges(z, quantile_bins, quantile_edges)

    pos_width = q_width[q_width > 1e-12]
    if pos_width.size:
        q_width_min_positive = float(pos_width.min())
        q_width_median_positive = float(np.median(pos_width))
        q_width_max_positive = float(pos_width.max())
    else:
        q_width_min_positive = None
        q_width_median_positive = None
        q_width_max_positive = None

    return {
        "feature": feature,
        "n": n,
        "num_bins": int(num_bins),
        "z_min": float(z.min()) if n else None,
        "z_max": float(z.max()) if n else None,
        "z_num_unique": z_unique,

        "uniform_bins_used": uniform_used,
        "quantile_bins_used": quantile_used,
        "delta_bins_used": int(quantile_used - uniform_used),

        "uniform_empty_bins": int(num_bins - uniform_used),
        "quantile_empty_bins": int(num_bins - quantile_used),

        "uniform_dominant_bin_ratio": uniform_dom,
        "quantile_dominant_bin_ratio": quantile_dom,
        "delta_dominant_bin_ratio": float(quantile_dom - uniform_dom),
        "dominant_ratio_reduction": float(uniform_dom - quantile_dom),

        "uniform_max_bin_count": int(uniform_counts.max()),
        "quantile_max_bin_count": int(quantile_counts.max()),

        "uniform_entropy_norm": uniform_entropy,
        "quantile_entropy_norm": quantile_entropy,
        "delta_entropy_norm": float(quantile_entropy - uniform_entropy),

        "quantile_duplicate_edge_count": q_dup,
        "quantile_zero_width_bin_count": q_zero,
        "quantile_zero_width_sample_ratio": 0.0,

        "quantile_width_min_positive": q_width_min_positive,
        "quantile_width_median_positive": q_width_median_positive,
        "quantile_width_max_positive": q_width_max_positive,

        "offset_min": float(off.min()) if n else None,
        "offset_max": float(off.max()) if n else None,
        "offset_mean": float(off.mean()) if n else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--num-bins", type=int, required=True)
    ap.add_argument("--train-preprocessed", default="")
    ap.add_argument("--out-dir", default="03_outputs/bin_diag")
    args = ap.parse_args()

    K = int(args.K)
    B = int(args.num_bins)

    train_path = (
        Path(args.train_preprocessed)
        if args.train_preprocessed
        else Path(f"03_outputs/preprocessing/train_preprocessed_K{K}.csv")
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(str(train_path))

    df = pd.read_csv(train_path)

    features = [
        c for c in df.columns
        if c not in TARGET_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]

    rows = []
    for feat in features:
        rows.append(
            diag_one_feature(
                df[feat].to_numpy(dtype=np.float64),
                feat,
                B,
            )
        )

    summary = {
        "stage": "quantile_vs_uniform_bin_diag",
        "K": K,
        "num_bins": B,
        "input_train_preprocessed": str(train_path),
        "n_features": len(rows),

        "mean_uniform_dominant_bin_ratio": float(np.mean([r["uniform_dominant_bin_ratio"] for r in rows])),
        "mean_quantile_dominant_bin_ratio": float(np.mean([r["quantile_dominant_bin_ratio"] for r in rows])),
        "mean_dominant_ratio_reduction": float(np.mean([r["dominant_ratio_reduction"] for r in rows])),

        "mean_uniform_entropy_norm": float(np.mean([r["uniform_entropy_norm"] for r in rows])),
        "mean_quantile_entropy_norm": float(np.mean([r["quantile_entropy_norm"] for r in rows])),
        "mean_delta_entropy_norm": float(np.mean([r["delta_entropy_norm"] for r in rows])),

        "features_dominant_ratio_improved": int(sum(r["dominant_ratio_reduction"] > 1e-12 for r in rows)),
        "features_dominant_ratio_worsened": int(sum(r["dominant_ratio_reduction"] < -1e-12 for r in rows)),
        "features_entropy_improved": int(sum(r["delta_entropy_norm"] > 1e-12 for r in rows)),
        "features_with_quantile_zero_width_bins": int(sum(r["quantile_zero_width_bin_count"] > 0 for r in rows)),

        "top_dominant_ratio_reductions": sorted(
            rows,
            key=lambda r: r["dominant_ratio_reduction"],
            reverse=True,
        )[:20],
    }

    out = {
        "summary": summary,
        "features": rows,
    }

    json_path = out_dir / f"quantile_vs_uniform_bin_diag_K{K}_B{B}.json"
    csv_path = out_dir / f"quantile_vs_uniform_bin_diag_K{K}_B{B}.csv"

    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("Wrote:", json_path)
    print("Wrote:", csv_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
33_audit_attention_gradient_rootcause.py

Conditional attention / gradient / occlusion audit for C2 D3.

Purpose
-------
This script does NOT train. It is a root-cause follow-up to the overfit audit.
It focuses on samples where raw/token geometry is mixed or near true class but
CLS/model prediction is wrong and confident.

It reports three complementary explanations:
  1) Occlusion / neutralization logit-delta per feature (most causal diagnostic).
  2) Gradient x activation attribution on the Transformer input feature tokens.
  3) Attention diagnostics: last-layer CLS attention and attention rollout.

Interpretation rules:
  - Occlusion positive delta_margin on wrong samples:
      neutralizing the feature reduces (logit_pred - logit_true), so the feature
      supports the wrong predicted class over the true class.
  - Grad x activation positive on wrong samples:
      local first-order support for (logit_pred - logit_true).
  - Attention is routing, not causality. Use it only together with occlusion/grad.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def norm_label(x: Any) -> str:
    return str(x).strip()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def import_train_module(root: Path):
    src = root / "02_src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    p = src / "10_train_fusion_ablation_D0_D7.py"
    if not p.exists():
        raise FileNotFoundError(f"Missing training module: {p}")
    spec = importlib.util.spec_from_file_location("_dacn_fusion", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_default_paths(root: Path, args: argparse.Namespace) -> Dict[str, Path]:
    dataset = Path(args.dataset_npz) if args.dataset_npz else root / "03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_dataset.npz"
    metadata = Path(args.metadata_json) if args.metadata_json else root / "03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_metadata.json"
    run_dir = Path(args.run_dir) if args.run_dir else root / "03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact"
    ckpt = Path(args.checkpoint) if args.checkpoint else run_dir / "best_model.pt"
    overfit_dir = Path(args.overfit_audit_dir) if args.overfit_audit_dir else root / "03_outputs/audit_overfit_rootcause"
    return {
        "dataset": dataset,
        "metadata": metadata,
        "run_dir": run_dir,
        "checkpoint": ckpt,
        "overfit_dir": overfit_dir,
    }


def load_dataset(dataset_path: Path, metadata_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset npz: {dataset_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata json: {metadata_path}")
    data = dict(np.load(dataset_path, allow_pickle=True))
    meta = load_json(metadata_path)
    need = ["X_train_bin", "X_train_offset", "y_train", "X_val_bin", "X_val_offset", "y_val"]
    miss = [k for k in need if k not in data]
    if miss:
        raise ValueError(f"Dataset missing arrays: {miss}")
    return data, meta


def resolve_raw_paths(root: Path, args: argparse.Namespace) -> Tuple[Path, Path]:
    train = Path(args.train_raw) if args.train_raw else root / "01_split/train_raw.csv"
    val = Path(args.val_raw) if args.val_raw else root / "01_split/val_raw.csv"
    if not train.exists():
        raise FileNotFoundError(f"Missing train_raw: {train}")
    if not val.exists():
        raise FileNotFoundError(f"Missing val_raw: {val}")
    return train, val


def load_raw_scaled(root: Path, meta: Dict[str, Any], args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    feature_names = [str(x) for x in meta["feature_names"]]
    train_path, val_path = resolve_raw_paths(root, args)
    tr = pd.read_csv(train_path)
    va = pd.read_csv(val_path)
    missing = [f for f in feature_names if f not in tr.columns or f not in va.columns]
    if missing:
        raise ValueError(f"Raw CSV missing features: {missing[:10]}")
    Xtr = tr.loc[:, feature_names].to_numpy(np.float64)
    Xva = va.loc[:, feature_names].to_numpy(np.float64)
    mn = Xtr.min(axis=0)
    mx = Xtr.max(axis=0)
    den = mx - mn
    const = np.isclose(den, 0.0)
    den[const] = 1.0
    Str = (Xtr - mn) / den
    Sva = (Xva - mn) / den
    Str[:, const] = 0.5
    Sva[:, const] = 0.5
    Str = np.clip(Str, 0.0, 1.0).astype(np.float32)
    Sva = np.clip(Sva, 0.0, 1.0).astype(np.float32)
    info = {
        "train_raw": str(train_path),
        "val_raw": str(val_path),
        "scale": "train_minmax_clip_val",
        "n_constant_features": int(const.sum()),
    }
    return Str, Sva, info


def load_feature_info(root: Path, meta: Dict[str, Any]) -> pd.DataFrame:
    feature_names = [str(x) for x in meta["feature_names"]]
    candidates = [
        root / "03_outputs/audit_group_pair_geometry/02_group_tokenization/token_feature_audit_C2_K512.csv",
        root / "03_outputs/audit_c2_best/02_token_audit/token_feature_audit_train.csv",
        root / "03_outputs/audit_c2_best/02_token_audit/token_feature_audit_val.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            if "split" in df.columns:
                train_df = df[df["split"].astype(str).str.lower().eq("train")]
                if len(train_df):
                    df = train_df
            # keep one row per feature
            if "feature_idx" not in df.columns:
                df["feature_idx"] = df["feature"].map({f: i for i, f in enumerate(feature_names)})
            df = df.sort_values("feature_idx").drop_duplicates("feature_idx")
            cols = [c for c in ["feature_idx", "feature", "strategy", "rare_cell_ratio_trainref_le5", "rare_used_bin_ratio_le5", "entropy_norm", "dominant_bin_ratio"] if c in df.columns]
            out = df[cols].copy()
            if "strategy" not in out.columns:
                out["strategy"] = "unknown"
            return out

    # metadata fallback
    rows = []
    fm = meta.get("feature_meta", {})
    if isinstance(fm, list):
        by_name = {str(r.get("feature", r.get("name", ""))): r for r in fm if isinstance(r, dict)}
    elif isinstance(fm, dict):
        by_name = {str(k): v for k, v in fm.items() if isinstance(v, dict)}
    else:
        by_name = {}
    for i, f in enumerate(feature_names):
        rec = by_name.get(f, {})
        rows.append({
            "feature_idx": i,
            "feature": f,
            "strategy": str(rec.get("strategy", rec.get("selected_strategy", "unknown"))),
        })
    return pd.DataFrame(rows)


def build_values(X_offset: np.ndarray, X_cont: np.ndarray) -> np.ndarray:
    mask = np.ones_like(X_offset, dtype=np.float32)
    return np.stack([X_offset.astype(np.float32), X_cont.astype(np.float32), mask], axis=-1)


def label_names_from_meta(meta: Dict[str, Any]) -> List[str]:
    mapping = meta.get("label_mapping", {})
    if isinstance(mapping, dict) and mapping:
        return [norm_label(k) for k, _ in sorted(mapping.items(), key=lambda kv: int(kv[1]))]
    raise ValueError("metadata.label_mapping missing")


def reconstruct_model(train_mod, ckpt: Dict[str, Any], meta: Dict[str, Any], diagnosis_path: Path | None, device: torch.device):
    diag = load_json(diagnosis_path) if diagnosis_path and diagnosis_path.exists() else {}
    mcfg = diag.get("model_config", {})
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    label_names = label_names_from_meta(meta)
    n_features = int(meta.get("n_features", len(meta["feature_names"])))

    def get(name: str, default: Any):
        if name in mcfg:
            return mcfg[name]
        if name in config:
            return config[name]
        return default

    model = train_mod.FusionAblationTransformer(
        run_id="D3",
        num_bins=int(get("num_bins", 512)),
        n_features=n_features,
        num_classes=len(label_names),
        value_dim=int(get("value_dim", 32)),
        feature_dim=int(get("feature_dim", 32)),
        hidden_dim=int(get("hidden_dim", 128)),
        num_layers=int(get("num_layers", 3)),
        num_heads=int(get("num_heads", 4)),
        dropout=float(get("dropout", 0.1)),
        classifier_hidden_dim=int(get("classifier_hidden_dim", 128)),
        classifier_dropout=float(get("classifier_dropout", 0.1)),
        norm_first=bool(get("norm_first", True)),
        gate_init=float(get("gate_init", 0.0)),
    )
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[warn] load_state_dict missing:", missing[:10], "unexpected:", unexpected[:10])
    model.to(device)
    model.eval()
    return model, label_names


def forward_logits(model, tokens: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return model(tokens, values)


def forward_from_projected_tokens(model, tokens: torch.Tensor, values: torch.Tensor, *, require_grad_x: bool = False):
    # Same as model.forward, but exposes Transformer input feature tokens x [B,F,H].
    cell_emb = model.embedding(tokens, values)
    x = model.input_proj(cell_emb)
    if require_grad_x:
        x.requires_grad_(True)
        x.retain_grad()
    B = x.shape[0]
    cls = model.cls_token.expand(B, 1, model.hidden_dim)
    enc_in = torch.cat([cls, x], dim=1)
    encoded = model.encoder(enc_in)
    cls_out = encoded[:, 0, :]
    logits = model.classifier(cls_out)
    return logits, x, cls_out


def predict_all(model, X_bin: np.ndarray, V: np.ndarray, device: torch.device, batch_size: int = 512):
    logits_all = []
    probs_all = []
    for s in range(0, len(X_bin), batch_size):
        xb = torch.as_tensor(X_bin[s:s+batch_size], dtype=torch.long, device=device)
        vv = torch.as_tensor(V[s:s+batch_size], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = forward_logits(model, xb, vv)
            probs = F.softmax(logits, dim=-1)
        logits_all.append(logits.detach().cpu())
        probs_all.append(probs.detach().cpu())
    logits = torch.cat(logits_all).numpy()
    probs = torch.cat(probs_all).numpy()
    pred = probs.argmax(axis=1).astype(np.int64)
    conf = probs.max(axis=1).astype(np.float32)
    return logits, probs, pred, conf


def make_selection(root: Path, paths: Dict[str, Path], y_val: np.ndarray, label_names: List[str], pred: np.ndarray, conf: np.ndarray, args: argparse.Namespace) -> pd.DataFrame:
    cross_path = paths["overfit_dir"] / "02_knn_raw_token_cls/val_cross_space_rootcause_per_sample.csv"
    if not cross_path.exists():
        raise FileNotFoundError(f"Missing overfit cross-space audit: {cross_path}\nRun overfit rootcause audit first.")
    df = pd.read_csv(cross_path)
    # Normalize labels and ensure numeric sample_idx.
    df["sample_idx"] = df["sample_idx"].astype(int)
    df["true_class"] = df["true_class"].map(norm_label)
    df["pred_class"] = df["pred_class"].map(norm_label)
    df["correct"] = df["correct"].astype(bool)
    # In case predictions were recomputed with slight formatting differences, keep audit labels as source of truth.
    wrong = ~df["correct"]

    raw_mixed_or_true = df["raw_scaled_category"].isin([
        "mixed_neighbors_ambiguous",
        "model_boundary_failure_knn_true_neighbors",
        "correct_but_neighbors_mixed",
    ])
    tok_mixed_or_true = df["token_bin_offset_category"].isin([
        "mixed_neighbors_ambiguous",
        "model_boundary_failure_knn_true_neighbors",
        "correct_but_neighbors_mixed",
    ])
    cls_pred_or_ood = df["cls_classifier_input_category"].isin([
        "feature_space_overlap_with_pred_class",
        "OOD_or_distribution_shift",
    ]) | df.get("cls_classifier_input_is_ood95", False).astype(bool)

    feature_overlap = wrong & (
        df["raw_scaled_category"].eq("feature_space_overlap_with_pred_class") |
        df["token_bin_offset_category"].eq("feature_space_overlap_with_pred_class")
    ) & df["cls_classifier_input_category"].eq("feature_space_overlap_with_pred_class")

    model_amplification = wrong & (raw_mixed_or_true | tok_mixed_or_true) & cls_pred_or_ood
    cls_ood_confident = wrong & df.get("cls_classifier_input_is_ood95", False).astype(bool)

    correct = df["correct"] & (df["confidence"] >= float(args.correct_conf_threshold))

    pieces = []
    def add_subset(name: str, mask, order_conf_desc: bool = True):
        sub = df[mask].copy()
        if len(sub) == 0:
            return
        # Balance by true->pred pair if wrong; by true class if correct.
        group_cols = ["true_class", "pred_class"] if name != "correct_high_conf_control" else ["true_class"]
        take_parts = []
        max_per_group = int(args.max_samples_per_pair)
        for _, g in sub.groupby(group_cols, dropna=False):
            g = g.sort_values("confidence", ascending=not order_conf_desc)
            take_parts.append(g.head(max_per_group))
        out = pd.concat(take_parts, ignore_index=True) if take_parts else sub.head(0)
        if len(out) > int(args.max_samples_per_subset):
            out = out.sort_values("confidence", ascending=not order_conf_desc).head(int(args.max_samples_per_subset))
        out["audit_subset"] = name
        pieces.append(out)

    add_subset("model_amplification", model_amplification, True)
    add_subset("feature_space_overlap", feature_overlap, True)
    add_subset("cls_ood_confident", cls_ood_confident, True)
    add_subset("all_wrong_high_conf", wrong & (df["confidence"] >= float(args.wrong_conf_threshold)), True)
    add_subset("correct_high_conf_control", correct, True)

    if not pieces:
        raise RuntimeError("No selected samples. Check overfit audit columns/thresholds.")
    sel = pd.concat(pieces, ignore_index=True)
    # Deduplicate within subset only; one sample can belong to multiple diagnostic subsets.
    sel = sel.drop_duplicates(["audit_subset", "sample_idx"])
    # numeric labels for downstream
    label_to_id = {name: i for i, name in enumerate(label_names)}
    sel["true_id"] = sel["true_class"].map(label_to_id).astype(int)
    sel["pred_id"] = sel["pred_class"].map(label_to_id).astype(int)
    return sel


def compute_occlusion_for_subset(model, X_val_bin, V_val, sel_df: pd.DataFrame, train_neutral: Dict[str, np.ndarray], features: pd.DataFrame, device: torch.device, batch_size: int) -> pd.DataFrame:
    idx = sel_df["sample_idx"].to_numpy(int)
    true_id = sel_df["true_id"].to_numpy(int)
    pred_id = sel_df["pred_id"].to_numpy(int)
    subset_names = sel_df["audit_subset"].to_numpy(str)
    pairs = (sel_df["true_class"].astype(str) + "→" + sel_df["pred_class"].astype(str)).to_numpy(str)
    correct_mask = sel_df["correct"].to_numpy(bool)

    X = X_val_bin[idx].copy()
    V = V_val[idx].copy()
    N, Fnum = X.shape
    xb = torch.as_tensor(X, dtype=torch.long, device=device)
    vv = torch.as_tensor(V, dtype=torch.float32, device=device)
    with torch.no_grad():
        base_logits = forward_logits(model, xb, vv).detach().cpu().numpy()

    # For correct controls pred=true; use alt = best non-true class from base logits.
    alt_id = base_logits.copy()
    for i in range(N):
        alt_id[i, true_id[i]] = -1e9
    alt_id = alt_id.argmax(axis=1).astype(int)
    target_a = pred_id.copy()  # wrong predicted class or true for correct controls
    target_b = true_id.copy()
    # Correct control: margin true - strongest alternative.
    target_a[correct_mask] = true_id[correct_mask]
    target_b[correct_mask] = alt_id[correct_mask]

    rows = []
    base_margin = base_logits[np.arange(N), target_a] - base_logits[np.arange(N), target_b]
    base_pred_logit = base_logits[np.arange(N), target_a]
    base_true_logit = base_logits[np.arange(N), target_b]

    neutral_bin = train_neutral["bin"]
    neutral_offset = train_neutral["offset"]
    neutral_cont = train_neutral["cont"]

    for f in range(Fnum):
        Xm = X.copy()
        Vm = V.copy()
        Xm[:, f] = int(neutral_bin[f])
        Vm[:, f, 0] = float(neutral_offset[f])
        Vm[:, f, 1] = float(neutral_cont[f])
        Vm[:, f, 2] = 1.0
        out_logits = []
        for s in range(0, N, batch_size):
            xb = torch.as_tensor(Xm[s:s+batch_size], dtype=torch.long, device=device)
            vv = torch.as_tensor(Vm[s:s+batch_size], dtype=torch.float32, device=device)
            with torch.no_grad():
                out_logits.append(forward_logits(model, xb, vv).detach().cpu().numpy())
        lg = np.concatenate(out_logits, axis=0)
        new_margin = lg[np.arange(N), target_a] - lg[np.arange(N), target_b]
        delta_margin = base_margin - new_margin
        delta_a_logit = base_pred_logit - lg[np.arange(N), target_a]
        delta_b_logit = base_true_logit - lg[np.arange(N), target_b]
        for subset in sorted(set(subset_names)):
            m = subset_names == subset
            if not m.any():
                continue
            rows.append({
                "scope": "subset",
                "audit_subset": subset,
                "pair": "ALL",
                "feature_idx": f,
                "n_samples": int(m.sum()),
                "delta_margin_mean": float(delta_margin[m].mean()),
                "delta_margin_median": float(np.median(delta_margin[m])),
                "delta_margin_positive_frac": float((delta_margin[m] > 0).mean()),
                "delta_target_a_logit_mean": float(delta_a_logit[m].mean()),
                "delta_target_b_logit_mean": float(delta_b_logit[m].mean()),
                "abs_delta_margin_mean": float(np.abs(delta_margin[m]).mean()),
            })
        for pair in sorted(set(pairs[~correct_mask])):
            m = (pairs == pair) & (~correct_mask)
            if not m.any():
                continue
            rows.append({
                "scope": "pair",
                "audit_subset": "ALL_WRONG",
                "pair": pair,
                "feature_idx": f,
                "n_samples": int(m.sum()),
                "delta_margin_mean": float(delta_margin[m].mean()),
                "delta_margin_median": float(np.median(delta_margin[m])),
                "delta_margin_positive_frac": float((delta_margin[m] > 0).mean()),
                "delta_target_a_logit_mean": float(delta_a_logit[m].mean()),
                "delta_target_b_logit_mean": float(delta_b_logit[m].mean()),
                "abs_delta_margin_mean": float(np.abs(delta_margin[m]).mean()),
            })
    res = pd.DataFrame(rows)
    res = res.merge(features, on="feature_idx", how="left")
    return res


def compute_gradxinput_for_subset(model, X_val_bin, V_val, sel_df: pd.DataFrame, features: pd.DataFrame, device: torch.device, batch_size: int) -> pd.DataFrame:
    rows = []
    for subset, sdf in sel_df.groupby("audit_subset"):
        # Process in batches, accumulate per sample feature attribution.
        all_attr = []
        all_abs = []
        pairs = []
        idx_all = sdf["sample_idx"].to_numpy(int)
        true_all = sdf["true_id"].to_numpy(int)
        pred_all = sdf["pred_id"].to_numpy(int)
        correct_all = sdf["correct"].to_numpy(bool)
        pair_all = (sdf["true_class"].astype(str) + "→" + sdf["pred_class"].astype(str)).to_numpy(str)

        for s in range(0, len(sdf), batch_size):
            idx = idx_all[s:s+batch_size]
            true_id = true_all[s:s+batch_size]
            pred_id = pred_all[s:s+batch_size]
            corr = correct_all[s:s+batch_size]
            xb = torch.as_tensor(X_val_bin[idx], dtype=torch.long, device=device)
            vv = torch.as_tensor(V_val[idx], dtype=torch.float32, device=device)

            model.zero_grad(set_to_none=True)
            logits, x, _ = forward_from_projected_tokens(model, xb, vv, require_grad_x=True)
            with torch.no_grad():
                tmp = logits.detach().clone()
                for i, t in enumerate(true_id):
                    tmp[i, int(t)] = -1e9
                alt = tmp.argmax(dim=1)
            a = torch.as_tensor(pred_id, dtype=torch.long, device=device)
            b = torch.as_tensor(true_id, dtype=torch.long, device=device)
            if bool(corr.any()):
                a = a.clone(); b = b.clone()
                cm = torch.as_tensor(corr, dtype=torch.bool, device=device)
                true_t = torch.as_tensor(true_id, dtype=torch.long, device=device)
                a[cm] = true_t[cm]
                b[cm] = alt[cm]
            margin = logits[torch.arange(len(idx), device=device), a] - logits[torch.arange(len(idx), device=device), b]
            margin.sum().backward()
            attr = (x.grad * x).sum(dim=-1).detach().cpu().numpy()
            all_attr.append(attr)
            all_abs.append(np.abs(attr))
            pairs.extend(list(pair_all[s:s+batch_size]))
        A = np.concatenate(all_attr, axis=0)
        Abs = np.concatenate(all_abs, axis=0)
        pairs = np.array(pairs)
        for f in range(A.shape[1]):
            rows.append({
                "scope": "subset",
                "audit_subset": subset,
                "pair": "ALL",
                "feature_idx": f,
                "n_samples": int(A.shape[0]),
                "gradxinput_mean": float(A[:, f].mean()),
                "gradxinput_median": float(np.median(A[:, f])),
                "gradxinput_positive_frac": float((A[:, f] > 0).mean()),
                "abs_gradxinput_mean": float(Abs[:, f].mean()),
            })
        # pair aggregation for wrong samples only
        for pair in sorted(set(pairs)):
            if "→" not in pair:
                continue
            m = pairs == pair
            if not m.any():
                continue
            for f in range(A.shape[1]):
                rows.append({
                    "scope": "pair_subset",
                    "audit_subset": subset,
                    "pair": pair,
                    "feature_idx": f,
                    "n_samples": int(m.sum()),
                    "gradxinput_mean": float(A[m, f].mean()),
                    "gradxinput_median": float(np.median(A[m, f])),
                    "gradxinput_positive_frac": float((A[m, f] > 0).mean()),
                    "abs_gradxinput_mean": float(Abs[m, f].mean()),
                })
    res = pd.DataFrame(rows).merge(features, on="feature_idx", how="left")
    return res


def install_attention_capture(model):
    store: Dict[int, torch.Tensor] = {}
    originals = []
    layers = list(model.encoder.layers)
    for li, layer in enumerate(layers):
        mha = layer.self_attn
        orig_forward = mha.forward
        originals.append((mha, orig_forward))
        def make_wrapped(layer_idx, orig):
            def wrapped(self, *args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                out, weights = orig(*args, **kwargs)
                if weights is not None:
                    store[layer_idx] = weights.detach().cpu()
                return out, weights
            return types.MethodType(wrapped, mha)
        mha.forward = make_wrapped(li, orig_forward)
    def restore():
        for mha, orig in originals:
            mha.forward = orig
    return store, restore, len(layers)


def rollout_from_store(store: Dict[int, torch.Tensor], n_layers: int) -> np.ndarray | None:
    if not store:
        return None
    mats = []
    for li in range(n_layers):
        if li not in store:
            return None
        w = store[li]  # [B,H,L,L]
        A = w.mean(dim=1)  # [B,L,L]
        B, L, _ = A.shape
        eye = torch.eye(L).unsqueeze(0)
        A = (A + eye) / 2.0
        A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        mats.append(A)
    joint = mats[0]
    for A in mats[1:]:
        joint = torch.bmm(A, joint)
    return joint[:, 0, 1:].numpy()


def compute_attention_for_subset(model, X_val_bin, V_val, sel_df: pd.DataFrame, features: pd.DataFrame, device: torch.device, batch_size: int) -> pd.DataFrame:
    rows = []
    store, restore, n_layers = install_attention_capture(model)
    try:
        for subset, sdf in sel_df.groupby("audit_subset"):
            idx_all = sdf["sample_idx"].to_numpy(int)
            pair_all = (sdf["true_class"].astype(str) + "→" + sdf["pred_class"].astype(str)).to_numpy(str)
            last_all = []
            rollout_all = []
            pairs = []
            for s in range(0, len(sdf), batch_size):
                store.clear()
                idx = idx_all[s:s+batch_size]
                xb = torch.as_tensor(X_val_bin[idx], dtype=torch.long, device=device)
                vv = torch.as_tensor(V_val[idx], dtype=torch.float32, device=device)
                with torch.no_grad():
                    _ = forward_logits(model, xb, vv)
                last = store.get(n_layers - 1)
                if last is None:
                    raise RuntimeError("Could not capture last-layer attention weights.")
                # last: [B,H,L,L], CLS query to feature tokens
                cls_last = last[:, :, 0, 1:].mean(dim=1).numpy()
                last_all.append(cls_last)
                ro = rollout_from_store(store, n_layers)
                if ro is not None:
                    rollout_all.append(ro)
                pairs.extend(list(pair_all[s:s+batch_size]))
            Last = np.concatenate(last_all, axis=0)
            Roll = np.concatenate(rollout_all, axis=0) if rollout_all else np.full_like(Last, np.nan)
            pairs = np.array(pairs)
            for f in range(Last.shape[1]):
                rows.append({
                    "scope": "subset",
                    "audit_subset": subset,
                    "pair": "ALL",
                    "feature_idx": f,
                    "n_samples": int(Last.shape[0]),
                    "last_cls_attention_mean": float(Last[:, f].mean()),
                    "last_cls_attention_median": float(np.median(Last[:, f])),
                    "rollout_attention_mean": float(np.nanmean(Roll[:, f])),
                    "rollout_attention_median": float(np.nanmedian(Roll[:, f])),
                })
            for pair in sorted(set(pairs)):
                m = pairs == pair
                if not m.any():
                    continue
                for f in range(Last.shape[1]):
                    rows.append({
                        "scope": "pair_subset",
                        "audit_subset": subset,
                        "pair": pair,
                        "feature_idx": f,
                        "n_samples": int(m.sum()),
                        "last_cls_attention_mean": float(Last[m, f].mean()),
                        "last_cls_attention_median": float(np.median(Last[m, f])),
                        "rollout_attention_mean": float(np.nanmean(Roll[m, f])),
                        "rollout_attention_median": float(np.nanmedian(Roll[m, f])),
                    })
        return pd.DataFrame(rows).merge(features, on="feature_idx", how="left")
    finally:
        restore()


def minmax01(s: pd.Series) -> pd.Series:
    x = s.astype(float)
    mn = x.min(); mx = x.max()
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
        return pd.Series(np.zeros(len(x)), index=s.index)
    return (x - mn) / (mx - mn)


def make_consensus(occ: pd.DataFrame, grad: pd.DataFrame, attn: pd.DataFrame) -> pd.DataFrame:
    # subset scope only for concise consensus
    o = occ[occ["scope"].eq("subset")][["audit_subset", "pair", "feature_idx", "feature", "strategy", "delta_margin_mean", "delta_margin_positive_frac", "abs_delta_margin_mean"]].copy()
    g = grad[grad["scope"].eq("subset")][["audit_subset", "pair", "feature_idx", "gradxinput_mean", "gradxinput_positive_frac", "abs_gradxinput_mean"]].copy()
    a = attn[attn["scope"].eq("subset")][["audit_subset", "pair", "feature_idx", "last_cls_attention_mean", "rollout_attention_mean"]].copy()
    df = o.merge(g, on=["audit_subset", "pair", "feature_idx"], how="outer").merge(a, on=["audit_subset", "pair", "feature_idx"], how="outer")
    # restore feature/strategy if merge made missing
    if "feature" not in df.columns or df["feature"].isna().any():
        fmap = o.drop_duplicates("feature_idx").set_index("feature_idx")[["feature", "strategy"]]
        df = df.merge(fmap, on="feature_idx", how="left", suffixes=("", "_fix"))
        for c in ["feature", "strategy"]:
            if c + "_fix" in df.columns:
                df[c] = df[c].fillna(df[c + "_fix"])
                df = df.drop(columns=[c + "_fix"])
    parts = []
    for subset, sdf in df.groupby("audit_subset"):
        sdf = sdf.copy()
        sdf["occ_support_score"] = minmax01(sdf["delta_margin_mean"].fillna(0.0).clip(lower=0.0))
        sdf["grad_support_score"] = minmax01(sdf["gradxinput_mean"].fillna(0.0).clip(lower=0.0))
        sdf["attn_score"] = minmax01(sdf["rollout_attention_mean"].fillna(sdf.get("last_cls_attention_mean", 0.0)).fillna(0.0))
        sdf["consensus_wrong_support_score"] = 0.50 * sdf["occ_support_score"] + 0.35 * sdf["grad_support_score"] + 0.15 * sdf["attn_score"]
        parts.append(sdf)
    return pd.concat(parts, ignore_index=True) if parts else df


def group_summary(df: pd.DataFrame, value_cols: List[str], group_cols: List[str]) -> pd.DataFrame:
    agg = {c: ["mean", "median", "max"] for c in value_cols if c in df.columns}
    if not agg:
        return pd.DataFrame()
    out = df.groupby(group_cols + ["strategy"], dropna=False).agg(agg)
    out.columns = ["_".join([str(x) for x in c if x]) for c in out.columns]
    out = out.reset_index()
    return out


def write_top_tables(out_dir: Path, consensus: pd.DataFrame, occ: pd.DataFrame, grad: pd.DataFrame, attn: pd.DataFrame):
    ensure_dir(out_dir / "04_consensus")
    for subset, sdf in consensus.groupby("audit_subset"):
        sdf.sort_values("consensus_wrong_support_score", ascending=False).head(30).to_csv(
            out_dir / "04_consensus" / f"top30_consensus_{subset}.csv", index=False
        )
    # Pair specific top by occlusion, wrong samples only.
    ensure_dir(out_dir / "01_occlusion")
    pair_occ = occ[occ["scope"].eq("pair")].copy()
    for pair, sdf in pair_occ.groupby("pair"):
        safe = pair.replace("→", "_to_").replace("/", "_").replace(" ", "")
        sdf.sort_values("delta_margin_mean", ascending=False).head(30).to_csv(
            out_dir / "01_occlusion" / f"top30_occlusion_{safe}.csv", index=False
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conditional attention/gradient/occlusion root-cause audit for C2 D3.")
    p.add_argument("--dataset-npz", default="")
    p.add_argument("--metadata-json", default="")
    p.add_argument("--run-dir", default="")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--overfit-audit-dir", default="")
    p.add_argument("--train-raw", default="")
    p.add_argument("--val-raw", default="")
    p.add_argument("--out-dir", default="03_outputs/audit_attention_gradient_rootcause")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-samples-per-subset", type=int, default=320)
    p.add_argument("--max-samples-per-pair", type=int, default=80)
    p.add_argument("--wrong-conf-threshold", type=float, default=0.70)
    p.add_argument("--correct-conf-threshold", type=float, default=0.95)
    return p.parse_args()


def main():
    args = parse_args()
    root = repo_root()
    out_dir = root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    ensure_dir(out_dir)
    print("[attn-grad-audit] root:", root)
    print("[attn-grad-audit] out:", out_dir)

    paths = find_default_paths(root, args)
    for k, p in paths.items():
        if k != "overfit_dir" and not p.exists():
            raise FileNotFoundError(f"Missing {k}: {p}")
    if not paths["overfit_dir"].exists():
        raise FileNotFoundError(f"Missing overfit audit dir: {paths['overfit_dir']}")

    data, meta = load_dataset(paths["dataset"], paths["metadata"])
    Xtr_bin = data["X_train_bin"].astype(np.int64)
    Xva_bin = data["X_val_bin"].astype(np.int64)
    Otr = data["X_train_offset"].astype(np.float32)
    Ova = data["X_val_offset"].astype(np.float32)
    y_val = data["y_val"].astype(np.int64)
    Xtr_cont, Xva_cont, raw_info = load_raw_scaled(root, meta, args)
    Vtr = build_values(Otr, Xtr_cont)
    Vva = build_values(Ova, Xva_cont)

    features = load_feature_info(root, meta)
    if "feature" not in features.columns:
        features["feature"] = [str(x) for x in meta["feature_names"]]
    if "strategy" not in features.columns:
        features["strategy"] = "unknown"
    features["feature_idx"] = features["feature_idx"].astype(int)
    features = features.sort_values("feature_idx").reset_index(drop=True)

    train_mod = import_train_module(root)
    device = pick_device(args.device)
    print("[attn-grad-audit] device:", device)
    ckpt = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    diagnosis_path = paths["run_dir"] / "diagnosis_summary.json"
    model, label_names = reconstruct_model(train_mod, ckpt, meta, diagnosis_path, device)
    print("[attn-grad-audit] labels:", label_names)

    # Recompute predictions to sanity check and to support fallback.
    _, _, pred, conf = predict_all(model, Xva_bin, Vva, device=device, batch_size=int(args.batch_size))

    sel = make_selection(root, paths, y_val, label_names, pred, conf, args)
    ensure_dir(out_dir / "00_sample_selection")
    sel.to_csv(out_dir / "00_sample_selection/selected_samples.csv", index=False)
    sel_summary = sel.groupby(["audit_subset", "true_class", "pred_class"], dropna=False).agg(
        n_samples=("sample_idx", "count"),
        confidence_mean=("confidence", "mean"),
        rare_mean=("rare_cell_count", "mean"),
        raw_true_frac_mean=("raw_scaled_knn_true_frac", "mean"),
        raw_pred_frac_mean=("raw_scaled_knn_pred_frac", "mean"),
        token_true_frac_mean=("token_bin_offset_knn_true_frac", "mean"),
        token_pred_frac_mean=("token_bin_offset_knn_pred_frac", "mean"),
        cls_true_frac_mean=("cls_classifier_input_knn_true_frac", "mean"),
        cls_pred_frac_mean=("cls_classifier_input_knn_pred_frac", "mean"),
    ).reset_index()
    sel_summary.to_csv(out_dir / "00_sample_selection/sample_selection_summary.csv", index=False)
    print("[attn-grad-audit] selected samples:", len(sel))
    print(sel["audit_subset"].value_counts().to_string())

    # Neutralization baselines: train medians per feature.
    train_neutral = {
        "bin": np.median(Xtr_bin, axis=0).round().astype(np.int64),
        "offset": np.median(Otr, axis=0).astype(np.float32),
        "cont": np.median(Xtr_cont, axis=0).astype(np.float32),
    }

    print("[attn-grad-audit] computing occlusion/logit-delta ...")
    occ = compute_occlusion_for_subset(model, Xva_bin, Vva, sel, train_neutral, features, device, batch_size=int(args.batch_size))
    ensure_dir(out_dir / "01_occlusion")
    occ.to_csv(out_dir / "01_occlusion/feature_occlusion_all.csv", index=False)
    group_summary(occ[occ["scope"].eq("subset")], ["delta_margin_mean", "delta_margin_positive_frac", "abs_delta_margin_mean"], ["audit_subset"]).to_csv(
        out_dir / "01_occlusion/group_occlusion_by_subset.csv", index=False
    )
    group_summary(occ[occ["scope"].eq("pair")], ["delta_margin_mean", "delta_margin_positive_frac", "abs_delta_margin_mean"], ["pair"]).to_csv(
        out_dir / "01_occlusion/group_occlusion_by_pair.csv", index=False
    )

    print("[attn-grad-audit] computing gradient x input ...")
    grad = compute_gradxinput_for_subset(model, Xva_bin, Vva, sel, features, device, batch_size=int(args.batch_size))
    ensure_dir(out_dir / "02_gradxinput")
    grad.to_csv(out_dir / "02_gradxinput/feature_gradxinput_all.csv", index=False)
    group_summary(grad[grad["scope"].eq("subset")], ["gradxinput_mean", "gradxinput_positive_frac", "abs_gradxinput_mean"], ["audit_subset"]).to_csv(
        out_dir / "02_gradxinput/group_gradxinput_by_subset.csv", index=False
    )

    print("[attn-grad-audit] computing attention capture ...")
    ensure_dir(out_dir / "03_attention")
    try:
        attn = compute_attention_for_subset(model, Xva_bin, Vva, sel, features, device, batch_size=int(args.batch_size))
        attention_status = {"attention_capture_ok": True, "attention_error": ""}
    except Exception as e:
        # Attention weights are optional for this audit. In some PyTorch TransformerEncoder
        # implementations, MultiheadAttention weights are not exposed through hooks because
        # the encoder calls self_attn with need_weights=False / optimized paths.
        # Occlusion and grad×input are still valid and are the primary causal evidence.
        print(f"[attn-grad-audit][WARN] attention capture failed; continuing with occlusion+grad only: {type(e).__name__}: {e}")
        base = occ[occ["scope"].eq("subset")][["audit_subset", "pair", "feature_idx", "feature", "strategy", "n_samples"]].drop_duplicates().copy()
        if base.empty:
            base = features[["feature_idx", "feature", "strategy"]].copy()
            base["audit_subset"] = "ALL"
            base["pair"] = "ALL"
            base["n_samples"] = 0
        base["scope"] = "subset"
        base["last_cls_attention_mean"] = np.nan
        base["last_cls_attention_median"] = np.nan
        base["rollout_attention_mean"] = np.nan
        base["rollout_attention_median"] = np.nan
        attn = base
        attention_status = {"attention_capture_ok": False, "attention_error": f"{type(e).__name__}: {e}"}
    attn.to_csv(out_dir / "03_attention/feature_attention_all.csv", index=False)
    group_summary(attn[attn["scope"].eq("subset")], ["last_cls_attention_mean", "rollout_attention_mean"], ["audit_subset"]).to_csv(
        out_dir / "03_attention/group_attention_by_subset.csv", index=False
    )

    print("[attn-grad-audit] making consensus tables ...")
    consensus = make_consensus(occ, grad, attn)
    ensure_dir(out_dir / "04_consensus")
    consensus.to_csv(out_dir / "04_consensus/feature_consensus_by_subset.csv", index=False)
    group_summary(consensus, ["consensus_wrong_support_score", "occ_support_score", "grad_support_score", "attn_score"], ["audit_subset"]).to_csv(
        out_dir / "04_consensus/group_consensus_by_subset.csv", index=False
    )
    write_top_tables(out_dir, consensus, occ, grad, attn)

    info = {
        "paths": {k: str(v) for k, v in paths.items()},
        "raw_info": raw_info,
        "device": str(device),
        "n_features": int(len(features)),
        "label_names": label_names,
        "selected_samples_total_rows": int(len(sel)),
        "selected_samples_by_subset": sel["audit_subset"].value_counts().to_dict(),
        "attention_status": attention_status,
        "notes": [
            "Occlusion delta_margin_mean > 0 on wrong samples means the feature supports the wrong predicted class over the true class.",
            "Gradient x input is local first-order evidence; use with occlusion.",
            "Attention is routing, not causality; interpret only when it agrees with occlusion/gradient.",
        ],
    }
    (out_dir / "audit_attention_gradient_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown summary with just file map and reading order, not conclusions.
    md = []
    md.append("# Attention / Gradient / Occlusion Root-Cause Audit")
    md.append("")
    md.append("This audit does not train. It explains selected samples from the overfit root-cause audit.")
    md.append("")
    md.append("## Reading order")
    md.append("1. `00_sample_selection/sample_selection_summary.csv`")
    md.append("2. `01_occlusion/feature_occlusion_all.csv` and `01_occlusion/top30_occlusion_*.csv`")
    md.append("3. `02_gradxinput/feature_gradxinput_all.csv`")
    md.append("4. `03_attention/feature_attention_all.csv`")
    md.append("5. `04_consensus/feature_consensus_by_subset.csv` and `04_consensus/top30_consensus_*.csv`")
    md.append("")
    md.append("## Key interpretation")
    md.append("- Occlusion is the strongest diagnostic here. Positive `delta_margin_mean` on wrong samples means neutralizing the feature reduces the wrong-vs-true logit margin.")
    md.append("- Grad×input should agree with occlusion for a stable feature-level explanation.")
    md.append("- Attention is not causal. It should be used as routing evidence only.")
    md.append("")
    md.append("## Selected samples")
    md.append(sel["audit_subset"].value_counts().to_string())
    (out_dir / "audit_attention_gradient_summary.md").write_text("\n".join(md), encoding="utf-8")

    print("[attn-grad-audit] done:", out_dir)


if __name__ == "__main__":
    main()

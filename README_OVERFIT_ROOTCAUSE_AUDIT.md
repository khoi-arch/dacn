# Overfit root-cause audit for C2 D3

This package does **not** train and does **not** change preprocessing/model.
It audits the existing C2 D3 best model to classify the overfit/generalization error into:

- train/val pair gap vs local underfit
- raw/token/CLS kNN overlap vs model-boundary failure vs OOD
- group-wise overfit by neutralizing feature groups at inference
- CLS/embedding centroid geometry
- rare token as cause vs symptom

## Files

- `02_src/32_audit_overfit_rootcause.py`
- `run_overfit_rootcause_audit_local.py`

## Required existing files in repo

The script auto-detects these paths:

```text
03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_dataset.npz
03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_metadata.json
03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/best_model.pt
03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/diagnosis_summary.json
01_split/train_raw.csv
01_split/val_raw.csv
```

## Run

```bash
cd ~/Documents/dacn
python -u run_overfit_rootcause_audit_local.py
```

If kNN is too slow, use a stratified train subset:

```bash
cd ~/Documents/dacn
python -u 02_src/32_audit_overfit_rootcause.py \
  --out-dir 03_outputs/audit_overfit_rootcause \
  --rare-threshold 5 \
  --knn-k 25 \
  --max-train-knn 30000
```

## Main outputs

```text
03_outputs/audit_overfit_rootcause/
  01_train_val_error_gap/
    split_summary.csv
    per_class_train_val.csv
    confusion_pair_train_val_long.csv
    pair_error_gap_train_vs_val.csv

  02_knn_raw_token_cls/
    raw_scaled/val_knn_sample_rootcause_raw_scaled.csv
    raw_scaled/val_wrong_pair_knn_rootcause_raw_scaled.csv
    token_bin_offset/val_wrong_pair_knn_rootcause_token_bin_offset.csv
    cls_classifier_input/val_wrong_pair_knn_rootcause_cls_classifier_input.csv
    wrong_pair_cross_space_rootcause_summary.csv
    val_cross_space_rootcause_per_sample.csv

  03_group_masking/
    group_neutralization_summary.csv
    group_neutralization_pair_changes.csv

  04_cls_geometry/
    val_centroid_distance_raw_scaled.csv
    val_centroid_distance_token_bin_offset.csv
    val_centroid_distance_cls_classifier_input.csv
    class_centroid_shift_*.csv
    wrong_pair_centroid_summary_*.csv

  05_rare_causal/
    rare_count_correct_vs_wrong.csv
    rare_by_knn_rootcause.csv
    wrong_pair_high_rare_rootcause.csv

  audit_overfit_rootcause_summary.md
```

## How to interpret

- Wrong val samples close to predicted class in raw/token kNN: feature overlap/ambiguity, not pure model overfit.
- Wrong val samples close to true class in raw/token but close to predicted class in CLS: learned representation/boundary failure.
- Wrong val samples OOD in raw/token: train-val distribution shift.
- Same pair wrong on train and val: local underfit or inherent class overlap.
- A group whose train neutralization drop is much larger than val drop: model overuses that group on train.
- High rare + true-neighbor kNN: rare-token model failure.
- High rare + pred/mixed/OOD kNN: rare is a symptom of overlap/sparsity, not a standalone cause.

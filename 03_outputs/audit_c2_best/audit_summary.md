# C2 D3 Best Audit Summary

## Inputs

- dataset: `/home/pak/Documents/dacn/03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_dataset.npz`
- metadata: `/home/pak/Documents/dacn/03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_metadata.json`
- run_dir: `/home/pak/Documents/dacn/03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact`
- checkpoint: `/home/pak/Documents/dacn/03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/best_model.pt`

## What this audit tests

Hypothesis: model has moderate overfit and the most likely driver is rare/sparse tokens. This audit does not change the model; it checks whether validation errors are actually enriched with rare/unseen tokens.

## Result snapshot

- train macro-F1: `0.910003459734893`
- val macro-F1: `0.8171466447478825`
- gap: `0.0928568149870106`

## Token snapshot

```json
{
  "train": {
    "split": "train",
    "n_rows": 46876,
    "n_features": 55,
    "num_bins": 512,
    "mean_bins_used": 251.23636363636365,
    "mean_rare_used_bin_ratio_le5": 0.22124805807051096,
    "mean_rare_cell_ratio_trainref_le5": 0.0014118486684405278,
    "mean_unseen_cell_ratio_trainref": 0.0,
    "mean_entropy_norm": 0.4896269568753839,
    "mean_dominant_bin_ratio": 0.3440706234630631
  },
  "val": {
    "split": "val",
    "n_rows": 11720,
    "n_features": 55,
    "num_bins": 512,
    "mean_bins_used": 229.07272727272726,
    "mean_rare_used_bin_ratio_le5": 0.3420249867091436,
    "mean_rare_cell_ratio_trainref_le5": 0.0017778467266521872,
    "mean_unseen_cell_ratio_trainref": 9.30809804529941e-05,
    "mean_entropy_norm": 0.48804786155319285,
    "mean_dominant_bin_ratio": 0.34247595408004966
  }
}
```

## Embedding snapshot

```json
{
  "shared_bin_embedding_num_embeddings": 513,
  "shared_bin_embedding_dim": 32,
  "bin_embedding_norm_mean": 0.46095719933509827,
  "bin_embedding_norm_std": 0.08225132524967194,
  "bin_step_norm_mean": 0.5913729667663574,
  "bin_step_norm_std": 0.10516534000635147,
  "cont_gate_min": 0.38421425223350525,
  "cont_gate_max": 0.5134873986244202,
  "cont_gate_mean": 0.45215266942977905,
  "cont_gate_std": 0.02945566549897194,
  "feature_embedding_norm_mean": 0.43944090604782104,
  "feature_embedding_norm_std": 0.1128695160150528
}
```

## Key files to inspect first

- `01_result_audit/history_best_vs_final.csv`
- `02_token_audit/token_feature_audit_train.csv`
- `02_token_audit/token_strategy_summary_train.csv`
- `03_embedding_audit/embedding_feature_audit_val.csv`
- `04_error_conditioned_audit/val_correct_vs_wrong_rare_by_true_class.csv`
- `04_error_conditioned_audit/val_feature_rare_wrong_vs_correct_by_class.csv`
- `04_error_conditioned_audit/val_top20_features_by_pair_rare_delta.csv`
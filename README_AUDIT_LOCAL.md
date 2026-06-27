# C2 Best Audit Local Package

Put this package at the root of your local `dacn` repo.

Required repo files/directories:

- `02_src/10_train_fusion_ablation_D0_D7.py`
- `01_split/train_raw.csv`
- `01_split/val_raw.csv`
- `03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_dataset.npz`
- `03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_metadata.json`
- `03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/best_model.pt`
- recommended: all json/csv files from the same C2 run folder, especially `diagnosis_summary.json`, `history.csv`, `*_classification_report_best.json`, `*_confusion_matrix_best.csv`.

Run:

```bash
cd ~/Documents/dacn
python -u run_audit_c2_best_local.py
```

Output:

- `03_outputs/audit_c2_best/01_result_audit/`
- `03_outputs/audit_c2_best/02_token_audit/`
- `03_outputs/audit_c2_best/03_embedding_audit/`
- `03_outputs/audit_c2_best/04_error_conditioned_audit/`

Zip output:

```bash
cd ~/Documents/dacn
zip -r c2_best_audit_outputs_FULL.zip \
  03_outputs/audit_c2_best \
  03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/diagnosis_summary.json \
  03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/history.csv \
  03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/val_classification_report_best.json \
  03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/val_confusion_matrix_best.csv \
  03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/train_classification_report_best.json \
  03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/train_confusion_matrix_best.csv
```

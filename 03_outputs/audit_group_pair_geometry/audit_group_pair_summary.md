# Group + Pair Geometry Audit

This audit adds group-wise tokenization comparisons and pair geometry analysis.

## Key files

- `01_result_tradeoff/result_summary_all_runs.csv`

- `01_result_tradeoff/confusion_pair_delta_vs_C2.csv`

- `02_group_tokenization/group_token_summary_all_runs.csv`

- `02_group_tokenization/group_token_delta_vs_C2_summary.csv`

- `02_group_tokenization/group_token_by_class_all_runs.csv`

- `03_pair_geometry/pair_geometry_all_runs.csv`

- `03_pair_geometry/pair_geometry_summary_by_strategy.csv`

- `03_pair_geometry/C2_Trojan_to_Ransomware_top30_closer_to_Ransomware_by_bin.csv`


## Result summary preview


| run               |   train_macro_f1 |   val_macro_f1 |   gap_macro_f1 |
|:------------------|-----------------:|---------------:|---------------:|
| C2_K512           |         0.910003 |       0.817147 |      0.0928568 |
| K1024_RANK_SAFE   |         0.904927 |       0.791326 |      0.113602  |
| K1024_ABS_CONTROL |         0.926791 |       0.80146  |      0.125331  |


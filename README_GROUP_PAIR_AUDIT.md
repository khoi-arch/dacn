# Group + Pair Geometry Audit

Run from repo root after you have:

1. C2 dataset + metadata:
   - `03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_dataset.npz`
   - `03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_metadata.json`

2. C2 trained run:
   - `03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/`

3. C2 full audit already generated:
   - `03_outputs/audit_c2_best/`

4. Optional but recommended: extract the full K1024 fixed rerun output zip into repo root, so paths exist:
   - `03_outputs/build_mixed_quantile_offset/K1024_B1024_C2policy_rank_safe_native/`
   - `03_outputs/build_mixed_quantile_offset/K1024_B1024_C2policy_abs_for_rank_control_native/`
   - `03_outputs/train_runs_k1024_fixed_c2policy/Keff1024/T1_K1024_C2POLICY_RANK_SAFE_NATIVE_D3/`
   - `03_outputs/train_runs_k1024_fixed_c2policy/Keff1024/T2_K1024_C2POLICY_ABS_FOR_RANK_CONTROL_NATIVE_D3/`

Run:

```bash
cd ~/Documents/dacn
python -u run_group_pair_audit_local.py
```

Or directly:

```bash
python -u 02_src/30_audit_group_pair_geometry.py \
  --out-dir 03_outputs/audit_group_pair_geometry \
  --rare-threshold 5
```

Zip output:

```bash
cd ~/Documents/dacn
zip -r group_pair_audit_outputs.zip \
  03_outputs/audit_group_pair_geometry \
  03_outputs/audit_c2_best/audit_summary.md \
  03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/diagnosis_summary.json \
  03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/history.csv
```

# Attention / Gradient / Occlusion Root-Cause Audit

This package does **not** train a new model. It loads the existing C2 D3 checkpoint and the previously generated overfit root-cause audit, then analyzes selected validation samples with:

1. feature occlusion / neutralization logit-delta,
2. gradient × activation on Transformer input feature tokens,
3. last-layer CLS attention and attention rollout.

## Run

```bash
cd ~/Documents/dacn
python -u run_attention_gradient_audit_local.py
```

Output:

```text
03_outputs/audit_attention_gradient_rootcause/
```

## If memory is tight

```bash
cd ~/Documents/dacn
python -u 02_src/33_audit_attention_gradient_rootcause.py \
  --out-dir 03_outputs/audit_attention_gradient_rootcause \
  --batch-size 64 \
  --max-samples-per-subset 220 \
  --max-samples-per-pair 50
```

## Zip outputs

```bash
cd ~/Documents/dacn
zip -r attention_gradient_audit_outputs.zip \
  03_outputs/audit_attention_gradient_rootcause \
  03_outputs/audit_overfit_rootcause/audit_overfit_rootcause_summary.md \
  03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/diagnosis_summary.json
```

Upload `~/Documents/dacn/attention_gradient_audit_outputs.zip`.

## Reading order

1. `00_sample_selection/sample_selection_summary.csv`
2. `01_occlusion/feature_occlusion_all.csv` and `01_occlusion/top30_occlusion_*.csv`
3. `02_gradxinput/feature_gradxinput_all.csv`
4. `03_attention/feature_attention_all.csv`
5. `04_consensus/feature_consensus_by_subset.csv`

## Interpretation

For wrong samples, `delta_margin_mean > 0` means:

```text
neutralizing this feature reduces logit(predicted_class) - logit(true_class)
```

So the feature supports the wrong prediction.

Attention is not causal. Use it only when it agrees with occlusion and gradient attribution.

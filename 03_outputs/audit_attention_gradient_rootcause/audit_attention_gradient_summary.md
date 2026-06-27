# Attention / Gradient / Occlusion Root-Cause Audit

This audit does not train. It explains selected samples from the overfit root-cause audit.

## Reading order
1. `00_sample_selection/sample_selection_summary.csv`
2. `01_occlusion/feature_occlusion_all.csv` and `01_occlusion/top30_occlusion_*.csv`
3. `02_gradxinput/feature_gradxinput_all.csv`
4. `03_attention/feature_attention_all.csv`
5. `04_consensus/feature_consensus_by_subset.csv` and `04_consensus/top30_consensus_*.csv`

## Key interpretation
- Occlusion is the strongest diagnostic here. Positive `delta_margin_mean` on wrong samples means neutralizing the feature reduces the wrong-vs-true logit margin.
- Grad×input should agree with occlusion for a stable feature-level explanation.
- Attention is not causal. It should be used as routing evidence only.

## Selected samples
audit_subset
model_amplification          320
feature_space_overlap        320
cls_ood_confident            320
all_wrong_high_conf          320
correct_high_conf_control    320
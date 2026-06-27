# Overfit root-cause audit summary

## Inputs
- dataset: `/home/pak/Documents/dacn/03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_dataset.npz`
- metadata: `/home/pak/Documents/dacn/03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_metadata.json`
- checkpoint: `/home/pak/Documents/dacn/03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/best_model.pt`
- run_dir: `/home/pak/Documents/dacn/03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact`

## 1. Train/val gap
| split   |     n |   accuracy |   macro_f1 |   wrong_n |   wrong_rate |   confidence_correct_mean |   confidence_wrong_mean |
|:--------|------:|-----------:|-----------:|----------:|-------------:|--------------------------:|------------------------:|
| train   | 46876 |   0.940545 |   0.90997  |      2787 |    0.0594547 |                  0.952499 |                0.632385 |
| val     | 11720 |   0.878584 |   0.817281 |      1423 |    0.121416  |                  0.949804 |                0.748175 |

## 2. Largest pair gaps: train vs val
Interpretation: large `val_minus_train_rate` means this pair is mainly a validation-generalization gap; high train and high val rates suggest local-underfit or true overlap.
| true_class   | pred_class   |   train_n |   val_n |   train_rate_within_true |   val_rate_within_true |   val_minus_train_rate | rootcause_hint          |
|:-------------|:-------------|----------:|--------:|-------------------------:|-----------------------:|-----------------------:|:------------------------|
| Trojan       | Ransomware   |       950 |     361 |                0.125165  |            0.1903      |            0.0651358   | generalization_gap_pair |
| Ransomware   | Spyware      |       555 |     287 |                0.0708631 |            0.146503    |            0.0756402   | generalization_gap_pair |
| Trojan       | Spyware      |       483 |     209 |                0.0636364 |            0.110174    |            0.0465376   | generalization_gap_pair |
| Ransomware   | Trojan       |       311 |     202 |                0.0397089 |            0.103114    |            0.0634049   | generalization_gap_pair |
| Spyware      | Trojan       |       296 |     196 |                0.0369261 |            0.0978044   |            0.0608782   | generalization_gap_pair |
| Spyware      | Ransomware   |       192 |     161 |                0.0239521 |            0.0803393   |            0.0563872   | generalization_gap_pair |
| Benign       | Spyware      |         0 |       4 |                0         |            0.000682594 |            0.000682594 | minor_pair              |
| Spyware      | Benign       |         0 |       1 |                0         |            0.000499002 |            0.000499002 | minor_pair              |
| Benign       | Ransomware   |         0 |       1 |                0         |            0.000170648 |            0.000170648 | minor_pair              |
| Benign       | Trojan       |         0 |       1 |                0         |            0.000170648 |            0.000170648 | minor_pair              |

## 3. kNN root-cause summaries
Categories:
- `feature_space_overlap_with_pred_class`: val wrong sample is closer to train samples of predicted class.
- `model_boundary_failure_knn_true_neighbors`: val wrong sample is closer to train samples of true class but model predicts other class.
- `OOD_or_distribution_shift`: val sample is farther than train reference q95.
- `mixed_neighbors_ambiguous`: no dominant nearby class.

### raw_scaled
| space      | correct   | rootcause_category                        |    n |   rate_all_val |   confidence_mean |   knn_true_frac_mean |   knn_pred_frac_mean |   ood95_rate |
|:-----------|:----------|:------------------------------------------|-----:|---------------:|------------------:|---------------------:|---------------------:|-------------:|
| raw_scaled | False     | mixed_neighbors_ambiguous                 |  747 |     0.0637372  |          0.730433 |             0.317269 |             0.337617 |            0 |
| raw_scaled | False     | feature_space_overlap_with_pred_class     |  356 |     0.0303754  |          0.763204 |             0.226854 |             0.635506 |            0 |
| raw_scaled | False     | model_boundary_failure_knn_true_neighbors |  222 |     0.018942   |          0.726363 |             0.611712 |             0.260541 |            0 |
| raw_scaled | False     | OOD_or_distribution_shift                 |   98 |     0.00836177 |          0.878228 |             0.428571 |             0.389796 |            1 |
| raw_scaled | True      | correct_and_knn_consistent                | 7713 |     0.658106   |          0.97704  |             0.931124 |             0.931124 |            0 |
| raw_scaled | True      | correct_but_neighbors_mixed               | 2092 |     0.178498   |          0.843671 |             0.339159 |             0.339159 |            0 |
| raw_scaled | True      | OOD_or_distribution_shift                 |  492 |     0.0419795  |          0.974116 |             0.694146 |             0.694146 |            1 |

### token_bin_offset
| space            | correct   | rootcause_category                        |    n |   rate_all_val |   confidence_mean |   knn_true_frac_mean |   knn_pred_frac_mean |   ood95_rate |
|:-----------------|:----------|:------------------------------------------|-----:|---------------:|------------------:|---------------------:|---------------------:|-------------:|
| token_bin_offset | False     | mixed_neighbors_ambiguous                 |  742 |     0.0633106  |          0.748704 |             0.321186 |             0.344744 |            0 |
| token_bin_offset | False     | feature_space_overlap_with_pred_class     |  400 |     0.0341297  |          0.740991 |             0.2347   |             0.6175   |            0 |
| token_bin_offset | False     | model_boundary_failure_knn_true_neighbors |  214 |     0.0182594  |          0.71701  |             0.593458 |             0.269533 |            0 |
| token_bin_offset | False     | OOD_or_distribution_shift                 |   67 |     0.00571672 |          0.884745 |             0.40597  |             0.410746 |            1 |
| token_bin_offset | True      | correct_and_knn_consistent                | 7718 |     0.658532   |          0.973978 |             0.917331 |             0.917331 |            0 |
| token_bin_offset | True      | correct_but_neighbors_mixed               | 2066 |     0.17628    |          0.850804 |             0.349293 |             0.349293 |            0 |
| token_bin_offset | True      | OOD_or_distribution_shift                 |  513 |     0.0437713  |          0.984816 |             0.785341 |             0.785341 |            1 |

### cls_classifier_input
| space                | correct   | rootcause_category                        |    n |   rate_all_val |   confidence_mean |   knn_true_frac_mean |   knn_pred_frac_mean |   ood95_rate |
|:---------------------|:----------|:------------------------------------------|-----:|---------------:|------------------:|---------------------:|---------------------:|-------------:|
| cls_classifier_input | False     | feature_space_overlap_with_pred_class     |  860 |     0.0733788  |          0.778298 |             0.158326 |             0.797442 |            0 |
| cls_classifier_input | False     | OOD_or_distribution_shift                 |  399 |     0.0340444  |          0.777073 |             0.147268 |             0.784261 |            1 |
| cls_classifier_input | False     | mixed_neighbors_ambiguous                 |   94 |     0.00802048 |          0.490888 |             0.289362 |             0.419574 |            0 |
| cls_classifier_input | False     | model_boundary_failure_knn_true_neighbors |   70 |     0.0059727  |          0.558876 |             0.589143 |             0.393143 |            0 |
| cls_classifier_input | True      | correct_and_knn_consistent                | 9674 |     0.825427   |          0.962179 |             0.966409 |             0.966409 |            0 |
| cls_classifier_input | True      | OOD_or_distribution_shift                 |  492 |     0.0419795  |          0.816586 |             0.837724 |             0.837724 |            1 |
| cls_classifier_input | True      | correct_but_neighbors_mixed               |  131 |     0.0111775  |          0.536259 |             0.415267 |             0.415267 |            0 |

## 4. Group neutralization
Large train drop but small val drop suggests group overfit. Negative val drop means masking/neutralizing group improves val, so group may be harmful for val boundary.
| strategy                 | mode       |   n_features |   train_macro_f1_drop_vs_base |   val_macro_f1_drop_vs_base |   drop_gap_train_minus_val |   val_wrong_to_correct_n |   val_correct_to_wrong_n |
|:-------------------------|:-----------|-------------:|------------------------------:|----------------------------:|---------------------------:|-------------------------:|-------------------------:|
| discrete_compact_offset0 | token_only |           23 |                   0.743498    |                 0.650644    |                 0.0928536  |                      374 |                     8270 |
| discrete_compact_offset0 | all        |           23 |                   0.74407     |                 0.651803    |                 0.0922666  |                      371 |                     8273 |
| keep_current             | all        |           14 |                   0.429239    |                 0.341293    |                 0.0879463  |                      538 |                     2771 |
| keep_current             | token_only |           14 |                   0.426807    |                 0.340971    |                 0.0858368  |                      526 |                     2767 |
| rank_uniform_offset      | token_only |           15 |                   0.118943    |                 0.0649704   |                 0.0539725  |                      303 |                      769 |
| rank_uniform_offset      | all        |           15 |                   0.117923    |                 0.0653811   |                 0.0525417  |                      303 |                      771 |
| keep_current             | raw_only   |           14 |                   0.00302997  |                 0.000194407 |                 0.00283557 |                       47 |                       48 |
| rank_uniform_offset      | raw_only   |           15 |                   0.000540281 |                -0.000499397 |                 0.00103968 |                       27 |                       23 |
| discrete_compact_offset0 | raw_only   |           23 |                   0.000678957 |                 0.000507197 |                 0.00017176 |                       16 |                       19 |
| constant                 | all        |            3 |                   0           |                 0           |                 0          |                        0 |                        0 |
| constant                 | token_only |            3 |                   0           |                 0           |                 0          |                        0 |                        0 |
| constant                 | raw_only   |            3 |                   0           |                 0           |                 0          |                        0 |                        0 |

## 5. CLS/embedding availability
```json
{
  "train": {
    "available": true,
    "embedding_name": "classifier_pre_hook_input",
    "shape": [
      46876,
      128
    ]
  },
  "val": {
    "available": true,
    "embedding_name": "classifier_pre_hook_input",
    "shape": [
      11720,
      128
    ]
  }
}
```

## How to decide after reading outputs
- Wrong samples close to predicted class in raw/token space ⇒ feature overlap/ambiguity, not pure model overfit.
- Wrong samples close to true class in raw/token but close to predicted class in CLS ⇒ learned representation/boundary failure.
- Wrong samples OOD in raw/token ⇒ train-val distribution shift.
- Same pair wrong on train and val ⇒ local underfit or inherent class overlap, not merely overfit.
- Group whose train masking drop ≫ val masking drop ⇒ model overuses that group on train.
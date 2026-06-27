# Test 1 — Confidence-only ablation

Mục tiêu: kiểm tra giả thuyết `overfit do CLS/logit quá tự tin ở vùng malware overlap`.

Giữ nguyên:
- C2 K512 tokenization
- D3 shared embedding + offset interpolation + raw FiLM
- Transformer CLS
- class weights
- optimizer/dropout/model size

Chỉ đổi:
- label smoothing

Runs:
- `T1CTRL_C2_D3_CE_LS000`: local control, label_smoothing=0.00
- `T1A_C2_D3_LS003`: label_smoothing=0.03
- `T1B_C2_D3_LS005`: label_smoothing=0.05

Outputs:
- training metrics and predictions for each run
- overfit root-cause audit for every run
- attention/gradient/occlusion audit for local control and best LS run
- one zip: `test1_confidence_only_outputs.zip`

Interpretation:
- If wrong confidence decreases but macro-F1 does not improve, overconfidence is a symptom but not sufficient cause.
- If wrong confidence decreases and CLS amplification decreases while macro/malware F1 improves, confidence control is useful.
- If errors merely move between malware pairs, label smoothing is changing bias rather than solving the root cause.

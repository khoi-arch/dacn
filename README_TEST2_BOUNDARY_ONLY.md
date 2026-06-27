# Test 2 — Boundary-only center-loss ablation

Mục tiêu: kiểm tra giả thuyết `overfit nằm ở CLS representation boundary`, tức raw/token còn ambiguous nhưng CLS kéo val-wrong mạnh về predicted class.

Giữ nguyên:
- C2 K512 tokenization
- D3 shared embedding + offset interpolation + raw FiLM
- Transformer CLS
- class weights
- optimizer/dropout/model size
- label_smoothing = 0

Chỉ đổi:
- thêm center loss nhỏ trên CLS embedding trong training. Center parameters không dùng ở inference.

Runs:
- `T2CTRL_C2_D3_CENTER000`: local control, center_loss_lambda=0.00
- `T2A_C2_D3_CENTER001`: center_loss_lambda=0.01
- `T2B_C2_D3_CENTER003`: center_loss_lambda=0.03

Outputs:
- training metrics and predictions for each run
- overfit root-cause audit for every run
- attention/gradient/occlusion audit for local control and best center-loss run
- one zip: `test2_boundary_only_outputs.zip`

Interpretation:
- If CLS centroid shift/amplification decreases and malware F1 improves, boundary representation is a fixable cause.
- If center loss improves train but worsens val, the malware overlap is too strong and forced compactness hurts.
- If errors merely move between malware pairs, center loss is changing bias rather than solving root cause.

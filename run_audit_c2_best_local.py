#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
cmd = [
    sys.executable, '-u', str(ROOT / '02_src' / '28_audit_c2_best.py'),
    '--dataset-npz', '03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_dataset.npz',
    '--metadata-json', '03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_metadata.json',
    '--run-dir', '03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact',
    '--checkpoint', '03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/best_model.pt',
    '--out-dir', '03_outputs/audit_c2_best',
    '--device', 'auto',
    '--batch-size', '512',
    '--rare-threshold', '5',
]
print('Running:', ' '.join(cmd))
subprocess.run(cmd, cwd=str(ROOT), check=True)
print('\nDONE. Audit output:')
print(ROOT / '03_outputs' / 'audit_c2_best')

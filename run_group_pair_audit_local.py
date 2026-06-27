#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cmd = [
    sys.executable,
    "-u",
    str(ROOT / "02_src" / "30_audit_group_pair_geometry.py"),
    "--out-dir", "03_outputs/audit_group_pair_geometry",
    "--rare-threshold", "5",
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, cwd=str(ROOT), check=True)
print("Output: 03_outputs/audit_group_pair_geometry")

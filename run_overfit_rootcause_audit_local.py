#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
cmd = [
    sys.executable,
    str(root / "02_src" / "32_audit_overfit_rootcause.py"),
    "--out-dir", "03_outputs/audit_overfit_rootcause",
    "--rare-threshold", "5",
    "--knn-k", "25",
    "--max-train-knn", "0",
]
print("[runner]", " ".join(cmd))
raise SystemExit(subprocess.call(cmd, cwd=root))

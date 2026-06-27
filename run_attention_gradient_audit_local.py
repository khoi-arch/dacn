#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
script = root / "02_src" / "33_audit_attention_gradient_rootcause.py"
if not script.exists():
    raise FileNotFoundError(script)

# Default args if user runs wrapper directly.
if len(sys.argv) == 1:
    sys.argv = [str(script), "--out-dir", "03_outputs/audit_attention_gradient_rootcause"]
else:
    sys.argv[0] = str(script)
runpy.run_path(str(script), run_name="__main__")

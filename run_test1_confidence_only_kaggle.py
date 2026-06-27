#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import runpy

root = Path.cwd()
script = root / "02_src" / "34_train_confidence_only_ablation.py"
if not script.exists():
    raise FileNotFoundError(script)
runpy.run_path(str(script), run_name="__main__")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import runpy

root = Path.cwd()
script = root / "02_src" / "35_train_boundary_only_center_loss.py"
if not script.exists():
    raise FileNotFoundError(script)
runpy.run_path(str(script), run_name="__main__")

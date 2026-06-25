#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_pipeline/final_config.py

Global config for final_pipeline.

Edit this file when changing K, thresholds, input/output paths, or special-feature rules.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
FINAL_PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FINAL_PIPELINE_DIR.parent

SPLIT_DIR = PROJECT_ROOT / "01_split"

TRAIN_CSV = SPLIT_DIR / "train_raw.csv"
VAL_CSV = SPLIT_DIR / "val_raw.csv"

OUTPUT_ROOT = PROJECT_ROOT / "03_outputs"
TOKEN_DIAG_DIR = OUTPUT_ROOT / "token_diag"
PREPROCESS_DIR = OUTPUT_ROOT / "preprocessing"
BUILD_TOKEN_DIR = OUTPUT_ROOT / "build_token"
TOKEN_OCCUPANCY_DIR = OUTPUT_ROOT / "token_occupancy"


# ---------------------------------------------------------------------
# Tokenization config
# ---------------------------------------------------------------------
# Main K used by token_diag, preprocessing decision, and build_token.
TOKEN_K = 1000

# Later experiment reminder: run two K settings as two separate runs.
# Change the second value later if needed.
EXPERIMENT_K_VALUES = [1000, 256]

TOKEN_QUANTILES = [0, 1, 5] + list(range(10, 95, 5)) + [95, 99, 100]


# ---------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------
TARGET_COLS = ["label_L1", "label_L2", "label_L3", "Class", "Category"]
DROP_COLS = []

DEFAULT_LABEL_COL = "label_L2"


# ---------------------------------------------------------------------
# Preprocessing decision config
# ---------------------------------------------------------------------
# Main piecewise condition:
#   possible_unique = min(raw_unique, TOKEN_K + 1)
#   unique_preserve_ratio = num_tokens_used / possible_unique
# If this ratio is lower than threshold, tokenization lost too many
# distinguishable raw values -> piecewise candidate.
UNIQUE_PRESERVE_THRESHOLD = 0.95

# If collisions exist, decide whether they mainly live in left/body/right.
# If one region accounts for >= this fraction of total collision loss,
# that region becomes the piecewise_side. Otherwise side = "mixed".
PIECEWISE_SIDE_DOMINANCE_RATIO = 0.60

# Special scaling for delay-like features:
#   z = x / (max + SPECIAL_DELAY_EPS)
#   token = round(TOKEN_K * z)
SPECIAL_DELAY_KEYWORDS = ["rtt", "delay", "invocation"]
SPECIAL_DELAY_EPS = 0.1

# Action for features whose minmax tokenization preserves too few raw unique
# values according to UNIQUE_PRESERVE_THRESHOLD.
# Recommended current default: blended_rank.
#   blended_rank    : z = (1-alpha)*minmax_z + alpha*rank_z
#   local_piecewise : legacy experimental local interval zoom
#   global_rank     : legacy full unique-rank transform
#   keep_minmax     : do not transform compressed features
COMPRESSED_FEATURE_ACTION = "blended_rank"
BLENDED_RANK_ALPHA = 0.25

# Legacy local piecewise token reallocation. Kept as an optional experiment,
# but no longer default because diagnostics showed it can over-edit wide
# collision intervals and damage useful regularization.
LOCAL_PIECEWISE_MARGIN = 1.10
LOCAL_PIECEWISE_MIN_WIDTH = 0.03
LOCAL_PIECEWISE_MAX_WIDTH = 0.50
LOCAL_PIECEWISE_MIN_UNIQUE = 3


# ---------------------------------------------------------------------
# Output file helpers
# ---------------------------------------------------------------------
def token_diag_json_path(k: int | None = None) -> Path:
    kk = TOKEN_K if k is None else int(k)
    return TOKEN_DIAG_DIR / f"token_diag_train_K{kk}.json"


def token_diag_latest_json_path() -> Path:
    return TOKEN_DIAG_DIR / "token_diag_train.json"


def preprocess_train_csv_path(k: int | None = None) -> Path:
    kk = TOKEN_K if k is None else int(k)
    return PREPROCESS_DIR / f"train_preprocessed_K{kk}.csv"


def preprocess_val_csv_path(k: int | None = None) -> Path:
    kk = TOKEN_K if k is None else int(k)
    return PREPROCESS_DIR / f"val_preprocessed_K{kk}.csv"


def preprocess_policy_json_path(k: int | None = None) -> Path:
    kk = TOKEN_K if k is None else int(k)
    return PREPROCESS_DIR / f"preprocess_policy_K{kk}.json"


def preprocess_report_json_path(k: int | None = None) -> Path:
    kk = TOKEN_K if k is None else int(k)
    return PREPROCESS_DIR / f"preprocess_report_K{kk}.json"


def build_token_dir(k: int | None = None) -> Path:
    kk = TOKEN_K if k is None else int(k)
    return BUILD_TOKEN_DIR / f"K{kk}"


def token_dataset_npz_path(k: int | None = None) -> Path:
    kk = TOKEN_K if k is None else int(k)
    return build_token_dir(kk) / f"token_dataset_K{kk}.npz"


def token_metadata_json_path(k: int | None = None) -> Path:
    kk = TOKEN_K if k is None else int(k)
    return build_token_dir(kk) / f"token_metadata_K{kk}.json"

# ---------------------------------------------------------------------
# Embedding config
# ---------------------------------------------------------------------
# V(cell) = V(value) || V(feature)
# Default: 32 + 32 = 64-dimensional cell embedding.
VALUE_EMBED_DIM = 32
FEATURE_EMBED_DIM = 32
VALUE_RANDOM_STD = 0.02

# Decouple numeric token resolution K from the number of learnable value vectors.
# token/K remains exact and monotonic, but the random learnable part is looked up
# through this many coarse value bins. Set to 0 or None to fall back to K+1 bins.
VALUE_NUM_BINS = 128

PREPROCESS_EVAL_DIR = OUTPUT_ROOT / "preprocess_eval"

# ---------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------
# TabularTransformerClassifier config.
# cell_dim = VALUE_EMBED_DIM + FEATURE_EMBED_DIM.
MODEL_HIDDEN_DIM = 128
MODEL_NUM_LAYERS = 3
MODEL_NUM_HEADS = 4
MODEL_DROPOUT = 0.1
CLASSIFIER_HIDDEN_DIM = 128
CLASSIFIER_DROPOUT = 0.1
TRANSFORMER_NORM_FIRST = True
MODEL_ACTIVATION = "gelu"

# ---------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------
TRAIN_RUN_DIR = OUTPUT_ROOT / "train_runs"
TRAIN_SEED = 42
TRAIN_DEVICE = "auto"

TRAIN_EPOCHS = 80
TRAIN_BATCH_SIZE = 256
TRAIN_LR = 1e-3
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_PATIENCE = 12
TRAIN_MIN_DELTA = 1e-4
TRAIN_NUM_WORKERS = 0
TRAIN_GRAD_CLIP_NORM = 1.0

# Macro-F1 matters for imbalanced L2 classification, so class weighting is on by default.
USE_CLASS_WEIGHTS = True

# ---------------------------------------------------------------------
# Scheduler config
# ---------------------------------------------------------------------
# Use warmup+cosine by default to make Transformer training more stable.
TRAIN_SCHEDULER = "warmup_cosine"
TRAIN_WARMUP_EPOCHS = 8
TRAIN_MIN_LR_RATIO = 0.05
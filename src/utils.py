"""
utils.py
--------
Shared utility functions used across every phase of the project
(data loading, path handling, label mapping, figure saving).

Keeping these in one place means every script (EDA, preprocessing,
training, evaluation, the Streamlit app) loads data in EXACTLY the
same way. This avoids a classic bug: subtle mismatches between how
train/val/test are loaded in different notebooks.
"""

import os
import csv
import pandas as pd

# ---------------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------------
# We compute paths relative to this file's location, NOT the current working
# directory. This way, the scripts work no matter where you run them from
# (terminal, notebook, Colab after cloning the repo, etc.)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

TRAIN_PATH = os.path.join(DATASET_DIR, "FinalTrainingOnly.tsv")
VAL_PATH = os.path.join(DATASET_DIR, "ValidationOnly.tsv")
TEST_PATH = os.path.join(DATASET_DIR, "FinalTest.tsv")
TEST_LABELS_PATH = os.path.join(DATASET_DIR, "FinalTest_labels.csv")

# Phase 2 outputs: cleaned datasets, saved alongside the raw files so every
# later phase can find them via these same constants.
CLEANED_TRAIN_PATH = os.path.join(DATASET_DIR, "cleaned_train.csv")
CLEANED_VAL_PATH = os.path.join(DATASET_DIR, "cleaned_validation.csv")
CLEANED_TEST_PATH = os.path.join(DATASET_DIR, "cleaned_test.csv")

# ---------------------------------------------------------------------------
# LABEL MAPPING
# ---------------------------------------------------------------------------
# FinalTrainingOnly.tsv / ValidationOnly.tsv encode the label as an integer:
#   0 = negative, 1 = neutral, 2 = positive
# Ty.txt (test ground truth) encodes it as a string instead
# ("negative" / "neutral" / "positive"). This mapping keeps both consistent.
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_raw_datasets():
    """
    Load the three official dataset files exactly as provided, with no
    column renaming assumptions beyond what we verified by inspecting the
    raw files (they ship with NO header row).

    Returns
    -------
    train_df : DataFrame [uid, text, label]         (label is int 0/1/2)
    val_df   : DataFrame [uid, text, label]          (label is int 0/1/2)
    test_df  : DataFrame [uid, text, label]          (label is int 0/1/2,
               obtained by joining FinalTest.tsv with its ground-truth
               file, FinalTest_labels.csv, on `uid`)
    """
    # quoting=csv.QUOTE_NONE prevents pandas from treating stray quote
    # characters inside tweets as the start/end of a quoted field, which
    # would otherwise silently corrupt rows.
    train_df = pd.read_csv(
        TRAIN_PATH, sep="\t", header=None,
        names=["uid", "text", "label"],
        quoting=csv.QUOTE_NONE, encoding="utf-8",
    )
    val_df = pd.read_csv(
        VAL_PATH, sep="\t", header=None,
        names=["uid", "text", "label"],
        quoting=csv.QUOTE_NONE, encoding="utf-8",
    )
    test_df = pd.read_csv(
        TEST_PATH, sep="\t", header=None,
        names=["uid", "text"],
        quoting=csv.QUOTE_NONE, encoding="utf-8",
    )

    # FinalTest.tsv ships WITHOUT labels. The true labels live in a
    # separate file (Ty.txt, renamed FinalTest_labels.csv) with columns
    # Uid, Sentiment (string labels). We merge them in on `uid` rather
    # than assuming row order, so we are safe even if the order ever
    # changes.
    test_labels_df = pd.read_csv(TEST_LABELS_PATH)
    test_labels_df.columns = [c.strip().lower() for c in test_labels_df.columns]
    test_labels_df["label"] = test_labels_df["sentiment"].map(LABEL2ID)

    test_df = test_df.merge(
        test_labels_df[["uid", "label"]], on="uid", how="left"
    )

    return train_df, val_df, test_df


def save_fig(fig, filename, dpi=150):
    """Save a matplotlib figure into the project's images/ directory."""
    import matplotlib.pyplot as plt  # lazy import: only training scripts call save_fig(),
    # so the Streamlit deploy app never needs matplotlib installed.
    os.makedirs(IMAGES_DIR, exist_ok=True)
    path = os.path.join(IMAGES_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved figure] {path}")
    return path
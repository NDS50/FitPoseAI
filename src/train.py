"""
train.py — AesCode PS4
Full training pipeline for UCF-101-style exercise video dataset.

Pipeline:
  1. Print mandatory §7 dataset dimensions
  2. Extract MediaPipe keypoints from exercise videos (with caching)
  3. Label frames as correct/incorrect using joint-angle rules
  4. Run feature engineering (angles, segment lengths, symmetry)
  5. Train Random Forest classifier
  6. Evaluate on val split
  7. Save model

Usage:
  python src/train.py
  python src/train.py --no-cache       # force re-extract all keypoints
  python src/train.py --report-angles  # print angle distribution first
"""

import os
import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    TRAIN_DIR, VAL_DIR, EXERCISE_CLASSES,
    RANDOM_STATE, MODEL_SAVE_PATH, LABEL_NAMES
)
from dataset_explorer import explore_dataset
from video_keypoint_extractor import extract_split
from form_labeller import build_dataset, angle_distribution_report
from feature_engineering import build_feature_names

# Import model
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model import PoseFormClassifier


# ─────────────────────────────────────────────

def run_training(use_cache: bool = True, report_angles: bool = False):

    # ══ STEP 1 — Dataset dimensions (mandatory §7) ══
    print("\n" + "="*60)
    print("  STEP 1 — DATASET DIMENSIONS")
    print("="*60)
    explore_dataset()

    # ══ STEP 2 — Extract keypoints ══════════════════
    print("\n" + "="*60)
    print("  STEP 2 — EXTRACTING KEYPOINTS FROM VIDEOS")
    print("  (Using MediaPipe Pose, cached to ./data/keypoints_cache)")
    print("="*60)

    train_sequences = extract_split(TRAIN_DIR, "train", use_cache=use_cache)
    val_sequences   = extract_split(VAL_DIR,   "val",   use_cache=use_cache)

    # ══ Optional: angle distribution report ═════════
    if report_angles:
        angle_distribution_report(train_sequences)

    # ══ STEP 3 — Label frames ══════════════════════
    print("\n" + "="*60)
    print("  STEP 3 — LABELLING FRAMES (rule-based form analysis)")
    print("="*60)

    print("\n  [TRAIN]")
    X_train, y_train, _ = build_dataset(train_sequences, aggregate="per_frame")

    print("\n  [VAL]")
    X_val,   y_val,   _ = build_dataset(val_sequences,   aggregate="per_frame")

    if len(X_train) == 0:
        print("\n[ERROR] No training data found. Check:")
        print("  1. DATASET_ROOT in config.py points to your dataset")
        print("  2. EXERCISE_CLASSES list matches actual folder names")
        print("  3. Videos are .avi or .mp4 format")
        return None

    # ══ STEP 4 — Summary ═══════════════════════════
    print("\n" + "="*60)
    print("  STEP 4 — FEATURE SUMMARY")
    print("="*60)
    print(f"  Train samples : {len(X_train):,}")
    print(f"  Val samples   : {len(X_val):,}")
    print(f"  Feature dim   : {X_train.shape[1]}")
    print()

    feature_names = build_feature_names()
    while len(feature_names) < X_train.shape[1]:
        feature_names.append(f"feat_{len(feature_names)}")
    feature_names = feature_names[:X_train.shape[1]]

    # ══ STEP 5 — Train ═════════════════════════════
    print("="*60)
    print("  STEP 5 — TRAINING RANDOM FOREST CLASSIFIER")
    print("="*60)

    clf = PoseFormClassifier(n_estimators=300, max_depth=15)
    clf.fit(X_train, y_train)

    # ══ STEP 6 — Evaluate ══════════════════════════
    print("\n" + "="*60)
    print("  STEP 6 — EVALUATION")
    print("="*60)

    clf.evaluate(X_train, y_train, split="Train")

    if len(X_val) > 0:
        clf.evaluate(X_val, y_val, split="Validation")
    else:
        print("  [WARN] No val data found — skipping val evaluation.")

    # ══ Feature importance ══════════════════════════
    top_feats = clf.get_feature_importances(feature_names)
    print("\n  Top 15 Most Important Features:")
    for name, imp in top_feats[:15]:
        bar = "█" * int(imp * 250)
        print(f"    {name:<35} {imp:.4f}  {bar}")

    # ══ STEP 7 — Save ══════════════════════════════
    print("\n" + "="*60)
    print("  STEP 7 — SAVING MODEL")
    print("="*60)
    clf.save(MODEL_SAVE_PATH)

    print("\n✅ Training complete.")
    return clf


# ── CLI ───────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AesCode PS4 — Training Pipeline")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-extraction of all keypoints")
    parser.add_argument("--report-angles", action="store_true",
                        help="Print joint angle distribution report")
    args = parser.parse_args()

    run_training(
        use_cache=not args.no_cache,
        report_angles=args.report_angles,
    )

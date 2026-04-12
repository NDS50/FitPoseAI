"""
setup_check.py — AesCode PS4
Run this ONCE after placing the dataset to verify everything is correct.
Checks: folder names, exercise class matches, file counts, imports.

Usage:
    python setup_check.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

PASS = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️ "

def check(label, ok, detail=""):
    icon = PASS if ok else FAIL
    msg  = f"{icon}  {label}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    return ok


print()
print("=" * 55)
print("  AesCode PS4 — Setup Check")
print("=" * 55)

all_ok = True

# ── 1. Imports ────────────────────────────────
print("\n[ Dependencies ]")
try:
    import cv2
    check("OpenCV", True, cv2.__version__)
except ImportError as e:
    all_ok = check("OpenCV", False, str(e))

try:
    import mediapipe
    check("MediaPipe", True, mediapipe.__version__)
except ImportError as e:
    all_ok = check("MediaPipe", False, str(e))

try:
    import sklearn
    check("scikit-learn", True, sklearn.__version__)
except ImportError as e:
    all_ok = check("scikit-learn", False, str(e))

try:
    import torch
    check("PyTorch", True, torch.__version__)
except ImportError as e:
    all_ok = check("PyTorch", False, str(e))

try:
    import streamlit
    check("Streamlit", True, streamlit.__version__)
except ImportError as e:
    all_ok = check("Streamlit", False, str(e))

try:
    import tqdm
    check("tqdm", True)
except ImportError as e:
    all_ok = check("tqdm", False, str(e))

# ── 2. Dataset folders ────────────────────────
print("\n[ Dataset ]")
from config import (
    DATASET_ROOT, TRAIN_DIR, VAL_DIR,
    EXERCISE_CLASSES, KEYPOINTS_CACHE_DIR,
)

train_exists = os.path.isdir(TRAIN_DIR)
val_exists   = os.path.isdir(VAL_DIR)
all_ok = check(f"train/ found at '{TRAIN_DIR}'", train_exists) and all_ok
all_ok = check(f"val/   found at '{VAL_DIR}'",   val_exists)   and all_ok

# ── 3. Exercise class matching ─────────────────
print("\n[ Exercise Class Matching ]")
if train_exists:
    actual_folders = set(os.listdir(TRAIN_DIR))
    matched   = [c for c in EXERCISE_CLASSES if c in actual_folders]
    unmatched = [c for c in EXERCISE_CLASSES if c not in actual_folders]

    check(f"{len(matched)}/{len(EXERCISE_CLASSES)} classes found in train/",
          len(matched) > 0,
          f"matched: {', '.join(matched[:5])}{'...' if len(matched)>5 else ''}")

    if unmatched:
        print(f"{WARN}  Classes in config NOT found in train/:")
        for u in unmatched:
            print(f"         — {u}")

    # Count videos per matched class
    print()
    total_train_videos = 0
    VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}
    for cls in sorted(matched):
        folder = os.path.join(TRAIN_DIR, cls)
        n = sum(1 for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in VIDEO_EXTS)
        total_train_videos += n
        bar = "█" * min(n // 3, 25)
        print(f"  ✅  {cls:<28} : {n:>4} videos  {bar}")

    print(f"\n  Total exercise videos (train) : {total_train_videos}")

# ── 4. Val folder exercise classes ────────────
print("\n[ Val Classes ]")
if val_exists:
    val_folders = set(os.listdir(VAL_DIR))
    val_matched = [c for c in EXERCISE_CLASSES if c in val_folders]
    check(f"{len(val_matched)}/{len(EXERCISE_CLASSES)} classes found in val/",
          len(val_matched) > 0)

# ── 5. Source files ───────────────────────────
print("\n[ Source Files ]")
src_files = [
    "src/dataset_explorer.py",
    "src/video_keypoint_extractor.py",
    "src/feature_engineering.py",
    "src/form_labeller.py",
    "src/model.py",
    "src/train.py",
    "app.py",
    "config.py",
    "requirements.txt",
]
for f in src_files:
    exists = os.path.isfile(f)
    all_ok = check(f, exists) and all_ok

# ── 6. Output dirs (auto-create) ──────────────
print("\n[ Directories ]")
for d in ["saved_model", KEYPOINTS_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)
    check(f"'{d}' directory ready", True)

# ── Summary ───────────────────────────────────
print()
print("=" * 55)
if all_ok:
    print("  ✅  All checks passed. Ready to train!")
    print()
    print("  Next command:")
    print("    python src/train.py --report-angles")
else:
    print("  ❌  Some checks failed.")
    print("  Fix the issues above, then re-run: python setup_check.py")
    print()
    print("  If dependencies are missing:")
    print("    pip install -r requirements.txt")
print("=" * 55)
print()

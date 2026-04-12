"""
dataset_explorer.py — AesCode PS4
Mandatory §7 dataset dimension output.
Handles UCF-101-style structure:
    dataset/
      train/
        PushUps/   *.avi
        PullUps/   *.avi
        Squat/     *.avi
        Haircut/   *.avi   ← ignored (not in EXERCISE_CLASSES)
        ...
      val/
        PushUps/   *.avi
        ...

Run this at the TOP of every script before any model training.
"""

import os
import sys
import cv2
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EXERCISE_CLASSES, TRAIN_DIR, VAL_DIR, DATASET_ROOT


VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def count_videos(folder: str) -> int:
    return sum(1 for f in os.listdir(folder)
               if Path(f).suffix.lower() in VIDEO_EXTS)


def get_folder_size_mb(folder: str) -> float:
    total = 0
    for dp, _, files in os.walk(folder):
        for f in files:
            fp = os.path.join(dp, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return round(total / (1024 * 1024), 2)


def get_video_resolution(video_path: str) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "N/A"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return f"{w} x {h} px  @  {fps:.1f} fps"


def find_sample_video(split_dir: str, exercise_classes: list) -> str:
    """Find any one video file to measure resolution."""
    for cls in exercise_classes:
        folder = os.path.join(split_dir, cls)
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if Path(f).suffix.lower() in VIDEO_EXTS:
                return os.path.join(folder, f)
    return ""


# ─────────────────────────────────────────────
# Main explorer
# ─────────────────────────────────────────────

def explore_dataset(dataset_root: str = DATASET_ROOT,
                    exercise_classes: list = None):
    """
    Prints the mandatory §7 dataset dimension block.
    Separates exercise-relevant classes from ignored classes.
    """
    if exercise_classes is None:
        exercise_classes = EXERCISE_CLASSES

    print()
    print("=" * 60)
    print("          === DATASET DIMENSIONS ===")
    print("=" * 60)
    print(f"  Dataset Root   : {os.path.abspath(dataset_root)}")
    print(f"  Format         : Video (.avi / .mp4)")
    print()

    splits = {}
    for split_name, split_dir in [("train", TRAIN_DIR), ("val", VAL_DIR)]:
        if not os.path.isdir(split_dir):
            continue

        all_classes   = sorted(d for d in os.listdir(split_dir)
                               if os.path.isdir(os.path.join(split_dir, d)))
        exercise_dirs = [c for c in all_classes
                         if c in exercise_classes]
        ignored_dirs  = [c for c in all_classes
                         if c not in exercise_classes]

        split_data = {}
        split_total = 0
        for cls in sorted(exercise_dirs):
            folder = os.path.join(split_dir, cls)
            n = count_videos(folder)
            split_data[cls] = n
            split_total += n

        splits[split_name] = {
            "dir": split_dir,
            "exercise_classes": split_data,
            "total_exercise": split_total,
            "total_all_classes": len(all_classes),
            "total_ignored": len(ignored_dirs),
            "ignored_classes": ignored_dirs,
        }

    # ── Print per-split breakdown ────────────
    grand_total = 0
    for split_name, info in splits.items():
        total = info["total_exercise"]
        grand_total += total
        print(f"  [{split_name.upper()}]  —  {total} exercise videos  "
              f"(from {info['total_all_classes']} total classes, "
              f"{info['total_ignored']} ignored)")
        print()

        max_count = max(info["exercise_classes"].values(), default=1)
        for cls, cnt in sorted(info["exercise_classes"].items(),
                               key=lambda x: -x[1]):
            bar = "█" * max(1, int(cnt / max(max_count, 1) * 28))
            print(f"    ├── {cls:<28} : {cnt:>4} videos  {bar}")
        print()

        # Show a few ignored examples
        sample_ignored = info["ignored_classes"][:5]
        more = len(info["ignored_classes"]) - 5
        ign_str = ", ".join(sample_ignored)
        if more > 0:
            ign_str += f"  ... +{more} more"
        print(f"    ✗  Ignored classes  : {ign_str}")
        print()

    print(f"  Total exercise videos used : {grand_total}")
    print()

    # ── Video resolution ─────────────────────
    sample_video = find_sample_video(TRAIN_DIR, EXERCISE_CLASSES)
    if sample_video:
        res = get_video_resolution(sample_video)
        print(f"  Video resolution   : {res}")

    # ── Disk size ────────────────────────────
    # Only count exercise class folders to be accurate
    exercise_size = 0
    for split_name, info in splits.items():
        split_dir = info["dir"]
        for cls in info["exercise_classes"]:
            folder = os.path.join(split_dir, cls)
            exercise_size += get_folder_size_mb(folder)
    size_str = (f"{exercise_size:.1f} MB" if exercise_size < 1024
                else f"{exercise_size/1024:.2f} GB")
    print(f"  Size (exercise classes) : {size_str}")

    print()
    print("=" * 60)
    print()

    return splits


# ── Standalone run ────────────────────────────
if __name__ == "__main__":
    explore_dataset()

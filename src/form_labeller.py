"""
form_labeller.py — AesCode PS4
Labels each video frame as correct (1) or incorrect (0) form
using joint-angle thresholds defined in config.py FORM_RULES.

No mediapipe import — works purely on numpy keypoint arrays.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import FORM_RULES, LABEL_CORRECT, LABEL_INCORRECT
from feature_engineering import (
    build_feature_vector, extract_joint_angles, build_feature_names,
    transform_sequence,
)


# ─────────────────────────────────────────────
# Rule lookup
# ─────────────────────────────────────────────

def get_rules(class_name: str) -> dict:
    """Return FORM_RULES entry for this class, falling back to _default."""
    if class_name in FORM_RULES:
        return FORM_RULES[class_name]
    for key in FORM_RULES:
        if key.lower() == class_name.lower():
            return FORM_RULES[key]
    return FORM_RULES["_default"]


# ─────────────────────────────────────────────
# Per-frame labelling
# ─────────────────────────────────────────────

def label_frame(kp: np.ndarray, rules: dict) -> int:
    """
    Label one frame. Returns CORRECT if ALL joints are within range,
    INCORRECT if any joint violates its threshold.
    """
    angles = extract_joint_angles(kp)
    for angle_name, (lo, hi) in rules.items():
        val = angles.get(angle_name)
        if val is not None and not (lo <= val <= hi):
            return LABEL_INCORRECT
    return LABEL_CORRECT


def label_frame_soft(kp: np.ndarray, rules: dict) -> float:
    """
    Soft confidence: fraction of angle rules that pass (0.0–1.0).
    Use this as a richer training signal than binary 0/1 labels.
    Returns 1.0 if no rules defined.
    """
    if not rules:
        return 1.0
    angles  = extract_joint_angles(kp)
    passing = sum(
        1 for angle_name, (lo, hi) in rules.items()
        if angles.get(angle_name) is not None
        and lo <= angles[angle_name] <= hi
    )
    return passing / len(rules)


def label_sequence(seq: np.ndarray, class_name: str) -> np.ndarray:
    """
    Label every frame in a video sequence.
    Returns np.ndarray (T,) of 0/1 labels.
    """
    rules = get_rules(class_name)
    return np.array([label_frame(frame, rules) for frame in seq],
                    dtype=np.int32)


# ─────────────────────────────────────────────
# Build ML-ready dataset
# ─────────────────────────────────────────────

def build_dataset(class_sequences: dict,
                  aggregate: str = "per_frame") -> tuple:
    """
    Convert {class_name: [seq, ...]} → (X, y, meta).

    Args:
        class_sequences : output of video_keypoint_extractor.extract_split()
        aggregate       : "per_frame" (one sample per frame, recommended)
                          "per_video" (one sample per video — mean+std)

    Returns:
        X    : np.ndarray (N, F)
        y    : np.ndarray (N,)
        meta : list of dicts
    """
    all_X, all_y, all_meta = [], [], []

    for class_name, sequences in class_sequences.items():
        rules = get_rules(class_name)
        print(f"  {class_name:<28} "
              f"({len(sequences)} videos, "
              f"rules: {list(rules.keys())})")

        for vid_idx, seq in enumerate(sequences):
            if len(seq) == 0:
                continue

            if aggregate == "per_frame":
                for frame_idx, kp in enumerate(seq):
                    all_X.append(build_feature_vector(kp))
                    all_y.append(label_frame(kp, rules))
                    all_meta.append({
                        "class": class_name,
                        "video_idx": vid_idx,
                        "frame_idx": frame_idx,
                    })

            elif aggregate == "per_video":
                fv    = transform_sequence(seq)
                lbls  = label_sequence(seq, class_name)
                label = int(np.mean(lbls) >= 0.5)   # majority vote
                all_X.append(fv)
                all_y.append(label)
                all_meta.append({
                    "class": class_name,
                    "video_idx": vid_idx,
                    "frame_idx": -1,
                })

    if not all_X:
        return np.empty((0,)), np.empty((0,)), []

    X = np.stack(all_X).astype(np.float32)
    y = np.array(all_y, dtype=np.int32)

    n_correct   = int((y == LABEL_CORRECT).sum())
    n_incorrect = int((y == LABEL_INCORRECT).sum())
    print(f"\n  Label distribution:")
    print(f"    Correct   (1) : {n_correct:>7,}  ({n_correct/len(y)*100:.1f}%)")
    print(f"    Incorrect (0) : {n_incorrect:>7,}  ({n_incorrect/len(y)*100:.1f}%)")
    print(f"    Total         : {len(y):>7,}")

    return X, y, all_meta


# ─────────────────────────────────────────────
# Angle distribution report (for threshold tuning)
# ─────────────────────────────────────────────

def angle_distribution_report(class_sequences: dict):
    """
    Print mean ± std of key angles per class.
    Use this output to calibrate FORM_RULES thresholds.
    """
    KEY = [
        "left_knee_angle", "right_knee_angle",
        "left_elbow_angle", "right_elbow_angle",
        "left_hip_angle", "right_hip_angle",
        "trunk_lean_angle", "neck_angle",
    ]

    print("\n" + "="*60)
    print("  JOINT ANGLE DISTRIBUTION REPORT")
    print("  Use these values to calibrate FORM_RULES in config.py")
    print("="*60)

    for cls, sequences in class_sequences.items():
        buckets = {a: [] for a in KEY}
        for seq in sequences:
            for kp in seq:
                angles = extract_joint_angles(kp)
                for a in KEY:
                    if a in angles:
                        buckets[a].append(angles[a])

        print(f"\n  [{cls}]")
        for a in KEY:
            vals = buckets[a]
            if vals:
                arr = np.array(vals)
                print(f"    {a:<28} : "
                      f"mean={arr.mean():6.1f}°  "
                      f"std={arr.std():5.1f}°  "
                      f"[{arr.min():.0f}°–{arr.max():.0f}°]")

    print("=" * 60)
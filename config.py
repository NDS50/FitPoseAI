"""
config.py — AesCode PS4
Central configuration. Updated for UCF-101-style multi-class action dataset.
"""

import os

# ── DATASET ───────────────────────────────────
# Matches your actual path: AESCode/dataset/train/
DATASET_ROOT        = "./dataset"
TRAIN_DIR           = os.path.join(DATASET_ROOT, "train")
VAL_DIR             = os.path.join(DATASET_ROOT, "val")
KEYPOINTS_CACHE_DIR = "./keypoints_cache"       # auto-created on first run

# ── Exercise classes to KEEP from the full dataset ────────────────────
# These are the folder names that are exercise-relevant.
# All other folders (Haircut, Billiards, CliffDiving, etc.) are ignored.
EXERCISE_CLASSES = [
    # ── Pushing movements ──────────────────────
    "PushUps",
    "WallPushups",
    "HandstandPushups",
    "BenchPress",
    # ── Pulling movements ──────────────────────
    "PullUps",
    "RopeClimbing",
    # ── Lower body ────────────────────────────
    "BodyWeightSquats",
    "Lunges",
    "CleanAndJerk",
    # ── Cardio / full body ────────────────────
    "JumpingJack",
    "JumpRope",
    "TrampolineJumping",
    # ── Gymnastics / core ─────────────────────
    "HandstandWalking",
    "FloorGymnastics",
    "ParallelBars",
    "StillRings",
    "UnevenBars",
    # ── Martial arts / balance ────────────────
    "TaiChi",
    # ── Combat sports ─────────────────────────
    "BoxingPunchingBag",
    "BoxingSpeedBag",
]

# ── Form rules: joint angle ranges for CORRECT form (degrees) ─────────
# Frames outside these ranges are labelled INCORRECT.
FORM_RULES = {
    # ── Pushing ────────────────────────────────
    "PushUps": {
        "left_elbow_angle":  (60, 110),
        "right_elbow_angle": (60, 110),
        "trunk_lean_angle":  (160, 180),
        "left_knee_angle":   (160, 180),
        "right_knee_angle":  (160, 180),
    },
    "WallPushups": {
        "left_elbow_angle":  (70, 130),
        "right_elbow_angle": (70, 130),
        "trunk_lean_angle":  (145, 180),
    },
    "HandstandPushups": {
        "left_elbow_angle":  (50, 110),
        "right_elbow_angle": (50, 110),
        "trunk_lean_angle":  (155, 180),
    },
    "BenchPress": {
        "left_elbow_angle":  (55, 115),
        "right_elbow_angle": (55, 115),
        "left_shoulder_angle":  (40, 120),
        "right_shoulder_angle": (40, 120),
    },
    # ── Pulling ────────────────────────────────
    "PullUps": {
        "left_elbow_angle":  (30, 90),
        "right_elbow_angle": (30, 90),
        "left_shoulder_angle":  (60, 160),
        "right_shoulder_angle": (60, 160),
    },
    "RopeClimbing": {
        "left_elbow_angle":  (40, 140),
        "right_elbow_angle": (40, 140),
    },
    # ── Lower body ─────────────────────────────
    "BodyWeightSquats": {
        "left_knee_angle":   (60, 120),
        "right_knee_angle":  (60, 120),
        "left_hip_angle":    (50, 115),
        "right_hip_angle":   (50, 115),
        "trunk_lean_angle":  (140, 180),
    },
    "Lunges": {
        "left_knee_angle":   (70, 115),
        "right_knee_angle":  (70, 115),
        "trunk_lean_angle":  (148, 180),
    },
    "CleanAndJerk": {
        "trunk_lean_angle":  (120, 180),
        "left_knee_angle":   (60, 180),
        "right_knee_angle":  (60, 180),
    },
    # ── Cardio / full body ─────────────────────
    "JumpingJack": {
        "left_shoulder_angle":  (70, 180),
        "right_shoulder_angle": (70, 180),
        "left_hip_angle":       (100, 180),
        "right_hip_angle":      (100, 180),
    },
    "JumpRope": {
        "trunk_lean_angle":  (148, 180),
        "left_elbow_angle":  (80, 160),
        "right_elbow_angle": (80, 160),
    },
    "TrampolineJumping": {
        "trunk_lean_angle":  (145, 180),
    },
    # ── Gymnastics / core ──────────────────────
    "HandstandWalking": {
        "trunk_lean_angle":  (155, 180),
        "left_elbow_angle":  (150, 180),
        "right_elbow_angle": (150, 180),
    },
    "FloorGymnastics": {
        "trunk_lean_angle":  (120, 180),
    },
    "ParallelBars": {
        "left_elbow_angle":  (80, 180),
        "right_elbow_angle": (80, 180),
        "trunk_lean_angle":  (148, 180),
    },
    "StillRings": {
        "left_elbow_angle":  (150, 180),
        "right_elbow_angle": (150, 180),
        "trunk_lean_angle":  (148, 180),
    },
    "UnevenBars": {
        "left_elbow_angle":  (100, 180),
        "right_elbow_angle": (100, 180),
    },
    # ── Martial arts / balance ─────────────────
    "TaiChi": {
        "trunk_lean_angle":  (140, 180),
        "left_knee_angle":   (110, 180),
        "right_knee_angle":  (110, 180),
    },
    # ── Combat sports ──────────────────────────
    "BoxingPunchingBag": {
        "trunk_lean_angle":  (130, 180),
        "left_elbow_angle":  (50, 175),
        "right_elbow_angle": (50, 175),
    },
    "BoxingSpeedBag": {
        "trunk_lean_angle":  (135, 180),
        "left_elbow_angle":  (60, 160),
        "right_elbow_angle": (60, 160),
    },
    # ── Fallback for any unmatched class ───────
    "_default": {
        "trunk_lean_angle": (120, 180),
    },
}

# ── Video processing ──────────────────────────
VIDEO_SAMPLE_RATE        = 5
MIN_FRAMES_PER_VIDEO     = 5
MEDIAPIPE_COMPLEXITY     = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE  = 0.5

# ── Training ──────────────────────────────────
TEST_SPLIT      = 0.2
RANDOM_STATE    = 42
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH    = 15

# ── Paths ─────────────────────────────────────
MODEL_SAVE_PATH = "./saved_model/pose_form_classifier.pkl"
LSTM_SAVE_PATH  = "./saved_model/lstm_weights.pt"

# ── Labels ────────────────────────────────────
LABEL_CORRECT   = 1
LABEL_INCORRECT = 0
LABEL_NAMES     = {0: "Incorrect Form", 1: "Correct Form"}

"""
video_keypoint_extractor.py — AesCode PS4
Extracts MediaPipe pose keypoints from .avi exercise videos.
Uses the NEW MediaPipe Tasks API (mediapipe >= 0.10.x).

For each video:
  → Sample every Nth frame
  → Extract 33 landmarks × 4 values (x, y, z, visibility) = 132-d per frame
  → Save as .npy cache file so videos are only processed once
"""

import os
import sys
import cv2
import numpy as np
import urllib.request
from pathlib import Path
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    EXERCISE_CLASSES, TRAIN_DIR, VAL_DIR,
    KEYPOINTS_CACHE_DIR, VIDEO_SAMPLE_RATE,
    MIN_FRAMES_PER_VIDEO, MEDIAPIPE_COMPLEXITY,
    MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
)

NUM_LANDMARKS = 33
FEATURE_DIM   = NUM_LANDMARKS * 4      # 132
VIDEO_EXTS    = {".avi", ".mp4", ".mov", ".mkv"}

MODEL_PATH = "./pose_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

# MediaPipe pose connections for skeleton drawing
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]


# ─────────────────────────────────────────────
# Model download
# ─────────────────────────────────────────────

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"  Downloading MediaPipe pose model → {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("  Download complete.")


# ─────────────────────────────────────────────
# Landmarker factories
# ─────────────────────────────────────────────

def _make_image_landmarker():
    ensure_model()
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def _make_video_landmarker():
    ensure_model()
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


# ─────────────────────────────────────────────
# Result → flat numpy array
# ─────────────────────────────────────────────

def _result_to_array(result) -> np.ndarray | None:
    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None
    lms = result.pose_landmarks[0]
    kp  = []
    for lm in lms:
        vis = float(getattr(lm, "visibility", 1.0) or 1.0)
        kp.extend([lm.x, lm.y, lm.z, vis])
    if len(kp) != FEATURE_DIM:
        return None
    return np.array(kp, dtype=np.float32)


# ─────────────────────────────────────────────
# Single-video extraction
# ─────────────────────────────────────────────

def extract_keypoints_from_video(video_path: str,
                                  sample_rate: int = VIDEO_SAMPLE_RATE
                                  ) -> np.ndarray:
    """
    Extract pose keypoints from every Nth frame of a video.
    Uses IMAGE mode (per-frame) — VIDEO mode requires strict timestamp
    continuity and tracker warm-up which fails on short/sampled clips.
    Returns np.ndarray (T, 132), or empty array if no poses found.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.empty((0, FEATURE_DIM), dtype=np.float32)

    sequences = []
    frame_idx = 0
    # IMAGE mode: each frame analysed independently — no timestamp needed,
    # no tracker warm-up required. Ideal for sampled exercise video frames.
    landmarker = _make_image_landmarker()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)          # IMAGE mode call
            kp     = _result_to_array(result)
            if kp is not None:
                sequences.append(kp)
        frame_idx += 1

    cap.release()
    landmarker.close()

    if not sequences:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)
    return np.stack(sequences)


# ─────────────────────────────────────────────
# Annotated frame extraction (for UI)
# ─────────────────────────────────────────────

def extract_annotated_frame(video_path: str, frame_no: int = 0):
    """Returns (annotated_bgr, kp_132d) or (None, None)."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        return None, None

    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    lm     = _make_image_landmarker()
    result = lm.detect(mp_img)
    lm.close()

    kp        = _result_to_array(result)
    annotated = frame_bgr.copy()
    if kp is not None:
        _draw_skeleton(annotated, kp)
    return annotated, kp


def _draw_skeleton(frame_bgr: np.ndarray, kp: np.ndarray):
    h, w = frame_bgr.shape[:2]
    def xy(i): return int(kp[i*4]*w), int(kp[i*4+1]*h)
    for a, b in POSE_CONNECTIONS:
        cv2.line(frame_bgr, xy(a), xy(b), (0, 255, 0), 2)
    for i in range(NUM_LANDMARKS):
        cv2.circle(frame_bgr, xy(i), 4, (0, 0, 255), -1)


# ─────────────────────────────────────────────
# Cached batch extraction
# ─────────────────────────────────────────────

def _cache_path(video_path: str, cache_dir: str) -> str:
    vid = Path(video_path)
    return os.path.join(cache_dir, f"{vid.parent.name}__{vid.stem}.npy")


def extract_split(split_dir: str,
                  split_name: str,
                  exercise_classes: list = None,
                  use_cache: bool = True) -> dict:
    """
    Extract keypoints from all exercise videos in one split.
    Returns { class_name → [np.ndarray(T, 132), ...] }
    """
    if exercise_classes is None:
        exercise_classes = EXERCISE_CLASSES

    cache_dir = os.path.join(KEYPOINTS_CACHE_DIR, split_name)
    os.makedirs(cache_dir, exist_ok=True)
    ensure_model()

    out = {}
    for cls in sorted(exercise_classes):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        videos = sorted(f for f in os.listdir(cls_dir)
                        if Path(f).suffix.lower() in VIDEO_EXTS)
        if not videos:
            continue

        seqs = []
        print(f"\n  [{split_name}] {cls}  ({len(videos)} videos)")
        for vid_name in tqdm(videos, desc=f"    {cls}", leave=False):
            vid_path   = os.path.join(cls_dir, vid_name)
            cache_file = _cache_path(vid_path, cache_dir)

            if use_cache and os.path.exists(cache_file):
                seq = np.load(cache_file)
            else:
                seq = extract_keypoints_from_video(vid_path)
                if use_cache:
                    np.save(cache_file, seq)

            if len(seq) >= MIN_FRAMES_PER_VIDEO:
                seqs.append(seq)

        out[cls] = seqs
        print(f"    → {len(seqs)} valid videos, "
              f"{sum(len(s) for s in seqs)} total frames")

    return out


# ── Standalone test ───────────────────────────
if __name__ == "__main__":
    ensure_model()
    print("Testing extraction on first 2 classes...")
    data = extract_split(TRAIN_DIR, "train",
                         exercise_classes=EXERCISE_CLASSES[:2],
                         use_cache=True)
    for cls, seqs in data.items():
        if seqs:
            print(f"  {cls}: {len(seqs)} videos, shape={seqs[0].shape}")
"""
feature_engineering.py — AesCode PS4
Converts raw 132-d MediaPipe keypoint vectors into interpretable
biomechanical features for the classifier.

Does NOT import mediapipe — works purely on the raw numpy arrays
produced by video_keypoint_extractor.py.

Features:
  1. Raw keypoints          (132 values)
  2. Joint angles           (degrees, 12 key joints)
  3. Segment lengths        (normalised by torso height)
  4. Symmetry ratios        (left / right balance)
"""

import numpy as np

# ─────────────────────────────────────────────
# MediaPipe landmark indices (hardcoded — no mp import needed)
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
# ─────────────────────────────────────────────
IDX = {
    "NOSE":             0,
    "LEFT_EYE_INNER":  1,  "LEFT_EYE":         2,  "LEFT_EYE_OUTER":   3,
    "RIGHT_EYE_INNER": 4,  "RIGHT_EYE":        5,  "RIGHT_EYE_OUTER":  6,
    "LEFT_EAR":        7,  "RIGHT_EAR":        8,
    "MOUTH_LEFT":      9,  "MOUTH_RIGHT":      10,
    "LEFT_SHOULDER":   11, "RIGHT_SHOULDER":   12,
    "LEFT_ELBOW":      13, "RIGHT_ELBOW":      14,
    "LEFT_WRIST":      15, "RIGHT_WRIST":      16,
    "LEFT_PINKY":      17, "RIGHT_PINKY":      18,
    "LEFT_INDEX":      19, "RIGHT_INDEX":      20,
    "LEFT_THUMB":      21, "RIGHT_THUMB":      22,
    "LEFT_HIP":        23, "RIGHT_HIP":        24,
    "LEFT_KNEE":       25, "RIGHT_KNEE":       26,
    "LEFT_ANKLE":      27, "RIGHT_ANKLE":      28,
    "LEFT_HEEL":       29, "RIGHT_HEEL":       30,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32,
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _get(kp: np.ndarray, name: str) -> np.ndarray:
    """Extract (x, y, z) for a named landmark from the 132-d vector."""
    i = IDX[name] * 4
    return kp[i:i+3]


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at joint B formed by A-B-C, in degrees [0, 180]."""
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ─────────────────────────────────────────────
# Joint angles
# ─────────────────────────────────────────────

def extract_joint_angles(kp: np.ndarray) -> dict:
    """
    Extract 12 clinically relevant joint angles from a 132-d vector.
    Returns dict {angle_name: degrees}.
    """
    g = lambda name: _get(kp, name)

    mid_shoulder = (g("LEFT_SHOULDER")  + g("RIGHT_SHOULDER")) / 2
    mid_hip      = (g("LEFT_HIP")       + g("RIGHT_HIP"))      / 2
    mid_knee     = (g("LEFT_KNEE")      + g("RIGHT_KNEE"))      / 2

    return {
        # ── Lower body ──────────────────────────
        "left_knee_angle":   _angle(g("LEFT_HIP"),       g("LEFT_KNEE"),   g("LEFT_ANKLE")),
        "right_knee_angle":  _angle(g("RIGHT_HIP"),      g("RIGHT_KNEE"),  g("RIGHT_ANKLE")),
        "left_hip_angle":    _angle(g("LEFT_SHOULDER"),  g("LEFT_HIP"),    g("LEFT_KNEE")),
        "right_hip_angle":   _angle(g("RIGHT_SHOULDER"), g("RIGHT_HIP"),   g("RIGHT_KNEE")),
        "left_ankle_angle":  _angle(g("LEFT_KNEE"),      g("LEFT_ANKLE"),  g("LEFT_FOOT_INDEX")),
        "right_ankle_angle": _angle(g("RIGHT_KNEE"),     g("RIGHT_ANKLE"), g("RIGHT_FOOT_INDEX")),
        # ── Upper body ──────────────────────────
        "left_elbow_angle":    _angle(g("LEFT_SHOULDER"),  g("LEFT_ELBOW"),   g("LEFT_WRIST")),
        "right_elbow_angle":   _angle(g("RIGHT_SHOULDER"), g("RIGHT_ELBOW"),  g("RIGHT_WRIST")),
        "left_shoulder_angle": _angle(g("LEFT_ELBOW"),     g("LEFT_SHOULDER"),  g("LEFT_HIP")),
        "right_shoulder_angle":_angle(g("RIGHT_ELBOW"),    g("RIGHT_SHOULDER"), g("RIGHT_HIP")),
        # ── Trunk ───────────────────────────────
        "trunk_lean_angle":  _angle(mid_knee, mid_hip, mid_shoulder),
        # Fixed: angle at mid-ear (B), between mid-shoulder (A) and NOSE (C)
        "neck_angle":        _angle(mid_shoulder,
                                    (g("LEFT_EAR") + g("RIGHT_EAR")) / 2,
                                    g("NOSE")),
    }


# ─────────────────────────────────────────────
# Segment lengths (torso-normalised)
# ─────────────────────────────────────────────

def extract_segment_lengths(kp: np.ndarray) -> dict:
    g      = lambda name: _get(kp, name)
    mid_sh = (g("LEFT_SHOULDER") + g("RIGHT_SHOULDER")) / 2
    mid_hp = (g("LEFT_HIP")      + g("RIGHT_HIP"))      / 2
    torso  = _dist(mid_sh, mid_hp) + 1e-8

    return {
        "left_upper_arm":  _dist(g("LEFT_SHOULDER"),  g("LEFT_ELBOW"))   / torso,
        "right_upper_arm": _dist(g("RIGHT_SHOULDER"), g("RIGHT_ELBOW"))  / torso,
        "left_forearm":    _dist(g("LEFT_ELBOW"),     g("LEFT_WRIST"))   / torso,
        "right_forearm":   _dist(g("RIGHT_ELBOW"),    g("RIGHT_WRIST"))  / torso,
        "left_thigh":      _dist(g("LEFT_HIP"),       g("LEFT_KNEE"))    / torso,
        "right_thigh":     _dist(g("RIGHT_HIP"),      g("RIGHT_KNEE"))   / torso,
        "left_shin":       _dist(g("LEFT_KNEE"),      g("LEFT_ANKLE"))   / torso,
        "right_shin":      _dist(g("RIGHT_KNEE"),     g("RIGHT_ANKLE"))  / torso,
        "shoulder_width":  _dist(g("LEFT_SHOULDER"),  g("RIGHT_SHOULDER")) / torso,
        "hip_width":       _dist(g("LEFT_HIP"),       g("RIGHT_HIP"))    / torso,
        "torso_height":    1.0,
    }


# ─────────────────────────────────────────────
# Symmetry ratios
# ─────────────────────────────────────────────

def extract_symmetry(angles: dict, segs: dict) -> dict:
    pairs = [
        ("sym_knee",     "left_knee_angle",     "right_knee_angle"),
        ("sym_hip",      "left_hip_angle",      "right_hip_angle"),
        ("sym_elbow",    "left_elbow_angle",    "right_elbow_angle"),
        ("sym_shoulder", "left_shoulder_angle", "right_shoulder_angle"),
        ("sym_thigh",    "left_thigh",          "right_thigh"),
        ("sym_shin",     "left_shin",           "right_shin"),
    ]
    out = {}
    for name, lk, rk in pairs:
        lv = angles.get(lk) or segs.get(lk)
        rv = angles.get(rk) or segs.get(rk)
        if lv is not None and rv is not None:
            out[name] = float(lv) / (float(rv) + 1e-8)
    return out


# ─────────────────────────────────────────────
# Velocity features (temporal, computed from a sequence)
# ─────────────────────────────────────────────

def extract_velocity(seq_angles: list) -> np.ndarray:
    """
    Compute per-joint angular velocity (degrees/frame) for a sequence.
    seq_angles : list of angle dicts (one per frame), len T.
    Returns np.ndarray shape (T-1, n_angles) — NaNs if < 2 frames.
    """
    if len(seq_angles) < 2:
        return np.zeros((1, len(seq_angles[0])), dtype=np.float32)
    keys = list(seq_angles[0].keys())
    arr  = np.array([[d[k] for k in keys] for d in seq_angles], dtype=np.float32)
    return np.diff(arr, axis=0)  # (T-1, n_angles)


# ─────────────────────────────────────────────
# Main feature builder
# ─────────────────────────────────────────────

def build_feature_vector(kp: np.ndarray) -> np.ndarray:
    """
    Build complete feature vector from a single 132-d keypoint array.
    Returns np.ndarray of shape (N_features,).
    """
    angles = extract_joint_angles(kp)
    segs   = extract_segment_lengths(kp)
    sym    = extract_symmetry(angles, segs)

    parts = (
        list(kp)                    # 132  raw keypoints
        + list(angles.values())     # 12   joint angles
        + list(segs.values())       # 11   segment lengths
        + list(sym.values())        # 6    symmetry ratios
    )
    return np.array(parts, dtype=np.float32)


def build_feature_names() -> list:
    """Feature names in the same order as build_feature_vector."""
    dummy  = np.zeros(132, dtype=np.float32)
    angles = extract_joint_angles(dummy)
    segs   = extract_segment_lengths(dummy)
    sym    = extract_symmetry(angles, segs)
    return (
        [f"kp_{i}" for i in range(132)]
        + list(angles.keys())
        + list(segs.keys())
        + list(sym.keys())
    )


# ─────────────────────────────────────────────
# Batch helpers
# ─────────────────────────────────────────────

def transform_batch(X_kp: np.ndarray) -> np.ndarray:
    """(N, 132) → (N, F)"""
    return np.stack([build_feature_vector(row) for row in X_kp])


def transform_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Aggregate a video sequence (T, 132) into one sample.
    Uses mean + std + p25 + p75 + min + max → captures full rep distribution.
    Also appends mean absolute velocity (angular speed) features.
    Returns 1-D array of shape (F*6 + V,).
    """
    fv = transform_batch(seq)                          # (T, F)
    agg = np.concatenate([
        fv.mean(axis=0),                               # F mean
        fv.std(axis=0),                                # F std
        np.percentile(fv, 25, axis=0),                 # F p25
        np.percentile(fv, 75, axis=0),                 # F p75
        fv.min(axis=0),                                # F min
        fv.max(axis=0),                                # F max
    ])

    # Angular velocity features (mean + std of Δangle/frame)
    if len(seq) >= 2:
        angle_dicts = [extract_joint_angles(kp) for kp in seq]
        vel = extract_velocity(angle_dicts)            # (T-1, n_angles)
        vel_feats = np.concatenate([np.abs(vel).mean(axis=0),
                                    np.abs(vel).std(axis=0)])
    else:
        n_angles = 12  # as defined in extract_joint_angles
        vel_feats = np.zeros(n_angles * 2, dtype=np.float32)

    return np.concatenate([agg, vel_feats]).astype(np.float32)


# ── Quick test ────────────────────────────────
if __name__ == "__main__":
    dummy = np.random.rand(132).astype(np.float32)
    fv    = build_feature_vector(dummy)
    names = build_feature_names()
    print(f"Feature vector dim : {len(fv)}")
    print(f"Feature names      : {len(names)}")
    assert len(fv) == len(names), "Mismatch!"
    print("feature_engineering.py OK")
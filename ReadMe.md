# 🏋️ FitPoseAI — AI-Based Exercise Form Detection

> **AesCode Nexus 2026** · PS4 · BVDUMC × KCDH IIT Bombay  
> Team **Synthexis** · Naushad Siddiqui · B.Tech CSE

---

## 📌 Problem Statement

**PS4 — AI-Based Detection of Incorrect Exercise Form Using Human Pose Estimation**

Most people exercise without professional supervision. Bad form leads to:
- Muscle imbalances and joint injuries
- Ineffective workouts (wrong muscles activated)
- Long-term chronic pain (especially knees, spine, shoulders)

FitPoseAI automatically analyses exercise videos and images, detects body posture, calculates joint angles, and classifies form as **correct or incorrect** — giving real-time feedback without a human trainer.

---

## 🧠 How It Works — Full Pipeline

```
Video / Image Input
        │
        ▼
┌───────────────────────────────┐
│  MediaPipe Pose Estimation    │  ← 33 body keypoints × (x, y, z, visibility)
│  (pose_landmarker.task)       │    = 132-dimensional vector per frame
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  Feature Engineering          │  ← 161-dimensional feature vector per frame
│  · 12 Joint Angles (degrees)  │    · Elbow, Knee, Hip, Shoulder, Trunk, Neck
│  · 11 Segment Lengths         │    · Torso-normalised (scale invariant)
│  · 6  Symmetry Ratios         │    · Left/right balance metrics
│  · Velocity (for LSTM)        │    · Rate of angle change between frames
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  Rule-Based Form Labelling    │  ← Joint angle thresholds per exercise class
│  (config.py FORM_RULES)       │    Correct = ALL joints within valid range
│                               │    Also: soft confidence score (% rules passed)
└───────────────────────────────┘
        │
        ▼
   ┌────┴────┐
   │         │
   ▼         ▼
Round 1   Round 2
   │         │
Random    BiLSTM
Forest    Sequence
   │      Classifier
   │         │
   └────┬────┘
        │
        ▼
┌───────────────────────────────┐
│  Streamlit UI                 │
│  · Frame-by-frame timeline    │
│  · Joint angle charts         │
│  · Correct / Incorrect verdict│
│  · Confidence scores          │
└───────────────────────────────┘
```

---

## 📂 Project Structure

```
AESCode/
│
├── app.py                      # Streamlit UI — main application
├── config.py                   # Central config: paths, rules, hyperparams
├── requirements.txt            # Python dependencies
├── pose_landmarker.task        # MediaPipe model file (5.7 MB)
│
├── src/
│   ├── video_keypoint_extractor.py   # MediaPipe pose extraction from videos
│   ├── feature_engineering.py        # Joint angles, segments, symmetry, velocity
│   ├── form_labeller.py              # Rule-based labelling (binary + soft)
│   ├── model.py                      # Random Forest + BiLSTM classifiers
│   ├── train.py                      # Round 1: RF training pipeline
│   ├── train_lstm.py                 # Round 2: BiLSTM training pipeline  ← NEW
│   ├── lstm_dataset.py               # PyTorch Dataset for sequences       ← NEW
│   ├── dataset_explorer.py           # Mandatory §7 dataset dimension report
│
├── dataset/
│   ├── train/                  # 1,105 videos across 20 exercise classes
│   └── val/                    # 304 videos for validation
│
├── keypoints_cache/            # .npy cached keypoints (auto-generated)
│   ├── train/
│   └── val/
│
└── saved_model/
    ├── pose_form_classifier.pkl    # Trained Random Forest (Round 1)
    └── lstm_weights.pt             # Trained BiLSTM + metadata (Round 2)
```

---

## 🎯 Exercise Classes Supported (20 Total)

| Category | Exercises |
|---|---|
| **Pushing** | PushUps, WallPushups, HandstandPushups, BenchPress |
| **Pulling** | PullUps, RopeClimbing |
| **Lower Body** | BodyWeightSquats, Lunges, CleanAndJerk |
| **Cardio** | JumpingJack, JumpRope, TrampolineJumping |
| **Gymnastics** | HandstandWalking, FloorGymnastics, ParallelBars, StillRings, UnevenBars |
| **Martial Arts** | TaiChi |
| **Combat Sports** | BoxingPunchingBag, BoxingSpeedBag |

---

## 📊 Dataset

- **Source:** UCF-101 Action Recognition Dataset (exercise-relevant subset)
- **Format:** `.avi` / `.mp4` video files, 320×240px at 25fps
- **Train:** 1,105 videos · **Val:** 304 videos · **Total:** 1,409 videos
- **Size:** 664.7 MB
- **Rule:** Only provided dataset used — No augmentation, no external data

---

## 🔬 Feature Engineering (161 dimensions per frame)

### Joint Angles (12)
Computed using the cosine rule on 3D landmark triplets:
```
angle at B = arccos( (BA · BC) / (|BA| × |BC|) )
```

| Joint | Landmarks Used |
|---|---|
| Knee (L/R) | Hip → Knee → Ankle |
| Hip (L/R) | Shoulder → Hip → Knee |
| Ankle (L/R) | Knee → Ankle → Foot Index |
| Elbow (L/R) | Shoulder → Elbow → Wrist |
| Shoulder (L/R) | Elbow → Shoulder → Hip |
| Trunk Lean | Mid-Knee → Mid-Hip → Mid-Shoulder |
| Neck | Mid-Shoulder → Mid-Ear → Nose |

### Segment Lengths (11)
Upper arm, forearm, thigh, shin, shoulder width, hip width — all **normalised by torso height** to be scale-invariant across different body sizes and camera distances.

### Symmetry Ratios (6)
`left_value / right_value` for knee, hip, elbow, shoulder, thigh, shin — captures imbalances during movement.

### Velocity Features (for BiLSTM)
`Δangle / frame` — rate of change of each joint angle. Captures movement speed and smoothness.

---

## 📐 Form Rules — How Correctness is Defined

Each exercise has a set of joint angle ranges in `config.py`. A frame is labelled **CORRECT** only if **all** defined joints are within range.

**Example — PushUps:**
```python
"PushUps": {
    "left_elbow_angle":  (60, 110),   # arms bent correctly
    "right_elbow_angle": (60, 110),
    "trunk_lean_angle":  (160, 180),  # body straight (plank position)
    "left_knee_angle":   (160, 180),  # legs straight
    "right_knee_angle":  (160, 180),
}
```

A **soft confidence score** is also computed as the fraction of rules satisfied (0.0 → 1.0), giving a more nuanced signal than binary 0/1.

---

## 🤖 Models

### Round 1 — Random Forest Classifier
- **Input:** 161-d per-frame feature vector (mean-pooled across all frames)
- **Architecture:** StandardScaler → RandomForest (300 trees, max_depth=15)
- **Class weighting:** Balanced (handles 63% incorrect / 37% correct imbalance)
- **Train Accuracy:** 97.33% · **Val Accuracy:** 81.25%
- **Val F1:** 0.8135 · **Val AUC:** 0.8961

### Round 2 — Bidirectional LSTM
- **Input:** (30, 161) — 30 padded frames of 161-d feature vectors
- **Architecture:**
  ```
  BiLSTM (128 hidden × 2 directions = 256)
    → Dropout (0.3)
    → Linear (256 → 2)
    → CrossEntropyLoss (class-weighted)
  ```
- **Training:** Adam + ReduceLROnPlateau + Early Stopping (patience=10)
- **Advantage:** Understands temporal patterns (speed, rhythm, range of motion over the full rep) — not just a snapshot

| Feature | Random Forest | BiLSTM |
|---|---|---|
| Input | Single aggregated vector | Full temporal sequence |
| Temporal awareness | ❌ None | ✅ Full sequence |
| Rep counting potential | ❌ | ✅ |
| Training speed | Fast (seconds) | Slower (minutes) |
| Preferred | Round 1 fallback | Round 2 primary |

---

## 🖥️ Streamlit Application — `app.py`

### Tab 1: Video Analysis
1. Upload `.avi` / `.mp4` exercise video
2. Extracts pose keypoints from every 5th frame (configurable)
3. Plots joint angle timelines (Knee, Hip, Elbow, Trunk)
4. Shows frame-level colour bar (green = correct, red = incorrect)
5. Applies **temporal smoothing** (5-frame sliding window) to reduce noise
6. Runs BiLSTM → RF → Rule-based prediction (whichever is available)
7. Shows confidence probabilities for correct and incorrect

### Tab 2: Image Analysis
- Upload a single exercise photo
- Overlays pose skeleton on the image
- Shows all 12 joint angles as metrics
- Gives instant rule-based form verdict

### Tab 3: Dataset Explorer
- Generates and displays the mandatory §7 dataset dimension block
- Required for hackathon compliance (screenshots for technical report)

### Sidebar Controls
| Control | Purpose |
|---|---|
| Exercise Type | Which rule set to apply |
| Confidence Threshold | Minimum probability for classification |
| Show Joint Angles | Toggle angle timeline charts |
| Sample every N frames | Processing speed vs accuracy tradeoff |

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.14 |
| Pose Estimation | MediaPipe Tasks API ≥ 0.10.x |
| Computer Vision | OpenCV (`opencv-contrib-python`) |
| ML Round 1 | scikit-learn (RandomForestClassifier) |
| ML Round 2 | PyTorch (Bidirectional LSTM) |
| UI | Streamlit |
| Data | NumPy, Pandas |
| Visualisation | Matplotlib |
| Smoothing | SciPy (`uniform_filter1d`) |

---

## 🚀 How to Run

```bash
# 1. Setup
cd /Users/naushad/AESCode
source venv/bin/activate
pip install -r requirements.txt

# 2. Verify dataset (mandatory §7)
python src/dataset_explorer.py

# 3. Train Round 1 — Random Forest
python src/train.py

# 4. Train Round 2 — BiLSTM (sequence-aware)
python src/train_lstm.py
# Options: --seq-len 40 --epochs 80 --hidden 256

# 5. Launch dashboard
streamlit run app.py
```

---

## 📈 Training Results

### Round 1 — Random Forest
```
Label distribution — Train: 28,950 frames
  Correct  (1) : 10,554  (36.5%)
  Incorrect(0) : 18,396  (63.5%)

[Train] Accuracy: 97.33%  F1: 0.9735  AUC: 0.9981
[Val]   Accuracy: 81.25%  F1: 0.8135  AUC: 0.8961

Top features: trunk_lean_angle, left_hip_angle, right_hip_angle
```

### Round 2 — BiLSTM
- Run `python src/train_lstm.py` to generate
- Per-video labels from majority vote of frame-level rules
- Early stopping prevents overfitting on small video count

---

## 🔮 Key Design Decisions

1. **IMAGE mode over VIDEO mode for uploaded clips** — MediaPipe VIDEO mode needs tracker warm-up across consecutive frames, which fails on short/sampled clips. IMAGE mode processes each frame independently and is more robust.

2. **Rule-based labels as supervision** — Since no ground-truth human annotations exist for this dataset (UCF-101 is action recognition, not form quality), joint angle thresholds derived from biomechanics literature are used as weak supervision.

3. **Keypoint caching** — All extracted `.npy` files are cached in `keypoints_cache/`. Re-running train or the UI never re-extracts previously processed videos. This saves 95% of compute time on subsequent runs.

4. **Soft confidence labels** — Beyond binary 0/1, the fraction of rules satisfied (0.0–1.0) is computed per frame, providing a richer quality signal.

5. **Temporal smoothing in UI** — A 5-frame sliding window filters single-frame MediaPipe glitches from the frame timeline display.

---

## 👨‍💻 Author

**Naushad Siddiqui**  
Team: Synthexis  
B.Tech Computer Science Engineering

---

## 📜 License

For educational and hackathon use only — AesCode Nexus 2026.
"""
app.py — AesCode PS4
Exercise Form Detection · Streamlit UI
Uses new MediaPipe Tasks API (>= 0.10.x)
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, "./src")

st.set_page_config(
    page_title="AesCode PS4 — Exercise Form Checker",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header{font-size:2.1rem;font-weight:700;color:#1a1a2e;text-align:center}
.sub-header{font-size:.95rem;color:#666;text-align:center;margin-bottom:1.8rem}
.correct-box{background:linear-gradient(135deg,#d4edda,#c3e6cb);border:2px solid #28a745;
             border-radius:12px;padding:1.5rem;text-align:center}
.incorrect-box{background:linear-gradient(135deg,#f8d7da,#f5c6cb);border:2px solid #dc3545;
               border-radius:12px;padding:1.5rem;text-align:center}
.info-card{background:#eef2ff;border-left:4px solid #6366f1;padding:1rem;
           border-radius:0 8px 8px 0;margin:.5rem 0}
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────
@st.cache_resource
def load_model():
    """Random Forest classifier (Round 1)."""
    try:
        from model import PoseFormClassifier
        clf = PoseFormClassifier()
        clf.load("./saved_model/pose_form_classifier.pkl")
        return clf
    except Exception:
        return None


@st.cache_resource
def load_lstm_model():
    """BiLSTM sequence classifier (Round 2) — preferred over RF."""
    try:
        from model import LSTMPoseClassifier
        lstm = LSTMPoseClassifier.from_checkpoint(
            "./saved_model/lstm_weights.pt")
        return lstm
    except Exception:
        return None


@st.cache_resource
def get_image_landmarker():
    """Cached image-mode PoseLandmarker — created once per app session."""
    from video_keypoint_extractor import ensure_model, MODEL_PATH
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    ensure_model()
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


# ── Header ────────────────────────────────────
st.markdown('<div class="main-header">🏋️ Exercise Form Checker</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Based Detection of Incorrect Exercise Form '
            '· AesCode Nexus PS4 · BVDUMC × KCDH IIT Bombay</div>',
            unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    exercise_type = st.selectbox("Exercise Type", [
        "PushUps", "WallPushups", "HandstandPushups", "BenchPress",
        "PullUps", "RopeClimbing", "BodyWeightSquats", "Lunges",
        "CleanAndJerk", "JumpingJack", "JumpRope", "TrampolineJumping",
        "HandstandWalking", "FloorGymnastics", "ParallelBars",
        "StillRings", "UnevenBars", "TaiChi",
        "BoxingPunchingBag", "BoxingSpeedBag",
    ])
    confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.99, 0.65, 0.01)
    show_angles  = st.checkbox("Show Joint Angles",  value=True)
    frame_sample = st.slider("Sample every N frames", 1, 15, 5)
    st.divider()
    st.markdown("**AesCode Nexus 2026**  \nBVDUMC × KCDH IIT Bombay")


# ── Tabs ──────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["🎬 Video Analysis", "📸 Image Analysis", "📊 Dataset Explorer"])


# ══════════════════════════════════════════════
# TAB 1 — Video
# ══════════════════════════════════════════════
with tab1:
    st.markdown("#### Upload an exercise video (.avi / .mp4)")
    video_file = st.file_uploader("Choose a video",
                                   type=["avi", "mp4", "mov"], key="vid")

    if video_file:
        suffix = Path(video_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        st.video(video_file)

        if st.button("🔍 Analyse Form", type="primary"):
            with st.spinner("Extracting pose from video frames..."):
                try:
                    from video_keypoint_extractor import extract_keypoints_from_video
                    seq = extract_keypoints_from_video(tmp_path,
                                                       sample_rate=frame_sample)
                    if len(seq) == 0:
                        st.warning("No pose detected. Ensure full body is visible.")
                    else:
                        st.success(f"✅ {len(seq)} frames with valid pose.")

                        from feature_engineering import extract_joint_angles
                        import pandas as pd
                        import matplotlib.pyplot as plt

                        angle_records = [extract_joint_angles(kp) for kp in seq]
                        angle_df = pd.DataFrame(angle_records)
                        angle_df["frame"] = range(len(angle_df))

                        if show_angles:
                            st.markdown("**📐 Joint Angle Timeline**")
                            fig, axes = plt.subplots(2, 2, figsize=(12, 6))
                            pairs = [
                                ("left_knee_angle",  "right_knee_angle",  "Knee Angles"),
                                ("left_hip_angle",   "right_hip_angle",   "Hip Angles"),
                                ("left_elbow_angle", "right_elbow_angle", "Elbow Angles"),
                                ("trunk_lean_angle", None,                "Trunk Lean"),
                            ]
                            for ax, (c1, c2, title) in zip(axes.flat, pairs):
                                if c1 in angle_df:
                                    ax.plot(angle_df["frame"], angle_df[c1],
                                            label="Left", color="#6366f1", lw=2)
                                if c2 and c2 in angle_df:
                                    ax.plot(angle_df["frame"], angle_df[c2],
                                            label="Right", color="#f59e0b",
                                            lw=2, linestyle="--")
                                ax.set_title(title, fontweight="bold")
                                ax.set_xlabel("Frame")
                                ax.set_ylabel("Degrees")
                                ax.legend(); ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)

                        # Frame-by-frame form labels
                        from form_labeller import label_sequence
                        frame_labels = label_sequence(seq, exercise_type)
                        pct_correct  = float(frame_labels.mean()) * 100

                        st.markdown("---")
                        st.markdown("**🎯 Frame-by-Frame Form Score**")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Frames Analysed", len(seq))
                        c2.metric("Correct Frames",
                                  f"{int(frame_labels.sum())} ({pct_correct:.0f}%)")
                        c3.metric("Incorrect Frames",
                                  f"{int((frame_labels==0).sum())} "
                                  f"({100-pct_correct:.0f}%)")

                        # Colour bar with temporal smoothing
                        try:
                            from scipy.ndimage import uniform_filter1d
                            smooth_labels = (
                                uniform_filter1d(
                                    frame_labels.astype(float), size=5
                                ) >= 0.5
                            ).astype(int)
                        except ImportError:
                            smooth_labels = frame_labels

                        bar = "".join(
                            f'<span style="color:{"#28a745" if l else "#dc3545"}'
                            f';font-size:14px">█</span>'
                            for l in smooth_labels)
                        st.markdown(
                            "<div><small><b>Frame timeline (smoothed)</b> &nbsp;"
                            "<span style='color:#28a745'>█ correct</span> &nbsp;"
                            "<span style='color:#dc3545'>█ incorrect</span></small></div>"
                            + bar, unsafe_allow_html=True)

                        # ── ML Prediction: BiLSTM → RF → Rule fallback ──
                        st.markdown("---")
                        lstm_clf = load_lstm_model()
                        rf_clf   = load_model()

                        from feature_engineering import transform_batch
                        fv_seq = transform_batch(seq)   # (T, 161)

                        if lstm_clf is not None:
                            # — Round 2: BiLSTM (sequence-aware) —————————
                            result = lstm_clf.predict_single(fv_seq)
                            label  = result["label"]
                            conf   = result["confidence"]
                            st.caption("🤖 **Model: Bidirectional LSTM** (Round 2)")
                            col_a, col_b = st.columns(2)
                            col_a.metric("✅ Correct",
                                f"{result['prob_correct']*100:.1f}%")
                            col_b.metric("❌ Incorrect",
                                f"{result['prob_incorrect']*100:.1f}%")
                            if "Correct" in label:
                                st.markdown(
                                    f'<div class="correct-box"><h2>✅ CORRECT FORM</h2>'
                                    f'<h3>Confidence: {conf*100:.1f}%</h3></div>',
                                    unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    f'<div class="incorrect-box"><h2>❌ INCORRECT FORM</h2>'
                                    f'<h3>Confidence: {conf*100:.1f}%</h3></div>',
                                    unsafe_allow_html=True)

                        elif rf_clf is not None:
                            # — Round 1: Random Forest (frame-averaged) ——————
                            agg    = fv_seq.mean(axis=0).reshape(1, -1)
                            result = rf_clf.predict_single(agg)
                            label  = result["label"]
                            conf   = result["confidence"]
                            st.caption("🌲 **Model: Random Forest** (Round 1)  "
                                       " — train BiLSTM for better results: "
                                       "`python src/train_lstm.py`")
                            col_a, col_b = st.columns(2)
                            col_a.metric("✅ Correct",
                                f"{result['prob_correct']*100:.1f}%")
                            col_b.metric("❌ Incorrect",
                                f"{result['prob_incorrect']*100:.1f}%")
                            if "Correct" in label:
                                st.markdown(
                                    f'<div class="correct-box"><h2>✅ CORRECT FORM</h2>'
                                    f'<h3>Confidence: {conf*100:.1f}%</h3></div>',
                                    unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    f'<div class="incorrect-box"><h2>❌ INCORRECT FORM</h2>'
                                    f'<h3>Confidence: {conf*100:.1f}%</h3></div>',
                                    unsafe_allow_html=True)

                        else:
                            # Rule-based fallback before model is trained
                            st.info("⚠️ ML model not trained yet — using rule-based fallback. "
                                    "Run `python src/train.py` to train.")
                            if pct_correct >= 70:
                                st.success(
                                    f"✅ **CORRECT FORM** — "
                                    f"{pct_correct:.0f}% of frames within correct angle ranges")
                            else:
                                st.error(
                                    f"❌ **INCORRECT FORM** — "
                                    f"Only {pct_correct:.0f}% of frames within correct ranges")

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback; st.code(traceback.format_exc())
                finally:
                    try: os.unlink(tmp_path)
                    except: pass

    else:
        st.markdown("""
        <div class="info-card">
        Upload a <b>.avi or .mp4</b> exercise video. The system will extract
        33 body keypoints per frame, compute joint angles, and classify
        each frame as correct or incorrect form.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — Image
# ══════════════════════════════════════════════
with tab2:
    st.markdown("#### Upload a single exercise image")
    img_file = st.file_uploader("Choose an image",
                                 type=["jpg", "jpeg", "png"], key="img")
    if img_file:
        from PIL import Image as PILImage
        pil_img = PILImage.open(img_file).convert("RGB")
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.image(pil_img, caption="Input", use_column_width=True)

        with col2:
            with st.spinner("Extracting pose..."):
                try:
                    import mediapipe as mp
                    from video_keypoint_extractor import _result_to_array, _draw_skeleton

                    landmarker = get_image_landmarker()
                    frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    rgb       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                    result = landmarker.detect(mp_img)
                    # Note: do NOT close the cached landmarker

                    kp = _result_to_array(result)

                    if kp is None:
                        st.warning("No pose detected. Upload a clear full-body image.")
                    else:
                        annotated = frame_bgr.copy()
                        _draw_skeleton(annotated, kp)
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                 caption="Pose Skeleton", use_column_width=True)

                        if show_angles:
                            from feature_engineering import extract_joint_angles
                            angles = extract_joint_angles(kp)
                            st.markdown("**📐 Joint Angles**")
                            cols = st.columns(2)
                            for i, (name, val) in enumerate(angles.items()):
                                cols[i%2].metric(
                                    name.replace("_", " ").title(), f"{val:.1f}°")

                        from form_labeller import label_frame, get_rules
                        rules = get_rules(exercise_type)
                        label = label_frame(kp, rules)
                        st.markdown("---")
                        if label == 1:
                            st.success("✅ **CORRECT FORM** (rule-based)")
                        else:
                            st.error("❌ **INCORRECT FORM** (rule-based)")

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback; st.code(traceback.format_exc())


# ══════════════════════════════════════════════
# TAB 3 — Dataset Explorer
# ══════════════════════════════════════════════
with tab3:
    st.markdown("#### Dataset Explorer — Mandatory §7 Output")
    st.markdown("""
    <div class="info-card">
    AesCode Nexus Rules §7 require dataset dimensions printed at runtime
    before any model training.
    </div>""", unsafe_allow_html=True)

    if st.button("🔍 Generate Dimension Block", type="primary"):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        try:
            from dataset_explorer import explore_dataset
            with redirect_stdout(buf):
                explore_dataset()
            st.code(buf.getvalue(), language="text")
            st.success("✅ Screenshot this for your technical report.")
        except Exception as e:
            st.error(f"Error: {e}")
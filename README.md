# 🧠 FitPoseAI – AI Exercise Form Assistant

## 🚧 Project Status

Currently under development for **AesCode Nexus 2026 **\
Core pipeline, dataset validation, and pose estimation modules are being actively built.

---

## 🚀 Overview

FitPoseAI is an AI-powered system being developed to detect incorrect exercise form using human pose estimation and provide real-time feedback to prevent injuries and improve workout quality.

---

## 🎯 Problem Statement

**PS4 – AI-Based Detection of Incorrect Exercise Form Using Human Pose Estimation**

---

## 🎯 Objectives

- Detect human body posture using AI
- Calculate joint angles (e.g., knee, hip, elbow)
- Identify incorrect exercise form
- Provide visual and textual feedback
- Ensure cross-platform compatibility (Mac, Windows, Linux)

---

## 🧠 Planned Tech Stack

| Component       | Technology          |
| --------------- | ------------------- |
| Language        | Python              |
| Computer Vision | OpenCV              |
| Pose Estimation | MediaPipe           |
| UI              | Streamlit (primary) |
| Data Handling   | NumPy, Pandas       |

---

## 📁 Project Structure

```
FitPoseAI/
├── dataset/              # Provided dataset (not included in repo)
├── src/
│   ├── app.py            # Main application (in progress)
│   ├── pose_utils.py     # Pose + angle logic (planned)
│   └── dataset_check.py  # Dataset verification (Round 1)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/your-username/FitPoseAI.git
cd FitPoseAI
```

### 2. Create virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Planned Execution

### Step 1: Dataset Verification (Mandatory)

```
python src/dataset_check.py
```

### Step 2: Run Application (in progress)

```
streamlit run src/app.py
```

---

## 📊 Dataset Compliance (Hackathon Rules)

- Uses only the provided dataset
- No augmentation or external data
- Prints dataset dimensions before execution

---

## 🧪 Development Plan (Round 1)

- [x] Repository setup
- [ ] Dataset verification script
- [ ] Basic pose detection pipeline
- [ ] Joint angle calculation logic
- [ ] Initial UI prototype

---

## 🔮 Future Scope

- Real-time rep counting
- Personalized feedback system
- Mobile integration
- Advanced ML-based posture scoring

---

## 👨‍💻 Author

**Naushad Siddiqui**\
Team: *Synthexis*\
B.Tech CSE

---

## 📜 License

For educational and hackathon use only.
# FitPoseAI

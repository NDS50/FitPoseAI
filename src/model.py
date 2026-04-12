"""
model.py — AesCode PS4
Two classifiers:
  1. PoseFormClassifier  — Random Forest pipeline (Round 1)
  2. LSTMPoseClassifier  — Bidirectional LSTM on sequences (Round 2)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RANDOM_STATE,
    MODEL_SAVE_PATH, LSTM_SAVE_PATH, LABEL_NAMES,
)


# ══════════════════════════════════════════════
# 1.  Random Forest  (Round 1 — primary model)
# ══════════════════════════════════════════════

class PoseFormClassifier:
    """
    StandardScaler → RandomForestClassifier pipeline.
    Input  : (N, F) feature vectors from feature_engineering.py
    Output : 0 = Incorrect Form | 1 = Correct Form
    """

    def __init__(self, n_estimators=RF_N_ESTIMATORS,
                 max_depth=RF_MAX_DEPTH,
                 random_state=RANDOM_STATE):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",   # handles imbalanced labels
                random_state=random_state,
                n_jobs=-1,
            )),
        ])
        self.is_trained = False

    # ── Training ──────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray):
        print(f"  Training RandomForest on {X.shape[0]:,} samples, "
              f"{X.shape[1]} features ...")
        self.model.fit(X, y)
        self.is_trained = True
        print("  Done.")

    # ── Inference ─────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns [[p_incorrect, p_correct], ...] for each sample."""
        return self.model.predict_proba(X)

    def predict_single(self, x: np.ndarray) -> dict:
        """
        Predict one sample. Returns result dict for the Streamlit UI.
        x : np.ndarray shape (F,) or (1, F)
        """
        x2d   = x.reshape(1, -1)
        pred  = self.predict(x2d)[0]
        proba = self.predict_proba(x2d)[0]
        return {
            "label":          LABEL_NAMES[int(pred)],
            "label_id":       int(pred),
            "confidence":     float(proba[pred]),
            "prob_correct":   float(proba[1]),
            "prob_incorrect": float(proba[0]),
        }

    # ── Evaluation ────────────────────────────

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 split: str = "Test") -> dict:
        y_pred  = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        acc = accuracy_score(y, y_pred)
        f1  = f1_score(y, y_pred, average="weighted")
        try:
            auc = roc_auc_score(y, y_proba)
        except Exception:
            auc = float("nan")
        cm = confusion_matrix(y, y_pred)

        print(f"\n{'='*50}")
        print(f"  [{split}] Evaluation")
        print(f"{'='*50}")
        print(f"  Accuracy  : {acc*100:.2f}%")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  ROC AUC   : {auc:.4f}")
        print(f"\n{classification_report(y, y_pred, target_names=['Incorrect','Correct'])}")
        print(f"  Confusion Matrix:\n    {cm}")
        print(f"{'='*50}\n")

        return {"accuracy": acc, "f1": f1, "auc": auc, "cm": cm}

    def get_feature_importances(self, feature_names: list) -> list:
        clf = self.model.named_steps["clf"]
        imp = clf.feature_importances_
        idx = np.argsort(imp)[::-1][:20]
        return [(feature_names[i], float(imp[i])) for i in idx]

    # ── Persistence ───────────────────────────

    def save(self, path: str = MODEL_SAVE_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"  Model saved → {path}")

    def load(self, path: str = MODEL_SAVE_PATH):
        self.model     = joblib.load(path)
        self.is_trained = True
        print(f"  Model loaded ← {path}")


# ══════════════════════════════════════════════
# 2.  BiLSTM  (Round 2 — sequence model)
# ══════════════════════════════════════════════

class LSTMNet(nn.Module):
    """Bidirectional LSTM: (batch, T, F) → (batch, 2) logits."""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3,
                 num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size * 2, num_classes)  # ×2 bidirectional

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))   # last time-step


class LSTMPoseClassifier:
    """
    Wrapper around LSTMNet with full train / predict / load interface.
    Used for per-video sequence classification (Round 2).
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3,
                 seq_len: int = 30,
                 lr: float = 1e-3, device: str = "cpu"):
        self.device     = torch.device(device)
        self.seq_len    = seq_len
        self.input_size = input_size
        self.net = LSTMNet(input_size, hidden_size,
                           num_layers, dropout).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    # ── Inference ─────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X : (N, T, F) or already-padded tensor."""
        self.net.eval()
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.net(Xt).argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return torch.softmax(self.net(Xt), dim=1).cpu().numpy()

    def predict_single(self, fv_seq: np.ndarray) -> dict:
        """
        Predict form quality for ONE video sequence.

        Args:
            fv_seq : np.ndarray (t, F) — per-frame feature vectors
                     (output of feature_engineering.transform_batch)
                     Will be padded / truncated to self.seq_len internally.

        Returns:
            dict with label, label_id, confidence, prob_correct,
                 prob_incorrect, model_type
        """
        from lstm_dataset import pad_or_truncate
        padded = pad_or_truncate(fv_seq, self.seq_len)   # (T, F)
        x3d    = torch.tensor(padded, dtype=torch.float32) \
                      .unsqueeze(0).to(self.device)       # (1, T, F)

        self.net.eval()
        with torch.no_grad():
            logits = self.net(x3d)
            proba  = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred   = int(logits.argmax(dim=1).cpu().item())

        return {
            "label":          LABEL_NAMES[pred],
            "label_id":       pred,
            "confidence":     float(proba[pred]),
            "prob_correct":   float(proba[1]),
            "prob_incorrect": float(proba[0]),
            "model_type":     "BiLSTM",
        }

    # ── Persistence ───────────────────────────

    def save(self, path: str = LSTM_SAVE_PATH, **extra_meta):
        """Save weights + architecture metadata dict."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict":  self.net.state_dict(),
            "input_size":  self.input_size,
            "hidden_size": self.net.lstm.hidden_size,
            "num_layers":  self.net.lstm.num_layers,
            "dropout":     self.net.drop.p,
            "seq_len":     self.seq_len,
            **extra_meta,
        }, path)
        print(f"  LSTM weights saved → {path}")

    def load(self, path: str = LSTM_SAVE_PATH):
        """
        Load from path.  Handles both:
          - old format : raw state_dict
          - new format : metadata dict with 'state_dict' key
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            self.net.load_state_dict(ckpt["state_dict"])
            self.seq_len    = ckpt.get("seq_len", self.seq_len)
            self.input_size = ckpt.get("input_size", self.input_size)
        else:
            self.net.load_state_dict(ckpt)   # raw state_dict fallback
        print(f"  LSTM weights loaded ← {path}")

    @classmethod
    def from_checkpoint(cls, path: str) -> "LSTMPoseClassifier":
        """
        Reconstruct a fully-ready classifier from a saved checkpoint.
        No need to know the hyperparameters — they are stored in the file.

        Usage:
            clf = LSTMPoseClassifier.from_checkpoint("saved_model/lstm_weights.pt")
            result = clf.predict_single(fv_seq)
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
            raise ValueError(
                f"Checkpoint at {path} is in old format (raw state_dict). "
                "Re-train with train_lstm.py to generate a full metadata checkpoint.")

        obj = cls(
            input_size=ckpt["input_size"],
            hidden_size=ckpt.get("hidden_size", 128),
            num_layers=ckpt.get("num_layers",  2),
            dropout=ckpt.get("dropout",     0.3),
            seq_len=ckpt.get("seq_len",     30),
        )
        obj.net.load_state_dict(ckpt["state_dict"])
        obj.net.eval()
        print(f"  BiLSTM loaded ← {path}  "
              f"(val_acc={ckpt.get('val_acc', float('nan'))*100:.1f}%  "
              f"val_f1={ckpt.get('val_f1', float('nan')):.4f})")
        return obj


# ── Quick sanity test ─────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    F = 155
    X_tr = np.random.randn(200, F).astype(np.float32)
    y_tr = np.random.randint(0, 2, 200)
    X_te = np.random.randn(50,  F).astype(np.float32)
    y_te = np.random.randint(0, 2, 50)

    clf = PoseFormClassifier()
    clf.fit(X_tr, y_tr)
    clf.evaluate(X_te, y_te)
    print("Single prediction:", clf.predict_single(X_te[0]))
    print("model.py OK")

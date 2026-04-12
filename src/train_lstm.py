"""
train_lstm.py — AesCode PS4 Round 2
Complete BiLSTM training pipeline for per-video exercise form classification.

Pipeline:
  1. Print mandatory §7 dataset dimensions
  2. Load cached keypoints (shared with train.py cache — no re-extraction)
  3. Build per-frame feature vectors (161-d) for every video
  4. Pad / truncate sequences to fixed length (default: 30 frames)
  5. Label each video via majority vote of frame-level angle rules
  6. Train Bidirectional LSTM with early stopping & LR scheduling
  7. Evaluate best checkpoint on validation split
  8. Save model weights + metadata dict

Usage:
  python src/train_lstm.py
  python src/train_lstm.py --seq-len 40 --epochs 80 --hidden 256
  python src/train_lstm.py --no-cache        # force re-extract keypoints
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
)

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    TRAIN_DIR, VAL_DIR, LSTM_SAVE_PATH, LABEL_NAMES,
)
from dataset_explorer    import explore_dataset
from video_keypoint_extractor import extract_split
from form_labeller       import get_rules, label_sequence
from feature_engineering import transform_batch
from model               import LSTMNet
from lstm_dataset        import PoseSequenceDataset, DEFAULT_SEQ_LEN


# ══════════════════════════════════════════════
# Per-video LSTM dataset builder
# ══════════════════════════════════════════════

def build_lstm_dataset(class_sequences: dict,
                       seq_len: int = DEFAULT_SEQ_LEN) -> tuple:
    """
    Convert {class_name → [raw_kp_seq, ...]} into padded LSTM inputs.

    Args:
        class_sequences : output of video_keypoint_extractor.extract_split()
        seq_len         : fixed frame count T

    Returns:
        feature_seqs : list of np.ndarray (t_i, 161) — per-video features
        labels       : list of int  (0 / 1 per video, majority vote)
        meta         : list of dicts
    """
    feature_seqs, labels, meta = [], [], []

    for class_name, sequences in class_sequences.items():
        rules = get_rules(class_name)
        n = len(sequences)
        print(f"  {class_name:<28}  {n} videos  "
              f"(rules: {list(rules.keys())})")

        for vid_idx, raw_seq in enumerate(sequences):
            if len(raw_seq) == 0:
                continue

            # Per-frame 161-d feature vectors  →  (t, 161)
            fv_seq = transform_batch(raw_seq)

            # Per-video label: majority vote of frame-level rule labels
            frame_labels = label_sequence(raw_seq, class_name)
            vid_label    = int(frame_labels.mean() >= 0.5)

            feature_seqs.append(fv_seq)
            labels.append(vid_label)
            meta.append({"class": class_name, "video_idx": vid_idx,
                          "n_frames": len(fv_seq)})

    if not labels:
        return [], [], []

    nc = sum(labels)
    ni = len(labels) - nc
    print(f"\n  Total videos : {len(labels)}")
    print(f"  Correct  (1) : {nc}  ({nc/len(labels)*100:.1f}%)")
    print(f"  Incorrect(0) : {ni}  ({ni/len(labels)*100:.1f}%)")

    return feature_seqs, labels, meta


# ══════════════════════════════════════════════
# Training / eval loops
# ══════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, device,
              training: bool = True):
    model.train() if training else model.eval()

    total_loss = 0.0
    all_preds, all_probs, all_true = [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            if training:
                optimizer.zero_grad()

            logits = model(X)
            loss   = criterion(logits, y)

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            probs = torch.softmax(logits, dim=1)[:, 1]
            total_loss += loss.item() * len(y)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_probs.extend(probs.cpu().detach().tolist())
            all_true.extend(y.cpu().tolist())

    n   = max(len(all_true), 1)
    acc = accuracy_score(all_true, all_preds)
    f1  = f1_score(all_true, all_preds, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(all_true, all_probs)
    except Exception:
        auc = float("nan")

    return total_loss / n, acc, f1, auc, all_preds, all_true


# ══════════════════════════════════════════════
# Main training function
# ══════════════════════════════════════════════

def run_lstm_training(
    seq_len:    int   = DEFAULT_SEQ_LEN,
    epochs:     int   = 50,
    batch_size: int   = 16,
    lr:         float = 1e-3,
    hidden_size: int  = 128,
    num_layers: int   = 2,
    dropout:    float = 0.3,
    patience:   int   = 10,
    use_cache:  bool  = True,
):
    # ── Device ──────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n🖥️   Device : {device}")

    # ══ STEP 1 — Dataset dimensions ════════════
    print("\n" + "="*60)
    print("  STEP 1 — DATASET DIMENSIONS")
    print("="*60)
    explore_dataset()

    # ══ STEP 2 — Extract keypoints (cached) ════
    print("\n" + "="*60)
    print("  STEP 2 — LOADING KEYPOINTS"
          "  (shared cache with train.py)")
    print("="*60)
    train_sequences = extract_split(TRAIN_DIR, "train", use_cache=use_cache)
    val_sequences   = extract_split(VAL_DIR,   "val",   use_cache=use_cache)

    # ══ STEP 3 — Build per-video LSTM datasets ═
    print("\n" + "="*60)
    print("  STEP 3 — BUILDING PER-VIDEO LSTM DATASETS")
    print(f"  Sequence length : {seq_len} frames  "
          f"| Feature dim : 161")
    print("="*60)

    print("\n  [TRAIN]")
    tr_seqs, tr_labels, _ = build_lstm_dataset(train_sequences, seq_len)
    print("\n  [VAL]")
    va_seqs, va_labels, _ = build_lstm_dataset(val_sequences,   seq_len)

    if not tr_seqs:
        print("\n[ERROR] No training data — check DATASET_ROOT in config.py")
        return None

    train_ds = PoseSequenceDataset(tr_seqs, tr_labels, seq_len)
    val_ds   = PoseSequenceDataset(va_seqs, va_labels, seq_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=0, drop_last=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, num_workers=0)

    # ══ STEP 4 — Build BiLSTM model ════════════
    print("\n" + "="*60)
    print("  STEP 4 — BUILDING BIDIRECTIONAL LSTM")
    print("="*60)

    input_size = train_ds.input_size   # 161
    model = LSTMNet(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Input dim    : {input_size}  (per-frame features)")
    print(f"  Seq length   : {seq_len}  frames")
    print(f"  Hidden size  : {hidden_size} × 2  (bidirectional)")
    print(f"  LSTM layers  : {num_layers}")
    print(f"  Dropout      : {dropout}")
    print(f"  Parameters   : {n_params:,}")

    # Class-weighted loss
    weights   = train_ds.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5)

    # ══ STEP 5 — Training loop ══════════════════
    print("\n" + "="*60)
    print(f"  STEP 5 — TRAINING  "
          f"(epochs={epochs}, patience={patience})")
    print("="*60 + "\n")

    best_val_f1  = -1.0
    best_state   = None
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1, _, _, _ = run_epoch(
            model, train_dl, criterion, optimizer, device, training=True)
        va_loss, va_acc, va_f1, va_auc, _, _ = run_epoch(
            model, val_dl, criterion, optimizer, device, training=False)
        scheduler.step(va_f1)

        marker = ""
        if va_f1 > best_val_f1:
            best_val_f1  = va_f1
            best_state   = {k: v.clone()
                            for k, v in model.state_dict().items()}
            patience_ctr = 0
            marker = "  ← best"
        else:
            patience_ctr += 1

        if epoch % 5 == 0 or epoch == 1 or patience_ctr == 0:
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc*100:.1f}%  "
                  f"| val_loss={va_loss:.4f}  val_acc={va_acc*100:.1f}%  "
                  f"val_f1={va_f1:.4f}  val_auc={va_auc:.4f}{marker}")

        if patience_ctr >= patience:
            print(f"\n  ⚡ Early stopping at epoch {epoch} "
                  f"(no val_f1 improvement for {patience} epochs)")
            break

    # ══ STEP 6 — Evaluate best checkpoint ═══════
    print("\n" + "="*60)
    print("  STEP 6 — FINAL EVALUATION  (best checkpoint)")
    print("="*60)

    if best_state:
        model.load_state_dict(best_state)

    _, tr_acc, tr_f1, tr_auc, _, _ = run_epoch(
        model, train_dl, criterion, optimizer, device, training=False)
    _, va_acc, va_f1, va_auc, va_preds, va_true = run_epoch(
        model, val_dl, criterion, optimizer, device, training=False)

    print(f"\n  [Train]  Acc={tr_acc*100:.2f}%  "
          f"F1={tr_f1:.4f}  AUC={tr_auc:.4f}")
    print(f"  [Val]    Acc={va_acc*100:.2f}%  "
          f"F1={va_f1:.4f}  AUC={va_auc:.4f}")
    print(f"\n{classification_report(va_true, va_preds, zero_division=0, target_names=['Incorrect','Correct'])}")

    # ══ STEP 7 — Save ═══════════════════════════
    print("\n" + "="*60)
    print("  STEP 7 — SAVING LSTM WEIGHTS + METADATA")
    print("="*60)

    os.makedirs(os.path.dirname(LSTM_SAVE_PATH), exist_ok=True)

    # Save weights AND all architecture / preprocessing metadata
    # so app.py can reconstruct the model without guessing hyperparams.
    torch.save({
        "state_dict":  best_state or model.state_dict(),
        "input_size":  input_size,
        "hidden_size": hidden_size,
        "num_layers":  num_layers,
        "dropout":     dropout,
        "seq_len":     seq_len,
        "val_acc":     va_acc,
        "val_f1":      va_f1,
        "val_auc":     va_auc,
    }, LSTM_SAVE_PATH)

    print(f"  Saved → {LSTM_SAVE_PATH}")
    print(f"\n✅ LSTM Training complete.  "
          f"Val Acc: {va_acc*100:.2f}%  F1: {va_f1:.4f}  AUC: {va_auc:.4f}")
    return model


# ── CLI ───────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AesCode PS4 Round 2 — BiLSTM Training Pipeline")
    parser.add_argument("--seq-len",    type=int,   default=DEFAULT_SEQ_LEN,
                        help=f"Frames per video (default: {DEFAULT_SEQ_LEN})")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--hidden",     type=int,   default=128)
    parser.add_argument("--layers",     type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--no-cache",   action="store_true",
                        help="Force re-extraction of all keypoints")
    args = parser.parse_args()

    run_lstm_training(
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        patience=args.patience,
        use_cache=not args.no_cache,
    )

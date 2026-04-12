"""
lstm_dataset.py — AesCode PS4 Round 2
PyTorch Dataset for per-video sequence classification.

Pads / truncates variable-length pose sequences to a fixed
temporal length T so they can be batched for the BiLSTM.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_SEQ_LEN = 30   # frames per video (pad short, truncate long)


# ─────────────────────────────────────────────
# Sequence normalisation
# ─────────────────────────────────────────────

def pad_or_truncate(seq: np.ndarray, T: int) -> np.ndarray:
    """
    seq : (t, F)  →  (T, F)
    t < T : zero-pad at the end
    t > T : keep the first T frames (covers the key exercise phase)
    """
    t, F = seq.shape
    if t >= T:
        return seq[:T].astype(np.float32)
    pad = np.zeros((T - t, F), dtype=np.float32)
    return np.vstack([seq.astype(np.float32), pad])


# ─────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────

class PoseSequenceDataset(Dataset):
    """
    Dataset of (padded_sequence, label) pairs.

    Args:
        sequences : list of np.ndarray  (t_i, F) — variable length
        labels    : list / array of int  0 = Incorrect | 1 = Correct
        seq_len   : fixed temporal length T to pad / truncate to
    """

    def __init__(self, sequences: list, labels: list,
                 seq_len: int = DEFAULT_SEQ_LEN):
        self.seq_len = seq_len
        self.X = [pad_or_truncate(s, seq_len) for s in sequences]
        self.y = list(labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

    # ── Helpers ───────────────────────────────

    @property
    def input_size(self) -> int:
        """Feature dimension F of each frame."""
        return self.X[0].shape[1]

    def class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency class weights [w_incorrect, w_correct].
        Pass to nn.CrossEntropyLoss(weight=...) for imbalanced data.
        """
        n  = len(self.y)
        nc = sum(self.y)
        ni = n - nc
        wi = n / (2 * ni + 1e-8)
        wc = n / (2 * nc + 1e-8)
        return torch.tensor([wi, wc], dtype=torch.float32)

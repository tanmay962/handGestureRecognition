"""
ml_models.py — PyTorch gesture recognition models

Static gestures: MLP with BatchNorm (GestureNet / PyTorchNN)
Dynamic gestures: bidirectional LSTM with attention (GestureLSTM / LSTMModel)

Both models are trained entirely on the Python side and serialised
to JSON for storage in SQLite. The JS frontend just sends feature
vectors over HTTP — it never touches the model weights directly.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from database import STATIC_FEATURES, DYNAMIC_FEATURES, DYNAMIC_FRAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_VERSION = "1.0"


# ── Feature scaler (z-score normalisation) ───────────────────────────────────

class FeatureScaler:
    """Fits a z-score scaler on training data and applies it consistently at inference time."""

    def __init__(self):
        self.mean_  = None
        self.std_   = None
        self.fitted = False

    def fit(self, X):
        arr        = np.array(X, dtype=np.float32)
        self.mean_ = arr.mean(axis=0)
        self.std_  = arr.std(axis=0) + 1e-8
        self.fitted = True

    def transform(self, X):
        if not self.fitted:
            return X
        arr = np.array(X, dtype=np.float32)
        return ((arr - self.mean_) / self.std_).tolist()

    def transform_one(self, vec):
        if not self.fitted:
            return vec
        arr = np.array(vec, dtype=np.float32)
        return ((arr - self.mean_) / self.std_).tolist()

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def to_dict(self):
        return {
            'mean': self.mean_.tolist() if self.mean_ is not None else None,
            'std':  self.std_.tolist()  if self.std_  is not None else None,
        }

    def from_dict(self, d):
        if d.get('mean') is not None:
            self.mean_  = np.array(d['mean'], dtype=np.float32)
            self.std_   = np.array(d['std'],  dtype=np.float32)
            self.fitted = True


# ── Data augmentation ─────────────────────────────────────────────────────────

def augment_static(inputs, labels, n_copies=2, noise_std=0.04):
    """Gaussian noise + curl scale jitter — roughly triples the training set."""
    aug_x = list(inputs)
    aug_y = list(labels)

    for vec, lbl in zip(inputs, labels):
        arr = np.array(vec, dtype=np.float32)
        for _ in range(n_copies):
            noisy = arr + np.random.normal(0, noise_std, arr.shape).astype(np.float32)
            scale = float(np.random.uniform(0.88, 1.12))
            # Scale curl values for both hands
            for ci in list(range(5)) + list(range(11, 16)):
                if ci < len(noisy):
                    noisy[ci] = float(np.clip(noisy[ci] * scale, 0.0, 1.0))
            aug_x.append(noisy.tolist())
            aug_y.append(lbl)

    return aug_x, aug_y


def augment_dynamic(inputs, labels, n_copies=1, noise_std=0.02):
    """Time-stretch ±20% + Gaussian noise — doubles the training set."""
    aug_x = list(inputs)
    aug_y = list(labels)

    for seq, lbl in zip(inputs, labels):
        arr    = np.array(seq, dtype=np.float32)
        frames = len(arr) // DYNAMIC_FEATURES
        if frames < 2:
            continue

        arr2d = arr[:frames * DYNAMIC_FEATURES].reshape(frames, DYNAMIC_FEATURES)

        for _ in range(n_copies):
            factor = float(np.random.uniform(0.8, 1.2))
            new_n  = max(2, int(frames * factor))
            idx_f  = np.linspace(0, frames - 1, new_n)

            stretched = np.zeros((new_n, DYNAMIC_FEATURES), dtype=np.float32)
            for fi, fi_f in enumerate(idx_f):
                lo = int(fi_f)
                hi = min(lo + 1, frames - 1)
                t  = fi_f - lo
                stretched[fi] = arr2d[lo] * (1.0 - t) + arr2d[hi] * t

            out = np.zeros((DYNAMIC_FRAMES, DYNAMIC_FEATURES), dtype=np.float32)
            copy_n = min(DYNAMIC_FRAMES, new_n)
            out[:copy_n] = stretched[:copy_n]
            out += np.random.normal(0, noise_std, out.shape).astype(np.float32)

            aug_x.append(out.flatten().tolist())
            aug_y.append(lbl)

    return aug_x, aug_y


# ── Static model: MLP with BatchNorm ─────────────────────────────────────────

class GestureNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev   = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.15)]
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PyTorchNN:
    """Wraps GestureNet with training, inference, and serialisation helpers."""

    def __init__(self, nn_id='static'):
        self.id              = nn_id
        self.model           = None
        self.gestures        = {}      # idx → name
        self.feature_version = FEATURE_VERSION
        self.trained         = False
        self.accuracy        = 0.0
        self.val_accuracy    = 0.0
        self.loss            = 1.0
        self.epochs          = 0
        self.per_gesture_acc = {}
        self.scaler          = FeatureScaler()
        self._input_size     = 0
        self._hidden_sizes   = []
        self._output_size    = 0

    def initialize(self, input_size, hidden_sizes, output_size):
        self._input_size   = input_size
        self._hidden_sizes = hidden_sizes
        self._output_size  = output_size
        self.model = GestureNet(input_size, hidden_sizes, output_size).to(DEVICE)

    def _pad_samples(self, samples):
        out = []
        for s in samples:
            if len(s) < self._input_size:
                s = list(s) + [0.0] * (self._input_size - len(s))
            out.append(s[:self._input_size])
        return out

    def train_batch(self, inputs, labels, epochs=10, lr=0.008):
        if not inputs or self.model is None:
            return

        n = len(inputs)

        # Hold out 20% as a validation set when we have enough samples
        if n >= 5:
            val_n      = max(1, int(n * 0.2))
            idx_all    = list(range(n))
            random.shuffle(idx_all)
            val_idx    = idx_all[:val_n]
            train_idx  = idx_all[val_n:]
        else:
            train_idx = list(range(n))
            val_idx   = []

        tr_in = [inputs[i] for i in train_idx]
        tr_lb = [labels[i] for i in train_idx]
        vl_in = [inputs[i] for i in val_idx]
        vl_lb = [labels[i] for i in val_idx]

        # Augment only the training split
        if len(tr_in) >= 4:
            tr_in, tr_lb = augment_static(tr_in, tr_lb)

        tr_pad = self._pad_samples(tr_in)
        vl_pad = self._pad_samples(vl_in) if vl_in else []

        # Fit scaler on augmented training data and apply to both splits
        self.scaler.fit(tr_pad)
        tr_sc = self.scaler.transform(tr_pad)
        vl_sc = self.scaler.transform(vl_pad) if vl_pad else []

        X = torch.tensor(tr_sc, dtype=torch.float32).to(DEVICE)
        y = torch.tensor(tr_lb, dtype=torch.long).to(DEVICE)

        # Class-balanced loss
        try:
            counts = torch.bincount(y, minlength=len(self.gestures)).float()
            w = 1.0 / (counts + 1e-6)
            w = w / w.sum() * len(self.gestures)
            crit = nn.CrossEntropyLoss(weight=w.to(DEVICE))
        except Exception:
            crit = nn.CrossEntropyLoss()

        opt   = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

        self.model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = crit(self.model(X), y)
            loss.backward()
            opt.step()
            sched.step()

        self.model.eval()
        with torch.no_grad():
            out   = self.model(X)
            preds = out.argmax(dim=1)
            self.accuracy = (preds == y).float().mean().item()
            self.loss     = crit(out, y).item()

            # Validation accuracy on unseen data (no augmentation applied)
            self.val_accuracy = 0.0
            if vl_sc:
                Xv = torch.tensor(vl_sc, dtype=torch.float32).to(DEVICE)
                yv = torch.tensor(vl_lb, dtype=torch.long).to(DEVICE)
                vp = self.model(Xv).argmax(dim=1)
                self.val_accuracy = float((vp == yv).float().mean().item())

            # Per-gesture breakdown
            self.per_gesture_acc = {}
            for idx, name in self.gestures.items():
                mask = (y == idx)
                if mask.sum() > 0:
                    self.per_gesture_acc[name] = (preds[mask] == y[mask]).float().mean().item()

        self.epochs  += epochs
        self.trained  = True

    def predict(self, features, conf_threshold=0.0):
        if not self.trained or self.model is None:
            return {"idx": -1, "conf": 0.0, "probs": [], "name": "Unknown", "below_threshold": False}

        self.model.eval()
        vec = features + [0.0] * max(0, self._input_size - len(features))
        vec = vec[:self._input_size]

        if self.scaler.fitted:
            vec = self.scaler.transform_one(vec)

        x = torch.tensor([vec], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]

        idx  = int(probs.argmax().item())
        conf = float(probs[idx].item())
        return {
            "idx":             idx,
            "conf":            conf,
            "probs":           probs.tolist(),
            "name":            self.gestures.get(idx, "Unknown"),
            "below_threshold": conf < conf_threshold,
        }

    def add_gesture(self, name, idx):
        self.gestures[idx] = name

    def reset(self):
        self.model           = None
        self.trained         = False
        self.gestures        = {}
        self.accuracy        = 0.0
        self.val_accuracy    = 0.0
        self.loss            = 1.0
        self.epochs          = 0
        self.per_gesture_acc = {}
        self.scaler          = FeatureScaler()

    def to_json(self):
        state = {k: v.tolist() for k, v in self.model.state_dict().items()} if self.model else {}
        return {
            "id":              self.id,
            "state_dict":      state,
            "gestures":        list(self.gestures.items()),
            "accuracy":        self.accuracy,
            "val_accuracy":    self.val_accuracy,
            "loss":            self.loss,
            "epochs":          self.epochs,
            "input_size":      self._input_size,
            "hidden_sizes":    self._hidden_sizes,
            "output_size":     self._output_size,
            "per_gesture_acc": self.per_gesture_acc,
            "scaler":          self.scaler.to_dict(),
        }

    def from_json(self, d):
        self._input_size   = d.get("input_size",   0)
        self._hidden_sizes = d.get("hidden_sizes",  [])
        self._output_size  = d.get("output_size",  0)

        if self._input_size and self._output_size:
            self.initialize(self._input_size, self._hidden_sizes, self._output_size)
            if d.get("state_dict"):
                sd = {k: torch.tensor(v) for k, v in d["state_dict"].items()}
                self.model.load_state_dict(sd)
                self.model.to(DEVICE)

        self.gestures        = {int(k): v for k, v in d.get("gestures", [])}
        self.accuracy        = d.get("accuracy",     0.0)
        self.val_accuracy    = d.get("val_accuracy", 0.0)
        self.loss            = d.get("loss",         1.0)
        self.epochs          = d.get("epochs",       0)
        self.per_gesture_acc = d.get("per_gesture_acc", {})
        self.trained         = bool(self._input_size and d.get("state_dict"))

        if d.get("scaler"):
            self.scaler.from_dict(d["scaler"])


# ── Dynamic model: bidirectional LSTM with attention ──────────────────────────

class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout=0.3, bidirectional=True):
        super().__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        nd = 2 if bidirectional else 1

        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0,
                               bidirectional=bidirectional)
        self.attn    = nn.Linear(hidden_size * nd, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(hidden_size * nd)
        self.fc      = nn.Linear(hidden_size * nd, output_size)

    def forward(self, x):
        nd = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * nd, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * nd, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # Temporal attention: weight frames by relevance
        attn_w = torch.softmax(self.attn(out), dim=1)
        ctx    = (attn_w * out).sum(dim=1)
        return self.fc(self.dropout(self.norm(ctx)))


class LSTMModel:
    """Wraps GestureLSTM with training, inference, and serialisation helpers."""

    def __init__(self):
        self.model           = None
        self.gestures        = {}
        self.trained         = False
        self.accuracy        = 0.0
        self.val_accuracy    = 0.0
        self.loss            = 1.0
        self.epochs          = 0
        self.hidden_size     = 128
        self.num_layers      = 2
        self._output_size    = 0
        self.per_gesture_acc = {}

    def initialize(self, num_classes, gesture_list):
        self._output_size = num_classes
        self.model = GestureLSTM(
            DYNAMIC_FEATURES, self.hidden_size, self.num_layers, num_classes,
            bidirectional=True
        ).to(DEVICE)
        self.gestures = {i: name for i, name in enumerate(gesture_list)}

    def _prepare(self, raw_sequences):
        """Pad/truncate each sequence to DYNAMIC_FRAMES × DYNAMIC_FEATURES."""
        seqs = []
        for s in raw_sequences:
            arr    = np.array(s, dtype=np.float32)
            frames = len(arr) // DYNAMIC_FEATURES
            if frames < DYNAMIC_FRAMES:
                pad = np.zeros(DYNAMIC_FRAMES * DYNAMIC_FEATURES, dtype=np.float32)
                pad[:len(arr)] = arr
                arr    = pad
                frames = DYNAMIC_FRAMES
            seqs.append(arr[:DYNAMIC_FRAMES * DYNAMIC_FEATURES].reshape(DYNAMIC_FRAMES, DYNAMIC_FEATURES))
        return np.array(seqs, dtype=np.float32)

    def train_batch(self, inputs, labels, epochs=10, lr=0.001):
        if not inputs or self.model is None:
            return

        n = len(inputs)

        if n >= 5:
            val_n     = max(1, int(n * 0.2))
            idx_all   = list(range(n))
            random.shuffle(idx_all)
            val_idx   = idx_all[:val_n]
            train_idx = idx_all[val_n:]
        else:
            train_idx = list(range(n))
            val_idx   = []

        tr_in = [inputs[i] for i in train_idx]
        tr_lb = [labels[i] for i in train_idx]
        vl_in = [inputs[i] for i in val_idx]
        vl_lb = [labels[i] for i in val_idx]

        if len(tr_in) >= 3:
            tr_in, tr_lb = augment_dynamic(tr_in, tr_lb)

        X = torch.tensor(self._prepare(tr_in), dtype=torch.float32).to(DEVICE)
        y = torch.tensor(tr_lb, dtype=torch.long).to(DEVICE)

        try:
            counts = torch.bincount(y, minlength=len(self.gestures)).float()
            w      = 1.0 / (counts + 1e-6)
            w      = w / w.sum() * len(self.gestures)
            crit   = nn.CrossEntropyLoss(weight=w.to(DEVICE))
        except Exception:
            crit = nn.CrossEntropyLoss()

        opt   = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

        self.model.train()
        for ep in range(epochs):
            opt.zero_grad()
            loss = crit(self.model(X), y)
            loss.backward()
            opt.step()
            if ep % 5 == 0:
                sched.step(loss)

        self.model.eval()
        with torch.no_grad():
            out   = self.model(X)
            preds = out.argmax(dim=1)
            self.accuracy = (preds == y).float().mean().item()
            self.loss     = crit(out, y).item()

            self.val_accuracy = 0.0
            if vl_in:
                Xv = torch.tensor(self._prepare(vl_in), dtype=torch.float32).to(DEVICE)
                yv = torch.tensor(vl_lb, dtype=torch.long).to(DEVICE)
                vp = self.model(Xv).argmax(dim=1)
                self.val_accuracy = float((vp == yv).float().mean().item())

            self.per_gesture_acc = {}
            for idx, name in self.gestures.items():
                mask = (y == idx)
                if mask.sum() > 0:
                    self.per_gesture_acc[name] = (preds[mask] == y[mask]).float().mean().item()

        self.epochs  += epochs
        self.trained  = True

    def predict(self, flat_sequence, conf_threshold=0.0):
        if not self.trained or self.model is None:
            return {"idx": -1, "conf": 0.0, "probs": [], "name": "Unknown", "below_threshold": False}

        self.model.eval()
        arr = np.array(flat_sequence, dtype=np.float32)
        expected = DYNAMIC_FRAMES * DYNAMIC_FEATURES
        if len(arr) < expected:
            pad        = np.zeros(expected, dtype=np.float32)
            pad[:len(arr)] = arr
            arr        = pad
        arr = arr[:expected].reshape(1, DYNAMIC_FRAMES, DYNAMIC_FEATURES)

        X = torch.tensor(arr, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(X), dim=1)[0]

        idx  = int(probs.argmax().item())
        conf = float(probs[idx].item())
        return {
            "idx":             idx,
            "conf":            conf,
            "probs":           probs.tolist(),
            "name":            self.gestures.get(idx, "Unknown"),
            "below_threshold": conf < conf_threshold,
        }

    def add_gesture(self, name, idx):
        self.gestures[idx] = name

    def reset(self):
        self.model           = None
        self.trained         = False
        self.gestures        = {}
        self.accuracy        = 0.0
        self.val_accuracy    = 0.0
        self.loss            = 1.0
        self.epochs          = 0
        self.per_gesture_acc = {}

    def to_json(self):
        state = {k: v.tolist() for k, v in self.model.state_dict().items()} if self.model else {}
        return {
            "type":            "lstm",
            "state_dict":      state,
            "gestures":        list(self.gestures.items()),
            "accuracy":        self.accuracy,
            "val_accuracy":    self.val_accuracy,
            "loss":            self.loss,
            "epochs":          self.epochs,
            "hidden_size":     self.hidden_size,
            "num_layers":      self.num_layers,
            "output_size":     self._output_size,
            "per_gesture_acc": self.per_gesture_acc,
            "bidirectional":   True,
        }

    def from_json(self, d):
        gesture_list = [v for _, v in sorted(d.get("gestures", []), key=lambda x: int(x[0]))]
        if not gesture_list:
            return

        self.hidden_size  = d.get("hidden_size",  128)
        self.num_layers   = d.get("num_layers",   2)
        self._output_size = d.get("output_size",  len(gesture_list))
        self.initialize(self._output_size, gesture_list)

        if d.get("state_dict"):
            try:
                sd = {k: torch.tensor(v) for k, v in d["state_dict"].items()}
                self.model.load_state_dict(sd)
                self.model.to(DEVICE)
            except Exception as e:
                print(f"[LSTM] Architecture mismatch — retrain needed: {e}")

        self.gestures        = {int(k): v for k, v in d.get("gestures", [])}
        self.accuracy        = d.get("accuracy",     0.0)
        self.val_accuracy    = d.get("val_accuracy", 0.0)
        self.loss            = d.get("loss",         1.0)
        self.epochs          = d.get("epochs",       0)
        self.per_gesture_acc = d.get("per_gesture_acc", {})
        self.trained         = bool(d.get("state_dict"))

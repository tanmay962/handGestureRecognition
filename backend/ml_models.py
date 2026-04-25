import copy
import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from database import DYNAMIC_FEATURES, DYNAMIC_FRAMES, score_sample_quality

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_VERSION = "1.0"


class FeatureScaler:
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


def augment_static(inputs, labels, n_copies=6, noise_std=0.04):
    aug_x = list(inputs)
    aug_y = list(labels)

    for vec, lbl in zip(inputs, labels):
        arr = np.array(vec, dtype=np.float32)
        for _ in range(n_copies):
            noisy = arr + np.random.normal(0, noise_std, arr.shape).astype(np.float32)
            scale = float(np.random.uniform(0.78, 1.22))
            for ci in list(range(5)) + list(range(11, 16)):
                if ci < len(noisy):
                    noisy[ci] = float(np.clip(noisy[ci] * scale, 0.0, 1.0))
            feat_mask = (np.random.random(noisy.shape) > 0.05).astype(np.float32)
            noisy = noisy * feat_mask
            aug_x.append(noisy.tolist())
            aug_y.append(lbl)

    return aug_x, aug_y


def augment_dynamic(inputs, labels, n_copies=3, noise_std=0.02):
    aug_x = list(inputs)
    aug_y = list(labels)

    for seq, lbl in zip(inputs, labels):
        arr    = np.array(seq, dtype=np.float32)
        frames = len(arr) // DYNAMIC_FEATURES
        if frames < 2:
            continue

        arr2d = arr[:frames * DYNAMIC_FEATURES].reshape(frames, DYNAMIC_FEATURES)

        for copy_i in range(n_copies):
            factor = float(np.random.uniform(0.75, 1.25))
            new_n  = max(2, int(frames * factor))
            idx_f  = np.linspace(0, frames - 1, new_n)

            stretched = np.zeros((new_n, DYNAMIC_FEATURES), dtype=np.float32)
            for fi, fi_f in enumerate(idx_f):
                lo = int(fi_f)
                hi = min(lo + 1, frames - 1)
                t  = fi_f - lo
                stretched[fi] = arr2d[lo] * (1.0 - t) + arr2d[hi] * t

            out    = np.zeros((DYNAMIC_FRAMES, DYNAMIC_FEATURES), dtype=np.float32)
            copy_n = min(DYNAMIC_FRAMES, new_n)

            # every other copy: phase-shift the gesture within the window
            if copy_i % 2 == 1 and copy_n < DYNAMIC_FRAMES:
                shift = np.random.randint(0, DYNAMIC_FRAMES - copy_n + 1)
                out[shift:shift + copy_n] = stretched[:copy_n]
            else:
                out[:copy_n] = stretched[:copy_n]

            out += np.random.normal(0, noise_std, out.shape).astype(np.float32)

            n_mask = np.random.randint(0, 4)
            if n_mask > 0:
                mask_idx = np.random.choice(DYNAMIC_FRAMES, n_mask, replace=False)
                out[mask_idx] = 0.0

            feat_mask = (np.random.random(out.shape) > 0.05).astype(np.float32)
            out = out * feat_mask

            aug_x.append(out.flatten().tolist())
            aug_y.append(lbl)

    return aug_x, aug_y


class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout=0.35, bidirectional=True):
        super().__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        nd  = 2 if bidirectional else 1
        dim = hidden_size * nd

        self.input_norm    = nn.LayerNorm(input_size)
        self.lstm          = nn.LSTM(input_size, hidden_size, num_layers,
                                     batch_first=True,
                                     dropout=dropout if num_layers > 1 else 0.0,
                                     bidirectional=bidirectional)
        # 4-head attention — each head learns different temporal patterns
        self.attn          = nn.MultiheadAttention(dim, num_heads=4,
                                                    dropout=dropout / 2,
                                                    batch_first=True)
        # Static path: MLP on mean-pooled frames (bypasses LSTM for static gestures)
        self.static_proj   = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Gate: learns to blend static vs dynamic path based on temporal variation
        # High std across frames → dynamic gesture → gate→1 (use LSTM)
        # Low std (tiled static) → static gesture → gate→0 (use MLP)
        self.gate_proj     = nn.Linear(1, 1)
        self.dropout       = nn.Dropout(dropout)
        self.norm          = nn.LayerNorm(dim)
        self.residual_proj = nn.Linear(input_size, dim)
        self.fc1           = nn.Linear(dim, dim // 2)
        self.act           = nn.GELU()
        self.fc2           = nn.Linear(dim // 2, output_size)

    def forward(self, x):
        x  = self.input_norm(x)
        nd = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * nd, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * nd, x.size(0), self.hidden_size).to(x.device)

        # Dynamic path: BiLSTM + multi-head attention pooling
        out, _       = self.lstm(x, (h0, c0))
        attn_out, _  = self.attn(out, out, out)
        dynamic_ctx  = attn_out.mean(dim=1)                          # (B, dim)

        # Static path: mean-pool frames directly through MLP
        static_ctx   = self.static_proj(x.mean(dim=1))               # (B, dim)

        # Gate from temporal variation: near-zero std → static, high std → dynamic
        variation    = x.std(dim=1).mean(dim=-1, keepdim=True)       # (B, 1)
        gate         = torch.sigmoid(self.gate_proj(variation))       # (B, 1)
        ctx          = gate * dynamic_ctx + (1.0 - gate) * static_ctx

        ctx = ctx + self.residual_proj(x.mean(dim=1))
        h   = self.norm(ctx)
        return self.fc2(self.dropout(self.act(self.fc1(self.dropout(h)))))


class UnifiedLSTMModel:
    # Handles both static (single frame tiled) and dynamic (sequence) gestures
    # in one BiLSTM. Static input: 41 features repeated 45x. Dynamic: 45x41.

    def __init__(self):
        self.model           = None
        self.gestures        = {}
        self.gesture_types   = {}
        self.feature_version = FEATURE_VERSION
        self.trained         = False
        self.accuracy        = 0.0
        self.val_accuracy    = 0.0
        self.loss            = 1.0
        self.epochs          = 0
        self.hidden_size     = 256
        self.num_layers      = 2
        self._output_size    = 0
        self._input_size     = DYNAMIC_FEATURES
        self.per_gesture_acc       = {}
        self.scaler                = FeatureScaler()
        # Post-training calibration
        self.temperature           = 1.0   # temperature scaling — >1 softens confidence
        self.per_gesture_threshold = {}    # auto-tuned per-class thresholds from val set
        self.confusion_matrix      = {}    # true→predicted counts on val set
        self.margin_threshold      = 0.10  # abstain if top1-top2 < this
        self.uncertainty_threshold = 0.025 # abstain if MC-dropout variance > this

    def initialize(self, num_classes, gesture_list, gesture_types=None):
        self._output_size = num_classes
        self.model = GestureLSTM(
            DYNAMIC_FEATURES, self.hidden_size, self.num_layers, num_classes,
            bidirectional=True,
        ).to(DEVICE)
        self.gestures      = {i: name for i, name in enumerate(gesture_list)}
        self.gesture_types = gesture_types or {name: 'static' for name in gesture_list}

    def _prepare_one(self, raw):
        arr = np.array(raw, dtype=np.float32)
        if len(arr) == DYNAMIC_FEATURES:
            return np.tile(arr, (DYNAMIC_FRAMES, 1))
        expected = DYNAMIC_FRAMES * DYNAMIC_FEATURES
        if len(arr) < expected:
            pad = np.zeros(expected, dtype=np.float32)
            pad[:len(arr)] = arr
            arr = pad
        return arr[:expected].reshape(DYNAMIC_FRAMES, DYNAMIC_FEATURES)

    def _prepare_batch(self, sequences):
        return np.array([self._prepare_one(s) for s in sequences], dtype=np.float32)

    def _scale_seq(self, seq_2d):
        if not self.scaler.fitted:
            return seq_2d
        return (seq_2d - self.scaler.mean_) / self.scaler.std_

    def _eval_val(self, vl_in, vl_lb):
        if not vl_in:
            return 0.0
        raw = self._prepare_batch(vl_in)
        sc  = np.array([self._scale_seq(s) for s in raw])
        Xv  = torch.tensor(sc,    dtype=torch.float32).to(DEVICE)
        yv  = torch.tensor(vl_lb, dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            return float((self.model(Xv).argmax(1) == yv).float().mean())

    # ── Post-training calibration helpers ───────────────────────────────────

    def _calibrate_temperature(self, vl_in, vl_lb):
        """Grid-search temperature T that minimises NLL on val set.
        T > 1 softens over-confident predictions; T < 1 sharpens under-confident ones."""
        if not vl_in or len(vl_in) < 5:
            self.temperature = 1.0
            return
        raw = self._prepare_batch(vl_in)
        sc  = np.array([self._scale_seq(s) for s in raw])
        Xv  = torch.tensor(sc, dtype=torch.float32).to(DEVICE)
        yv  = torch.tensor(vl_lb, dtype=torch.long).to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(Xv)
        best_nll, best_T = float('inf'), 1.0
        for T in [0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
            nll = nn.CrossEntropyLoss()(logits / T, yv).item()
            if nll < best_nll:
                best_nll = nll
                best_T   = T
        self.temperature = best_T

    def _auto_thresholds(self, vl_in, vl_lb):
        """Sweep confidence thresholds per gesture on val set and pick the one
        that maximises F1.  Stored in per_gesture_threshold."""
        if not vl_in or len(vl_in) < 5:
            return
        raw = self._prepare_batch(vl_in)
        sc  = np.array([self._scale_seq(s) for s in raw])
        Xv  = torch.tensor(sc, dtype=torch.float32).to(DEVICE)
        T   = self.temperature
        self.model.eval()
        with torch.no_grad():
            probs_np = torch.softmax(self.model(Xv) / T, dim=1).cpu().numpy()
        labels_np = np.array(vl_lb)
        self.per_gesture_threshold = {}
        for idx, name in self.gestures.items():
            true_mask = (labels_np == idx)
            if true_mask.sum() < 2:
                continue
            cls_conf = probs_np[:, idx]
            best_f1, best_t = -1.0, 0.60
            for t in np.arange(0.30, 0.86, 0.05):
                tp = int(((cls_conf >= t) &  true_mask).sum())
                fp = int(((cls_conf >= t) & ~true_mask).sum())
                fn = int(((cls_conf <  t) &  true_mask).sum())
                if tp == 0:
                    continue
                prec = tp / (tp + fp) if tp + fp > 0 else 0.0
                rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
                f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_t  = float(t)
            self.per_gesture_threshold[name] = round(best_t, 2)

    def _build_confusion_matrix(self, vl_in, vl_lb):
        """Compute true→predicted confusion counts on val set."""
        if not vl_in:
            return
        raw = self._prepare_batch(vl_in)
        sc  = np.array([self._scale_seq(s) for s in raw])
        Xv  = torch.tensor(sc, dtype=torch.float32).to(DEVICE)
        yv  = torch.tensor(vl_lb, dtype=torch.long).to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(Xv).argmax(dim=1)
        self.confusion_matrix = {}
        for true_idx in range(len(self.gestures)):
            true_name = self.gestures.get(true_idx)
            if not true_name:
                continue
            mask = (yv == true_idx)
            if mask.sum() == 0:
                continue
            row = {}
            for pred_idx in range(len(self.gestures)):
                count = int((preds[mask] == pred_idx).sum().item())
                if count > 0:
                    row[self.gestures[pred_idx]] = count
            self.confusion_matrix[true_name] = row

    def train_batch(self, inputs, labels, epochs=30, lr=0.001, progress_cb=None):
        if not inputs or self.model is None:
            return

        n = len(inputs)

        if n >= 5:
            by_class = defaultdict(list)
            for i, lbl in enumerate(labels):
                by_class[lbl].append(i)
            train_idx, val_idx = [], []
            for lbl, idxs in sorted(by_class.items()):
                random.shuffle(idxs)
                n_val = max(1, int(len(idxs) * 0.2)) if len(idxs) >= 3 else 0
                val_idx.extend(idxs[:n_val])
                train_idx.extend(idxs[n_val:])
            random.shuffle(train_idx)
            random.shuffle(val_idx)
        else:
            train_idx = list(range(n))
            val_idx   = []

        tr_in = [inputs[i] for i in train_idx]
        tr_lb = [labels[i] for i in train_idx]
        vl_in = [inputs[i] for i in val_idx]
        vl_lb = [labels[i] for i in val_idx]

        s_in, s_lb, d_in, d_lb = [], [], [], []
        for inp, lbl in zip(tr_in, tr_lb):
            if len(inp) == DYNAMIC_FEATURES:
                s_in.append(inp); s_lb.append(lbl)
            else:
                d_in.append(inp); d_lb.append(lbl)

        # Quality gate — drop clearly bad samples (all-zero hand, clipped features)
        # Safety: only apply if no class loses all its samples
        def _quality_filter(inputs, labels, is_dynamic=False):
            pairs = [(i, l) for i, l in zip(inputs, labels)
                     if score_sample_quality(i[:DYNAMIC_FEATURES] if is_dynamic else i) >= 0.25]
            if not pairs:
                return inputs, labels
            if {l for _, l in pairs} != set(labels):
                return inputs, labels  # would drop a class — keep all
            return [p[0] for p in pairs], [p[1] for p in pairs]

        s_in, s_lb = _quality_filter(s_in, s_lb)
        d_in, d_lb = _quality_filter(d_in, d_lb, is_dynamic=True)

        # Fit scaler on original data before augmentation so the distribution
        # isn't skewed by the artificially inflated augmented set
        orig_raw   = self._prepare_batch(s_in + d_in)
        all_frames = orig_raw.reshape(-1, DYNAMIC_FEATURES)
        self.scaler.fit(all_frames)

        if len(s_in) >= 4:
            s_in, s_lb = augment_static(s_in, s_lb)
        if len(d_in) >= 3:
            d_in, d_lb = augment_dynamic(d_in, d_lb)

        tr_in = s_in + d_in
        tr_lb = s_lb + d_lb

        tr_raw = self._prepare_batch(tr_in)
        tr_sc  = np.array([self._scale_seq(s) for s in tr_raw])
        X = torch.tensor(tr_sc, dtype=torch.float32).to(DEVICE)
        y = torch.tensor(tr_lb, dtype=torch.long).to(DEVICE)

        try:
            counts = torch.bincount(y, minlength=len(self.gestures)).float()
            w      = 1.0 / (counts + 1e-6)
            w      = w / w.sum() * len(self.gestures)
            crit   = nn.CrossEntropyLoss(weight=w.to(DEVICE), label_smoothing=0.1)
        except Exception:
            crit = nn.CrossEntropyLoss(label_smoothing=0.1)

        opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)

        # Linear warmup for first ~12% of epochs, cosine decay for the rest
        warmup_ep = max(3, epochs // 8)
        def lr_lambda(ep):
            if ep < warmup_ep:
                return (ep + 1) / warmup_ep
            progress = (ep - warmup_ep) / max(1, epochs - warmup_ep)
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        BATCH_SIZE  = min(32, len(tr_in))
        n_train     = len(tr_in)
        # Validation + weight-rebalance interval (every 10% of epochs)
        check_every = max(1, epochs // 10)
        # Lightweight progress-only update interval (every ~2% of epochs so bar moves smoothly)
        prog_every  = max(1, epochs // 50)
        swa_start   = int(epochs * 0.6)   # collect SWA snapshots from 60% onward

        best_val_acc      = -1.0
        best_state        = None
        swa_states        = []
        last_ep_loss      = 1.0
        last_v_acc        = None
        no_improve_checks = 0
        patience_checks   = 3   # stop after 3 consecutive val checks with no improvement

        self.model.train()
        for ep in range(epochs):
            ep_perm     = torch.randperm(n_train)
            ep_loss_sum = 0.0
            n_steps     = 0

            for start in range(0, n_train, BATCH_SIZE):
                idx_b = ep_perm[start:start + BATCH_SIZE]
                xb    = X[idx_b]
                yb    = y[idx_b]
                opt.zero_grad()

                if len(xb) >= 2 and torch.rand(1).item() < 0.6:
                    lam   = float(np.random.beta(0.4, 0.4))
                    idx_m = torch.randperm(len(xb), device=DEVICE)
                    xb_m  = lam * xb + (1 - lam) * xb[idx_m]
                    logit = self.model(xb_m)
                    loss  = lam * crit(logit, yb) + (1 - lam) * crit(logit, yb[idx_m])
                else:
                    loss = crit(self.model(xb), yb)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()
                ep_loss_sum += loss.item()
                n_steps     += 1

            sched.step()
            last_ep_loss = ep_loss_sum / max(1, n_steps)

            # Collect SWA snapshot
            if ep >= swa_start:
                swa_states.append({k: v.clone().cpu() for k, v in self.model.state_dict().items()})

            # ── Heavy validation check (every check_every epochs) ───────────
            if (ep + 1) % check_every == 0:
                if vl_in:
                    self.model.eval()
                    last_v_acc = self._eval_val(vl_in, vl_lb)

                    # Confusion-aware dynamic weights — only in second half of training
                    if ep >= epochs // 2:
                        vl_r_ = self._prepare_batch(vl_in)
                        vl_s_ = np.array([self._scale_seq(s) for s in vl_r_])
                        Xv_   = torch.tensor(vl_s_, dtype=torch.float32).to(DEVICE)
                        yv_   = torch.tensor(vl_lb,  dtype=torch.long).to(DEVICE)
                        with torch.no_grad():
                            vp_ = self.model(Xv_).argmax(dim=1)
                        new_w = w.clone()
                        for ci in range(len(self.gestures)):
                            m_ = (yv_ == ci)
                            if m_.sum() > 0:
                                cls_acc = (vp_[m_] == yv_[m_]).float().mean().item()
                                new_w[ci] = w[ci] * (1.0 + max(0.0, 0.8 - cls_acc))
                        new_w = new_w / new_w.sum() * len(self.gestures)
                        crit  = nn.CrossEntropyLoss(weight=new_w.to(DEVICE), label_smoothing=0.1)

                    if last_v_acc >= best_val_acc:
                        best_val_acc      = last_v_acc
                        best_state        = copy.deepcopy(self.model.state_dict())
                        no_improve_checks = 0
                    else:
                        no_improve_checks += 1
                    self.model.train()

                if vl_in and no_improve_checks >= patience_checks:
                    # Send final progress before early-stop break
                    if progress_cb:
                        progress_cb({
                            "epoch":        ep + 1,
                            "total_epochs": epochs,
                            "progress_pct": round((ep + 1) / epochs * 100),
                            "train_loss":   round(last_ep_loss, 4),
                            "val_acc":      round(last_v_acc, 4) if last_v_acc is not None else None,
                        })
                    break

            # ── Lightweight progress callback (every prog_every epochs) ─────
            # Use elif to prevent double-firing when check_every and prog_every land on same epoch (BUG #9 fix)
            elif progress_cb and (ep + 1) % prog_every == 0:
                progress_cb({
                    "epoch":        ep + 1,
                    "total_epochs": epochs,
                    "progress_pct": round((ep + 1) / epochs * 100),
                    "train_loss":   round(last_ep_loss, 4),
                    "val_acc":      round(last_v_acc, 4) if last_v_acc is not None else None,
                })

        # Build SWA-averaged weights and compare against best-val checkpoint
        if swa_states:
            swa_avg = {
                k: torch.stack([s[k].float() for s in swa_states]).mean(0).to(DEVICE)
                for k in swa_states[0]
            }
            if vl_in and best_state is not None:
                self.model.load_state_dict(swa_avg)
                self.model.eval()
                swa_acc = self._eval_val(vl_in, vl_lb)
                if best_val_acc >= swa_acc:
                    self.model.load_state_dict(best_state)
                # else keep swa_avg (already loaded)
            else:
                self.model.load_state_dict(swa_avg)
        elif best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        with torch.no_grad():
            out   = self.model(X)
            preds = out.argmax(dim=1)
            self.accuracy = (preds == y).float().mean().item()
            self.loss     = crit(out, y).item()

            self.val_accuracy = 0.0
            if vl_in:
                self.val_accuracy = self._eval_val(vl_in, vl_lb)

            # Per-gesture accuracy on val set (honest) or training as fallback
            self.per_gesture_acc = {}
            if vl_in:
                vl_raw = self._prepare_batch(vl_in)
                vl_sc  = np.array([self._scale_seq(s) for s in vl_raw])
                Xv     = torch.tensor(vl_sc, dtype=torch.float32).to(DEVICE)
                yv     = torch.tensor(vl_lb, dtype=torch.long).to(DEVICE)
                val_p  = self.model(Xv).argmax(dim=1)
                for idx, name in self.gestures.items():
                    mask = (yv == idx)
                    if mask.sum() > 0:
                        self.per_gesture_acc[name] = (val_p[mask] == yv[mask]).float().mean().item()
            else:
                for idx, name in self.gestures.items():
                    mask = (y == idx)
                    if mask.sum() > 0:
                        self.per_gesture_acc[name] = (preds[mask] == y[mask]).float().mean().item()

        # Post-training calibration (all three use val set; safe to call when vl_in is empty)
        self._calibrate_temperature(vl_in, vl_lb)
        self._auto_thresholds(vl_in, vl_lb)
        self._build_confusion_matrix(vl_in, vl_lb)

        self.epochs  += epochs
        self.trained  = True

    def predict(self, features, conf_threshold=0.0):
        if not self.trained or self.model is None:
            return {"idx": -1, "conf": 0.0, "probs": [], "name": "Unknown",
                    "below_threshold": False, "margin": 0.0, "uncertainty": 0.0}

        seq = self._scale_seq(self._prepare_one(features))
        X   = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(DEVICE)
        T   = self.temperature  # calibrated temperature (1.0 = uncalibrated)

        all_probs = []
        with torch.no_grad():
            # Pass 1: clean, eval mode (no dropout — deterministic baseline)
            self.model.eval()
            all_probs.append(torch.softmax(self.model(X) / T, dim=1)[0])

            # Passes 2-5: MC Dropout — train mode (dropout active) + small noise
            # High variance across these passes = model is genuinely uncertain
            self.model.train()
            for _ in range(4):
                noisy = seq + np.random.normal(0, 0.008, seq.shape).astype(np.float32)
                Xn = torch.tensor(noisy[np.newaxis], dtype=torch.float32).to(DEVICE)
                all_probs.append(torch.softmax(self.model(Xn) / T, dim=1)[0])
            self.model.eval()

        stacked     = torch.stack(all_probs)           # (5, C)
        probs       = stacked.mean(0)                   # mean probability
        uncertainty = float(stacked.var(0).mean())      # mean variance = prediction instability

        idx  = int(probs.argmax().item())
        conf = float(probs[idx].item())

        # Margin gate: when top-2 classes are too close, the model is unsure
        sorted_p = sorted(probs.tolist(), reverse=True)
        margin   = sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0)

        below = (
            conf < conf_threshold
            or margin    < self.margin_threshold
            or uncertainty > self.uncertainty_threshold
        )

        return {
            "idx":             idx,
            "conf":            conf,
            "probs":           probs.tolist(),
            "name":            self.gestures.get(idx, "Unknown"),
            "below_threshold": below,
            "margin":          round(margin, 4),
            "uncertainty":     round(uncertainty, 6),
        }

    def add_gesture(self, name, idx):
        self.gestures[idx] = name

    def reset(self):
        self.model                 = None
        self.trained               = False
        self.gestures              = {}
        self.gesture_types         = {}
        self.accuracy              = 0.0
        self.val_accuracy          = 0.0
        self.loss                  = 1.0
        self.epochs                = 0
        self.per_gesture_acc       = {}
        self.scaler                = FeatureScaler()
        self.temperature           = 1.0
        self.per_gesture_threshold = {}
        self.confusion_matrix      = {}

    def to_json(self):
        state = {k: v.tolist() for k, v in self.model.state_dict().items()} if self.model else {}
        return {
            "type":            "unified_bilstm",
            "state_dict":      state,
            "gestures":        list(self.gestures.items()),
            "gesture_types":   self.gesture_types,
            "accuracy":        self.accuracy,
            "val_accuracy":    self.val_accuracy,
            "loss":            self.loss,
            "epochs":          self.epochs,
            "hidden_size":     self.hidden_size,
            "num_layers":      self.num_layers,
            "output_size":     self._output_size,
            "per_gesture_acc":       self.per_gesture_acc,
            "feature_version":       self.feature_version,
            "scaler":                self.scaler.to_dict(),
            "temperature":           self.temperature,
            "per_gesture_threshold": self.per_gesture_threshold,
            "confusion_matrix":      self.confusion_matrix,
        }

    def from_json(self, d):
        gesture_list = [v for _, v in sorted(d.get("gestures", []), key=lambda x: int(x[0]))]
        if not gesture_list:
            return

        self.hidden_size     = d.get("hidden_size",     256)
        self.num_layers      = d.get("num_layers",      2)
        self._output_size    = d.get("output_size",     len(gesture_list))
        self.feature_version = d.get("feature_version", FEATURE_VERSION)
        self.initialize(self._output_size, gesture_list, d.get("gesture_types", {}))

        self.trained = False
        if d.get("state_dict"):
            try:
                sd = {k: torch.tensor(v) for k, v in d["state_dict"].items()}
                self.model.load_state_dict(sd)
                self.model.to(DEVICE)
                self.trained = True
            except Exception as e:
                print(f"[UnifiedBiLSTM] retrain needed: {e}")

        self.gestures              = {int(k): v for k, v in d.get("gestures", [])}
        self.accuracy              = d.get("accuracy",              0.0)
        self.val_accuracy          = d.get("val_accuracy",          0.0)
        self.loss                  = d.get("loss",                  1.0)
        self.epochs                = d.get("epochs",                0)
        self.per_gesture_acc       = d.get("per_gesture_acc",       {})
        self.temperature           = d.get("temperature",           1.0)
        self.per_gesture_threshold = d.get("per_gesture_threshold", {})
        self.confusion_matrix      = d.get("confusion_matrix",      {})
        if d.get("scaler"):
            self.scaler.from_dict(d["scaler"])

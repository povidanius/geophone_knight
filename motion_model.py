"""Small, low-latency geophone motion classifier shared by training and game."""

from collections import deque
import time

import numpy as np

CLASSES = ("still", "walk")
MODEL_SAMPLES = 128
WINDOW_SECONDS = 2.50


def prepare_window(samples, fs, n=MODEL_SAMPLES, window_s=WINDOW_SECONDS):
    """Resample the newest causal window and make amplitude-robust features."""
    needed = max(8, int(round(fs * window_s)))
    x = np.asarray(samples[-needed:], dtype=np.float32)
    if x.size < needed:
        x = np.pad(x, (needed - x.size, 0))
    x = x - np.median(x)
    old = np.linspace(0.0, 1.0, x.size)
    x = np.interp(np.linspace(0.0, 1.0, n), old, x).astype(np.float32)
    rms = float(np.sqrt(np.mean(x * x) + 1e-8))
    scale = float(np.percentile(np.abs(x), 90)) + 1e-4
    raw = np.clip(x / scale, -5.0, 5.0)
    envelope = np.abs(raw)
    # Absolute signal level remains useful for separating true stillness.
    level = np.full_like(raw, np.clip(np.log1p(rms) / 5.0, 0.0, 2.0))
    return np.stack((raw, envelope, level)).astype(np.float32)


def build_network():
    import torch
    import torch.nn as nn

    class MotionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(3, 32, 9, stride=2, padding=4),
                nn.BatchNorm1d(32), nn.ReLU(inplace=True),
                nn.Conv1d(32, 64, 7, stride=2, padding=3),
                nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                nn.Conv1d(64, 96, 5, stride=2, padding=2),
                nn.BatchNorm1d(96), nn.ReLU(inplace=True),
                nn.Conv1d(96, 128, 3, stride=2, padding=1),
                nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            )
            # Combine sustained activity with brief transient information.
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)
            self.classifier = nn.Sequential(
                nn.Linear(256, 96),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.15),
                nn.Linear(96, len(CLASSES)),
            )

        def forward(self, x):
            x = self.features(x)
            pooled = torch.cat((self.avg_pool(x), self.max_pool(x)), dim=1)
            return self.classifier(pooled.squeeze(-1))

    return MotionNet()


class MotionClassifier:
    """Streaming still/walk inference with a short probability EMA."""

    def __init__(self, model_path, fs_hint=100.0, smooth=0.35):
        import torch

        torch.set_num_threads(1)
        checkpoint = torch.load(model_path, map_location="cpu")
        self.model = build_network()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.window_s = float(checkpoint.get("window_seconds", WINDOW_SECONDS))
        self.n = int(checkpoint.get("model_samples", MODEL_SAMPLES))
        # Covers the ADS1115 maximum rate (860 SPS) even when the initial rate
        # hint is the conservative 100 Hz default.
        self.buffer = deque(maxlen=max(4096, int(fs_hint * 3)))
        self.probs = np.zeros(len(CLASSES), dtype=np.float32)
        self.probs[CLASSES.index("still")] = 1.0
        self.smooth = smooth
        self._last_t = None
        self.label = "still"
        self.features = {"confidence": 1.0, "probabilities": self.probs}

    def add(self, samples):
        self.buffer.extend(samples)

    def update(self, fs):
        import torch

        needed = max(8, int(round(fs * self.window_s)))
        if len(self.buffer) < needed:
            return self.label
        feat = prepare_window(list(self.buffer), fs, self.n, self.window_s)
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(feat).unsqueeze(0))
            current = torch.softmax(logits, dim=1)[0].numpy()
        now = time.monotonic()
        dt = 0.033 if self._last_t is None else min(0.2, now - self._last_t)
        self._last_t = now
        alpha = 1.0 - np.exp(-dt / max(0.001, self.smooth))
        self.probs += alpha * (current - self.probs)
        self.label = CLASSES[int(np.argmax(self.probs))]
        self.features = {
            "confidence": float(np.max(self.probs)),
            "probabilities": self.probs.copy(),
        }
        return self.label

#!/usr/bin/env python3
"""
Smart Geophone — diagnostics / live scope.

A standalone tool (no game) to test the two detectors on the live geophone
stream and to *see* the raw signal:

* a scrolling **raw-waveform scope** (auto-scaled, with jump markers);
* the **still/walk classifier** output (probability bars + label);
* the **jump onset detector** (fires, count, and its internal levels — the
  step-peak follower, the walking-activity gate and the fire threshold — so you
  can see *why* a jump does or does not trigger).

It reuses exactly the same classes as the game (`MotionClassifier`,
`OnsetDetector`), so what you see here is what the game sees.

    ./diagnostics.py                 # auto-detect serial port
    ./diagnostics.py --simulate      # no hardware: synthetic walker
    #   press [j] to inject a synthetic jump (handy with --simulate)

Keys:  [j] inject test jump   [Up/Down] jump threshold   [c] clear count
       [q]/[Esc] quit
"""

import argparse
from collections import deque
import math
from pathlib import Path
import sys
import time
import tkinter as tk

import numpy as np

from geophone_io import (OnsetDetector, SerialSource, SimulatedSource,
                         auto_detect_port)

try:
    from motion_model import MotionClassifier
except Exception:                        # torch missing etc.
    MotionClassifier = None


# -- colours -----------------------------------------------------------------
BG = "#10151c"
GRID = "#20303f"
INK = "#dce6f0"
DIM = "#7a8aa0"
WAVE = "#4fd1ff"
ZERO = "#33465c"
STILL_C = "#8a94a6"
WALK_C = "#2ec96b"
JUMP_C = "#ff5a3c"
JUMP_HOT = "#ffd23f"
PEAK_C = "#f4c430"
ACT_C = "#9b6cff"
FIRE_C = "#ff5a3c"
ABS_C = "#ff9f1c"


class Diagnostics:
    W = 940
    WAVE_H = 250
    METER_H = 210
    FPS = 30
    DISPLAY_S = 6.0                      # seconds of signal shown in the scope

    def __init__(self, root, source, classifier, onset, fs, label,
                 model_path=None):
        self.root = root
        self.source = source
        self.classifier = classifier
        self.onset = onset
        self.fs = fs
        self.meas_fs = fs
        self.src_label = label
        self.model_path = model_path

        self.wave = tk.Canvas(root, width=self.W, height=self.WAVE_H,
                              bg=BG, highlightthickness=0)
        self.wave.pack(fill="both", expand=True)
        self.meter = tk.Canvas(root, width=self.W, height=self.METER_H,
                               bg=BG, highlightthickness=0)
        self.meter.pack(fill="both", expand=True)

        self.buf = deque(maxlen=9000)    # raw samples for the scope
        self.total = 0                   # total samples ever seen
        self.jump_marks = deque()        # global sample index of each jump

        self.label = "still"
        self.probs = (1.0, 0.0)
        self.confidence = 1.0

        self.jumps = 0
        self.last_strength = 0.0
        self._flash_until = 0.0
        self.dev_now = 0.0

        self._inject = []                # queued synthetic-jump samples
        self._dc = 0.0                   # clean DC estimate (real samples only)
        self._step_ref = 0.0             # jump-resistant step-peak estimate
        self._samp_count = 0
        self._samp_t0 = None
        self._last = time.time()

        root.bind("<Key>", self._on_key)

    # -- input ---------------------------------------------------------------
    def _on_key(self, e):
        k = (e.keysym or "").lower()
        if k in ("q", "escape"):
            self.root.destroy()
        elif k == "c":
            self.jumps = 0
            self.jump_marks.clear()
        elif k == "j":
            self._queue_test_jump()
        elif k in ("up", "plus", "equal"):
            self.onset.threshold = round(self.onset.threshold + 0.5, 2)
        elif k in ("down", "minus"):
            self.onset.threshold = max(0.5, round(self.onset.threshold - 0.5, 2))

    def _queue_test_jump(self):
        """Enqueue a damped burst shaped like a *realistic* jump take-off: ~3.5×
        your recent step-peak. It fires only if that clears the fire line, so
        the tool honestly shows whether the current threshold is set well."""
        fs = max(20.0, self.meas_fs)
        n = int(0.18 * fs)
        burst = [math.exp(-i / (0.05 * fs)) * math.sin(2 * math.pi * 12 * i / fs)
                 for i in range(n)]
        peak = max((abs(b) for b in burst), default=1.0) or 1.0
        ref = max(self._step_ref, self.onset.min_level, self.onset.abs_level, 8.0)
        target = 3.5 * ref                              # a plausible jump
        self._inject.extend(b * (target / peak) for b in burst)

    # -- main loop -----------------------------------------------------------
    def tick(self):
        now = time.time()
        self._last = now
        real = self.source.drain()
        # A clean, jump-resistant step-peak estimate from the *real* stream only
        # (never the synthetic injects), used to size a realistic test jump.
        for v in real:
            self._dc += 0.02 * (v - self._dc)
            dev = abs(v - self._dc)
            self._step_ref = dev if dev > self._step_ref else self._step_ref * 0.999
        samples = list(real)
        if self._inject:
            samples = samples + self._inject
            self._inject = []

        if samples:
            if self._samp_t0 is None:
                self._samp_t0 = now
            self._samp_count += len(samples)
            el = now - self._samp_t0
            if el > 1.5:                 # rolling sample-rate estimate
                self.meas_fs = self._samp_count / el
                self._samp_count, self._samp_t0 = 0, now

            # still/walk classifier
            if self.classifier is not None:
                self.classifier.add(samples)
                self.label = self.classifier.update(self.meas_fs)
                f = self.classifier.features
                self.probs = tuple(float(p) for p in f["probabilities"])
                self.confidence = float(f["confidence"])

            # jump onset detector
            fired = self.onset.process(samples, self.meas_fs)
            self.dev_now = max(abs(v - self.onset._center) for v in samples)
            if fired:
                self.jumps += 1
                self.last_strength = self.onset.last_strength
                self._flash_until = now + 0.35
                self.jump_marks.append(self.total + len(samples))

            self.buf.extend(samples)
            self.total += len(samples)

        self._draw_wave()
        self._draw_meter()
        self.root.after(int(1000 / self.FPS), self.tick)

    # -- drawing -------------------------------------------------------------
    def _draw_wave(self):
        c = self.wave
        c.delete("all")
        W, H = self.W, self.WAVE_H
        mid = H * 0.55
        c.create_line(0, mid, W, mid, fill=ZERO)

        show_n = min(len(self.buf), int(self.DISPLAY_S * max(1.0, self.meas_fs)))
        title = "Smart Geophone — Diagnostics   ·   %s   ·   %.0f samples/s" % (
            self.src_label, self.meas_fs)
        c.create_text(12, 14, anchor="w", text=title, fill=INK,
                      font=("TkDefaultFont", 12, "bold"))
        if show_n < 2:
            c.create_text(W / 2, mid, text="waiting for samples…", fill=DIM,
                          font=("TkDefaultFont", 13))
            return
        data = np.fromiter((self.buf[i] for i in range(len(self.buf) - show_n,
                                                        len(self.buf))),
                           dtype=np.float64, count=show_n)
        base = float(np.median(data))
        # Robust scale (98th percentile, not max) so one big jump spike does not
        # crush the rest of the trace; the spike simply clips at the edges.
        amax = max(1e-3, float(np.percentile(np.abs(data - base), 98)))
        scale = (H * 0.42) / amax
        lim = H * 0.46
        start_global = self.total - show_n

        c.create_text(W - 10, 14, anchor="e",
                      text="±%.1f mV/div  (%.1fs window)" % (amax, self.DISPLAY_S),
                      fill=DIM, font=("TkDefaultFont", 9))

        # waveform, decimated to one point per pixel column (y clamped to canvas)
        idx = (np.arange(W) * (show_n - 1) / (W - 1)).astype(int)
        ys = np.clip((data[idx] - base) * scale, -lim, lim)
        pts = []
        for x in range(W):
            pts.append(x)
            pts.append(mid - float(ys[x]))
        c.create_line(*pts, fill=WAVE, width=1)

        # jump markers scrolling with the trace
        while self.jump_marks and self.jump_marks[0] < start_global:
            self.jump_marks.popleft()
        for g in self.jump_marks:
            x = (g - start_global) / show_n * W
            c.create_line(x, 20, x, H, fill=JUMP_C, width=2)
            c.create_text(x, 28, text="jump", fill=JUMP_C,
                          font=("TkDefaultFont", 8), anchor="n")

    def _bar(self, c, x, y, w, h, frac, col, bg=GRID):
        frac = max(0.0, min(1.0, frac))
        c.create_rectangle(x, y, x + w, y + h, fill=bg, outline="")
        c.create_rectangle(x, y, x + w * frac, y + h, fill=col, outline="")

    def _draw_meter(self):
        c = self.meter
        c.delete("all")
        W = self.W
        now = time.time()

        # ---- still / walk classifier ----
        c.create_text(12, 16, anchor="w", text="STILL / WALK classifier",
                      fill=INK, font=("TkDefaultFont", 11, "bold"))
        if self.classifier is None:
            c.create_text(12, 40, anchor="w",
                          text="model not loaded (%s)" % (self.model_path or "—"),
                          fill=JUMP_C, font=("TkDefaultFont", 10))
        else:
            walking = self.label == "walk"
            still_p, walk_p = self.probs
            c.create_text(300, 16, anchor="w",
                          text=("WALKING" if walking else "STANDING"),
                          fill=(WALK_C if walking else STILL_C),
                          font=("TkDefaultFont", 12, "bold"))
            self._bar(c, 60, 34, 360, 16, still_p, STILL_C)
            c.create_text(12, 42, anchor="w", text="still", fill=DIM,
                          font=("TkDefaultFont", 9))
            c.create_text(430, 42, anchor="w", text="%.0f%%" % (100 * still_p),
                          fill=INK, font=("TkDefaultFont", 9))
            self._bar(c, 60, 56, 360, 16, walk_p, WALK_C)
            c.create_text(12, 64, anchor="w", text="walk", fill=DIM,
                          font=("TkDefaultFont", 9))
            c.create_text(430, 64, anchor="w", text="%.0f%%" % (100 * walk_p),
                          fill=INK, font=("TkDefaultFont", 9))
            c.create_text(60, 86, anchor="w",
                          text="confidence %.0f%%" % (100 * self.confidence),
                          fill=DIM, font=("TkDefaultFont", 9))

        # ---- jump onset detector ----
        y0 = 116
        c.create_text(12, y0, anchor="w", text="JUMP onset detector", fill=INK,
                      font=("TkDefaultFont", 11, "bold"))
        hot = now < self._flash_until
        c.create_rectangle(300, y0 - 12, 430, y0 + 12,
                           fill=(JUMP_HOT if hot else GRID), outline=JUMP_C,
                           width=2)
        c.create_text(365, y0, text=("JUMP!" if hot else "watching"),
                      fill=(BG if hot else DIM),
                      font=("TkDefaultFont", 11, "bold"))
        c.create_text(450, y0, anchor="w",
                      text="count %d   ·   last strength %.1f×" % (
                          self.jumps, self.last_strength),
                      fill=INK, font=("TkDefaultFont", 10))

        # level bars: scale everything against a common reference
        d = self.onset
        ref = max(self.dev_now, d._peak, d.min_level, d.abs_level,
                  d.threshold * d._peak, 1e-3) * 1.15
        established = d._activity >= d.min_level
        rows = [
            ("transient", self.dev_now, WAVE),
            ("step-peak", d._peak, PEAK_C),
            ("fire line", d.threshold * d._peak, FIRE_C),
        ]
        yy = y0 + 22
        for name, val, col in rows:
            c.create_text(12, yy + 7, anchor="w", text=name, fill=DIM,
                          font=("TkDefaultFont", 9))
            self._bar(c, 90, yy, 470, 14, val / ref, col)
            c.create_text(568, yy + 7, anchor="w", text="%.1f" % val, fill=INK,
                          font=("TkDefaultFont", 9))
            yy += 20
        # walking-activity gate
        c.create_text(12, yy + 7, anchor="w", text="activity", fill=DIM,
                      font=("TkDefaultFont", 9))
        self._bar(c, 90, yy, 470, 14, d._activity / ref, ACT_C)
        gx = 90 + 470 * min(1.0, d.min_level / ref)
        c.create_line(gx, yy - 2, gx, yy + 16, fill=INK, width=2)     # min_level
        c.create_text(568, yy + 7, anchor="w",
                      text="%.1f  gate:%s" % (
                          d._activity, "WALKING" if established else "idle"),
                      fill=(WALK_C if established else DIM),
                      font=("TkDefaultFont", 9, "bold"))
        yy += 24

        thr = "threshold %.1f×" % d.threshold
        if d.abs_level > 0:
            thr += "   ·   abs_level %.1f (trained)" % d.abs_level
        c.create_text(12, yy + 6, anchor="w", text=thr, fill=INK,
                      font=("TkDefaultFont", 9, "bold"))
        c.create_text(W - 12, self.METER_H - 10, anchor="e",
                      text="[j] test jump   [↑/↓] threshold   [c] clear   [q] quit",
                      fill=DIM, font=("TkDefaultFont", 9))


# ---------------------------------------------------------------------------
def build_source(args):
    fs_hint = args.fs if args.fs else 100.0
    if args.simulate:
        return SimulatedSource(fs=fs_hint), fs_hint, "SIMULATED"
    port = args.port or auto_detect_port()
    if port is None:
        sys.exit("No serial port found. Plug in the geophone, pass --port, or "
                 "use --simulate.")
    src = SerialSource(port, args.baud)
    try:
        src.open()
    except Exception as exc:
        sys.exit("Could not open %s: %s" % (port, exc))
    return src, fs_hint, port


def main():
    ap = argparse.ArgumentParser(
        description="Live diagnostics for the geophone still/walk + jump detectors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0", help="Geophone serial port.")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--fs", type=float, default=None, help="Sample-rate hint (Hz).")
    ap.add_argument("--simulate", action="store_true",
                    help="Use the synthetic walker instead of the serial port.")
    ap.add_argument("--model",
                    default=str(Path(__file__).with_name("geophone_model.pt")),
                    help="still/walk model from learn_model.py.")
    ap.add_argument("--smoothing", type=float, default=0.12,
                    help="Probability smoothing time (s); lower is faster.")
    ap.add_argument("--jump-threshold", type=float, default=6.0,
                    help="Jump onset threshold (× recent step peak).")
    ap.add_argument("--jump-refractory", type=float, default=0.6)
    ap.add_argument("--jump-model", default=None,
                    help="Optional trained jump model (jump_model.json).")
    args = ap.parse_args()

    source, fs, label = build_source(args)

    classifier = None
    if MotionClassifier is not None:
        try:
            classifier = MotionClassifier(args.model, fs, smooth=args.smoothing)
        except Exception as exc:
            sys.stderr.write("still/walk model not loaded: %s\n" % exc)

    onset = None
    if args.jump_model:
        onset = OnsetDetector.from_model(args.jump_model, fs,
                                         refractory=args.jump_refractory)
    if onset is None:
        onset = OnsetDetector(fs, threshold=args.jump_threshold,
                              refractory=args.jump_refractory)

    source.start()
    root = tk.Tk()
    root.title("Smart Geophone — Diagnostics  (%s)" % label)
    app = Diagnostics(root, source, classifier, onset, fs, label,
                      model_path=args.model)
    app.tick()
    try:
        root.mainloop()
    finally:
        source.stop()


if __name__ == "__main__":
    main()

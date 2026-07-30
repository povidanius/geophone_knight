#!/usr/bin/env python3
"""Train the adaptive jump detector from your own footsteps and jumps.

The jump detector (``geophone_io.OnsetDetector``) is not a CNN -- it is a small,
self-calibrating threshold model. This script records a short WALK phase and a
JUMP phase from the geophone, measures how much bigger your jumps are than your
walking steps, and writes the learned levels to ``jump_model.json``. This model
is **opt-in**: the game uses its responsive built-in detector by default and only
uses the trained model when you pass ``--jump-model jump_model.json``
(``OnsetDetector.from_model``).

    ./collect_jumps.py                 # guided walk + jump capture, then fit
    ./collect_jumps.py --from-data     # refit from saved sessions (no hardware)

The raw WALK/JUMP recordings are also saved into ``training_data/`` (labels
``walk`` / ``jump``) so you can refit later with ``--from-data``.
"""

import argparse
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np

from geophone_io import SerialSource, auto_detect_port


def _dev(x):
    """Absolute deviation of the signal from its DC centre (the median)."""
    x = np.asarray(x, dtype=np.float64)
    return np.abs(x - np.median(x))


def fit_jump_model(walk, jump, fs):
    """Return adaptive ``OnsetDetector`` params from walk-step and jump samples.

    * ``walk_peak`` -- a strong footstep amplitude (92nd pct of |deviation|).
    * ``jump_peak`` -- a jump transient amplitude (96th pct in the jump phase).
    * ``abs_level`` -- an absolute floor set *between* the two, so an ordinary
      (or cold first) step can never reach it but a jump does.
    * ``threshold`` -- how many recent step-peaks a jump must exceed.
    * ``min_level`` -- the walking-activity floor for the sustained-activity gate.
    """
    dw, dj = _dev(walk), _dev(jump)
    if dw.size < 32 or dj.size < 32:
        raise ValueError("need more walk and jump samples to fit a model")
    walk_peak = float(np.percentile(dw, 92)) + 1e-6      # strong footstep
    walk_activity = float(np.mean(dw))                   # sustained walk level
    jump_peak = float(np.percentile(dj, 96))             # jump transient
    ratio = jump_peak / walk_peak
    # Absolute floor: above ordinary steps, below the jump transient. Kept
    # deliberately low (only just above steps) so real jumps trigger easily.
    abs_level = min(walk_peak * 1.2, jump_peak * 0.5)
    abs_level = float(max(abs_level, walk_peak * 1.1))
    # Relative multiplier: a jump must exceed this many recent step-peaks.
    threshold = float(min(6.0, max(1.8, 0.4 * ratio)))
    # Activity floor that means "you are actually walking" (between still/walk).
    min_level = float(max(0.3, walk_activity * 0.4))
    return {
        "fs": float(fs),
        "threshold": round(threshold, 3),
        "abs_level": round(abs_level, 4),
        "min_level": round(min_level, 4),
        "refractory": 0.6,
        "walk_peak": round(walk_peak, 4),
        "jump_peak": round(jump_peak, 4),
        "walk_activity": round(walk_activity, 4),
        "jump_over_walk_ratio": round(ratio, 3),
    }


def _record_phase(source, prompt, seconds, warmup):
    """Record `seconds` of samples after a warmup countdown; return (array, fs)."""
    print("\n>>> %s\n    Recording starts in %.0f s..." % (prompt, warmup),
          flush=True)
    time.sleep(warmup)
    source.drain()
    chunks = []
    started = time.monotonic()
    nxt = started
    while time.monotonic() - started < seconds:
        chunks.extend(source.drain())
        now = time.monotonic()
        if now >= nxt:
            print("\r    %5.1f/%5.1f s  (%d samples)" %
                  (now - started, seconds, len(chunks)), end="", flush=True)
            nxt = now + 0.25
        time.sleep(0.01)
    chunks.extend(source.drain())
    print()
    elapsed = max(1e-3, time.monotonic() - started)
    return np.asarray(chunks, dtype=np.float32), len(chunks) / elapsed


def _load_sessions(data_dir, label):
    arrays = []
    for p in sorted(Path(data_dir).glob("*.npz")):
        try:
            with np.load(p) as d:
                if str(d["label"].item()) == label:
                    arrays.append(np.asarray(d["samples"], dtype=np.float32))
        except (OSError, KeyError, ValueError):
            pass
    return arrays


def _save_session(data_dir, label, arr, fs):
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("%s-%s-%s.npz" % (label, time.strftime("%Y%m%d-%H%M%S"),
                                    uuid.uuid4().hex[:8]))
    np.savez_compressed(path, samples=np.asarray(arr, dtype=np.float32),
                        fs=np.float32(fs), label=label)
    return path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0", help="Geophone serial port")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--walk-seconds", type=float, default=15.0)
    ap.add_argument("--jump-seconds", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=3.0)
    ap.add_argument("--data-dir",
                    default=str(Path(__file__).with_name("training_data")))
    ap.add_argument("--model",
                    default=str(Path(__file__).with_name("jump_model.json")))
    ap.add_argument("--from-data", action="store_true",
                    help="Skip recording; refit from saved walk/jump sessions.")
    ap.add_argument("--fs", type=float, default=100.0,
                    help="Sample-rate hint used with --from-data.")
    args = ap.parse_args()

    if args.from_data:
        walks = _load_sessions(args.data_dir, "walk")
        jumps = _load_sessions(args.data_dir, "jump")
        if not walks or not jumps:
            sys.exit("Need at least one 'walk' and one 'jump' session in %s. "
                     "Run collect_jumps.py (with hardware) first." % args.data_dir)
        walk, jump, fs = np.concatenate(walks), np.concatenate(jumps), args.fs
    else:
        port = args.port or auto_detect_port()
        if not port:
            sys.exit("No serial port found; connect the geophone or pass --port.")
        source = SerialSource(port, args.baud)
        source.open()
        source.start()
        try:
            print("=== Adaptive jump-detector training ===")
            print("Do it as you will in the game: same sensor spot, same shoes.")
            walk, fsw = _record_phase(
                source, "WALK in place, naturally and continuously.",
                args.walk_seconds, args.warmup)
            jump, fsj = _record_phase(
                source, "JUMP straight up ~6-10 times, landing where you stand.",
                args.jump_seconds, args.warmup)
        finally:
            source.stop()
        if walk.size < 32 or jump.size < 32:
            sys.exit("\nToo few samples were received; nothing saved.")
        fs = 0.5 * (fsw + fsj)
        print("Saved raw sessions: %s , %s" % (
            _save_session(args.data_dir, "walk", walk, fsw),
            _save_session(args.data_dir, "jump", jump, fsj)))

    model = fit_jump_model(walk, jump, fs)
    with open(args.model, "w") as f:
        json.dump(model, f, indent=2)
    print("\nLearned adaptive jump model -> %s" % args.model)
    for k in ("threshold", "abs_level", "min_level", "walk_peak", "jump_peak",
              "jump_over_walk_ratio"):
        print("  %-22s %s" % (k, model[k]))
    if model["jump_over_walk_ratio"] < 1.6:
        print("\nWARNING: your jumps were only %.2fx your steps -- detection may "
              "be unreliable.\n  Jump harder, or move the sensor closer to where "
              "you land." % model["jump_over_walk_ratio"])
    # The model is opt-in: the game uses its responsive built-in detector by
    # default. Try the trained model with:
    print("\nTo try this trained model in the game, run it with:")
    print("    python geophone_game.py --jump-model %s" % args.model)
    print("Omit --jump-model to keep the default (more responsive) jump detector.")


if __name__ == "__main__":
    main()

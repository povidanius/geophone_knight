"""Command-line data collection shared by collect_still.py and collect_walk.py."""

import argparse
from pathlib import Path
import sys
import time
import uuid

import numpy as np

from geophone_io import SerialSource, auto_detect_port


def collect(label):
    ap = argparse.ArgumentParser(
        description="Record geophone examples for class %r." % label,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0",
                    help="Geophone serial port")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--output-dir",
                    default=str(Path(__file__).with_name("training_data")))
    args = ap.parse_args()

    port = args.port or auto_detect_port()
    if not port:
        sys.exit("No serial port found; connect the geophone or pass --port.")
    source = SerialSource(port, args.baud)
    source.open()
    source.start()
    try:
        action = {
            "still": "stand completely still",
            "walk": "walk naturally and continuously near the sensor",
            "jump": "jump straight up repeatedly near the sensor",
        }[label]
        print("Get ready to %s. Recording starts in %.1f s..." %
              (action, args.warmup), flush=True)
        time.sleep(args.warmup)
        source.drain()
        chunks = []
        started = time.monotonic()
        next_report = started
        while time.monotonic() - started < args.seconds:
            chunks.extend(source.drain())
            now = time.monotonic()
            if now >= next_report:
                print("\rRecording %-5s %5.1f/%5.1f s  (%d samples)" %
                      (label, now - started, args.seconds, len(chunks)),
                      end="", flush=True)
                next_report = now + 0.25
            time.sleep(0.01)
        chunks.extend(source.drain())
    except KeyboardInterrupt:
        print("\nStopped early; saving what was recorded.")
    finally:
        source.stop()

    elapsed = max(1e-3, time.monotonic() - started)
    fs = len(chunks) / elapsed
    if len(chunks) < 32:
        sys.exit("\nToo few samples were received; no file saved.")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # Each run is an independent session. The suffix guarantees that even
    # sessions completed in the same second cannot replace one another.
    path = out_dir / ("%s-%s-%s.npz" %
                      (label, stamp, uuid.uuid4().hex[:8]))
    np.savez_compressed(path, samples=np.asarray(chunks, dtype=np.float32),
                        fs=np.float32(fs), label=label)
    print("\nSaved %s (%.1f samples/s)." % (path, fs))
    session_count = 0
    for existing in out_dir.glob("*.npz"):
        try:
            with np.load(existing) as data:
                session_count += str(data["label"].item()) == label
        except (OSError, KeyError, ValueError):
            pass
    print("Dataset now contains %d separate %s session(s); training uses all "
          "of them." % (session_count, label))

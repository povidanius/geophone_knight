#!/usr/bin/env python3
"""Procedural "Cave" soundtrack for the Step Runner game.

A companion to :mod:`medieval_music` and :mod:`zombie_music`: this synthesizes
an original, royalty-free subterranean ambience for the Cave stage (stage 3+)
and writes a cleanly-looping WAV the game can repeat while skeletons close in.

Sound recipe
------------
* **Depth:** a very low, detuned **A1 drone** with a hollow fifth, the pressure
  of tonnes of rock overhead.
* **Drips:** scattered high **water drips**, each with a long cave **echo** — the
  single most "cave" sound there is.
* **Rumble:** a slow filtered-noise **rumble** that swells like distant
  cave-ins.
* **Bones:** dry **bone rattles** (bandpassed noise clicks) — the skeletons
  stirring in the dark.
* **Toll:** a lonely low **gong** and a couple of hollow, echoing tones.

Run it directly to regenerate the WAV::

    python cave_music.py                   # -> cave_theme.wav
    python cave_music.py out.wav
"""

from pathlib import Path
import sys
import wave

import numpy as np


SAMPLE_RATE = 44100
LOOP_SECONDS = 20.0


def _reverb(x, taps):
    """Add attenuated delayed copies of `x` for a big, echoing cave space."""
    out = x.copy()
    for delay_s, gain in taps:
        d = int(SAMPLE_RATE * delay_s)
        if 0 < d < len(out):
            out[d:] += gain * x[:len(x) - d]
    return out


def _drone(freq, dur, amp=0.16, detune=0.005):
    """A detuned sub-bass drone with a very slow swell — the weight of the rock."""
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    trem = 0.8 + 0.2 * np.sin(2.0 * np.pi * 0.08 * t) ** 2
    w = (np.sin(2.0 * np.pi * freq * t)
         + np.sin(2.0 * np.pi * freq * (1.0 + detune) * t)
         + 0.4 * np.sin(2.0 * np.pi * 1.5 * freq * t))     # hollow fifth
    return amp * w * trem


def _drip(freq=2100.0, dur=0.45, amp=0.32):
    """A single water drip: a bright plink with a fast decay and a shimmer."""
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    phase = 2.0 * np.pi * freq * t
    env = np.exp(-t / 0.05)
    body = np.sin(phase) * env
    body += 0.3 * np.sin(2.0 * phase) * np.exp(-t / 0.03)  # high shimmer
    return amp * body


def _rumble(dur, amp=0.13):
    """A slow, filtered-noise rumble that swells like a distant cave-in."""
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    noise = np.random.default_rng(7).normal(0, 1, n)
    low = np.convolve(noise, np.ones(800) / 800.0, mode="same")
    swell = 0.4 + 0.6 * (np.sin(2.0 * np.pi * 0.06 * t) * 0.5 + 0.5)
    return amp * low * swell


def _rattle(dur=0.9, amp=0.16, seed=0, clicks=10):
    """A dry bone rattle: bandpassed noise gated into a burst of little clicks."""
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, n)
    hp = noise - np.convolve(noise, np.ones(24) / 24.0, mode="same")   # highpass
    gate = np.zeros(n)
    for c in range(clicks):
        s = int((c / clicks) * n + rng.uniform(0.0, 0.02) * SAMPLE_RATE)
        e = min(n, s + int(0.018 * SAMPLE_RATE))
        gate[s:e] = 1.0
    return amp * hp * gate * np.exp(-t / (dur * 0.7))


def _hollow(freq, dur, amp=0.13):
    """A hollow, vibrato tone for a lonely echo drifting down the tunnels."""
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    vib = 1.0 + 0.010 * np.sin(2.0 * np.pi * 4.5 * t)
    phase = 2.0 * np.pi * freq * t * vib
    w = np.sin(phase) + 0.2 * np.sin(3.0 * phase)
    env = np.sin(np.pi * np.clip(t / dur, 0.0, 1.0)) ** 1.2
    return amp * w * env


def _gong(freq, dur, amp=0.16):
    """A low, inharmonic gong toll with a long decay."""
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    w = (np.sin(2.0 * np.pi * freq * t)
         + 0.7 * np.sin(2.0 * np.pi * freq * 1.48 * t)
         + 0.4 * np.sin(2.0 * np.pi * freq * 2.61 * t))
    env = np.exp(-t / 1.1)
    return amp * w * env


def synthesize():
    """Render the full looping Cave bed as float32 in [-1, 1]."""
    song_len = int(SAMPLE_RATE * LOOP_SECONDS)
    dur_s = LOOP_SECONDS
    out = np.zeros(song_len, dtype=np.float64)

    def add(chunk, start_s):
        start = int(SAMPLE_RATE * start_s)
        end = min(song_len, start + len(chunk))
        if end > start:
            out[start:end] += chunk[:end - start]

    # -- Continuous beds: sub drone (A1) + a beating twin + rumble.
    out += _drone(55.0, dur_s, amp=0.17)[:song_len]
    out += _drone(82.4, dur_s, amp=0.08, detune=0.004)[:song_len]     # E2 fifth
    out += _rumble(dur_s)[:song_len]

    # -- A low gong at the top of the loop to anchor it.
    add(_reverb(_gong(55.0, 4.5), [(0.28, 0.4), (0.6, 0.22)]), 0.2)

    # -- Scattered, echoing water drips (varied pitch and timing).
    drips = [(0.8, 2350.0), (2.6, 1850.0), (4.1, 2600.0), (5.9, 2050.0),
             (7.7, 1650.0), (9.2, 2450.0), (11.3, 1950.0), (13.0, 2700.0),
             (14.8, 1750.0), (16.6, 2250.0), (18.1, 1550.0), (19.0, 2400.0)]
    for start_s, freq in drips:
        add(_reverb(_drip(freq), [(0.21, 0.45), (0.44, 0.24), (0.7, 0.12)]),
            start_s)

    # -- Bone rattles: the skeletons stir a few times per loop.
    for i, start_s in enumerate((3.4, 8.6, 12.9, 17.4)):
        add(_reverb(_rattle(0.9, seed=i + 1), [(0.19, 0.35)]), start_s)

    # -- Hollow echoing tones drifting down distant tunnels.
    for start_s, freq in ((6.0, 146.83), (13.5, 110.0)):              # D3, A2
        add(_reverb(_hollow(freq, 3.0), [(0.33, 0.4), (0.66, 0.2)]), start_s)

    # Normalize with headroom, then soft-clip for a warm, saturated murk.
    out /= max(1e-9, np.max(np.abs(out)))
    out = np.tanh(1.5 * out) / np.tanh(1.5)
    out *= 0.82
    return out.astype(np.float32)


def write_wav(path, samples):
    """Write a mono 16-bit PCM WAV."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def ensure_cave_theme(path=None):
    """Return the path to the cave theme WAV, generating it once if missing."""
    if path is None:
        path = Path(__file__).with_name("cave_theme.wav")
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        write_wav(path, synthesize())
    return path


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).with_name("cave_theme.wav"))
    write_wav(out, synthesize())
    print("Wrote cave theme to %s" % out)


if __name__ == "__main__":
    main()

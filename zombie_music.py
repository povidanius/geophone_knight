#!/usr/bin/env python3
"""Procedural "Zombie Village" soundtrack for the Step Runner game.

A companion to :mod:`medieval_music`: instead of a lilting minstrel tune this
synthesizes an original, royalty-free horror-ambient bed for the Zombie Village
stage (stage 2+). It writes a WAV built to loop cleanly so the game can play it
on repeat while the zombies close in.

Sound recipe
------------
* **Key:** E Phrygian-ish darkness — a low **E** drone with a grinding minor
  second (**F**) beneath it, the classic "something is wrong here" dissonance.
* **Drone:** a detuned sub-bass pad (E1 + a beating twin + a distant fifth)
  that slowly swells like a cold wind through the village.
* **Groans:** every couple of bars a low, pitch-bending vocal *moan* — a
  shambling zombie somewhere in the fog.
* **Heartbeat:** a soft "lub-dub" kick on every bar — your own pulse as the
  horde nears.
* **Bells & wind:** sparse, inharmonic bell tolls and a filtered wind hiss for
  the graveyard atmosphere.

Run it directly to regenerate the WAV::

    python zombie_music.py                 # -> zombie_theme.wav
    python zombie_music.py out.wav
"""

from pathlib import Path
import sys
import wave

import numpy as np


SAMPLE_RATE = 44100
BEAT = 0.60                      # slow, dragging pulse (~100 BPM, half-time feel)
BARS = 8                         # length of the loop
BEATS_PER_BAR = 4


def _drone(freq, dur, amp=0.14, detune=0.006, swell=0.30):
    """A detuned sub-bass pad with a slow swell — the cold-wind bed."""
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    trem = (1.0 - swell) + swell * np.sin(2.0 * np.pi * 0.11 * t) ** 2
    w = (np.sin(2.0 * np.pi * freq * t)
         + np.sin(2.0 * np.pi * freq * (1.0 + detune) * t)
         + 0.4 * np.sin(2.0 * np.pi * 2.0 * freq * t))
    return amp * w * trem


def _moan(freq, dur, amp=0.22, seed=0):
    """A low, pitch-bending zombie groan (a rough voice that swells in and out)."""
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    p = t / dur
    # Sag downward then lurch back up: the shape of a drawn-out groan.
    bend = freq * (1.0 + 0.10 * np.sin(2.0 * np.pi * 0.7 * p) - 0.14 * p)
    phase = 2.0 * np.pi * np.cumsum(bend) / SAMPLE_RATE
    rough = 1.0 + 0.5 * np.sin(2.0 * np.pi * 6.5 * t)      # vocal-fry roughness
    env = np.sin(np.pi * np.clip(p, 0.0, 1.0)) ** 1.3       # swell in / out
    voice = np.tanh(1.6 * np.sin(phase) * rough)
    breath = np.random.default_rng(seed).normal(0, 1, n) * 0.18
    return amp * env * (voice + breath)


def _heartbeat(dur, amp=0.5):
    """A soft kick-drum thump: a fast pitch-dropping sine (one beat of a pulse)."""
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    f = 90.0 * np.exp(-t / 0.05) + 38.0
    phase = 2.0 * np.pi * np.cumsum(f) / SAMPLE_RATE
    env = np.exp(-t / 0.09)
    return amp * np.sin(phase) * env


def _bell(freq, dur, amp=0.11):
    """An inharmonic, decaying bell toll for a lonely graveyard chime."""
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    w = (np.sin(2.0 * np.pi * freq * t)
         + 0.6 * np.sin(2.0 * np.pi * freq * 2.76 * t)
         + 0.4 * np.sin(2.0 * np.pi * freq * 5.40 * t))
    env = np.exp(-t / 0.55)
    return amp * w * env


def _wind(dur, amp=0.06):
    """A filtered noise bed with slow gusts — wind through the dead village."""
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    noise = np.random.default_rng(42).normal(0, 1, n)
    k = np.ones(300) / 300.0
    low = np.convolve(noise, k, mode="same")              # lowpass hiss
    gust = 0.45 + 0.55 * (np.sin(2.0 * np.pi * 0.13 * t
                                 + np.sin(2.0 * np.pi * 0.05 * t)) * 0.5 + 0.5)
    return amp * low * gust


def synthesize():
    """Render the full looping Zombie Village bed as float32 in [-1, 1]."""
    beat_s = BEAT
    bar_s = beat_s * BEATS_PER_BAR
    song_len = int(SAMPLE_RATE * bar_s * BARS)
    dur_s = song_len / SAMPLE_RATE
    out = np.zeros(song_len, dtype=np.float64)

    def add(chunk, start):
        end = min(song_len, start + len(chunk))
        if end > start:
            out[start:end] += chunk[:end - start]

    # -- Continuous beds: sub drone (E1), beating fifth (B1), a low, quiet
    #    minor-second (F1) grinding underneath, and wind.
    e1, b1, f1 = 41.20, 61.74, 43.65
    out += _drone(e1, dur_s, amp=0.16)[:song_len]
    out += _drone(b1, dur_s, amp=0.09, detune=0.004)[:song_len]
    out += _drone(f1, dur_s, amp=0.05, detune=0.010, swell=0.6)[:song_len]
    out += _wind(dur_s)[:song_len]

    # -- Heartbeat "lub-dub" on every bar.
    for bar in range(BARS):
        base = int(bar * bar_s * SAMPLE_RATE)
        add(_heartbeat(0.30, amp=0.42), base)
        add(_heartbeat(0.30, amp=0.26), base + int(0.32 * SAMPLE_RATE))

    # -- Zombie groans every second bar, alternating pitch.
    for i, bar in enumerate(range(0, BARS, 2)):
        start = int((bar * bar_s + 0.5 * beat_s) * SAMPLE_RATE)
        freq = 78.0 if i % 2 == 0 else 96.0
        add(_moan(freq, 2.4, amp=0.20, seed=bar + 1), start)

    # -- Sparse, dissonant bell tolls (E5 and its unsettled minor-second F5).
    for bar, freq in ((1, 659.26), (3, 698.46), (5, 493.88), (6, 659.26)):
        start = int((bar * bar_s + 2.0 * beat_s) * SAMPLE_RATE)
        add(_bell(freq, 1.6), start)

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


def ensure_zombie_theme(path=None):
    """Return the path to the zombie theme WAV, generating it once if missing."""
    if path is None:
        path = Path(__file__).with_name("zombie_theme.wav")
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        write_wav(path, synthesize())
    return path


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).with_name("zombie_theme.wav"))
    write_wav(out, synthesize())
    print("Wrote zombie village theme to %s" % out)


if __name__ == "__main__":
    main()

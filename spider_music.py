#!/usr/bin/env python3
"""Procedural "Spider Cave" soundtrack for the Step Runner game.

A companion to :mod:`medieval_music`, :mod:`zombie_music` and :mod:`cave_music`:
this synthesizes an original, royalty-free, skin-crawling ambience for the
Spider Cave stage (stage 4+) and writes a cleanly-looping WAV.

Sound recipe
------------
* **Dread:** a very low **D1 drone** with a slow beating twin — the pit you are
  trapped in.
* **Tension:** a high, dissonant **tremolo cluster** (clustered minor seconds,
  fast amplitude flutter) — the classic horror "strings" that make skin crawl.
* **Skittering:** dense bursts of tiny high **clicks** at irregular timing — a
  thousand legs moving in the dark.
* **Web twangs:** resonant **plucks** with a downward pitch bend — a taut strand
  of silk being plucked.
* **Glissando:** an occasional slow rising **glissando** whine for unease.

Run it directly to regenerate the WAV::

    python spider_music.py                 # -> spider_theme.wav
    python spider_music.py out.wav
"""

from pathlib import Path
import sys
import wave

import numpy as np


SAMPLE_RATE = 44100
LOOP_SECONDS = 18.0


def _reverb(x, taps):
    """Add attenuated delayed copies of `x` for a hollow, echoing cave space."""
    out = x.copy()
    for delay_s, gain in taps:
        d = int(SAMPLE_RATE * delay_s)
        if 0 < d < len(out):
            out[d:] += gain * x[:len(x) - d]
    return out


def _drone(freq, dur, amp=0.16, detune=0.004):
    """A detuned sub-bass drone with a very slow swell — the pit of dread."""
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    trem = 0.8 + 0.2 * np.sin(2.0 * np.pi * 0.07 * t) ** 2
    w = (np.sin(2.0 * np.pi * freq * t)
         + np.sin(2.0 * np.pi * freq * (1.0 + detune) * t)
         + 0.35 * np.sin(2.0 * np.pi * 2.0 * freq * t))
    return amp * w * trem


def _tremolo(freqs, dur, amp=0.10, rate=11.0):
    """A dissonant, fast-fluttering high cluster — the crawling-skin strings."""
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    w = sum(np.sin(2.0 * np.pi * f * t) for f in freqs) / len(freqs)
    trem = 0.5 + 0.5 * np.sin(2.0 * np.pi * rate * t)
    env = np.sin(np.pi * np.clip(t / dur, 0.0, 1.0)) ** 1.2
    return amp * w * trem * env


def _skitter(dur=1.2, amp=0.13, seed=0, clicks=34):
    """A burst of tiny, irregular high clicks — countless little legs moving."""
    n = int(SAMPLE_RATE * dur)
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    click_n = int(0.012 * SAMPLE_RATE)
    tt = np.arange(click_n) / SAMPLE_RATE
    for _ in range(clicks):
        s = int(rng.uniform(0.0, dur) * SAMPLE_RATE)
        e = min(n, s + click_n)
        f = rng.uniform(2600.0, 5200.0)
        out[s:e] += (np.sin(2.0 * np.pi * f * tt[:e - s])
                     * np.exp(-tt[:e - s] / 0.0035))
    return amp * out


def _web_pluck(freq=180.0, dur=1.3, amp=0.24):
    """A taut silk-strand twang: a pluck whose pitch bends down as it resonates."""
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    bend = freq * (1.0 + 0.5 * np.exp(-t / 0.08))
    phase = 2.0 * np.pi * np.cumsum(bend) / SAMPLE_RATE
    w = (np.sin(phase) + 0.4 * np.sin(2.0 * phase)
         + 0.2 * np.sin(3.0 * phase))
    env = np.exp(-t / 0.5)
    return amp * w * env


def _gliss(f0, f1, dur=2.2, amp=0.09):
    """A slow rising glissando whine (theremin-like) for pure unease."""
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    f = f0 * (f1 / f0) ** (t / dur)
    phase = 2.0 * np.pi * np.cumsum(f) / SAMPLE_RATE
    vib = 1.0 + 0.02 * np.sin(2.0 * np.pi * 6.0 * t)
    env = np.sin(np.pi * np.clip(t / dur, 0.0, 1.0))
    return amp * np.sin(phase * vib) * env


def synthesize():
    """Render the full looping Spider Cave bed as float32 in [-1, 1]."""
    song_len = int(SAMPLE_RATE * LOOP_SECONDS)
    dur_s = LOOP_SECONDS
    out = np.zeros(song_len, dtype=np.float64)

    def add(chunk, start_s):
        start = int(SAMPLE_RATE * start_s)
        end = min(song_len, start + len(chunk))
        if end > start:
            out[start:end] += chunk[:end - start]

    # -- Continuous beds: sub drone (D1) + a beating twin.
    out += _drone(36.71, dur_s, amp=0.18)[:song_len]
    out += _drone(55.0, dur_s, amp=0.07, detune=0.003)[:song_len]     # A1

    # -- Pulsing dissonant tremolo clusters (clustered minor seconds up high).
    for start_s in (0.0, 4.5, 9.0, 13.5):
        add(_tremolo([1174.7, 1244.5, 1318.5], 4.2, amp=0.09, rate=11.0),
            start_s)
    # a second, higher, faster cluster woven against it
    for start_s in (2.2, 6.7, 11.2, 15.7):
        add(_tremolo([1760.0, 1864.7], 2.4, amp=0.055, rate=15.0), start_s)

    # -- Dense skittering bursts (the legs), scattered across the loop.
    for i, start_s in enumerate((1.0, 3.6, 5.4, 7.9, 10.1, 12.6, 14.9, 16.8)):
        add(_reverb(_skitter(1.3, seed=i + 1), [(0.16, 0.3)]), start_s)

    # -- Web twangs: a taut strand plucked a few times.
    for start_s, freq in ((2.0, 196.0), (8.4, 146.83), (13.0, 220.0),
                          (16.2, 174.6)):
        add(_reverb(_web_pluck(freq), [(0.22, 0.35), (0.47, 0.18)]), start_s)

    # -- Two slow rising glissando whines for unease.
    add(_reverb(_gliss(300.0, 900.0, 2.4), [(0.3, 0.3)]), 5.0)
    add(_reverb(_gliss(260.0, 780.0, 2.4), [(0.3, 0.3)]), 12.2)

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


def ensure_spider_theme(path=None):
    """Return the path to the spider theme WAV, generating it once if missing."""
    if path is None:
        path = Path(__file__).with_name("spider_theme.wav")
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        write_wav(path, synthesize())
    return path


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).with_name("spider_theme.wav"))
    write_wav(out, synthesize())
    print("Wrote spider cave theme to %s" % out)


if __name__ == "__main__":
    main()

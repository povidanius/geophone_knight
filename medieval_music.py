#!/usr/bin/env python3
"""Procedural medieval soundtrack for the Step Runner game.

Rather than ship (and license) a real audio file, this synthesizes an original,
royalty-free medieval-sounding theme entirely with numpy and writes it to a WAV.
The tune is built to loop cleanly, so the game can play it on repeat as
background music.

Musical recipe
--------------
* **Mode:** D Dorian (D E F G A B C) — the archetypal "minstrel" church mode.
* **Lead:** a plucked lute/harp timbre (a few decaying harmonics with a fast
  attack and a little vibrato) playing a wandering melody.
* **Drone:** a sustained tonic+fifth (D + A), the bagpipe/hurdy-gurdy bed that
  gives the whole thing its ancient feel.
* **Percussion:** a soft frame-drum thump on the strong beats.

Run it directly to regenerate the WAV::

    python medieval_music.py               # -> medieval_theme.wav
    python medieval_music.py out.wav
"""

from pathlib import Path
import sys
import wave

import numpy as np


SAMPLE_RATE = 44100
BEAT = 0.42                      # seconds per beat (~143 BPM feel, lilting)
A4 = 440.0

# D Dorian melody as (semitones-relative-to-D4, duration-in-beats). The phrase
# opens and closes on the tonic D so the loop point is seamless.
MELODY = [
    # Phrase A — rising question
    (0, 1), (2, 1), (3, 1), (5, 1),
    (7, 1.5), (5, 0.5), (7, 1), (9, 1),
    (10, 1), (9, 1), (7, 1), (5, 1),
    (3, 2), (2, 2),
    # Phrase B — answer that settles home
    (7, 1), (9, 1), (10, 1), (12, 1),
    (10, 1.5), (9, 0.5), (7, 1), (9, 1),
    (5, 1), (7, 1), (3, 1), (5, 1),
    (2, 1), (3, 1), (0, 2),
]


def _midi_freq(semitones_from_d4):
    """Frequency for a note given as semitones relative to D4."""
    # D4 is 3 semitones above A3, i.e. 9 semitones below A4 ... use A4 anchor:
    # A4 = 440, D4 = A4 * 2**(-7/12) (D4 is 7 semitones below A4).
    return A4 * 2.0 ** ((semitones_from_d4 - 7) / 12.0)


def _pluck(freq, dur, amp=0.5):
    """A plucked-string note: decaying harmonics, soft attack, gentle vibrato."""
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    vib = 1.0 + 0.006 * np.sin(2.0 * np.pi * 5.5 * t)
    phase = 2.0 * np.pi * freq * t * vib
    # A lute-ish spectrum: fundamental plus a few softening overtones.
    wave_ = (1.00 * np.sin(phase)
             + 0.45 * np.sin(2 * phase)
             + 0.28 * np.sin(3 * phase)
             + 0.12 * np.sin(4 * phase))
    # Pluck envelope: near-instant attack, exponential decay, quick release.
    env = np.exp(-t / (0.28 * dur + 0.12))
    attack = np.clip(t / 0.005, 0.0, 1.0)
    rel = np.clip((dur - t) / 0.03, 0.0, 1.0)
    return amp * wave_ * env * attack * rel


def _drone(freq, dur, amp=0.16):
    """A sustained reed/bagpipe drone with a slow tremolo and light overtones."""
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    phase = 2.0 * np.pi * freq * t
    trem = 0.85 + 0.15 * np.sin(2.0 * np.pi * 3.0 * t)
    wave_ = (np.sin(phase) + 0.5 * np.sin(2 * phase)
             + 0.22 * np.sin(3 * phase))
    return amp * wave_ * trem


def _drum(dur, amp=0.35):
    """A soft frame-drum thump: a short pitch-dropping sine with fast decay."""
    n = int(SAMPLE_RATE * dur)
    t = np.arange(n) / SAMPLE_RATE
    f = 150.0 * np.exp(-t / 0.06) + 55.0
    phase = 2.0 * np.pi * np.cumsum(f) / SAMPLE_RATE
    env = np.exp(-t / 0.10)
    body = np.sin(phase) * env
    noise = np.random.default_rng(0).normal(0, 1, n) * np.exp(-t / 0.02) * 0.3
    return amp * (body + noise)


def synthesize():
    """Render the full looping theme as a float32 array in [-1, 1]."""
    total_beats = sum(d for _, d in MELODY)
    total = int(SAMPLE_RATE * total_beats * BEAT) + SAMPLE_RATE
    lead = np.zeros(total, dtype=np.float64)

    # Lead melody.
    cursor = 0
    for semis, beats in MELODY:
        dur = beats * BEAT
        note = _pluck(_midi_freq(semis), dur)
        lead[cursor:cursor + len(note)] += note
        cursor += int(SAMPLE_RATE * dur)

    song_len = cursor
    out = lead[:song_len].copy()

    # Drone: tonic D3 + fifth A3, filling the whole phrase.
    d3 = _midi_freq(-12)
    a3 = _midi_freq(-5)
    dur_s = song_len / SAMPLE_RATE
    out += _drone(d3, dur_s)[:song_len]
    out += _drone(a3, dur_s, amp=0.11)[:song_len]

    # Drum on beats 1 and 3 of every 4/4 bar.
    beat_samples = BEAT * SAMPLE_RATE
    n_beats = int(song_len / beat_samples)
    for b in range(n_beats):
        if b % 4 in (0, 2):
            start = int(b * beat_samples)
            hit = _drum(0.25, amp=0.28 if b % 4 == 0 else 0.18)
            end = min(song_len, start + len(hit))
            out[start:end] += hit[:end - start]

    # Normalize with a touch of headroom, then soft-clip for warmth.
    out /= max(1e-9, np.max(np.abs(out)))
    out = np.tanh(1.4 * out) / np.tanh(1.4)
    out *= 0.85
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


def ensure_theme(path=None):
    """Return the path to the theme WAV, generating it once if it is missing."""
    if path is None:
        path = Path(__file__).with_name("medieval_theme.wav")
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        write_wav(path, synthesize())
    return path


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).with_name("medieval_theme.wav"))
    write_wav(out, synthesize())
    print("Wrote medieval theme to %s" % out)


if __name__ == "__main__":
    main()

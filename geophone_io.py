"""Serial acquisition and test simulation for the geophone game/collectors."""

from collections import deque
import math
import sys
import threading
import time

import numpy as np


class SampleSource:
    def __init__(self):
        self._queue = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _push(self, value):
        with self._lock:
            self._queue.append(value)

    def drain(self):
        with self._lock:
            result = list(self._queue)
            self._queue.clear()
        return result


class SerialSource(SampleSource):
    """Read firmware lines formatted as ``E<value_in_mV>``."""

    def __init__(self, port, baud):
        super().__init__()
        self.port = port
        self.baud = baud
        self._serial = None

    def open(self):
        import serial

        self._serial = serial.Serial(self.port, self.baud, timeout=1)
        time.sleep(0.2)
        self._serial.reset_input_buffer()

    def _run(self):
        while not self._stop.is_set():
            try:
                line = self._serial.readline().decode(
                    "ascii", errors="ignore").strip()
            except Exception as exc:
                sys.stderr.write("Serial read error: %s\n" % exc)
                time.sleep(0.5)
                continue
            if line[:1] in ("E", "e"):
                line = line[1:]
            try:
                self._push(float(line))
            except ValueError:
                continue

    def stop(self):
        super().stop()
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass


class SimulatedSource(SampleSource):
    """Alternate quiet and walking-like signals for UI testing."""

    def __init__(self, fs=100.0):
        super().__init__()
        self.fs = fs

    def _run(self):
        rng = np.random.default_rng(7)
        index = 0
        while not self._stop.is_set():
            t = index / self.fs
            value = rng.normal(0.0, 0.3)
            if 4.0 <= t % 10.0 < 9.0:
                since_step = t % 0.5
                value += (35.0 * np.exp(-since_step / 0.06)
                          * np.sin(2.0 * np.pi * 14.0 * since_step))
            self._push(float(value))
            index += 1
            time.sleep(1.0 / self.fs)


class OnsetDetector:
    """Low-latency impulse detector for jumps/stomps on the raw stream.

    The still/walk CNN is deliberately not used here: its 2.5 s window plus
    probability EMA make it slow, and it locks onto the biggest amplitude in
    the window, which for a jump is the *landing* spike. That is inherently
    late -- by the time you land you have already been airborne for a few
    hundred milliseconds.

    Instead this watches each freshly drained batch of raw samples and fires on
    the first strong transient, which for a jump is the take-off push-off (you
    load the floor with well over body weight before leaving it). A refractory
    window after a trigger then deliberately swallows the larger, later landing
    spike so one jump produces exactly one event.

    Walk steps are transients too, so a fixed threshold would fire on every
    footstep. The reference here is instead a *peak follower* -- it tracks your
    recent step-peak amplitude (instant attack, slow release). A transient only
    triggers when it exceeds ``threshold`` times that recent peak, i.e. when it
    is markedly stronger than an ordinary step, which a jump is. An absolute
    ``min_level`` keeps small blips from firing while you stand still.

    Because the reference follows your own footsteps, it is self-calibrating
    across people, floors and sensor placement, but ``threshold`` may still need
    field tuning: too low and hard footsteps count as jumps, too high and gentle
    jumps are missed. ``collect_jumps.py`` learns these levels from your own walk
    and jumps and writes ``jump_model.json`` (see :meth:`from_model`), giving an
    adaptive ``threshold`` plus an absolute jump floor ``abs_level``.

    Two gates keep it honest:

    * **Sustained-activity gate.** A jump may fire only once a slow activity
      envelope (``activity_tau``) shows you are *already walking* -- above
      ``min_level``. A single transient from a standstill (the first step when
      you start walking) barely moves that slow envelope, so it can no longer
      trigger a jump. This fixes the "knight jumps when I start walking" bug. The
      trade-off is that a jump from a dead standstill is intentionally ignored;
      in play you are always moving when you jump.
    * **Absolute floor.** With a trained ``abs_level`` a transient must also
      exceed that learned magnitude, so a merely hard footstep never counts.
    """

    def __init__(self, fs=100.0, threshold=2.5, refractory=0.6,
                 release_tau=1.2, center_tau=0.3, warmup=0.5, min_level=0.5,
                 abs_level=0.0, activity_tau=0.8):
        self.fs = max(1.0, float(fs))
        self.threshold = float(threshold)   # multiples of the recent step peak
        self.refractory = float(refractory)  # seconds silenced after a trigger
        self.release_tau = float(release_tau)
        self.center_tau = float(center_tau)
        self.warmup = float(warmup)
        self.min_level = float(min_level)    # walking-activity floor for the gate
        self.abs_level = float(abs_level)    # learned absolute jump floor (0=off)
        self.activity_tau = float(activity_tau)
        self._center = 0.0
        self._peak = 0.0                   # recent step-peak follower (fast)
        self._activity = 0.0               # slow "am I walking?" envelope
        self._since_fire = 1e9
        self._warmed = 0.0
        self.last_strength = 0.0           # dev/reference of the last trigger

    @classmethod
    def from_model(cls, path, fs, **overrides):
        """Build a detector from a trained ``jump_model.json``; None if absent.

        The JSON is written by ``collect_jumps.py`` and holds the adaptive
        ``threshold`` / ``abs_level`` / ``min_level`` calibrated to the player.
        Keyword ``overrides`` win over the file (e.g. ``refractory=``).
        """
        import json
        import os

        if not path or not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                m = json.load(f)
        except (OSError, ValueError):
            return None
        params = dict(threshold=float(m.get("threshold", 2.5)),
                      abs_level=float(m.get("abs_level", 0.0)),
                      min_level=float(m.get("min_level", 0.5)),
                      refractory=float(m.get("refractory", 0.6)))
        params.update(overrides)
        return cls(fs, **params)

    def process(self, samples, fs=None):
        """Feed the newest raw samples; return True if an onset just started."""
        if fs and fs > 1.0:
            self.fs = float(fs)
        dt = 1.0 / self.fs
        a_center = 1.0 - math.exp(-dt / self.center_tau)
        a_act = 1.0 - math.exp(-dt / self.activity_tau)
        release = math.exp(-dt / self.release_tau)
        fired = False
        for v in samples:
            self._since_fire += dt
            self._warmed += dt
            dev = abs(v - self._center)
            established = self._activity >= self.min_level     # already walking?
            strong_rel = dev > self.threshold * self._peak     # vs recent steps
            strong_abs = dev >= self.abs_level                 # 0 => always ok
            if (self._warmed >= self.warmup
                    and self._since_fire >= self.refractory
                    and established and strong_rel and strong_abs):
                self._since_fire = 0.0
                self.last_strength = dev / max(self._peak, 1e-6)
                fired = True
            # Track the DC centre, the fast step-peak reference, and the slow
            # walking-activity envelope on every sample. The reference must keep
            # learning even right after a trigger: freezing it would let one
            # stray fire pin it low and make the next footstep fire too.
            self._center += a_center * (v - self._center)
            self._activity += a_act * (dev - self._activity)
            self._peak = max(dev, self._peak * release)
        return fired


class ScreenRecorder:
    """Record the game window to an MP4 with ffmpeg's ``x11grab`` (Linux/X11).

    The capture runs entirely inside a separate ffmpeg process, so it adds no
    per-frame cost to the game loop. It is best-effort: if ffmpeg or an X
    display is unavailable it disables itself with a warning rather than
    breaking the game. The window region is sampled at fixed screen
    coordinates, so moving the window mid-game is not followed.
    """

    def __init__(self, path, fps=20):
        self.path = path
        self.fps = fps
        self._proc = None
        self._tmp = None

    def start(self, x, y, width, height, display=None):
        import os
        import shutil
        import subprocess
        import tempfile

        if shutil.which("ffmpeg") is None:
            sys.stderr.write("Recording disabled: ffmpeg not found.\n")
            return False
        display = display or os.environ.get("DISPLAY")
        if not display:
            sys.stderr.write("Recording disabled: no X DISPLAY set.\n")
            return False
        # h264 + yuv420p needs even dimensions.
        w = max(2, int(width) & ~1)
        h = max(2, int(height) & ~1)
        out_dir = os.path.dirname(os.path.abspath(self.path)) or "."
        fd, self._tmp = tempfile.mkstemp(suffix=".mp4", dir=out_dir)
        os.close(fd)
        cmd = ["ffmpeg", "-y", "-f", "x11grab", "-framerate", str(self.fps),
               "-draw_mouse", "0",              # keep the cursor out of the video
               "-video_size", "%dx%d" % (w, h),
               "-i", "%s+%d,%d" % (display, int(x), int(y)),
               "-pix_fmt", "yuv420p", "-loglevel", "error", self._tmp]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except Exception as exc:
            sys.stderr.write("Recording disabled: %s\n" % exc)
            self._proc, self._tmp = None, None
            return False
        return True

    def stop(self):
        """Finalise the recording and move it into place as the saved movie."""
        import os

        if self._proc is None:
            return
        try:
            # 'q' on stdin tells ffmpeg to flush and close the file cleanly.
            self._proc.communicate(input=b"q", timeout=10)
        except Exception:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._proc = None
        if (self._tmp and os.path.exists(self._tmp)
                and os.path.getsize(self._tmp) > 0):
            try:
                os.replace(self._tmp, self.path)
                sys.stderr.write("Saved game recording to %s\n" % self.path)
            except Exception as exc:
                sys.stderr.write("Could not save recording: %s\n" % exc)
        elif self._tmp and os.path.exists(self._tmp):
            try:
                os.remove(self._tmp)
            except OSError:
                pass
        self._tmp = None


class MusicPlayer:
    """Loop a background-music WAV in a separate audio-player process.

    Playback runs entirely inside an external CLI player (``ffplay`` for gapless
    looping, falling back to ``paplay``/``aplay`` re-launched on each pass), so
    it costs the tkinter game loop nothing. It is best-effort: if no player or
    audio device is available it disables itself with a warning rather than
    breaking the game.
    """

    def __init__(self, path, volume=0.7):
        self.path = str(path)
        self.volume = float(volume)
        self._proc = None
        self._stop = threading.Event()
        self._thread = None
        self._mode = None          # "ffplay" | "loop-thread" | None

    def start(self):
        import shutil

        self._stop.clear()             # allow restart after a previous stop()
        if shutil.which("ffplay"):
            self._mode = "ffplay"
            self._start_ffplay()
            return True
        player = shutil.which("paplay") or shutil.which("aplay")
        if player:
            self._mode = "loop-thread"
            self._player = player
            self._thread = threading.Thread(target=self._loop_player, daemon=True)
            self._thread.start()
            return True
        sys.stderr.write("Music disabled: no ffplay/paplay/aplay found.\n")
        return False

    def _start_ffplay(self):
        import subprocess

        # ffplay loops the file natively (-loop 0) with no gap, runs headless
        # (-nodisp) and applies a playback volume in [0, 100].
        cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
               "-loop", "0", "-volume", str(int(self.volume * 100)), self.path]
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            sys.stderr.write("Music disabled: %s\n" % exc)
            self._proc = None

    def _loop_player(self):
        import subprocess

        # paplay/aplay play once, so relaunch until asked to stop.
        while not self._stop.is_set():
            try:
                self._proc = subprocess.Popen(
                    [self._player, self.path], stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as exc:
                sys.stderr.write("Music disabled: %s\n" % exc)
                return
            self._proc.wait()

    def switch(self, path):
        """Change the looping track (best-effort): stop, swap the file, restart.

        Used to move between per-stage themes (e.g. the medieval bed and the
        Zombie Village bed). A no-op if the requested track is already playing.
        """
        path = str(path)
        if path == self.path and self._proc is not None:
            return
        self.stop()
        self.path = path
        self.start()

    def stop(self):
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None


def list_serial_ports():
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return list(list_ports.comports())


def auto_detect_port():
    ports = list_serial_ports()
    if not ports:
        return None
    for port in ports:
        if any(key in port.device.lower()
               for key in ("usb", "acm", "wch", "ftdi")):
            return port.device
    return ports[0].device

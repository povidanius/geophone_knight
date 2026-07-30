# Smart Geophone game — Medieval edition

A geophone-driven side-scroller. A compact PyTorch model classifies the signal
window as **still** or **walk** to move a **knight** (his shield bears the gold
**Columns of Gediminas**); a separate onset detector fires **jumps**. An evil
wizard chases you and gains ground when you stop. Collect **10 coins** per stage
to clear it — a friendly dragon carries you to a **shop**, then on to the next
stage. High scores and game videos accumulate in your personal folder.

Each stage has its own environment, hazard and synthesized soundtrack:

| Stage | Theme | Hazard |
|-------|-------|--------|
| 1 | Medieval countryside | rolling barrels |
| 2 | Zombie Village | shambling zombies |
| 3 | Cave | attacking skeletons |
| 4+ | Spider Cave | attacking spiders |

You must **jump exactly over** each hazard (feet airborne and clear as it
passes). A contact costs **half a heart**; lose all hearts and the round ends.
Background creatures are pure atmosphere and never interact.

## Requirements

- **Python 3.7**; packages `numpy`, `pyserial`, `torch` (`requirements.txt`).
- An **SM-32 vertical geophone** over serial (firmware emits `E<mV>` lines), or
  run without hardware via `--simulate`.
- Optional (best-effort): `ffmpeg` for recording, and `ffplay`/`paplay`/`aplay`
  for music.

```bash
pip install -r requirements.txt
```

Serial port defaults to `/dev/ttyACM0` (override with `--port`).

## Collect data and train

Each run adds a session under `training_data/` (never overwritten). Collect
several sessions per class, matching the people, distances and floors of play.

```bash
python collect_still.py --seconds 30
python collect_walk.py --seconds 30
python learn_model.py --window-seconds 1.5
```

The window length is stored in `geophone_model.pt` and used automatically.

## Play

```bash
python geophone_game.py                # auto-detect port, player Povilas
python geophone_game.py --simulate     # no hardware
python geophone_game.py --name Jonas   # different player
```

**Controls:** walk near the sensor to move; jump on the sensor to jump (or
**Up**/**J**). Hold **Space** to walk manually. In the shop, `[1]`/`[2]`/`[3]`
buy upgrades (higher jump, extra heart, full heal) and `[Enter]` starts the next
stage. **R** restarts, **Q**/**Esc** quits (score saved).

**Players:** name is CLI-only (`--player`/`--name`, default Povilas). Each player
gets `players/<name>/` holding scores and one timestamped `game_*.mp4` per
session. `--players-dir PATH` changes the location.

**Recording** uses ffmpeg `x11grab` in a separate process (needs `ffmpeg` + X11).
Options: `--record-file`, `--record-fps`, `--no-record`.

**Music** is synthesized and looped per theme, swapping on stage change.
Options: `--no-music`, `--music-volume`. Regenerate with `python <name>_music.py`.

## Jump detection

Jumps use a small, self-calibrating **onset detector** (not the CNN) that fires
on the take-off transient. A **sustained-activity gate** means jumps only fire
while you are already walking (so the first step doesn't trigger one). Tune with
`--jump-threshold` (lower = more sensitive); disable with `--no-sensor-jump`.

The built-in detector is the default. To train a personal model:

```bash
python collect_jumps.py              # guided walk + jump capture
python collect_jumps.py --from-data  # refit from saved sessions (no hardware)
```

This writes `jump_model.json`, opt-in via `--jump-model jump_model.json`.

## Diagnostics

```bash
python diagnostics.py            # live scope: raw waveform + detectors
python diagnostics.py --simulate # press [j] to inject a test jump
```

Shows the raw waveform, still/walk probabilities, and the jump detector's levels
and gate. Keys: `[j]` test jump, `[↑]`/`[↓]` tune threshold, `[c]` clear, `[q]`
quit.

## Reset

Model and datasets are local, regenerable, and untracked:

```bash
rm -f geophone_model.pt jump_model.json
rm -rf training_data/
```

## Authors

Povilas Daniušis (geophone interface, classifier) and Albertas Daniušis (agenda and stages design)

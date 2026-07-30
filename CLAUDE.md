# CLAUDE.md — Smart Geophone Step Runner

Geophone-driven side-scroller (Chrome-dinosaur style). A single **SM-32 vertical
geophone** is read over serial; a small PyTorch CNN classifies the signal window
as **still/walk** to move the character, and a fast raw-stream onset detector
fires **jumps**. The game is themed in stages with per-stage art, hazards and
music.

- Working dir: `/home/tank/smart_geo1/smart_geophone/python`
- Python **3.7** (bytecode is `cpython-37`). Deps: `numpy`, `pyserial`, `torch`
  (`requirements.txt`).
- Git: work happens on branch **`medieval`** (main branch is `main`). Commit/push
  only when asked.

## Run

```bash
python geophone_game.py                 # auto-detect serial port; player=Povilas
python geophone_game.py --simulate      # no hardware: synthetic walker
python geophone_game.py --name Jonas    # player name via CLI (alias of --player)
```

- Player **name is CLI-only** (`--player`/`--name`, default **Povilas**); there
  is no GUI name prompt. Personal folder: `players/<name>/`.

- Hold **Space** to walk manually (works with `--simulate`, handy for testing).
  **Up/J** jump, **R** restart, **1/2/3** buy in shop, **Enter** leave shop,
  **Q/Esc** quit.
- Model file `geophone_model.pt` must exist (train via `learn_model.py` from
  `training_data/`). Collectors: `collect_still.py`, `collect_walk.py`,
  `collect_motion.py`.
- Jump detection is a separate, adaptive threshold model (NOT the CNN). Train it
  with `./collect_jumps.py` (guided walk + jump capture → `jump_model.json`);
  `./collect_jumps.py --from-data` refits from saved sessions with no hardware.
  The trained model is **opt-in**: the game uses the built-in `--jump-threshold`
  detector by default and only loads the model when run with
  `--jump-model jump_model.json` (`OnsetDetector.from_model`). Default was made
  opt-in because an over-conservative fit could make jumps barely trigger.

## Files

- `geophone_game.py` — **everything gameplay/rendering** lives in the `Game`
  class (tkinter Canvas, per-frame `after()` loop in `tick()`, `_render()`).
  `main()` wires source + classifier + music + recorder.
- `geophone_io.py` — acquisition + side processes: `SampleSource`/`SerialSource`
  (`E<mV>` firmware lines)/`SimulatedSource`; `OnsetDetector` (jump); 
  `ScreenRecorder` (ffmpeg x11grab); `MusicPlayer` (ffplay→paplay/aplay loop).
- `motion_model.py` — `prepare_window`, `MotionNet`, `MotionClassifier`
  (`MODEL_SAMPLES=128`, `WINDOW_SECONDS=2.50`, prob EMA smoothing).
- `learn_model.py` — trains `geophone_model.pt`.
- `diagnostics.py` — standalone live scope (tkinter, no game): raw-waveform
  display + still/walk bars + jump-detector levels/gate. `--simulate` + `[j]`
  injects a realistic test jump; `[↑/↓]` tunes threshold live. Reuses
  `MotionClassifier`/`OnsetDetector`, so it shows exactly what the game sees.
- `player_profile.py` — per-player folder `players/<name>/` (scores + videos).
- `medieval_music.py`, `zombie_music.py`, `cave_music.py`, `spider_music.py` —
  procedural soundtracks (numpy → WAV, cached next to the module).

## Stage / theme system (the recurring feature)

`Game._theme()` maps stage number → theme string:

| Stage | theme | hazard | soundtrack |
|-------|-------|--------|------------|
| 1 | `medieval` | rolling barrels | `medieval_theme.wav` |
| 2 | `zombie` | zombies | `zombie_theme.wav` |
| 3 | `cave` | skeletons | `cave_theme.wav` |
| 4+ | `spider` | spiders | `spider_theme.wav` |

Key mechanics:
- The **hazard reuses the `barrels` list** for all themes; only the *drawing*
  differs. `_creature_stage()` is true for zombie/cave/spider. Hazard drawn via
  `_draw_hazard_creature` → `_draw_creature` → theme dispatch (`_draw_zombie` /
  `_draw_skeleton` / `_draw_spider`, all `(x, base_y, s, phase, ghost, lunge)`).
- **Background creatures** (`bg_creatures`, `_populate_bg_creatures`,
  `_draw_bg_creature`) are transparent (stipple `gray50`), non-colliding
  atmosphere. Same `_draw_creature` with `ghost=True`.
- Jump-over is **exact** (not forgiving): the runner clears a hazard only while
  his feet are lifted `>= HAZARD_CLEAR` (40px) during overlap — see the `cleared`
  local in `_update_entities` (`jump_offset >= HAZARD_CLEAR`). A contact costs
  `HAZARD_DAMAGE` (0.5 = half a heart; health is a float, hearts render half via
  `_draw_heart(cx,cy,s,fill)` with fill in {0,0.5,1}) and triggers a `_draw_impact`
  "OUCH!" burst (`_hit_fx`/`_hit_fx_until`) plus the red border (`_hurt_until`).
  0 health → dead (6 hits). `JUMP_DUR=0.78`. Collect **10 coins** → rescue-dragon
  cutscene → shop → next stage.
- Music swap: `Game` holds `theme_medieval/zombie/cave/spider` paths + `self.music`
  (a `MusicPlayer`). `_update_music()` (called from `_reset_round`, i.e. on every
  stage change) calls `music.switch(path)` (no-op if already playing that track).

### To add a new stage/theme
1. New `<name>_music.py` mirroring the others: `synthesize()` (numpy, clean loop,
   normalize + `tanh` soft-clip), `write_wav`, `ensure_<name>_theme()`.
2. In `geophone_game.py`: add palette constants; extend `_theme()` and
   `_creature_stage()`; add branches in `_draw_static_background` (palette dict +
   celestial/ceiling), `_draw_cloud`, `_draw_mountain`, `_draw_ground_obj`
   (→ new `_draw_<name>_obj`); add `_draw_<creature>` + dispatch in
   `_draw_creature`; add death text in `_draw_overlay` and hazard word in
   `_draw_hud`; add `self.theme_<name>` init + `_update_music` entry.
3. In `main()`: `game.theme_<name> = str(ensure_<name>_theme())` (best-effort).
4. Update README stage list + music paragraph + controls hazard bullet.

## Recording

Every session is saved to an MP4 via `ScreenRecorder` (ffmpeg `x11grab`, separate
process = zero game-loop cost, `-draw_mouse 0` so no cursor). Best-effort; needs
`ffmpeg` + an X11 `DISPLAY`. Saved to the player's folder (timestamped) or
`last_game.mp4`. Options: `--record-file`, `--record-fps`, `--no-record`.
x11grab samples a **fixed** screen rectangle, so `main()` must sample the window
geometry only after it settles: it `resizable(False,False)` + `lift/focus_force`,
then loops until `winfo_rootx/y/width/height` is stable (two identical reads,
viewable) before `recorder.start(*geo)`. Sampling too early captured an offset
rectangle (part desktop, part game) — that was the bug.

## Testing recipe (no interactive play needed)

This environment has `DISPLAY=:1`, plus `ffmpeg` and `gs` (ghostscript). Drive the
real `Game` headlessly and screenshot via postscript → PNG:

```python
import time, tkinter as tk
from geophone_io import SimulatedSource
from motion_model import MotionClassifier
from geophone_game import Game
src = SimulatedSource(100.0); src.start()
clf = MotionClassifier("geophone_model.pt", 100.0, smooth=0.12)
root = tk.Tk(); g = Game(root, src, clf, 100.0, source_label="T")
g.stage = 4; g._reset_round(); g.space = True; g.tick()     # jump to a stage
for _ in range(60): root.update(); time.sleep(0.008)
g.barrels=[{"x":g.char_x+150,"spin":1.2,"hit":False,"cleared":False}]  # force a hazard
g.coins=[{"x":g.char_x+60,"y":g.ground_y-118,"phase":0.5,"got":False}]
g._render(); root.update_idletasks()
g.canvas.postscript(file="/tmp/x.ps", colormode="color")
src.stop(); root.destroy()
```
Then: `gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r96 -dEPSCrop -sOutputFile=/tmp/x.png /tmp/x.ps`
and Read the PNG. Run scripts from the project dir so imports resolve. Always
`python -m py_compile geophone_game.py geophone_io.py *_music.py` after edits.
For music: unit-test switching with a fake `music` object exposing `switch(path)`
and calling `g._advance_stage()`.

## Gotchas learned

- **Canvas z-order:** static-background items go in `_bg_ids` and are
  `tag_lower`ed under all `"frame"` items — do NOT put things there that must sit
  *above* the sky gradient (the spider corner cobwebs are drawn per-frame in
  `_render` instead, via `_draw_corner_webs`).
- **OnsetDetector:** peak-follower (instant attack, slow release) comparing
  transients to the recent step-peak; `self._peak` must update **every** sample
  (freezing it during the refractory window causes repeated false fires). Fires on
  jump take-off, not the later landing. Two gates: a **sustained-activity**
  envelope (`_activity`/`activity_tau`, floor `min_level`) means a jump only
  fires once you are *already walking* — this killed the "knight jumps when I
  start walking" bug (trade-off: a jump from a dead standstill is ignored); and a
  trained **absolute floor** `abs_level`. `from_model()` loads `jump_model.json`
  (keys: threshold/abs_level/min_level/refractory) — but only when the game is
  run with `--jump-model`; default jump detection is the CLI `--jump-threshold`
  (default 6.0). For an in-motion jump the fallback fires identically to the
  original detector (activity gate only changes the cold-start first step).
- **Knight shield:** bears the **Columns of Gediminas** (`_draw_columns`, gold
  bars, geometry traced from `/home/tank/opengdl/static/img/columns_gold.png`).
- **MusicPlayer.switch/restart:** `start()` calls `self._stop.clear()` so a
  loop-thread player can restart after `stop()`.
- Headless drives that `cd` into a scratchpad dir break imports — run from the
  project dir with an absolute script path.

#!/usr/bin/env python3
"""
Smart Geophone — Step Runner.

A little side-scrolling game (in the spirit of the Chrome "dinosaur" game) driven
by the geophone: the character WALKS forward while footsteps are detected on the
sensor stream, and stands still when they are not. As he walks, a 1-D world of
mountains, trees and rocks scrolls past with parallax.

It uses a small trained PyTorch classifier to distinguish stillness from
walking. Jumping is driven by a fast raw-stream onset detector (triggering on
the take-off push-off, not the laggy landing impact) and remains available from
the keyboard as well.

Examples
--------
    python geophone_game.py                 # auto-detect the serial port
    python geophone_game.py --port /dev/ttyUSB0
    python geophone_game.py --simulate      # no hardware: synthetic walker
    #   hold SPACE at any time to walk manually (handy for testing)
"""

import argparse
from pathlib import Path
import random
import sys
import time
import tkinter as tk

from geophone_io import (
    MusicPlayer, OnsetDetector, ScreenRecorder, SerialSource, SimulatedSource,
    auto_detect_port)
from medieval_music import ensure_theme
from zombie_music import ensure_zombie_theme
from cave_music import ensure_cave_theme
from spider_music import ensure_spider_theme
from motion_model import MotionClassifier
from player_profile import PlayerProfile


# ----------------------------------------------------------------------------
# Colours (simple flat "daytime" palette)
# ----------------------------------------------------------------------------
SKY_TOP = "#8ec9ef"
SKY_BOTTOM = "#dcefff"
SUN = "#ffd75a"
CLOUD = "#ffffff"
MOUNTAIN_FAR = "#9fb3c8"
MOUNTAIN_NEAR = "#7d97b0"
GROUND = "#7ec850"
GROUND_DARK = "#5aa63f"
DIRT = "#c9a06a"
TRUNK = "#8b5a2b"
PINE = "#2e7d32"
LEAF = "#43a047"
ROCK = "#9aa0a6"
ROCK_DARK = "#6f7479"
# -- the pursuer: an evil wizard --
WIZ_ROBE = "#3b2a63"
WIZ_ROBE_DARK = "#241542"
WIZ_HAT = "#241542"
WIZ_SKIN = "#d9c9a8"
WIZ_BEARD = "#eef0f6"
WIZ_EYE = "#ff3b3b"
WIZ_SHOE = "#1c1230"
WIZ_ORB = "#7cf5b0"
WIZ_ORB_CORE = "#e8fff3"
WIZ_SPELL = "#b061ff"
WIZ_STAR = "#f4c430"
# -- the hero: a knight --
KNIGHT_ARMOR = "#c7ccd6"
KNIGHT_ARMOR_DARK = "#8a94a6"
KNIGHT_TUNIC = "#8a1f2b"
KNIGHT_TUNIC_DARK = "#5e1420"
KNIGHT_PLUME = "#e23b3b"
SWORD = "#e8edf5"
SWORD_HILT = "#d4af37"
SHIELD = "#2f5aa8"
SHIELD_TRIM = "#d4af37"
SKIN = "#f2c79b"
SHIRT = "#3f6fd1"
LEGS = "#2b2b2b"
HUD_INK = "#1d2733"
COIN = "#f4c430"
COIN_EDGE = "#b8860b"
COIN_SHINE = "#fff3b0"
BARREL = "#a9682e"
BARREL_DARK = "#6e3f16"
BARREL_HOOP = "#d9a066"
HEART = "#e23b3b"
HEART_EMPTY = "#c9ced6"
# -- the rescue mount: a friendly dragon (replaces the old helicopter) --
DRAGON = "#3f9d5a"
DRAGON_DARK = "#256b39"
DRAGON_BELLY = "#cfe8b0"
DRAGON_WING = "#5cc47a"
DRAGON_WING_DARK = "#2f8a4d"
DRAGON_EYE = "#14261a"
DRAGON_HORN = "#f0e6c8"
DRAGON_FIRE = "#ff8a3c"
LADDER = "#8a5a2b"
# -- the between-stage shop --
SHOP_WOOD = "#8a5a2b"
SHOP_WOOD_DARK = "#5e3c1c"
SHOP_ROOF = "#7a1f2b"
SHOP_ROOF_DARK = "#571018"
SHOP_AWNING = "#c9a06a"
SHOP_SIGN = "#f4e4c1"
SHOP_PANEL = "#fff7e6"
SHOP_PANEL_DARK = "#e7d3ac"
SHOP_INK = "#3a2410"
SHOP_AFFORD = "#166534"
SHOP_TOODEAR = "#991b1b"
# -- Zombie Village (stage 2+): a murky, moonlit graveyard palette --
Z_SKY_TOP = "#232733"
Z_SKY_BOTTOM = "#55603f"       # sickly greenish horizon glow
Z_MOON = "#e9edd6"
Z_MOON_CRATER = "#c9cfb4"
Z_GROUND = "#464f36"
Z_GROUND_DARK = "#2f3724"
Z_HILL_FAR = "#39414c"
Z_HILL_NEAR = "#2c333d"
Z_FOG = "#7c8a6b"
Z_HUT = "#6b5236"
Z_HUT_DARK = "#453322"
Z_HUT_ROOF = "#53442c"
Z_HUT_WIN = "#d98a2b"          # a lone lit window
Z_GRAVE = "#9298a1"
Z_GRAVE_DARK = "#5f646d"
Z_DEADTREE = "#4a3d2c"
Z_DEADTREE_DARK = "#332a1d"
# -- zombies (attacking hazard + drifting background ghouls) --
ZOMB_SKIN = "#7fa86b"
ZOMB_SKIN_DARK = "#547046"
ZOMB_SHIRT = "#465262"
ZOMB_SHIRT_DARK = "#2c343f"
ZOMB_EYE = "#f2ec52"
ZOMB_MOUTH = "#241616"
ZOMB_GHOST = "#9fb58c"         # muted tone for the transparent background ghouls
# -- Cave (stage 3+): a dark, dripping underground palette --
C_SKY_TOP = "#141013"          # near-black cave ceiling
C_SKY_BOTTOM = "#2b2420"       # dim brown depths near the floor
C_GLOW = "#c8781f"             # a distant torch glow
C_GROUND = "#3a332c"
C_GROUND_DARK = "#221d18"
C_ROCK_FAR = "#332d2a"
C_ROCK_NEAR = "#26211f"
C_STALACTITE = "#4a4038"
C_STALACTITE_DARK = "#2e2721"
C_CRYSTAL = "#5fb3c9"
C_CRYSTAL_DARK = "#2f6f80"
# Coloured crystal gems (fill, dark edge, bright tip) — blue, purple, red, green.
C_CRYSTALS = [
    ("#4d7fff", "#26407f", "#bcd0ff"),   # blue
    ("#a15cff", "#4f2b80", "#e2ccff"),   # purple
    ("#ff4d5e", "#7a232f", "#ffc7cd"),   # red
    ("#48d17e", "#22693f", "#c4f4d7"),   # green
]
C_BONE = "#d8d2be"
C_BONE_DARK = "#9a9075"
C_BAT = "#160f14"
# -- skeleton (attacking hazard + drifting background bones) --
SKEL_BONE = "#e8e3d1"
SKEL_BONE_DARK = "#a49d84"
SKEL_EYE = "#e0532a"           # glowing eye socket
SKEL_GHOST = "#b9c2c8"         # muted tone for the transparent background bones
# -- Spider Cave (stage 4+): a webbed, purple-black lair palette --
S_SKY_TOP = "#0f0b14"          # near-black webbed ceiling
S_SKY_BOTTOM = "#241b2e"       # dim purple depths
S_GLOW = "#3a2f45"             # faint sickly glow
S_GROUND = "#2e2733"
S_GROUND_DARK = "#191420"
S_ROCK_FAR = "#2b2433"
S_ROCK_NEAR = "#1f1a27"
S_WEB = "#c9c6d6"              # spider-silk web
S_EGG = "#d9d3c2"             # egg sac / cocoon silk
S_EGG_DARK = "#a39b83"
# -- spiders (attacking hazard + drifting background crawlers) --
SPID_BODY = "#251a2e"
SPID_BODY_DARK = "#120b18"
SPID_LEG = "#181020"
SPID_MARK = "#d43a3a"          # red hourglass marking
SPID_EYE = "#e6d84a"           # cluster of glowing eyes
SPID_GHOST = "#6f6280"         # muted tone for the transparent background spiders


# ----------------------------------------------------------------------------
# Parallax scenery layer
# ----------------------------------------------------------------------------
class Layer:
    """A horizontal band of world objects that scrolls at `parallax` speed."""

    def __init__(self, parallax, gap, spawn_fn):
        self.parallax = parallax
        self.gap = gap                # (min, max) world-px between objects
        self.spawn_fn = spawn_fn      # wx -> object dict
        self.offset = 0.0
        self.items = []
        self.next_wx = 0.0

    def populate(self, width):
        self.offset = 0.0
        self.items = []
        wx = random.uniform(0, self.gap[0])
        while wx < width + self.gap[1]:
            self.items.append(self.spawn_fn(wx))
            wx += random.uniform(*self.gap)
        self.next_wx = wx

    def advance(self, dx):
        self.offset += dx * self.parallax

    def maintain(self, width):
        self.items = [it for it in self.items
                      if it["wx"] - self.offset > -500]
        while self.next_wx - self.offset < width + self.gap[1]:
            self.items.append(self.spawn_fn(self.next_wx))
            self.next_wx += random.uniform(*self.gap)

    def screen_x(self, it):
        return it["wx"] - self.offset


# ----------------------------------------------------------------------------
# Object factories
# ----------------------------------------------------------------------------
def make_cloud(wx):
    return {"wx": wx, "y": random.uniform(30, 120), "s": random.uniform(0.7, 1.4)}


def make_mountain(wx):
    return {"wx": wx, "w": random.uniform(220, 420), "h": random.uniform(80, 190),
            "near": random.random() < 0.5}


def make_ground_obj(wx):
    r = random.random()
    if r < 0.55:
        kind = "pine" if random.random() < 0.5 else "tree"
    else:
        kind = "rock"
    return {"wx": wx, "kind": kind, "s": random.uniform(0.8, 1.3),
            "cross": random.random() < 0.5, "gem": random.randint(0, 3)}


# ----------------------------------------------------------------------------
# The game
# ----------------------------------------------------------------------------
class Game:
    WIDTH = 900
    HEIGHT = 420
    SPEED = 230.0          # world px / second while walking
    STRIDE = 46.0          # world px per half-stride (leg-swing cadence)
    PX_PER_M = 90.0        # world px -> metres, for the distance readout
    FPS = 30
    JUMP_DUR = 0.78        # seconds a jump lasts (25% longer airtime)
    JUMP_HEIGHT = 92       # px peak jump height
    DINO_CHASE_SPEED = 72.0
    DINO_FALLBACK_SPEED = 105.0

    # Collectibles / hazards
    COINS_TO_CLEAR = 10        # coins needed to pass the stage
    MAX_HEALTH = 3
    COIN_R = 12                # coin radius (px)
    COIN_GAP = (240.0, 420.0)  # world-px between coins
    BARREL_R = 18              # rolling-barrel radius (px)
    BARREL_ROLL = 95.0         # extra leftward roll speed (px/s)
    BARREL_GAP = (520.0, 900.0)  # world-px between barrels
    GUY_H = 78                 # runner height for collision (feet->head)
    HAZARD_CLEAR = 40          # px the feet must be lifted to clear a hazard
    HAZARD_DAMAGE = 0.5        # hearts lost per hazard contact (half a heart)

    # Stage-clear pickup cutscene geometry
    HELI_HALF_H = 20           # helicopter body half-height (px)
    LADDER_LEN = 92            # rope-ladder length (px)
    GRAB_LIFT = 95             # how high the runner climbs onto the ladder (px)

    def __init__(self, root, source, classifier, fs, confidence=0.55,
                 baud=None, source_label="", jump_detector=None, profile=None):
        self.root = root
        self.source = source
        self.classifier = classifier
        self.jump_detector = jump_detector
        self.fs = fs                    # measured live below
        self.confidence = confidence
        self.baud = baud
        self.source_label = source_label

        # Player profile: personal folder for high scores and saved videos.
        self.profile = profile
        self.player_name = profile.name if profile else "Player"
        self.best_score = profile.best() if profile else None

        self.canvas = tk.Canvas(root, width=self.WIDTH, height=self.HEIGHT,
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.ground_y = int(self.HEIGHT * 0.80)
        self.char_x = int(self.WIDTH * 0.20)

        self.clouds = Layer(0.15, (240, 520), make_cloud)
        self.mountains = Layer(0.30, (280, 620), make_mountain)
        self.ground = Layer(1.00, (150, 340), make_ground_obj)
        for layer in (self.clouds, self.mountains, self.ground):
            layer.populate(self.WIDTH)

        self.world_x = 0.0
        self.distance_m = 0.0
        self.walk_phase = 0.0
        self.walking = False
        self.space = False
        self.eaten = False
        self.dino_x = self.char_x - 180.0
        self.dino_phase = 0.0
        self._dino_grace_until = (
            time.time() + getattr(classifier, "window_s", 1.5) + 1.0)
        self._last = time.time()

        # Jump state (manual keyboard control).
        self.jump_t = None         # seconds into the current jump, or None
        self.jump_offset = 0.0     # current vertical lift (px)
        self.jumps = 0
        self._cooldown_until = 0.0

        # Coins (jump to grab), rolling barrels (jump over), health and stage.
        self.coins = []            # {x, y, phase, got}
        self.barrels = []          # {x, spin, hit}
        self.coins_collected = 0
        self.coins_bank = 0        # total coins kept as shop currency
        self.health = self.MAX_HEALTH
        self.dead = False          # ran out of health
        self.stage_cleared = False
        self.stage = 1

        # Between-stage shop: after the rescue cutscene the knight arrives at a
        # shop where collected coins buy upgrades (bigger jumps, more health).
        # Upgrades bump the instance copies of the JUMP_HEIGHT / JUMP_DUR /
        # MAX_HEALTH class constants, so they persist for the rest of the run.
        self.shop_open = False
        self.shop_msg = ""
        self._shop_item_rects = []   # (x0,y0,x1,y1,item) clickable rows
        self.shop_items = [
            {"key": "1", "name": "Higher Jump", "desc": "+22 px jump height",
             "base_cost": 6, "step": 4, "count": 0, "max": 5,
             "apply": self._buy_higher_jump},
            {"key": "2", "name": "Extra Heart", "desc": "+1 max health",
             "base_cost": 8, "step": 6, "count": 0, "max": 3,
             "apply": self._buy_extra_heart},
            {"key": "3", "name": "Full Heal", "desc": "refill all hearts",
             "base_cost": 3, "step": 0, "count": 0, "max": None,
             "apply": self._buy_full_heal},
        ]
        self._hurt_until = 0.0     # red flash timer after a hazard hit
        self._hit_fx_until = 0.0   # impact-burst effect timer after a hit
        self._hit_fx = (0.0, 0.0)  # screen position of the impact burst
        self._next_coin_wx = random.uniform(*self.COIN_GAP)
        self._next_barrel_wx = random.uniform(700.0, 1100.0)

        # Themed stages (2 = Zombie Village, 3+ = Cave) add purely decorative
        # creatures shuffling in the background; they do not collide with
        # anything -- they are atmosphere. The attacking hazard reuses the
        # `barrels` list (drawn as a lunging zombie / skeleton on those stages).
        self.bg_creatures = []

        # Per-stage background music (set from main(); best-effort). When the
        # stage theme changes the looping track is swapped.
        self.music = None
        self.theme_medieval = None
        self.theme_zombie = None
        self.theme_cave = None
        self.theme_spider = None

        # Stage-clear pickup cutscene: a friendly dragon with a rope ladder
        # flies in, the knight boards it and is carried off into the next stage.
        self.cutscene = None       # None | "in" | "board" | "out"
        self.heli_x = 0.0
        self.heli_y = 0.0
        self.heli_rotor = 0.0
        self.guy_lift = 0.0        # how high the runner has climbed the ladder
        self.guy_on_ladder = False
        self._board_t = 0.0

        # live sample-rate meter (the stream can run much faster than 100 Hz)
        self.meas_fs = fs
        self._samp_count = 0
        self._samp_t0 = None

        self._bg_ids = []
        self._draw_static_background()
        self._populate_bg_creatures()

        root.bind("<KeyPress-space>", lambda e: setattr(self, "space", True))
        root.bind("<KeyRelease-space>", lambda e: setattr(self, "space", False))
        root.bind("<Up>", lambda e: self.start_jump())
        root.bind("j", lambda e: self.start_jump())
        root.bind("r", lambda e: self.restart())
        root.bind("<Escape>", lambda e: root.destroy())
        root.bind("q", lambda e: root.destroy())
        # Shop controls: number keys buy an item, Enter leaves for the next stage.
        for _k in ("1", "2", "3"):
            root.bind(_k, lambda e: self._shop_buy(e.char))
        root.bind("<Return>", lambda e: self._shop_continue())
        root.bind("n", lambda e: self._shop_continue())
        # Mouse: click a shop item row to buy it.
        self.canvas.bind("<Button-1>", self._on_shop_click)
        self.canvas.bind("<Configure>", self._on_resize)

    # -- stage theme ----------------------------------------------------------
    def _theme(self):
        """The visual/audio theme for the current stage."""
        if self.stage >= 4:
            return "spider"
        if self.stage == 3:
            return "cave"
        if self.stage == 2:
            return "zombie"
        return "medieval"

    def _creature_stage(self):
        """True on stages that have background creatures + creature hazards."""
        return self._theme() in ("zombie", "cave", "spider")

    def _make_bg_creature(self, x):
        """A decorative, non-colliding background creature at screen-x `x`."""
        return {"x": float(x),
                "s": random.uniform(0.55, 0.82),
                "phase": random.uniform(0.0, 6.28),
                "sway": random.uniform(0.5, 1.1)}

    def _populate_bg_creatures(self):
        """(Re)fill the background with drifting creatures, or clear off-theme."""
        self.bg_creatures = []
        if not self._creature_stage():
            return
        x = random.uniform(40.0, 160.0)
        while x < self.WIDTH + 80.0:
            self.bg_creatures.append(self._make_bg_creature(x))
            x += random.uniform(150.0, 300.0)

    def _update_music(self):
        """Swap the looping soundtrack to match the current stage theme."""
        if self.music is None:
            return
        want = {"cave": self.theme_cave,
                "zombie": self.theme_zombie,
                "spider": self.theme_spider}.get(self._theme(),
                                                 self.theme_medieval)
        if want is None:
            return
        try:
            self.music.switch(want)
        except Exception:
            pass

    # -- background (sky gradient + ground band), redrawn only on resize ------
    def _on_resize(self, event):
        if event.width < 50:
            return
        self.WIDTH, self.HEIGHT = event.width, event.height
        self.ground_y = int(self.HEIGHT * 0.80)
        self.char_x = int(self.WIDTH * 0.20)
        self._draw_static_background()

    def _corner_web(self, ox, oy, r, ang0, ang1, spokes=6):
        """A cobweb anchored at corner (ox, oy): radial spokes + connecting
        threads, drawn as (static) per-frame scenery."""
        import math
        angs = [ang0 + (ang1 - ang0) * i / (spokes - 1) for i in range(spokes)]
        for a in angs:
            self.canvas.create_line(
                ox, oy, ox + r * math.cos(a), oy + r * math.sin(a),
                fill=S_WEB, width=1, tags="frame")
        for rr in (r * 0.32, r * 0.58, r * 0.84):
            pts = []
            for a in angs:
                pts += [ox + rr * math.cos(a), oy + rr * math.sin(a)]
            self.canvas.create_line(*pts, fill=S_WEB, width=1, tags="frame")

    def _draw_corner_webs(self):
        """Big cobwebs strung across the top corners of the spider lair."""
        import math
        r = min(self.WIDTH * 0.32, 260)
        self._corner_web(0, 0, r, 0.0, math.pi / 2)
        self._corner_web(self.WIDTH, 0, r, math.pi / 2, math.pi)

    def _draw_static_background(self):
        for i in self._bg_ids:
            self.canvas.delete(i)
        self._bg_ids = []
        theme = self._theme()
        sky_top, sky_bottom, ground, ground_dark = {
            "zombie": (Z_SKY_TOP, Z_SKY_BOTTOM, Z_GROUND, Z_GROUND_DARK),
            "cave": (C_SKY_TOP, C_SKY_BOTTOM, C_GROUND, C_GROUND_DARK),
            "spider": (S_SKY_TOP, S_SKY_BOTTOM, S_GROUND, S_GROUND_DARK),
        }.get(theme, (SKY_TOP, SKY_BOTTOM, GROUND, GROUND_DARK))
        # vertical sky gradient (contiguous bands, no gaps)
        steps = 60
        r1, g1, b1 = self.root.winfo_rgb(sky_top)
        r2, g2, b2 = self.root.winfo_rgb(sky_bottom)
        for k in range(steps):
            t = k / (steps - 1)
            col = "#%02x%02x%02x" % (
                int((r1 + (r2 - r1) * t) / 256),
                int((g1 + (g2 - g1) * t) / 256),
                int((b1 + (b2 - b1) * t) / 256))
            y0 = round(self.ground_y * k / steps)
            y1 = round(self.ground_y * (k + 1) / steps)
            self._bg_ids.append(self.canvas.create_rectangle(
                0, y0, self.WIDTH, y1, fill=col, outline=col))
        if theme == "zombie":
            # a pale full moon with a couple of craters
            mx0, my0 = self.WIDTH - 135, 34
            self._bg_ids.append(self.canvas.create_oval(
                mx0, my0, mx0 + 78, my0 + 78, fill=Z_MOON, width=0))
            for cx, cy, cr in ((26, 24, 9), (52, 44, 6), (34, 56, 5)):
                self._bg_ids.append(self.canvas.create_oval(
                    mx0 + cx - cr, my0 + cy - cr, mx0 + cx + cr, my0 + cy + cr,
                    fill=Z_MOON_CRATER, width=0))
        elif theme == "cave":
            # a jagged ceiling of stalactites and a distant torch glow
            self._bg_ids.append(self.canvas.create_oval(
                self.WIDTH - 160, 8, self.WIDTH - 40, 120, fill=C_GLOW,
                stipple="gray12", width=0))
            n = 11
            for i in range(n):
                sx = self.WIDTH * i / (n - 1)
                h = 16 + (i * 37 % 5) * 8      # deterministic, varied lengths
                self._bg_ids.append(self.canvas.create_polygon(
                    sx - 13, 0, sx + 13, 0, sx, h, fill=C_STALACTITE,
                    outline=C_STALACTITE_DARK, width=1))
        else:
            # sun
            self._bg_ids.append(self.canvas.create_oval(
                self.WIDTH - 130, 40, self.WIDTH - 60, 110, fill=SUN, width=0))
        # ground
        self._bg_ids.append(self.canvas.create_rectangle(
            0, self.ground_y, self.WIDTH, self.HEIGHT, fill=ground, width=0))
        self._bg_ids.append(self.canvas.create_rectangle(
            0, self.ground_y, self.WIDTH, self.ground_y + 8, fill=ground_dark,
            width=0))
        for i in self._bg_ids:
            self.canvas.tag_lower(i)

    # -- per-frame drawing ----------------------------------------------------
    def _draw_cloud(self, x, it):
        y, s = it["y"], it["s"]
        theme = self._theme()
        if theme == "spider":
            # a little spider dangling from a silk thread off the ceiling
            import math
            y = y + math.sin(x * 0.04) * 6
            self.canvas.create_line(x, 0, x, y, fill=S_WEB, width=1,
                                    tags="frame")
            r = 4.5 * s
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=SPID_BODY,
                                    outline=SPID_BODY_DARK, width=1,
                                    tags="frame")
            for dx in (-2.0, -1.0, 1.0, 2.0):
                self.canvas.create_line(x, y, x + dx * r, y + r * 1.5,
                                        fill=SPID_LEG, width=1, tags="frame")
            return
        if theme == "cave":
            # a little flitting bat instead of a cloud
            import math
            bob = math.sin(x * 0.05) * 4
            y = y + bob
            b = 5 * s
            self.canvas.create_oval(x - b * 0.5, y - b * 0.4, x + b * 0.5,
                                    y + b * 0.5, fill=C_BAT, width=0,
                                    tags="frame")
            self.canvas.create_polygon(
                x, y, x - b * 2.2, y - b, x - b * 1.3, y + b * 0.3,
                fill=C_BAT, width=0, tags="frame")
            self.canvas.create_polygon(
                x, y, x + b * 2.2, y - b, x + b * 1.3, y + b * 0.3,
                fill=C_BAT, width=0, tags="frame")
            return
        r = 18 * s
        zombie = theme == "zombie"
        col = Z_FOG if zombie else CLOUD
        extra = {"stipple": "gray50"} if zombie else {}
        for dx, dy, rr in [(-r, 0, r), (0, -r * 0.6, r * 1.2), (r, 0, r),
                           (0, r * 0.2, r)]:
            self.canvas.create_oval(x + dx - rr, y + dy - rr,
                                    x + dx + rr, y + dy + rr,
                                    fill=col, width=0, tags="frame", **extra)

    def _draw_mountain(self, x, it):
        w, h = it["w"], it["h"]
        base = self.ground_y
        theme = self._theme()
        if theme == "spider":
            # dark rock spires draped with sheets of web
            col = S_ROCK_NEAR if it["near"] else S_ROCK_FAR
            self.canvas.create_polygon(
                x - w / 2, base, x - w * 0.18, base - h * 0.85,
                x, base - h, x + w * 0.2, base - h * 0.8,
                x + w / 2, base, fill=col, width=0, tags="frame")
            # a couple of slack web strands draped down the near side
            for dx in (-w * 0.22, w * 0.12):
                self.canvas.create_line(
                    x + dx, base - h * 0.7, x + dx * 0.5, base - h * 0.35,
                    x + dx * 0.9, base, fill=S_WEB, width=1, smooth=True,
                    tags="frame")
            return
        if theme == "cave":
            # big dark rock formations / stone spires rising from the floor
            col = C_ROCK_NEAR if it["near"] else C_ROCK_FAR
            self.canvas.create_polygon(
                x - w / 2, base, x - w * 0.18, base - h * 0.85,
                x, base - h, x + w * 0.2, base - h * 0.8,
                x + w / 2, base, fill=col, width=0, tags="frame")
            return
        if theme == "zombie":
            # bleak, bare hills silhouetted against the moon (no snow cap)
            col = Z_HILL_NEAR if it["near"] else Z_HILL_FAR
            self.canvas.create_polygon(
                x - w / 2, base, x - w * 0.12, base - h,
                x + w * 0.08, base - h * 0.9, x + w / 2, base,
                fill=col, width=0, tags="frame")
            return
        col = MOUNTAIN_NEAR if it["near"] else MOUNTAIN_FAR
        self.canvas.create_polygon(
            x - w / 2, base, x, base - h, x + w / 2, base,
            fill=col, width=0, tags="frame")
        # snow cap
        cap = h * 0.28
        self.canvas.create_polygon(
            x - w * 0.14, base - h + cap, x, base - h,
            x + w * 0.14, base - h + cap, fill="#f4f8fb", width=0, tags="frame")

    def _draw_ground_obj(self, x, it):
        theme = self._theme()
        if theme == "spider":
            self._draw_spider_obj(x, it)
            return
        if theme == "cave":
            self._draw_cave_obj(x, it)
            return
        if theme == "zombie":
            self._draw_village_obj(x, it)
            return
        s = it["s"]
        g = self.ground_y
        if it["kind"] == "rock":
            w, h = 34 * s, 20 * s
            self.canvas.create_polygon(
                x - w / 2, g, x - w * 0.3, g - h, x + w * 0.1, g - h * 1.1,
                x + w / 2, g, fill=ROCK, outline=ROCK_DARK, width=2,
                tags="frame")
        elif it["kind"] == "pine":
            th = 46 * s
            self.canvas.create_rectangle(
                x - 5, g - 12, x + 5, g, fill=TRUNK, width=0, tags="frame")
            for lvl in range(3):
                yb = g - 12 - lvl * th * 0.28
                wv = (30 - lvl * 6) * s
                self.canvas.create_polygon(
                    x - wv, yb, x, yb - th * 0.5, x + wv, yb,
                    fill=PINE, width=0, tags="frame")
        else:  # round tree
            th = 50 * s
            self.canvas.create_rectangle(
                x - 6, g - th * 0.5, x + 6, g, fill=TRUNK, width=0, tags="frame")
            r = 26 * s
            top = g - th * 0.5
            for dx, dy in [(-r * 0.6, 0), (r * 0.6, 0), (0, -r * 0.7)]:
                self.canvas.create_oval(
                    x + dx - r, top + dy - r, x + dx + r, top + dy + r,
                    fill=LEAF, width=0, tags="frame")

    def _draw_village_obj(self, x, it):
        """Zombie-Village scenery: gravestones, dead trees and ruined huts."""
        s = it["s"]
        g = self.ground_y
        kind = it["kind"]
        if kind == "rock":                 # -> gravestone
            w, h = 26 * s, 40 * s
            if it.get("cross"):
                self.canvas.create_line(x, g, x, g - h, fill=Z_GRAVE, width=6,
                                        tags="frame")
                self.canvas.create_line(x - w * 0.5, g - h * 0.72,
                                        x + w * 0.5, g - h * 0.72, fill=Z_GRAVE,
                                        width=6, tags="frame")
                self.canvas.create_line(x, g, x, g - h, fill=Z_GRAVE_DARK,
                                        width=2, tags="frame")
            else:
                # rounded headstone
                self.canvas.create_rectangle(
                    x - w / 2, g - h * 0.7, x + w / 2, g, fill=Z_GRAVE,
                    outline=Z_GRAVE_DARK, width=2, tags="frame")
                self.canvas.create_arc(
                    x - w / 2, g - h, x + w / 2, g - h * 0.4, start=0,
                    extent=180, fill=Z_GRAVE, outline=Z_GRAVE_DARK, width=2,
                    tags="frame")
                self.canvas.create_text(x, g - h * 0.42, text="R.I.P.",
                                        fill=Z_GRAVE_DARK,
                                        font=("TkDefaultFont", max(6, int(7 * s))),
                                        tags="frame")
        elif kind == "pine":               # -> bare dead tree
            th = 66 * s
            top = g - th
            self.canvas.create_line(x, g, x, top, fill=Z_DEADTREE,
                                    width=max(3, int(7 * s)), capstyle="round",
                                    tags="frame")
            for frac, dx, up in ((0.55, -1, 0.30), (0.68, 1, 0.24),
                                 (0.82, -1, 0.18), (0.9, 1, 0.14)):
                by = g - th * frac
                self.canvas.create_line(
                    x, by, x + dx * 22 * s, by - th * up, fill=Z_DEADTREE,
                    width=max(2, int(4 * s)), capstyle="round", tags="frame")
        else:                              # -> ruined hut with a lit window
            w, h = 60 * s, 42 * s
            left, right = x - w / 2, x + w / 2
            top = g - h
            self.canvas.create_rectangle(left, top, right, g, fill=Z_HUT,
                                         outline=Z_HUT_DARK, width=2,
                                         tags="frame")
            # sagging plank shading
            self.canvas.create_line(left, top + h * 0.5, right, top + h * 0.5,
                                    fill=Z_HUT_DARK, width=1, tags="frame")
            # roof
            self.canvas.create_polygon(
                left - 6 * s, top, right + 6 * s, top, x + 4 * s, top - 22 * s,
                fill=Z_HUT_ROOF, outline=Z_HUT_DARK, width=2, tags="frame")
            # a single ominous lit window
            wx, wy = x + w * 0.14, top + h * 0.32
            ww = 10 * s
            self.canvas.create_rectangle(wx - ww, wy - ww, wx + ww, wy + ww,
                                         fill=Z_HUT_WIN, outline=Z_HUT_DARK,
                                         width=2, tags="frame")
            self.canvas.create_line(wx, wy - ww, wx, wy + ww, fill=Z_HUT_DARK,
                                    width=1, tags="frame")
            # dark doorway
            self.canvas.create_rectangle(
                left + w * 0.16, g - h * 0.5, left + w * 0.36, g,
                fill=Z_HUT_DARK, width=0, tags="frame")

    def _draw_cave_obj(self, x, it):
        """Cave scenery: stalagmites, crystal clusters, boulders and bone piles."""
        s = it["s"]
        g = self.ground_y
        kind = it["kind"]
        if kind == "pine":                 # -> stalagmite rising from the floor
            h, w = 60 * s, 26 * s
            self.canvas.create_polygon(
                x - w / 2, g, x - w * 0.15, g - h * 0.6, x, g - h,
                x + w * 0.2, g - h * 0.55, x + w / 2, g, fill=C_STALACTITE,
                outline=C_STALACTITE_DARK, width=2, tags="frame")
            self.canvas.create_line(x - 2 * s, g - h * 0.3, x + 3 * s,
                                    g - h * 0.7, fill=C_STALACTITE_DARK,
                                    width=2, tags="frame")
        elif kind == "rock" and it.get("cross"):   # -> glowing crystal cluster
            # Each shard takes a different gem colour (blue/purple/red/green),
            # rotating from this cluster's stable `gem` seed so the cave glows
            # with all four hues.
            gem0 = it.get("gem", 0)
            shards = ((-9 * s, 30 * s, 7 * s), (3 * s, 46 * s, 8 * s),
                      (15 * s, 24 * s, 6 * s))
            for i, (dx, hh, ww) in enumerate(shards):
                fill, edge, tip = C_CRYSTALS[(gem0 + i) % len(C_CRYSTALS)]
                self.canvas.create_polygon(
                    x + dx - ww, g, x + dx - ww * 0.4, g - hh * 0.7,
                    x + dx, g - hh, x + dx + ww * 0.4, g - hh * 0.7,
                    x + dx + ww, g, fill=fill, outline=edge, width=1,
                    tags="frame")
                # a bright faceted highlight running up to the tip
                self.canvas.create_line(x + dx, g - hh * 0.15, x + dx,
                                        g - hh * 0.95, fill=tip, width=1,
                                        tags="frame")
                self.canvas.create_polygon(
                    x + dx, g - hh, x + dx - ww * 0.4, g - hh * 0.7,
                    x + dx, g - hh * 0.5, fill=tip, outline="", tags="frame")
        elif kind == "rock":               # -> boulder
            w, h = 44 * s, 28 * s
            self.canvas.create_oval(x - w / 2, g - h, x + w / 2, g + 6,
                                    fill=C_ROCK_FAR, outline=C_ROCK_NEAR,
                                    width=2, tags="frame")
            self.canvas.create_arc(x - w * 0.36, g - h * 0.9, x + w * 0.1,
                                   g - h * 0.2, start=40, extent=120,
                                   style="arc", outline=C_ROCK_NEAR, width=2,
                                   tags="frame")
        else:                              # -> a pile of bones and skulls
            for dx in (-15 * s, 15 * s):
                self.canvas.create_line(x - 16 * s + dx * 0.1, g - 4,
                                        x + 16 * s + dx * 0.1, g - 2,
                                        fill=C_BONE, width=max(2, int(4 * s)),
                                        capstyle="round", tags="frame")
            for sx in (-9 * s, 8 * s):
                self.canvas.create_oval(x + sx - 7 * s, g - 16 * s, x + sx + 7 * s,
                                        g - 2 * s, fill=C_BONE,
                                        outline=C_BONE_DARK, width=1,
                                        tags="frame")
                for ox in (-4 * s, 1 * s):
                    self.canvas.create_oval(
                        x + sx + ox, g - 12 * s, x + sx + ox + 3 * s, g - 9 * s,
                        fill=C_BONE_DARK, width=0, tags="frame")

    def _draw_spider_obj(self, x, it):
        """Spider-lair scenery: egg-sac clusters, silk-wrapped cocoons and
        web-draped rock columns."""
        s = it["s"]
        g = self.ground_y
        kind = it["kind"]
        if kind == "rock":                 # -> a cluster of egg sacs
            for dx, r in ((-9 * s, 8 * s), (8 * s, 7 * s), (-1 * s, 6 * s),
                          (2 * s, 5 * s)):
                cyc = g - r
                self.canvas.create_oval(x + dx - r, cyc - r, x + dx + r,
                                        cyc + r, fill=S_EGG, outline=S_EGG_DARK,
                                        width=1, tags="frame")
                self.canvas.create_arc(x + dx - r * 0.6, cyc - r * 0.6,
                                       x + dx + r * 0.2, cyc + r * 0.2,
                                       start=40, extent=120, style="arc",
                                       outline=S_EGG_DARK, width=1,
                                       tags="frame")
        elif kind == "pine":               # -> web-draped rock column
            h, w = 54 * s, 24 * s
            self.canvas.create_polygon(
                x - w / 2, g, x, g - h, x + w / 2, g, fill=S_ROCK_NEAR,
                outline=S_ROCK_FAR, width=2, tags="frame")
            for dx in (-w * 0.3, w * 0.22):
                self.canvas.create_line(x + dx, g - h * 0.82, x + dx * 0.3,
                                        g - h * 0.4, x + dx * 0.7, g,
                                        fill=S_WEB, width=1, smooth=True,
                                        tags="frame")
        else:                              # -> a silk-wrapped cocoon of prey
            w, h = 18 * s, 42 * s
            top = g - h
            self.canvas.create_line(x, top - 20 * s, x, top, fill=S_WEB,
                                    width=1, tags="frame")     # anchor thread
            self.canvas.create_oval(x - w / 2, top, x + w / 2, g, fill=S_EGG,
                                    outline=S_EGG_DARK, width=1, tags="frame")
            yy = top + 6 * s
            while yy < g:
                self.canvas.create_line(x - w / 2, yy, x + w / 2, yy - 4 * s,
                                        fill=S_WEB, width=1, tags="frame")
                yy += 9 * s

    def _draw_creature(self, x, base_y, s, phase, ghost=False, lunge=0.0):
        """Draw the themed creature: spider, skeleton (cave), or zombie."""
        theme = self._theme()
        if theme == "spider":
            self._draw_spider(x, base_y, s, phase, ghost=ghost, lunge=lunge)
        elif theme == "cave":
            self._draw_skeleton(x, base_y, s, phase, ghost=ghost, lunge=lunge)
        else:
            self._draw_zombie(x, base_y, s, phase, ghost=ghost, lunge=lunge)

    def _draw_skeleton(self, x, base_y, s, phase, ghost=False, lunge=0.0):
        """A shambling skeleton. `ghost` draws a transparent background bone-walker
        (stippled, hollow sockets); otherwise a solid, lunging attacker."""
        import math
        st = {"stipple": "gray50"} if ghost else {}
        bone = SKEL_GHOST if ghost else SKEL_BONE
        bone_edge = SKEL_GHOST if ghost else SKEL_BONE_DARK
        g = base_y
        sway = math.sin(phase)

        def w(px):
            return max(2, int(px * s))

        hip_y = g - 30 * s
        # stiff bony legs (thigh + shin)
        for sign in (1, -1):
            foot_dx = sign * 6 * s + (sway * 5 * s) * (1 if sign > 0 else -1)
            knee_x = x + sign * 4 * s
            self.canvas.create_line(x, hip_y, knee_x, hip_y + 14 * s,
                                    x + foot_dx, g, fill=bone, width=w(4),
                                    capstyle="round", joinstyle="round",
                                    tags="frame", **st)
        shoulder_y = hip_y - 28 * s
        # spine
        self.canvas.create_line(x, hip_y, x, shoulder_y, fill=bone, width=w(4),
                                capstyle="round", tags="frame", **st)
        # rib cage
        for i in range(3):
            ry = shoulder_y + 6 * s + i * 7 * s
            rw = (9 - i * 1.5) * s
            self.canvas.create_line(x - rw, ry, x + rw, ry, fill=bone,
                                    width=w(2), tags="frame", **st)
        # both arms outstretched forward (to the left, toward the runner)
        arm_y = shoulder_y + 3 * s
        reach = (16 + lunge * 12) * s
        self.canvas.create_line(x, shoulder_y + 2 * s, x - reach,
                                arm_y + 3 * s * sway, fill=bone, width=w(3),
                                capstyle="round", tags="frame", **st)
        self.canvas.create_line(x, shoulder_y + 2 * s, x - reach * 0.6,
                                arm_y - 5 * s, fill=bone, width=w(3),
                                capstyle="round", tags="frame", **st)
        # skull + jaw
        hr = 10 * s
        hx = x - 2 * s
        hy = shoulder_y - hr
        self.canvas.create_oval(hx - hr, hy - hr, hx + hr, hy + hr * 0.9,
                                fill=bone, outline=bone_edge, width=1,
                                tags="frame", **st)
        self.canvas.create_rectangle(hx - hr * 0.5, hy + hr * 0.7, hx + hr * 0.5,
                                     hy + hr * 1.2, fill=bone, outline=bone_edge,
                                     width=1, tags="frame", **st)
        if not ghost:
            for ex in (-4, 4):
                self.canvas.create_oval(
                    hx + ex * s - 3 * s, hy - 3 * s, hx + ex * s + 2 * s,
                    hy + 2 * s, fill=SKEL_EYE, width=0, tags="frame")
            self.canvas.create_line(hx, hy + 1 * s, hx, hy + 6 * s,
                                    fill=SKEL_BONE_DARK, width=w(1),
                                    tags="frame")

    def _draw_spider(self, x, base_y, s, phase, ghost=False, lunge=0.0):
        """A skittering spider. `ghost` draws a transparent background crawler
        (stippled, eyeless); otherwise a solid attacker with a red marking and
        a cluster of glowing eyes. It faces left, toward the runner."""
        import math
        st = {"stipple": "gray50"} if ghost else {}
        body = SPID_GHOST if ghost else SPID_BODY
        body_edge = SPID_GHOST if ghost else SPID_BODY_DARK
        leg = SPID_GHOST if ghost else SPID_LEG
        g = base_y
        cy = g - 15 * s + math.sin(phase * 2.0) * 2.0 * s

        def w(px):
            return max(1, int(px * s))

        # Eight arched legs: four per side, each a two-jointed line that lifts
        # in turn so the spider looks like it is skittering.
        for side in (-1, 1):
            for k in range(4):
                ph = phase * 4.0 + k * 1.1 + (0.0 if side < 0 else 0.55)
                lift = max(0.0, math.sin(ph)) * 5.0 * s
                spread = (11 + k * 8) * s + (lunge * 6 * s if side < 0 else 0.0)
                foot_x = x + side * spread
                knee_x = x + side * spread * 0.5
                knee_y = cy - (11 - k * 1.6) * s          # arch up above the body
                self.canvas.create_line(
                    x + side * 4 * s, cy, knee_x, knee_y, foot_x, g - lift,
                    fill=leg, width=w(2), capstyle="round", joinstyle="round",
                    tags="frame", **st)

        # Abdomen (bulbous, to the rear/right) and cephalothorax (front/left).
        abx = x + 6 * s
        self.canvas.create_oval(abx - 11 * s, cy - 9 * s, abx + 13 * s,
                                cy + 11 * s, fill=body, outline=body_edge,
                                width=1, tags="frame", **st)
        chx = x - 8 * s
        self.canvas.create_oval(chx - 8 * s, cy - 6 * s, chx + 8 * s,
                                cy + 7 * s, fill=body, outline=body_edge,
                                width=1, tags="frame", **st)
        if not ghost:
            # red hourglass marking on the abdomen
            self.canvas.create_polygon(
                abx, cy + 1 * s, abx - 5 * s, cy - 5 * s, abx + 5 * s, cy - 5 * s,
                fill=SPID_MARK, outline="", tags="frame")
            self.canvas.create_polygon(
                abx, cy + 1 * s, abx - 5 * s, cy + 7 * s, abx + 5 * s, cy + 7 * s,
                fill=SPID_MARK, outline="", tags="frame")
            # a cluster of little glowing eyes on the front
            for ex, ey in ((-13, -3), (-10, -4), (-13, 0), (-10, 1),
                           (-7, -3), (-7, 1)):
                self.canvas.create_oval(
                    chx + ex * s * 0.4, cy + ey * s * 0.6,
                    chx + ex * s * 0.4 + 2.4 * s, cy + ey * s * 0.6 + 2.4 * s,
                    fill=SPID_EYE, width=0, tags="frame")
            # two fangs reaching forward
            for fy in (-1, 3):
                self.canvas.create_line(
                    chx - 6 * s, cy + fy * s, chx - 12 * s, cy + (fy + 3) * s,
                    fill=SPID_BODY_DARK, width=w(2), capstyle="round",
                    tags="frame")

    def _draw_zombie(self, x, base_y, s, phase, ghost=False, lunge=0.0):
        """A shambling zombie. `ghost` draws it as a transparent background
        ghoul (stippled, eyeless); otherwise it is a solid, lunging attacker."""
        import math
        st = {"stipple": "gray50"} if ghost else {}
        skin = ZOMB_GHOST if ghost else ZOMB_SKIN
        skin_edge = ZOMB_GHOST if ghost else ZOMB_SKIN_DARK
        shirt = ZOMB_GHOST if ghost else ZOMB_SHIRT
        shirt_edge = ZOMB_GHOST if ghost else ZOMB_SHIRT_DARK
        g = base_y
        sway = math.sin(phase)

        def w(px):
            return max(2, int(px * s))

        hip_y = g - 30 * s
        # stiff, dragging legs
        for sign in (1, -1):
            foot_dx = sign * 6 * s + (sway * 5 * s) * (1 if sign > 0 else -1)
            self.canvas.create_line(x, hip_y, x + foot_dx, g, fill=shirt_edge,
                                    width=w(6), capstyle="round", tags="frame",
                                    **st)
        shoulder_y = hip_y - 26 * s
        # tattered torso
        self.canvas.create_line(x, hip_y, x, shoulder_y, fill=shirt,
                                width=w(11), capstyle="round", tags="frame",
                                **st)
        # both arms outstretched forward (to the left, toward the runner)
        arm_y = shoulder_y + 5 * s
        reach = (16 + lunge * 12) * s
        self.canvas.create_line(x, shoulder_y + 2 * s, x - reach,
                                arm_y + 3 * s * sway, fill=skin, width=w(5),
                                capstyle="round", tags="frame", **st)
        self.canvas.create_line(x, shoulder_y + 3 * s, x - reach * 0.6,
                                arm_y - 4 * s, fill=skin, width=w(5),
                                capstyle="round", tags="frame", **st)
        # lolling head
        hr = 10 * s
        hx = x - 3 * s
        hy = shoulder_y - hr + 2 * s
        self.canvas.create_oval(hx - hr, hy - hr, hx + hr, hy + hr, fill=skin,
                                outline=skin_edge, width=1, tags="frame", **st)
        if not ghost:
            for ex in (-4, 3):
                self.canvas.create_oval(
                    hx + ex * s - 2 * s, hy - 2 * s, hx + ex * s + 2 * s,
                    hy + 2 * s, fill=ZOMB_EYE, width=0, tags="frame")
            self.canvas.create_line(hx - 5 * s, hy + 5 * s, hx + 4 * s,
                                    hy + 6 * s, fill=ZOMB_MOUTH, width=w(2),
                                    tags="frame")

    def _draw_bg_creature(self, z):
        self._draw_creature(z["x"], self.ground_y, z["s"], z["phase"],
                            ghost=True)

    def _draw_hazard_creature(self, b):
        self._draw_creature(b["x"], self.ground_y, 1.0, b["spin"], ghost=False,
                            lunge=0.6)

    def _draw_coin(self, c):
        import math
        x, y, r = c["x"], c["y"], self.COIN_R
        # spinning shimmer: squash the width like a flipping coin
        w = max(2.0, r * (0.35 + 0.65 * abs(math.cos(c["phase"]))))
        self.canvas.create_oval(x - w, y - r, x + w, y + r,
                                fill=COIN, outline=COIN_EDGE, width=2,
                                tags="frame")
        if w > r * 0.45:
            self.canvas.create_oval(x - w * 0.45, y - r * 0.55,
                                    x + w * 0.1, y - r * 0.05,
                                    fill=COIN_SHINE, width=0, tags="frame")

    def _draw_barrel(self, b):
        import math
        x = b["x"]
        cy = self.ground_y - self.BARREL_R
        r = self.BARREL_R
        self.canvas.create_oval(x - r * 1.05, cy - r, x + r * 1.05, cy + r,
                                fill=BARREL, outline=BARREL_DARK, width=2,
                                tags="frame")
        # horizontal hoops
        for dy in (-r * 0.45, r * 0.45):
            self.canvas.create_line(x - r, cy + dy, x + r, cy + dy,
                                    fill=BARREL_DARK, width=2, tags="frame")
        # a rotating diameter conveys the roll
        ex = r * 0.82 * math.cos(b["spin"])
        ey = r * 0.82 * math.sin(b["spin"])
        self.canvas.create_line(x - ex, cy - ey, x + ex, cy + ey,
                                fill=BARREL_HOOP, width=3, tags="frame")

    def _draw_heart(self, cx, cy, s, fill):
        """Draw a heart filled by `fill` in {0.0, 0.5, 1.0} (0.5 = left half)."""
        r = s * 0.42

        def shape(col, half):
            # left lobe (always) + right lobe / body (skipped for a half heart)
            self.canvas.create_oval(cx - s * 0.5 - r, cy - r, cx - s * 0.5 + r,
                                    cy + r, fill=col, width=0, tags="frame")
            if half:
                self.canvas.create_polygon(cx - s * 0.92, cy, cx, cy,
                                           cx, cy + s * 1.05, fill=col, width=0,
                                           tags="frame")
            else:
                self.canvas.create_oval(cx + s * 0.5 - r, cy - r, cx + s * 0.5 + r,
                                        cy + r, fill=col, width=0, tags="frame")
                self.canvas.create_polygon(cx - s * 0.92, cy, cx + s * 0.92, cy,
                                           cx, cy + s * 1.05, fill=col, width=0,
                                           tags="frame")

        shape(HEART_EMPTY, False)              # empty base
        if fill >= 1.0:
            shape(HEART, False)
        elif fill >= 0.5:
            shape(HEART, True)

    def _draw_impact(self, cx, cy):
        """A comic-book "OUCH!" impact burst, shown where an enemy struck the
        knight, so a contact is clearly depicted as an attack."""
        import math
        pts = []
        for k in range(16):
            rad = 26 if k % 2 == 0 else 12
            ang = k * math.pi / 8 + 0.20
            pts += [cx + rad * math.cos(ang), cy + rad * math.sin(ang)]
        self.canvas.create_polygon(*pts, fill="#ffd23f", outline="#e23b3b",
                                   width=2, tags="frame")
        # a few short radiating shock lines
        for a in (0.6, 2.1, 3.7, 5.2):
            self.canvas.create_line(
                cx + 24 * math.cos(a), cy + 24 * math.sin(a),
                cx + 34 * math.cos(a), cy + 34 * math.sin(a),
                fill="#e23b3b", width=2, capstyle="round", tags="frame")
        self.canvas.create_text(cx, cy, text="OUCH!", fill="#991b1b",
                                font=("TkDefaultFont", 10, "bold"),
                                tags="frame")

    def _draw_rescue_dragon(self, cx, cy):
        """Draw the friendly dragon that flies in to carry the knight away.

        Reuses the old helicopter geometry (`heli_x/heli_y`, `HELI_HALF_H`) so
        the fly-in / board / fly-out cutscene and the rope ladder still line up;
        the flapping wings replace the spinning rotor as the hover animation.
        """
        import math
        bw, bh = 46, self.HELI_HALF_H
        flap = math.sin(self.heli_rotor * 2.2)

        # Tail sweeping to the left, ending in a spade tip.
        tail_x = cx - bw - 42
        ty = cy + 4 + flap * 5
        self.canvas.create_line(
            cx - bw + 8, cy, cx - bw - 16, cy - 6, tail_x, ty, fill=DRAGON,
            width=7, capstyle="round", joinstyle="round", smooth=True,
            tags="frame")
        self.canvas.create_polygon(
            tail_x, ty - 8, tail_x - 13, ty, tail_x, ty + 8, fill=DRAGON_WING,
            outline=DRAGON_DARK, width=1, tags="frame")

        # Dangling legs with little claws.
        for lx in (cx - 16, cx + 12):
            self.canvas.create_line(lx, cy + bh - 2, lx, cy + bh + 12,
                                    fill=DRAGON_DARK, width=5, capstyle="round",
                                    tags="frame")
            self.canvas.create_oval(lx - 4, cy + bh + 9, lx + 6, cy + bh + 15,
                                    fill=DRAGON_DARK, width=0, tags="frame")

        # Body + pale belly + a little spine ridge.
        self.canvas.create_oval(cx - bw, cy - bh, cx + bw, cy + bh, fill=DRAGON,
                                outline=DRAGON_DARK, width=2, tags="frame")
        self.canvas.create_oval(cx - bw * 0.72, cy - 1, cx + bw * 0.62,
                                cy + bh, fill=DRAGON_BELLY, outline="",
                                tags="frame")
        for rx in (-30, -16, -2):
            self.canvas.create_polygon(
                cx + rx - 4, cy - bh + 3, cx + rx, cy - bh - 7, cx + rx + 4,
                cy - bh + 3, fill=DRAGON_DARK, width=0, tags="frame")

        # A membranous wing, fanning up from the shoulder and flapping.
        sh_x, sh_y = cx - 4, cy - bh + 2
        up = 34 + 16 * flap
        self.canvas.create_polygon(
            sh_x, sh_y,
            sh_x - 12, sh_y - up * 0.9,
            sh_x + 16, sh_y - up,
            sh_x + 40, sh_y - up * 0.55,
            sh_x + 26, sh_y - up * 0.2,
            sh_x + 34, sh_y + 2,
            sh_x + 14, sh_y - 2,
            fill=DRAGON_WING, outline=DRAGON_WING_DARK, width=2, tags="frame")
        for fx, fy in ((sh_x - 12, sh_y - up * 0.9), (sh_x + 16, sh_y - up),
                       (sh_x + 40, sh_y - up * 0.55)):
            self.canvas.create_line(sh_x, sh_y, fx, fy, fill=DRAGON_WING_DARK,
                                    width=2, tags="frame")

        # Neck + head reaching to the right, with a horn, smile and kind eye.
        hx = cx + bw + 2
        hy = cy - bh - 2
        self.canvas.create_line(cx + bw - 12, cy - bh + 6, hx, hy,
                                fill=DRAGON, width=11, capstyle="round",
                                tags="frame")
        self.canvas.create_oval(hx - 13, hy - 13, hx + 15, hy + 11,
                                fill=DRAGON, outline=DRAGON_DARK, width=2,
                                tags="frame")
        self.canvas.create_polygon(
            hx + 8, hy - 9, hx + 26, hy - 4, hx + 8, hy + 5, fill=DRAGON,
            outline=DRAGON_DARK, width=2, tags="frame")
        self.canvas.create_polygon(
            hx - 7, hy - 11, hx - 2, hy - 26, hx + 5, hy - 11,
            fill=DRAGON_HORN, outline=DRAGON_DARK, width=1, tags="frame")
        self.canvas.create_oval(hx + 1, hy - 8, hx + 8, hy - 1, fill="white",
                                width=0, tags="frame")
        self.canvas.create_oval(hx + 3, hy - 6, hx + 7, hy - 2,
                                fill=DRAGON_EYE, width=0, tags="frame")
        # A friendly puff of flame from the snout.
        if flap > 0.3:
            for k in range(3):
                px = hx + 24 + k * 9
                pr = 6 - k * 1.5
                self.canvas.create_oval(px - pr, hy - pr, px + pr, hy + pr,
                                        fill=DRAGON_FIRE, outline="",
                                        tags="frame")

    def _draw_ladder(self, x, top, bottom):
        for dx in (-7, 7):
            self.canvas.create_line(x + dx, top, x + dx, bottom, fill=LADDER,
                                    width=3, tags="frame")
        y = top + 12
        while y < bottom:
            self.canvas.create_line(x - 7, y, x + 7, y, fill=LADDER, width=3,
                                    tags="frame")
            y += 14

    def _draw_cutscene(self):
        ladder_x = self.heli_x - 6
        top = self.heli_y + self.HELI_HALF_H
        self._draw_ladder(ladder_x, top, top + self.LADDER_LEN)
        self._draw_rescue_dragon(self.heli_x, self.heli_y)
        if self.cutscene == "in":
            # still on the ground while the dragon descends
            self._draw_character(self.char_x, 0.0, False)
        else:
            # riding the ladder: reuse the tucked, arms-up jump pose
            saved_off, saved_t = self.jump_offset, self.jump_t
            self.jump_offset, self.jump_t = self.guy_lift, 0.0
            self._draw_character(ladder_x, 0.0, False)
            self.jump_offset, self.jump_t = saved_off, saved_t
        self.canvas.create_text(
            self.WIDTH / 2, self.HEIGHT * 0.14,
            text="STAGE %d CLEARED!" % self.stage, fill="#166534",
            font=("TkDefaultFont", 20, "bold"), tags="frame")

    def start_jump(self):
        """Begin a jump if grounded and past the cooldown."""
        now = time.time()
        if self.jump_t is None and now >= self._cooldown_until:
            self.jump_t = 0.0
            self.jumps += 1
            self._cooldown_until = now + self.JUMP_DUR + 0.12

    def finished(self):
        """True while play is paused: an end-of-round overlay or the shop."""
        return self.eaten or self.dead or self.stage_cleared or self.shop_open

    def _update_entities(self, dt):
        """Spawn, move and collide coins and rolling barrels."""
        # Spawn as the world scrolls past (measured in walked world-px), so
        # density is consistent per metre regardless of frame rate.
        while self.world_x >= self._next_coin_wx:
            self.coins.append({"x": self.WIDTH + 40.0,
                               "y": self.ground_y - random.uniform(108, 128),
                               "phase": random.uniform(0.0, 6.28), "got": False})
            self._next_coin_wx += random.uniform(*self.COIN_GAP)
        while self.world_x >= self._next_barrel_wx:
            self.barrels.append({"x": self.WIDTH + 40.0, "spin": 0.0,
                                 "hit": False})
            self._next_barrel_wx += random.uniform(*self.BARREL_GAP)

        world_dx = self.SPEED * dt if self.walking else 0.0
        roll = self.BARREL_ROLL * dt

        # Runner collision box for this frame (feet lift with the jump arc).
        feet_y = self.ground_y - self.jump_offset
        head_y = feet_y - self.GUY_H
        gx0, gx1 = self.char_x - 15, self.char_x + 15

        # Coins: drift with the world; grabbed when the runner's box (raised by
        # a jump) reaches them. They sit above head height, so a jump is needed.
        cr = self.COIN_R
        for c in self.coins:
            c["x"] -= world_dx
            c["phase"] += dt * 6.0
            if (not c["got"] and gx0 - cr < c["x"] < gx1 + cr
                    and head_y - cr < c["y"] < feet_y + cr):
                c["got"] = True
                self.coins_collected += 1
                self.coins_bank += 1        # kept as spendable shop currency
                if (self.coins_collected >= self.COINS_TO_CLEAR
                        and self.cutscene is None):
                    self._begin_pickup()
        self.coins = [c for c in self.coins if c["x"] > -50 and not c["got"]]

        # Hazards (barrels / zombies / skeletons / spiders): roll leftward
        # (faster than the scenery, and even while the runner stands still). The
        # runner must *actually* jump over them -- his feet have to be lifted at
        # least HAZARD_CLEAR px while overlapping, so precise timing matters. A
        # contact now costs only HAZARD_DAMAGE (half a heart) and shows an impact
        # burst where the enemy struck.
        br = self.BARREL_R
        cleared = self.jump_offset >= self.HAZARD_CLEAR
        for b in self.barrels:
            step = world_dx + roll
            b["x"] -= step
            b["spin"] += step / self.BARREL_R
            if (not b["hit"] and not cleared
                    and gx0 - br < b["x"] < gx1 + br):
                b["hit"] = True
                self.health -= self.HAZARD_DAMAGE
                now = time.time()
                self._hurt_until = now + 0.5
                self._hit_fx_until = now + 0.5
                self._hit_fx = (self.char_x + 18.0, self.ground_y - 46.0)
                if self.health <= 0:
                    self.dead = True
        self.barrels = [b for b in self.barrels if b["x"] > -50 and not b["hit"]]

        # Background creatures (themed stages only): shuffle leftward with a
        # slow parallax drift plus their own idle shamble; respawn on the right
        # so the scene stays populated. They never collide with anything.
        if self._creature_stage():
            parallax = (self.SPEED * dt * 0.5) if self.walking else 0.0
            for z in self.bg_creatures:
                z["phase"] += dt * 2.4 * z["sway"]
                z["x"] -= parallax + 7.0 * dt * z["sway"]
            self.bg_creatures = [z for z in self.bg_creatures if z["x"] > -70.0]
            while len(self.bg_creatures) < 5:
                self.bg_creatures.append(self._make_bg_creature(
                    self.WIDTH + random.uniform(30.0, 200.0)))

    def _begin_pickup(self):
        """Enter the stage-clear cutscene: the rescue dragon flies in."""
        self.stage_cleared = True
        self.cutscene = "in"
        self.heli_x = self.WIDTH + 180.0
        self.heli_y = -60.0
        self.heli_rotor = 0.0
        self.guy_lift = 0.0
        self.guy_on_ladder = False
        self._board_t = 0.0

    def _update_cutscene(self, dt):
        """Drive the fly-in / board / fly-out dragon pickup animation."""
        self.heli_rotor += dt * 32.0
        # Hover target: ladder hanging just above the runner's reach.
        tx = self.char_x + 12.0
        ty = self.ground_y - (self.GRAB_LIFT + self.LADDER_LEN
                              + self.HELI_HALF_H)
        if self.cutscene == "in":
            k = min(1.0, 2.5 * dt)
            self.heli_x += (tx - self.heli_x) * k
            self.heli_y += (ty - self.heli_y) * k
            if abs(self.heli_x - tx) < 4 and abs(self.heli_y - ty) < 4:
                self.heli_x, self.heli_y = tx, ty
                self.cutscene = "board"
                self._board_t = 0.0
        elif self.cutscene == "board":
            self._board_t += dt
            p = min(1.0, self._board_t / 0.55)
            self.guy_lift = self.GRAB_LIFT * p
            if p >= 1.0:
                self.guy_on_ladder = True
                self.cutscene = "out"
        else:  # "out": climb away to the top-right, carrying the runner
            self.heli_x += 70.0 * dt
            self.heli_y -= 95.0 * dt
            ladder_bottom = self.heli_y + self.HELI_HALF_H + self.LADDER_LEN
            self.guy_lift = self.ground_y - ladder_bottom
            if self.heli_y < -160.0:
                self._open_shop()

    # -- between-stage shop ---------------------------------------------------
    def _open_shop(self):
        """Cutscene finished: the knight arrives at the shop to spend coins."""
        self.cutscene = None
        self.stage_cleared = False
        self.shop_open = True
        self.shop_msg = ""

    def _item_cost(self, item):
        """Current price of a shop item (rises each time it is bought)."""
        return item["base_cost"] + item["step"] * item["count"]

    def _item_maxed(self, item):
        return item["max"] is not None and item["count"] >= item["max"]

    def _shop_buy(self, key):
        """Buy the item bound to `key` if the shop is open and it's affordable."""
        if not self.shop_open:
            return
        for item in self.shop_items:
            if item["key"] == key:
                self._buy_item(item)
                return

    def _on_shop_click(self, event):
        """Buy whichever shop item row was clicked with the mouse."""
        if not self.shop_open:
            return
        for x0, y0, x1, y1, item in self._shop_item_rects:
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                self._buy_item(item)
                return

    def _buy_item(self, item):
        """Buy `item` if the shop is open and it's affordable."""
        if not self.shop_open:
            return
        if self._item_maxed(item):
            self.shop_msg = "%s is maxed out." % item["name"]
            return
        cost = self._item_cost(item)
        if self.coins_bank < cost:
            self.shop_msg = "Not enough coins for %s (need %d)." % (
                item["name"], cost)
            return
        self.coins_bank -= cost
        item["count"] += 1
        item["apply"]()
        self.shop_msg = "Bought %s!" % item["name"]

    def _shop_continue(self):
        """Leave the shop and start the next stage."""
        if not self.shop_open:
            return
        self.shop_open = False
        self.shop_msg = ""
        self._advance_stage()

    def _buy_higher_jump(self):
        self.JUMP_HEIGHT = self.JUMP_HEIGHT + 22
        self.JUMP_DUR = self.JUMP_DUR + 0.04

    def _buy_extra_heart(self):
        self.MAX_HEALTH = self.MAX_HEALTH + 1
        self.health = self.MAX_HEALTH

    def _buy_full_heal(self):
        self.health = self.MAX_HEALTH

    def _advance_stage(self):
        """Start the next stage (upgrades and the coin bank carry over)."""
        self.stage += 1
        self._reset_round()

    def restart(self):
        """Retry after a loss, or skip the cutscene / start the next stage."""
        if self.shop_open:
            return                 # the shop is left with [Enter], not [R]
        if not self.finished():
            return
        if self.stage_cleared:
            self.stage += 1
        self._reset_round()

    def _reset_round(self):
        """Reset all per-round state (keeps the current stage number)."""
        self.eaten = False
        self.dead = False
        self.stage_cleared = False
        self.cutscene = None
        self.guy_on_ladder = False
        self.guy_lift = 0.0
        self.health = self.MAX_HEALTH
        self.coins = []
        self.barrels = []
        self.coins_collected = 0
        self._hurt_until = 0.0
        self._hit_fx_until = 0.0
        self._next_coin_wx = random.uniform(*self.COIN_GAP)
        self._next_barrel_wx = random.uniform(700.0, 1100.0)
        self.dino_x = self.char_x - 180.0
        self._dino_grace_until = (
            time.time() + getattr(self.classifier, "window_s", 1.5) + 1.0)
        self.world_x = 0.0
        self.distance_m = 0.0
        self.walk_phase = 0.0
        self.jump_t = None
        self.jump_offset = 0.0
        self.jumps = 0
        self._last = time.time()
        # Re-theme for the (possibly new) stage: repaint the static backdrop,
        # (re)spawn or clear the background creatures, and swap the soundtrack.
        self._draw_static_background()
        self._populate_bg_creatures()
        self._update_music()

    def _draw_star(self, cx, cy, r, colour):
        """A small five-pointed star (hat decoration / spell spark)."""
        import math
        pts = []
        for k in range(10):
            rad = r if k % 2 == 0 else r * 0.42
            ang = -math.pi / 2 + k * math.pi / 5
            pts += [cx + rad * math.cos(ang), cy + rad * math.sin(ang)]
        self.canvas.create_polygon(*pts, fill=colour, width=0, tags="frame")

    def _draw_evil_wizard(self):
        """Draw the evil wizard who chases the knight, casting as he closes in."""
        import math

        x = self.dino_x
        g = self.ground_y
        bob = abs(math.sin(self.dino_phase)) * 2.5
        casting = self.eaten or (math.sin(self.dino_phase * 0.6) > 0.55)

        # Feet peeking from under the robe, with an alternating gliding stride.
        swing = math.sin(self.dino_phase) * 6
        for sign in (-1, 1):
            fx = x + sign * 7 + (swing if sign > 0 else -swing) * 0.5
            self.canvas.create_oval(
                fx - 6, g - 5, fx + 8, g, fill=WIZ_SHOE,
                outline=WIZ_ROBE_DARK, width=1, tags="frame")

        top = g - 58 - bob                 # shoulder line
        # Flowing robe: narrow shoulders, wide wavy hem.
        self.canvas.create_polygon(
            x - 10, top, x + 10, top,
            x + 26, g, x + 13, g - 4, x, g, x - 13, g - 4, x - 26, g,
            fill=WIZ_ROBE, outline=WIZ_ROBE_DARK, width=2, tags="frame")
        self.canvas.create_line(x, top + 6, x, g - 3, fill=WIZ_ROBE_DARK,
                                width=2, tags="frame")

        # Head, glaring eyes and a long beard.
        hy = top - 8
        self.canvas.create_oval(x - 9, hy - 9, x + 9, hy + 9, fill=WIZ_SKIN,
                                outline=WIZ_ROBE_DARK, width=1, tags="frame")
        self.canvas.create_polygon(
            x - 8, hy + 3, x + 8, hy + 3, x + 3, top + 22, x, top + 28,
            x - 3, top + 22, fill=WIZ_BEARD, outline="#c9ccd6", width=1,
            tags="frame")
        for ex in (-4, 4):
            self.canvas.create_oval(x + ex - 2, hy - 3, x + ex + 2, hy + 1,
                                    fill=WIZ_EYE, width=0, tags="frame")

        # Tall pointed hat with a drooping tip, a brim and a star.
        brim_y = hy - 7
        self.canvas.create_polygon(
            x - 16, brim_y, x + 16, brim_y, x + 11, brim_y - 3,
            x - 11, brim_y - 3, fill=WIZ_HAT, outline=WIZ_ROBE_DARK,
            width=1, tags="frame")
        self.canvas.create_polygon(
            x - 10, brim_y - 2, x + 10, brim_y - 2, x + 12, brim_y - 46,
            fill=WIZ_HAT, outline=WIZ_ROBE_DARK, width=2, tags="frame")
        self._draw_star(x, brim_y - 20, 4.5, WIZ_STAR)

        # Staff held forward in the leading hand, topped with a glowing orb.
        hand_x, hand_y = x + 13, top + 12
        orb_x, orb_y = x + 44, top - 24
        self.canvas.create_line(x + 6, g - 3, orb_x, orb_y, fill="#6b4a2b",
                                width=4, capstyle="round", tags="frame")
        self.canvas.create_oval(hand_x - 4, hand_y - 4, hand_x + 4, hand_y + 4,
                                fill=WIZ_SKIN, outline=WIZ_ROBE_DARK, width=1,
                                tags="frame")
        glow = 12 if casting else 8
        self.canvas.create_oval(orb_x - glow, orb_y - glow, orb_x + glow,
                                orb_y + glow, fill="", outline=WIZ_ORB,
                                width=2, tags="frame")
        self.canvas.create_oval(orb_x - 6, orb_y - 6, orb_x + 6, orb_y + 6,
                                fill=WIZ_ORB, outline="", tags="frame")
        self.canvas.create_oval(orb_x - 3, orb_y - 3, orb_x + 3, orb_y + 3,
                                fill=WIZ_ORB_CORE, outline="", tags="frame")

        # Casting: a crackling spell-bolt hurled forward toward the knight.
        if casting:
            by = top - 6
            bolt = [orb_x, orb_y]
            bx = orb_x
            for k in range(4):
                bx += 16
                bolt += [bx, by + (6 if k % 2 else -6)]
            self.canvas.create_line(*bolt, fill=WIZ_SPELL, width=3,
                                    capstyle="round", tags="frame")
            for k in range(3):
                sx = orb_x + 20 + k * 16 + random.uniform(-3, 3)
                sy = by + random.uniform(-8, 8)
                self._draw_star(sx, sy, 3, WIZ_SPELL)

    def _draw_sword(self, hand):
        """A raised knightly sword held at `hand` (blade up and forward)."""
        hx, hy = hand
        self.canvas.create_line(hx, hy + 6, hx + 11, hy - 30, fill=SWORD,
                                width=4, capstyle="round", tags="frame")
        # crossguard and pommel
        self.canvas.create_line(hx - 6, hy + 4, hx + 8, hy - 2, fill=SWORD_HILT,
                                width=3, capstyle="round", tags="frame")
        self.canvas.create_oval(hx - 3, hy + 5, hx + 3, hy + 11,
                                fill=SWORD_HILT, width=0, tags="frame")

    def _draw_columns(self, cx, cy, hw, hh, col):
        """The Columns of Gediminas (Gediminaičių stulpai), the Lithuanian
        heraldic emblem, as gold bars centred at (cx, cy). Geometry traced from
        the reference: two full-height outer posts on a base, a central top stub
        on a wide lintel, and two inner posts framing an open square."""
        def gx(g):
            return cx + (g - 11.0) / 11.0 * hw
        def gy(g):
            return cy + (g - 11.0) / 11.0 * hh
        bars = (
            (0, 2.5, 0, 19),        # left outer post
            (19.5, 22, 0, 19),      # right outer post
            (0, 22, 19, 22),        # base bar
            (9.5, 12.5, 0, 8.5),    # central top stub
            (5, 17, 8, 10.5),       # lintel (crossbar)
            (5, 7, 10, 19),         # inner-left post
            (15, 17, 10, 19),       # inner-right post
        )
        for a, b, c, d in bars:
            self.canvas.create_rectangle(gx(a), gy(c), gx(b), gy(d),
                                         fill=col, outline="", tags="frame")

    def _draw_shield(self, hand):
        """A heater shield bearing the Columns of Gediminas, centred at `hand`."""
        hx, hy = hand
        w, h = 12, 15
        self.canvas.create_polygon(
            hx - w, hy - h, hx + w, hy - h, hx + w, hy + 3, hx, hy + h,
            hx - w, hy + 3, fill=SHIELD, outline=SHIELD_TRIM, width=2,
            tags="frame")
        self._draw_columns(hx, hy - 4.5, 7.6, 10.5, SHIELD_TRIM)

    def _draw_helm(self, x, shoulder_y):
        """A plumed steel helm with a visor slit, crowning the knight."""
        hr = 11
        cy = shoulder_y - hr
        self.canvas.create_line(
            x - 2, cy - hr, x - 14, cy - hr - 8, x - 20, cy - 2,
            fill=KNIGHT_PLUME, width=5, capstyle="round", joinstyle="round",
            tags="frame")
        self.canvas.create_oval(x - hr, cy - hr, x + hr, cy + hr,
                                fill=KNIGHT_ARMOR, outline=KNIGHT_ARMOR_DARK,
                                width=1, tags="frame")
        self.canvas.create_rectangle(x - hr + 2, cy - 2, x + hr - 2, cy + 1,
                                     fill="#3a4048", width=0, tags="frame")
        self.canvas.create_line(x, cy + 3, x, cy + hr - 1,
                                fill=KNIGHT_ARMOR_DARK, width=1, tags="frame")

    def _draw_character(self, x, phase, walking):
        """Draw the hero as an armoured knight, in walking/jumping/riding poses."""
        import math
        g = self.ground_y - self.jump_offset
        jumping = self.jump_t is not None
        if jumping:
            # tucked pose: bent armored legs, weapon and shield held in
            hip_y = g - 26
            shoulder_y = hip_y - 30
            for sign in (1, -1):
                knee_x = x + sign * 9
                self.canvas.create_line(
                    x, hip_y, knee_x, hip_y + 10, knee_x - sign * 5, hip_y + 20,
                    fill=KNIGHT_ARMOR_DARK, width=6, capstyle="round",
                    joinstyle="round", tags="frame")
            sword_hand = (x + 15, shoulder_y - 6)
            shield_hand = (x - 11, shoulder_y + 8)
        else:
            bob = -abs(math.sin(phase)) * 4 if walking else 0.0
            hip_y = g - 26 + bob
            swing = math.sin(phase) * (14 if walking else 0)
            for sign in (1, -1):
                foot_dx = sign * swing
                foot_y = g + bob - max(0, sign * math.sin(phase)) * 4
                self.canvas.create_line(
                    x, hip_y, x + foot_dx, foot_y, fill=KNIGHT_ARMOR_DARK,
                    width=6, capstyle="round", tags="frame")
            shoulder_y = hip_y - 30
            sword_hand = (x + 12 - swing, shoulder_y - 4)
            shield_hand = (x - 6 + swing, shoulder_y + 12)

        # Torso: a red surcoat over a steel breastplate.
        self.canvas.create_line(x, hip_y, x, shoulder_y, fill=KNIGHT_TUNIC,
                                width=13, capstyle="round", tags="frame")
        self.canvas.create_line(x, hip_y - 4, x, shoulder_y, fill=KNIGHT_ARMOR,
                                width=5, capstyle="round", tags="frame")
        # Sword arm (raising the blade) and shield arm (facing forward).
        self.canvas.create_line(x, shoulder_y + 3, sword_hand[0], sword_hand[1],
                                fill=KNIGHT_ARMOR, width=5, capstyle="round",
                                tags="frame")
        self._draw_sword(sword_hand)
        self.canvas.create_line(x, shoulder_y + 5, shield_hand[0],
                                shield_hand[1], fill=KNIGHT_ARMOR, width=5,
                                capstyle="round", tags="frame")
        self._draw_shield(shield_hand)
        self._draw_helm(x, shoulder_y)

    def _draw_hud(self):
        f = self.classifier.features if self.classifier else {}
        state = "WALKING" if self.walking else "STANDING"
        colour = "#1a7f37" if self.walking else "#8a94a6"
        self.canvas.create_text(
            16, 18, anchor="w", text="Smart Geophone — %s" % self.player_name,
            fill=HUD_INK, font=("TkDefaultFont", 13, "bold"), tags="frame")
        self.canvas.create_text(
            16, 40, anchor="w", text=state, fill=colour,
            font=("TkDefaultFont", 12, "bold"), tags="frame")
        baud_txt = ("baud %d" % self.baud) if self.baud else self.source_label
        self.canvas.create_text(
            16, 60, anchor="w",
            text="%s   ·   %.0f samples/s" % (baud_txt, self.meas_fs),
            fill="#8a94a6", font=("TkDefaultFont", 9), tags="frame")
        if self.best_score:
            self.canvas.create_text(
                16, 78, anchor="w",
                text="best: stage %d · %d coins" % (
                    self.best_score.get("stage", 0),
                    self.best_score.get("coins", 0)),
                fill="#b8860b", font=("TkDefaultFont", 9, "bold"),
                tags="frame")
        self.canvas.create_text(
            self.WIDTH - 16, 18, anchor="e",
            text="Stage %d   ·   %.1f m   ·   %d/%d coins   ·   purse %d" % (
                self.stage, self.distance_m, self.coins_collected,
                self.COINS_TO_CLEAR, self.coins_bank),
            fill=HUD_INK, font=("TkDefaultFont", 13, "bold"), tags="frame")
        jump_col = "#c2410c" if self.jump_t is not None else "#8a94a6"
        probs = f.get("probabilities", (1.0, 0.0))
        self.canvas.create_text(
            self.WIDTH - 16, 40, anchor="e",
            text="NN  still %.0f%%  walk %.0f%%" % tuple(
                100.0 * float(p) for p in probs),
            fill=jump_col, font=("TkDefaultFont", 10), tags="frame")
        for i in range(self.MAX_HEALTH):
            remaining = self.health - i
            fill = 1.0 if remaining >= 1.0 else (0.5 if remaining >= 0.5 else 0.0)
            self._draw_heart(self.WIDTH - 24 - i * 22, 62, 7, fill)
        hazard = {"spider": "jump the spiders",
                  "cave": "jump the skeletons",
                  "zombie": "jump the zombies"}.get(self._theme(),
                                                    "jump over barrels")
        self.canvas.create_text(
            self.WIDTH / 2, self.HEIGHT - 14, anchor="s",
            text="walk to move  ·  jump for coins  ·  %s  ·  "
                 "spend coins at the shop  ·  [space] walk  ·  [↑] jump  ·  [q] quit"
                 % hazard,
            fill="#7a8698", font=("TkDefaultFont", 9), tags="frame")

    def _render(self):
        self.canvas.delete("frame")
        if self._theme() == "spider":
            self._draw_corner_webs()
        for it in self.clouds.items:
            self._draw_cloud(self.clouds.screen_x(it), it)
        for it in self.mountains.items:
            self._draw_mountain(self.mountains.screen_x(it), it)
        for it in self.ground.items:
            self._draw_ground_obj(self.ground.screen_x(it), it)
        creatures = self._creature_stage()
        if creatures:
            for z in self.bg_creatures:
                self._draw_bg_creature(z)
        self._draw_evil_wizard()
        for b in self.barrels:
            if creatures:
                self._draw_hazard_creature(b)
            else:
                self._draw_barrel(b)
        for c in self.coins:
            self._draw_coin(c)
        if (not self.eaten and not self.dead and self.cutscene is None
                and not self.shop_open):
            self._draw_character(self.char_x, self.walk_phase, self.walking)
        if time.time() < self._hurt_until:
            self.canvas.create_rectangle(
                0, 0, self.WIDTH, self.HEIGHT, outline="#e23b3b", width=8,
                tags="frame")
        if time.time() < self._hit_fx_until:
            self._draw_impact(*self._hit_fx)
        self._draw_hud()
        if self.cutscene is not None:
            self._draw_cutscene()
        elif self.shop_open:
            self._draw_shop()
        elif self.finished():
            self._draw_overlay()

    def _draw_overlay(self):
        if self.stage_cleared:
            border, title_col, title = "#166534", "#166534", \
                "STAGE %d CLEARED!" % self.stage
            sub = "Collected %d coins  ·  press [R] for the next stage" % \
                self.COINS_TO_CLEAR
            fill = "#e7fbef"
        elif self.dead:
            border, title_col = "#7f1d1d", "#991b1b"
            theme = self._theme()
            if theme == "spider":
                title = "THE SPIDERS GOT YOU!"
                sub = "Wrapped in silk  ·  press [R] to restart"
            elif theme == "cave":
                title = "THE SKELETONS GOT YOU!"
                sub = "The bones dragged you down  ·  press [R] to restart"
            elif theme == "zombie":
                title = "THE HORDE GOT YOU!"
                sub = "The zombies overwhelmed you  ·  press [R] to restart"
            else:
                title = "OUT OF HEALTH!"
                sub = "The barrels got you  ·  press [R] to restart"
            fill = "#fff7e6"
        else:
            border, title_col, title = "#7f1d1d", "#991b1b", \
                "THE EVIL WIZARD GOT YOU!"
            sub = "Keep walking to escape his spell  ·  press [R] to restart"
            fill = "#fff7e6"
        self.canvas.create_rectangle(
            self.WIDTH * 0.22, self.HEIGHT * 0.33,
            self.WIDTH * 0.78, self.HEIGHT * 0.61,
            fill=fill, outline=border, width=4, tags="frame")
        self.canvas.create_text(
            self.WIDTH / 2, self.HEIGHT * 0.42, text=title, fill=title_col,
            font=("TkDefaultFont", 22, "bold"), tags="frame")
        self.canvas.create_text(
            self.WIDTH / 2, self.HEIGHT * 0.53, text=sub, fill=HUD_INK,
            font=("TkDefaultFont", 12), tags="frame")

    def _draw_shop_stall(self, cx, base_y):
        """A little market stall (striped awning, sign, counter) for the shop."""
        w, h = 150, 78
        left, right = cx - w / 2, cx + w / 2
        top = base_y - h
        # posts + counter
        for px in (left + 8, right - 8):
            self.canvas.create_rectangle(px - 4, top, px + 4, base_y,
                                         fill=SHOP_WOOD, outline=SHOP_WOOD_DARK,
                                         width=1, tags="frame")
        self.canvas.create_rectangle(left, base_y - 24, right, base_y - 8,
                                     fill=SHOP_WOOD, outline=SHOP_WOOD_DARK,
                                     width=2, tags="frame")
        # peaked roof
        self.canvas.create_polygon(
            left - 10, top, right + 10, top, cx, top - 34,
            fill=SHOP_ROOF, outline=SHOP_ROOF_DARK, width=2, tags="frame")
        # striped awning under the roof
        n = 6
        for i in range(n):
            x0 = left + (right - left) * i / n
            x1 = left + (right - left) * (i + 1) / n
            col = SHOP_ROOF if i % 2 == 0 else SHOP_AWNING
            self.canvas.create_polygon(
                x0, top, x1, top, x1 - 6, top + 12, x0 - 6, top + 12,
                fill=col, outline=SHOP_ROOF_DARK, width=1, tags="frame")
        # hanging sign
        self.canvas.create_rectangle(cx - 26, top + 20, cx + 26, top + 40,
                                     fill=SHOP_SIGN, outline=SHOP_WOOD_DARK,
                                     width=2, tags="frame")
        self.canvas.create_text(cx, top + 30, text="SHOP", fill=SHOP_INK,
                                font=("TkDefaultFont", 11, "bold"),
                                tags="frame")

    def _draw_shop(self):
        """The between-stage shop: spend collected coins on upgrades."""
        # Dim the world, then draw the stall and a purchase panel.
        self.canvas.create_rectangle(0, 0, self.WIDTH, self.HEIGHT,
                                     fill="#0b1a2b", stipple="gray50",
                                     width=0, tags="frame")
        self._draw_shop_stall(self.WIDTH * 0.5, self.ground_y)
        # the knight stands at the counter
        saved_off, saved_t = self.jump_offset, self.jump_t
        self.jump_offset, self.jump_t = 0.0, None
        self._draw_character(int(self.WIDTH * 0.32), 0.0, False)
        self.jump_offset, self.jump_t = saved_off, saved_t

        px0, py0 = self.WIDTH * 0.16, self.HEIGHT * 0.30
        px1, py1 = self.WIDTH * 0.84, self.HEIGHT * 0.88
        self.canvas.create_rectangle(px0, py0, px1, py1, fill=SHOP_PANEL,
                                     outline=SHOP_WOOD_DARK, width=4,
                                     tags="frame")
        self.canvas.create_text(
            self.WIDTH / 2, py0 + 22,
            text="SHOP  ·  Stage %d cleared!" % self.stage, fill=SHOP_ROOF,
            font=("TkDefaultFont", 18, "bold"), tags="frame")
        self.canvas.create_text(
            self.WIDTH / 2, py0 + 46,
            text="Coins available: %d" % self.coins_bank, fill=SHOP_INK,
            font=("TkDefaultFont", 13, "bold"), tags="frame")

        self._shop_item_rects = []
        row_y = py0 + 82
        row_h = 34
        for item in self.shop_items:
            maxed = self._item_maxed(item)
            cost = self._item_cost(item)
            if maxed:
                price_txt, price_col = "MAX", SHOP_INK
            elif self.coins_bank >= cost:
                price_txt, price_col = "%d coins" % cost, SHOP_AFFORD
            else:
                price_txt, price_col = "%d coins" % cost, SHOP_TOODEAR
            # A subtly tinted, clickable row spanning the panel width.
            rx0, rx1 = px0 + 14, px1 - 14
            ry0, ry1 = row_y - row_h / 2 + 2, row_y + row_h / 2 - 2
            row_fill = SHOP_PANEL_DARK if maxed else SHOP_PANEL
            self.canvas.create_rectangle(
                rx0, ry0, rx1, ry1, fill=row_fill,
                outline=SHOP_AWNING, width=1, tags="frame")
            if not maxed:
                self._shop_item_rects.append((rx0, ry0, rx1, ry1, item))
            owned = ("  (owned x%d)" % item["count"]) if item["count"] else ""
            self.canvas.create_text(
                px0 + 26, row_y, anchor="w",
                text="[%s]  %s — %s%s" % (item["key"], item["name"],
                                          item["desc"], owned),
                fill=SHOP_INK, font=("TkDefaultFont", 12), tags="frame")
            self.canvas.create_text(
                px1 - 26, row_y, anchor="e", text=price_txt, fill=price_col,
                font=("TkDefaultFont", 12, "bold"), tags="frame")
            row_y += row_h

        if self.shop_msg:
            self.canvas.create_text(
                self.WIDTH / 2, py1 - 44, text=self.shop_msg, fill=SHOP_ROOF,
                font=("TkDefaultFont", 11, "italic"), tags="frame")
        self.canvas.create_text(
            self.WIDTH / 2, py1 - 20,
            text="click an item or press [1]/[2]/[3] to buy   ·   "
                 "[Enter] to start stage %d" % (self.stage + 1),
            fill=SHOP_INK, font=("TkDefaultFont", 11, "bold"), tags="frame")

    # -- main loop ------------------------------------------------------------
    def tick(self):
        now = time.time()
        dt = min(0.1, now - self._last)
        self._last = now

        # Pull and classify once per rendered frame. Inference is a tiny CPU
        # model and uses only the newest causal window (no future/look-ahead).
        samples = self.source.drain()

        # live sample-rate measurement (throughput)
        if samples:
            now2 = time.time()
            if self._samp_t0 is None:
                self._samp_t0 = now2
            self._samp_count += len(samples)
            el = now2 - self._samp_t0
            if el > 1.0:
                self.meas_fs = self._samp_count / el

        # Low-latency jump: fire on the raw take-off transient, independent of
        # the (slower) still/walk classifier below.
        if (self.jump_detector is not None and samples and not self.finished()
                and self.jump_detector.process(samples, self.meas_fs)):
            self.start_jump()

        detected = "still"
        confident = False
        if self.classifier is not None:
            self.classifier.add(samples)
            detected = self.classifier.update(self.meas_fs)
            confident = self.classifier.features["confidence"] >= self.confidence
        self.walking = bool((detected == "walk" and confident) or self.space)

        # The evil wizard gains ground whenever the player stops. Walking moves
        # the knight faster than the pursuer, opening a small safety gap.
        if not self.finished() and now >= self._dino_grace_until:
            if self.walking:
                self.dino_x -= self.DINO_FALLBACK_SPEED * dt
                self.dino_x = max(self.char_x - 230.0, self.dino_x)
            else:
                self.dino_x += self.DINO_CHASE_SPEED * dt
            self.dino_phase += dt * (10.0 if self.walking else 7.0)
            if self.dino_x + 58 >= self.char_x - 9:
                self.eaten = True
                self.walking = False
        elif not self.finished():
            self.dino_phase += dt * 7.0

        # advance the jump arc (parabola, independent of walking)
        if self.jump_t is not None:
            self.jump_t += dt
            p = self.jump_t / self.JUMP_DUR
            if p >= 1.0:
                self.jump_t = None
                self.jump_offset = 0.0
            else:
                self.jump_offset = self.JUMP_HEIGHT * 4.0 * p * (1.0 - p)

        if self.walking and not self.finished():
            dx = self.SPEED * dt
            self.world_x += dx
            self.distance_m += dx / self.PX_PER_M
            self.walk_phase += dx / self.STRIDE
            for layer in (self.clouds, self.mountains, self.ground):
                layer.advance(dx)
                layer.maintain(self.WIDTH)

        if not self.finished():
            self._update_entities(dt)

        if self.cutscene is not None:
            self._update_cutscene(dt)

        self._render()
        self.root.after(int(1000 / self.FPS), self.tick)


# ----------------------------------------------------------------------------
def build_source(args):
    fs_hint = args.fs if args.fs else 100.0
    if args.simulate:
        return SimulatedSource(fs=fs_hint), fs_hint, "SIMULATED"
    port = args.port or auto_detect_port()
    if port is None:
        sys.stderr.write(
            "No serial port found. Plug in the geophone, pass --port, or use "
            "--simulate. (You can still hold [space] to walk.)\n")
        sys.exit(1)
    src = SerialSource(port, args.baud)
    try:
        src.open()
    except Exception as exc:
        sys.stderr.write("Could not open %s: %s\n" % (port, exc))
        sys.exit(1)
    return src, fs_hint, port


def main():
    ap = argparse.ArgumentParser(
        description="Geophone-driven side-scroller: walk to make the knight walk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0",
                    help="Geophone serial port.")
    ap.add_argument("--baud", type=int, default=9600,
                    help="Serial baud rate. MUST match the firmware's "
                         "Serial.begin(...) (stock = 9600).")
    ap.add_argument("--fs", type=float, default=None, help="Sample rate hint (Hz).")
    ap.add_argument("--simulate", action="store_true",
                    help="Use the synthetic walker instead of the serial port.")
    ap.add_argument("--model",
                    default=str(Path(__file__).with_name("geophone_model.pt")),
                    help="Model produced by learn_model.py.")
    ap.add_argument("--confidence", type=float, default=0.55,
                    help="Minimum probability before acting on a prediction.")
    ap.add_argument("--smoothing", type=float, default=0.12,
                    help="Probability smoothing time in seconds; lower is faster.")
    ap.add_argument("--jump-threshold", type=float, default=6.0,
                    help="Jump onset threshold in multiples of the recent step "
                         "peak; lower is more sensitive. Ignored when a trained "
                         "jump model is loaded.")
    ap.add_argument("--jump-refractory", type=float, default=0.6,
                    help="Seconds silenced after a jump trigger so the landing "
                         "impact is not counted as a second jump.")
    ap.add_argument("--jump-model", default=None,
                    help="Path to an adaptive jump model from collect_jumps.py. "
                         "Off by default (uses --jump-threshold); pass a path "
                         "like jump_model.json to opt in.")
    ap.add_argument("--no-sensor-jump", action="store_true",
                    help="Disable sensor-driven jumps (keyboard only).")
    ap.add_argument("--player", "--name", dest="player", default="Povilas",
                    help="Player name, passed on the command line (default "
                         "Povilas). A personal folder under "
                         "python/players/<name>/ collects this player's high "
                         "scores and game videos.")
    ap.add_argument("--players-dir",
                    default=str(Path(__file__).with_name("players")),
                    help="Base folder holding per-player profile folders.")
    ap.add_argument("--record-file", default=None,
                    help="Where the screen movie of the session is saved. "
                         "Defaults to a timestamped file in the player's "
                         "folder (or last_game.mp4 if no player).")
    ap.add_argument("--record-fps", type=int, default=20,
                    help="Frame rate of the saved screen movie.")
    ap.add_argument("--no-record", action="store_true",
                    help="Do not save a screen movie of the game.")
    ap.add_argument("--no-music", action="store_true",
                    help="Do not play the medieval background soundtrack.")
    ap.add_argument("--music-volume", type=float, default=0.7,
                    help="Background-music volume in [0, 1].")
    args = ap.parse_args()

    source, fs, label = build_source(args)
    try:
        classifier = MotionClassifier(args.model, fs, smooth=args.smoothing)
    except FileNotFoundError:
        sys.exit("Model not found: %s\nCollect training data and run "
                 "learn_model.py first." % args.model)
    jump_detector = None
    if not args.no_sensor_jump:
        # Default: the hand-tuned onset detector (the responsive behaviour that
        # works out of the box). The adaptive model from collect_jumps.py is
        # opt-in via --jump-model, since an over-conservative fit can make jumps
        # too hard to trigger.
        if args.jump_model:
            jump_detector = OnsetDetector.from_model(
                args.jump_model, fs, refractory=args.jump_refractory)
            if jump_detector is not None:
                sys.stderr.write("Using trained jump model %s\n" % args.jump_model)
            else:
                sys.stderr.write(
                    "Jump model %s not found; using --jump-threshold.\n"
                    % args.jump_model)
        if jump_detector is None:
            jump_detector = OnsetDetector(
                fs, threshold=args.jump_threshold,
                refractory=args.jump_refractory)
    source.start()

    root = tk.Tk()
    root.title("Smart Geophone — Step Runner  (%s)" % label)

    # Player name comes from the command line (--player/--name, default Povilas);
    # no GUI prompt. Load / create that player's personal folder.
    profile = PlayerProfile(args.player, base_dir=args.players_dir)
    root.title("Smart Geophone — Step Runner  ·  %s  (%s)"
               % (profile.name, label))

    game = Game(root, source, classifier, fs, confidence=args.confidence,
                baud=None if args.simulate else args.baud, source_label=label,
                jump_detector=jump_detector, profile=profile)

    # Per-stage background soundtrack (best-effort, looped in a player process):
    # a medieval bed for stage 1 and an eerie Zombie Village bed for stage 2+.
    # The game swaps the looping track when the stage theme changes.
    music = None
    if not args.no_music:
        try:
            theme = ensure_theme()
            game.theme_medieval = str(theme)
            try:
                game.theme_zombie = str(ensure_zombie_theme())
            except Exception as exc:
                sys.stderr.write("Zombie theme unavailable: %s\n" % exc)
                game.theme_zombie = None
            try:
                game.theme_cave = str(ensure_cave_theme())
            except Exception as exc:
                sys.stderr.write("Cave theme unavailable: %s\n" % exc)
                game.theme_cave = None
            try:
                game.theme_spider = str(ensure_spider_theme())
            except Exception as exc:
                sys.stderr.write("Spider theme unavailable: %s\n" % exc)
                game.theme_spider = None
            music = MusicPlayer(theme, volume=args.music_volume)
            if music.start():
                game.music = music
            else:
                music = None
        except Exception as exc:
            sys.stderr.write("Music disabled: %s\n" % exc)
            music = None

    # Always record the session to a "last game" movie (best-effort). ffmpeg
    # grabs the window region directly, so it costs the game loop nothing.
    # A player's videos accumulate (one timestamped file per session) in their
    # personal folder; --record-file overrides this and a bare run without a
    # profile falls back to the shared last_game.mp4.
    if args.record_file:
        record_file = args.record_file
    elif profile is not None:
        record_file = profile.new_video_path()
    else:
        record_file = str(Path(__file__).with_name("last_game.mp4"))

    recorder = None
    if not args.no_record:
        # x11grab captures a FIXED screen rectangle, so we must sample the game
        # window's REAL on-screen position and size only after the window
        # manager has finished mapping, raising and placing it. Sampling too
        # early grabbed a stale/offset rectangle -- part desktop, part game.
        # Lock the window size first so a later resize can't desync the region
        # from the game content, then wait for the geometry to stop changing
        # (two identical reads in a row) before starting the grab.
        root.resizable(False, False)
        root.deiconify()
        root.lift()
        root.focus_force()
        root.update_idletasks()
        prev, geo = None, (0, 0, 0, 0)
        for _ in range(80):            # up to ~1.6s for the WM to settle
            root.update()
            geo = (root.winfo_rootx(), root.winfo_rooty(),
                   root.winfo_width(), root.winfo_height())
            if (geo == prev and geo[2] > 1 and geo[3] > 1
                    and root.winfo_viewable()):
                break                  # position + size stable and mapped
            prev = geo
            time.sleep(0.02)
        recorder = ScreenRecorder(record_file, fps=args.record_fps)
        if not recorder.start(*geo):
            recorder = None

    game.tick()
    try:
        root.mainloop()
    finally:
        source.stop()
        if music is not None:
            music.stop()
        if recorder is not None:
            recorder.stop()
        # Record this session's high score in the player's personal folder.
        if profile is not None:
            profile.add_score(stage=game.stage, coins=game.coins_bank,
                              distance_m=game.distance_m)


if __name__ == "__main__":
    main()

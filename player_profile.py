"""Per-player profiles: a personal folder holding scores and game videos.

Each player gets a folder under ``python/players/<slug>/`` where their high
scores (``scores.json``) and recorded game movies accumulate across sessions.
The folder name is a filesystem-safe slug of the entered name; the original
name is preserved inside ``scores.json`` for display.
"""

import json
import re
import time
from pathlib import Path


def _slug(name):
    """A filesystem-safe folder name derived from a player's typed name."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_")
    return s or "guest"


class PlayerProfile:
    """A player's personal folder for high scores and recorded videos."""

    def __init__(self, name, base_dir=None):
        self.name = (name or "").strip() or "guest"
        base = Path(base_dir) if base_dir else Path(__file__).with_name("players")
        self.slug = _slug(self.name)
        self.folder = base / self.slug
        self.folder.mkdir(parents=True, exist_ok=True)
        self.scores_path = self.folder / "scores.json"

    # -- videos ---------------------------------------------------------------
    def new_video_path(self):
        """A fresh, timestamped MP4 path in this player's folder."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        return str(self.folder / ("game_%s.mp4" % ts))

    # -- scores ---------------------------------------------------------------
    def load_scores(self):
        """Return the saved score list (best first), or [] if none/unreadable."""
        if self.scores_path.exists():
            try:
                data = json.loads(self.scores_path.read_text())
                if isinstance(data, list):
                    return data
            except (ValueError, OSError):
                pass
        return []

    def _rank_key(self, entry):
        # Higher stage wins; then more total coins; then more distance.
        return (entry.get("stage", 0), entry.get("coins", 0),
                entry.get("distance_m", 0.0))

    def add_score(self, stage, coins, distance_m):
        """Append this session's result and persist the sorted list."""
        scores = self.load_scores()
        scores.append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": int(stage),
            "coins": int(coins),
            "distance_m": round(float(distance_m), 1),
        })
        scores.sort(key=self._rank_key, reverse=True)
        try:
            self.scores_path.write_text(json.dumps(scores, indent=2))
        except OSError:
            pass
        return scores

    def best(self):
        """The best saved score entry, or None if the player has none yet."""
        scores = self.load_scores()
        return scores[0] if scores else None

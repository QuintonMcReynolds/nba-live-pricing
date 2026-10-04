"""Versioned model + trading configuration, and a lock that detects unreviewed changes.

config/model.json holds every number that moves a price. config/model.lock records its
hash and the prices it produces for a fixed set of golden game states. CI fails if either
drifts, so a hand-edited parameter cannot reach production without a reviewed lock update
(scripts/update_lock.py), which shows exactly which prices moved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .live import TotalParams, WinProbParams

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@dataclass(frozen=True)
class Automation:
    vig: float = 0.045
    min_prob_move: float = 0.005        # publish only if fair prob moved this much...
    jump_suspend: float = 0.12          # ...and suspend if one event moves it this much
    suspend_events: int = 2             # events to wait before reopening after a jump
    stale_feed_seconds: float = 45.0    # default wall-clock silence that suspends...
    stale_seconds_by_action: dict = field(default_factory=dict)  # ...learned per last action
    decided_prob: float = 0.995         # suspend (not alert) once the outcome is ~decided
    max_override_pts: float = 4.0
    override_half_life_secs: float = 360.0  # game seconds; None-like (0) = no decay
    shadow_tolerance: float = 0.02      # unexplained |live - shadow| prob gap that alerts


@dataclass(frozen=True)
class ModelConfig:
    version: str
    win: WinProbParams
    total: TotalParams
    automation: Automation

    @staticmethod
    def load(path: Path = CONFIG_DIR / "model.json") -> ModelConfig:
        raw = json.loads(Path(path).read_text())
        return ModelConfig(raw["version"], WinProbParams(**raw["win"]),
                           TotalParams(**raw["total"]), Automation(**raw["automation"]))


def config_hash(path: Path = CONFIG_DIR / "model.json") -> str:
    canonical = json.dumps(json.loads(Path(path).read_text()), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


GOLDEN_STATES = [
    # (margin, secs_left, period, mu0, poss, points, elapsed, total0)
    (0, 2880, 1, 0.0, 0, 0, 0, 228.0),
    (0, 2880, 1, 6.0, 0, 0, 0, 228.0),
    (5, 1440, 3, 0.0, 1, 110, 1440, 228.0),
    (-8, 720, 4, 3.0, -1, 170, 2160, 228.0),
    (2, 60, 4, 0.0, 1, 214, 2820, 228.0),
    (-1, 10, 4, -2.0, 1, 220, 2870, 228.0),
    (12, 300, 4, -4.0, 0, 200, 2580, 235.0),
]


def golden_outputs(cfg: ModelConfig) -> list[dict]:
    from .live import total_projection, win_prob

    out = []
    for m, s, per, mu0, poss, pts, el, t0 in GOLDEN_STATES:
        p = float(win_prob(m, s, per, mu0, poss, cfg.win))
        tm, tsd = total_projection(pts, el, s, per, t0, cfg.total, m)
        out.append({"p_home": round(p, 6), "total_mean": round(float(tm), 4),
                    "total_sd": round(float(tsd), 4)})
    return out


def write_lock(path: Path = CONFIG_DIR / "model.lock") -> dict:
    cfg = ModelConfig.load()
    lock = {"config_sha256": config_hash(), "version": cfg.version,
            "golden": golden_outputs(cfg)}
    Path(path).write_text(json.dumps(lock, indent=2) + "\n")
    return lock

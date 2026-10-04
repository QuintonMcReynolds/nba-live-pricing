"""Turn fair probabilities and lines into two-way prices."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from scipy.special import ndtr


def american(p: float) -> int:
    """Implied probability (vig included) -> American odds."""
    p = min(max(p, 1e-4), 1 - 1e-4)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def implied(odds: int) -> float:
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def half_point(x: float) -> float:
    """Round to the nearest .5 - books hang half-point lines to avoid pushes."""
    return math.floor(x) + 0.5


@dataclass
class Quote:
    game_id: int
    seq: int
    ts: str
    secs_left: float
    margin: int                 # home minus away, current score
    status: str                 # OPEN | SUSPENDED | CLOSED
    p_home: float               # fair (no-vig) probability, override included
    home_ml: int
    away_ml: int
    spread_home: float          # home handicap, e.g. -4.5 = home favored by 4.5
    p_home_cover: float
    total_line: float
    p_over: float
    model_p_home: float         # untouched shadow model, for audit
    override_pts: float
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def two_way(p: float, vig: float) -> tuple[int, int]:
    """Proportional overround: each side's implied probability is scaled by (1 + vig)."""
    return american(p * (1 + vig)), american((1 - p) * (1 + vig))


def spread_quote(mean_margin: float, sd: float) -> tuple[float, float]:
    """Home handicap line at the median margin, and P(home covers it)."""
    line = -half_point(mean_margin)
    return line, float(ndtr((mean_margin + line) / sd))


def total_quote(mean_total: float, sd: float) -> tuple[float, float]:
    line = half_point(mean_total)
    return line, float(ndtr((mean_total - line) / sd))

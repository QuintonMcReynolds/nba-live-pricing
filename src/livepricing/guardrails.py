"""Checks that stop a bad price or a bad manual input from reaching customers."""

from __future__ import annotations

from .pricing import Quote, implied
from .state import GameState


def validate_override(points: float, trader: str, reason: str, a) -> tuple[bool, str]:
    if not trader or not reason:
        return False, "override needs a trader id and a reason"
    if abs(points) > a.max_override_pts:
        return False, f"|{points}| exceeds the {a.max_override_pts}-point override limit"
    return True, "ok"


def check_quote(q: Quote, st: GameState, a) -> list[str]:
    """Internal-consistency invariants every published quote must satisfy."""
    v = []
    if not 0.0 < q.p_home < 1.0:
        v.append(f"p_home {q.p_home} outside (0, 1)")
    over = implied(q.home_ml) + implied(q.away_ml) - 1
    if q.p_home * (1 + a.vig) < 0.99 and (1 - q.p_home) * (1 + a.vig) < 0.99 \
            and not (0.5 * a.vig <= over <= 2 * a.vig):
        v.append(f"moneyline overround {over:.3f} off target {a.vig}")
    # Coherence: the favourite on the moneyline must be the favourite on the spread.
    if q.p_home > 0.55 and q.spread_home > 0.5:
        v.append(f"home favored on ML ({q.p_home:.2f}) but getting points ({q.spread_home})")
    if q.p_home < 0.45 and q.spread_home < -0.5:
        v.append(f"home underdog on ML ({q.p_home:.2f}) but laying points ({q.spread_home})")
    if st.secs_left > 60 and not 0.3 <= q.p_home_cover <= 0.7:
        v.append(f"spread not centered: P(cover) {q.p_home_cover:.2f}")
    if q.total_line < st.points:
        v.append(f"total line {q.total_line} below points already scored {st.points}")
    if abs(q.override_pts) > a.max_override_pts + 1e-9:
        v.append(f"override {q.override_pts} beyond limit")
    return v

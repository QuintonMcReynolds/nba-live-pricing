"""Game state from live-feed events. One code path for training data and the live engine,
so features are computed identically offline and in production (no train/serve skew)."""

from __future__ import annotations

from dataclasses import dataclass, field

FG = {"2pt", "3pt"}
LAST_FT = {"1 of 1", "2 of 2", "3 of 3"}


@dataclass
class Event:
    game_id: int
    seq: int
    ts: str                 # ISO-8601 wall-clock time the action happened
    period: int
    secs_left: float        # regulation seconds left (OT: seconds left in the OT period)
    elapsed: float
    home_score: int
    away_score: int
    side: int               # +1 home team's action, -1 away, 0 none
    poss: int               # feed possession during the action: +1 home, -1 away, 0 unknown
    action_type: str
    sub_type: str = ""
    description: str = ""

    @staticmethod
    def from_row(r, home: str) -> Event:
        side = 0 if not isinstance(r.teamTricode, str) else (1 if r.teamTricode == home else -1)
        return Event(int(r.game_id), int(r.orderNumber), r.ts.isoformat(), int(r.period),
                     float(r.secs_left), float(r.elapsed), int(r.scoreHome), int(r.scoreAway),
                     side, int(r.poss), str(r.actionType),
                     r.subType if isinstance(r.subType, str) else "",
                     r.description if isinstance(r.description, str) else "")


def possession_after(e: Event) -> int:
    """Who has the ball once this action is over, from basketball rules (not the next event)."""
    made = not e.description.startswith("MISS")
    if e.action_type in FG and made and e.side:
        return -e.side
    if e.action_type == "freethrow" and e.sub_type in LAST_FT and made and e.side:
        return -e.side
    if e.action_type == "rebound" and e.side:
        return e.side
    if e.action_type == "turnover" and e.side:
        return -e.side
    if e.action_type == "period":
        return 0
    return e.poss


@dataclass
class GameState:
    game_id: int
    home: str
    away: str
    mu0: float              # pregame expected home margin
    total0: float           # pregame expected total
    period: int = 1
    secs_left: float = 2880.0
    elapsed: float = 0.0
    home_score: int = 0
    away_score: int = 0
    poss: int = 0
    final: bool = False
    last_seq: int = -1
    last_ts: str = ""
    n_events: int = 0
    last_action: str = ""
    history: list = field(default_factory=list, repr=False)

    @property
    def margin(self) -> int:
        return self.home_score - self.away_score

    @property
    def points(self) -> int:
        return self.home_score + self.away_score

    def apply(self, e: Event) -> bool:
        """Advance the state. Returns False for duplicate or out-of-order events (ignored)."""
        if e.seq <= self.last_seq:
            return False
        self.last_seq, self.last_ts = e.seq, e.ts
        self.period, self.secs_left, self.elapsed = e.period, e.secs_left, e.elapsed
        self.home_score, self.away_score = e.home_score, e.away_score
        self.poss = possession_after(e)
        self.final = e.action_type == "game" and e.sub_type == "end"
        self.n_events += 1
        self.last_action = e.action_type
        return True

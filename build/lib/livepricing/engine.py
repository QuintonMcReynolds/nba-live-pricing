"""Live pricing engine with trader automation.

For every feed event the engine:
  1. advances the game state (duplicate and out-of-order events are dropped),
  2. prices moneyline / spread / total from the live models,
  3. applies any trader override, decaying it as the game moves on,
  4. runs automation: suspend on large single-event jumps or a stale feed, reopen once
     the market has settled, and only publish when the price actually moved,
  5. re-prices with the locked "shadow" config and alerts if the live price has drifted
     from it by more than active overrides explain (an unreviewed manual change),
  6. checks the quote against guardrails before it is published.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np

from . import guardrails, pricing
from .config import ModelConfig
from .live import margin_mean_sd, remaining_fraction, total_projection, win_prob
from .state import Event, GameState


@dataclass
class Override:
    points: float               # shift to the expected home margin (+ = home better)
    trader: str
    reason: str
    applied_elapsed: float      # game seconds when applied
    decays: bool = True


@dataclass
class Market:
    state: GameState
    status: str = "OPEN"
    reason: str = ""
    cooldown: int = 0
    last_quote: pricing.Quote | None = None
    override: Override | None = None
    last_recv: float = 0.0


class PricingEngine:
    def __init__(self, config: ModelConfig, pregame: dict[int, dict],
                 shadow: ModelConfig | None = None, clock=time.monotonic):
        self.clock = clock                 # injectable for tests and simulated replays
        self.cfg = config
        self.shadow = shadow or config     # the locked, reviewed config
        self.pregame = pregame             # game_id -> {home, away, mu0, total0}
        self.markets: dict[int, Market] = {}
        self.latency_us: list[float] = []
        self.counters: dict[str, int] = defaultdict(int)
        self.audit: list[dict] = []

    # ---- trader / operator inputs ------------------------------------------------------

    def apply_override(self, game_id: int, points: float, trader: str, reason: str,
                       decays: bool = True) -> tuple[bool, str]:
        ok, why = guardrails.validate_override(points, trader, reason, self.cfg.automation)
        m = self.markets.get(game_id)
        if ok and m is None:
            ok, why = False, "unknown or not-started game"
        self.audit.append({"type": "override", "game_id": game_id, "points": points,
                           "trader": trader, "reason": reason, "accepted": ok, "why": why})
        if ok:
            m.override = Override(points, trader, reason, m.state.elapsed, decays)
            self.counters["overrides"] += 1
        return ok, why

    def set_params(self, config: ModelConfig, who: str) -> None:
        """Hot-swap live parameters (audited). The shadow config is left untouched."""
        self.audit.append({"type": "params", "who": who, "version": config.version})
        self.cfg = config

    # ---- feed --------------------------------------------------------------------------

    def on_event(self, e: Event) -> list[dict]:
        t0 = time.perf_counter()
        out = self._handle(e)
        self.latency_us.append((time.perf_counter() - t0) * 1e6)
        return out

    def on_tick(self, now: float | None = None) -> list[dict]:
        """Wall-clock check: suspend markets whose feed has gone quiet while the clock runs."""
        now = self.clock() if now is None else now
        out = []
        for m in self.markets.values():
            st = m.state
            quiet = now - m.last_recv
            a = self.cfg.automation
            limit = a.stale_seconds_by_action.get(st.last_action, a.stale_feed_seconds)
            if m.status == "OPEN" and not st.final and quiet > limit:
                out += self._suspend(m, f"stale feed ({quiet:.0f}s)")
        return out

    def _handle(self, e: Event) -> list[dict]:
        m = self.markets.get(e.game_id)
        if m is None:
            info = self.pregame.get(e.game_id)
            if info is None:
                self.counters["unknown_game"] += 1
                return []
            m = Market(GameState(e.game_id, info["home"], info["away"], info["mu0"],
                                 info["total0"]))
            self.markets[e.game_id] = m
        m.last_recv = self.clock()
        if not m.state.apply(e):
            self.counters["dropped_out_of_order"] += 1
            return []
        self.counters["events"] += 1
        msgs: list[dict] = []
        if m.state.final:
            q = self._quote(m, e, self.cfg)
            q.status, q.reason = "CLOSED", "final"
            m.status, m.last_quote = "CLOSED", q
            return [{"topic": "prices", **q.to_dict()}]

        q = self._quote(m, e, self.cfg)
        shadow = self._quote(m, e, self.shadow, with_override=False)
        q.model_p_home = shadow.p_home

        # Shadow audit: the live price should differ from the reviewed model only by the
        # override the trader declared. Anything else is an unexplained change.
        a = self.cfg.automation
        explained = self._quote(m, e, self.shadow).p_home
        divergent = abs(q.p_home - explained) > a.shadow_tolerance
        if divergent:
            reason = "unexplained divergence from reviewed model"
            if m.reason == reason:
                return msgs                      # already suspended and alerted
            self.counters["shadow_alerts"] += 1
            msgs.append({"topic": "alerts", "game_id": e.game_id, "seq": e.seq,
                         "type": "unexplained_divergence",
                         "live": round(q.p_home, 4), "expected": round(explained, 4),
                         "config_version": self.cfg.version})
            msgs += self._suspend(m, reason)
            return msgs

        if max(q.p_home, 1 - q.p_home) >= a.decided_prob:
            # Not an error: the outcome is all but settled, so stop taking bets.
            msgs += self._suspend(m, "outcome decided")
            return msgs

        violations = guardrails.check_quote(q, m.state, a)
        if violations:
            reason = "guardrail: " + "; ".join(violations)
            if not m.reason.startswith("guardrail"):
                self.counters["guardrail_violations"] += 1
                msgs.append({"topic": "alerts", "game_id": e.game_id, "seq": e.seq,
                             "type": "guardrail", "violations": violations})
            msgs += self._suspend(m, reason)
            return msgs

        prev = m.last_quote
        jump = abs(q.p_home - prev.p_home) if prev else 0.0
        if m.status == "SUSPENDED" and m.reason.startswith(
                ("unexplained", "guardrail", "outcome decided")):
            self._reopen(m)                       # the blocking condition has cleared
        elif m.status == "OPEN" and prev and jump >= a.jump_suspend and m.state.secs_left > 120:
            msgs += self._suspend(m, f"price jump {jump:.0%}")
            m.cooldown = a.suspend_events
        elif m.status == "SUSPENDED":
            m.cooldown -= 1                       # jump settled / feed resumed
            if m.cooldown <= 0:
                self._reopen(m)

        q.status, q.reason = m.status, m.reason
        changed = (prev is None or jump >= a.min_prob_move or q.status != prev.status
                   or q.spread_home != prev.spread_home or q.total_line != prev.total_line)
        if changed:
            m.last_quote = q
            self.counters["quotes"] += 1
            msgs.append({"topic": "prices", **q.to_dict()})
        return msgs

    # ---- pricing -----------------------------------------------------------------------

    def _override_pts(self, m: Market) -> float:
        o = m.override
        if o is None:
            return 0.0
        hl = self.cfg.automation.override_half_life_secs
        if not o.decays or hl <= 0:
            return o.points
        return o.points * 0.5 ** ((m.state.elapsed - o.applied_elapsed) / hl)

    def _quote(self, m: Market, e: Event, cfg: ModelConfig, with_override=True) -> pricing.Quote:
        st = m.state
        ov = self._override_pts(m) if with_override else 0.0
        f = float(remaining_fraction(st.secs_left, st.period))
        # A trader's view on team strength shifts the drift over the remaining game.
        mu0 = st.mu0 + ov
        p = float(win_prob(st.margin, st.secs_left, st.period, mu0, st.poss, cfg.win))
        mean_margin, sd_margin = margin_mean_sd(st.margin, max(f, 1e-6), mu0, st.poss, cfg.win)
        spread, p_cover = pricing.spread_quote(float(mean_margin), float(sd_margin))
        tm, tsd = total_projection(st.points, st.elapsed, st.secs_left, st.period,
                                   st.total0, cfg.total, st.margin)
        total, p_over = pricing.total_quote(float(tm), float(tsd))
        home_ml, away_ml = pricing.two_way(p, cfg.automation.vig)
        return pricing.Quote(e.game_id, e.seq, e.ts, st.secs_left, st.margin, m.status, p,
                             home_ml, away_ml, spread,
                             p_cover, total, p_over, p, round(ov, 3), m.reason)

    def _reopen(self, m: Market) -> None:
        m.status, m.reason, m.cooldown = "OPEN", "", 0
        self.counters["reopens"] += 1

    def _suspend(self, m: Market, reason: str) -> list[dict]:
        if m.status == "SUSPENDED" and m.reason == reason:
            return []
        m.status, m.reason = "SUSPENDED", reason
        self.counters["suspensions"] += 1
        if m.last_quote is not None:
            m.last_quote = replace(m.last_quote, status="SUSPENDED", reason=reason)
            return [{"topic": "prices", **m.last_quote.to_dict()}]
        return []

    # ---- reporting ---------------------------------------------------------------------

    def metrics(self) -> dict:
        lat = np.asarray(self.latency_us) if self.latency_us else np.zeros(1)
        return {**self.counters,
                "latency_us_p50": float(np.percentile(lat, 50)),
                "latency_us_p99": float(np.percentile(lat, 99)),
                "latency_us_max": float(lat.max()),
                "markets": len(self.markets),
                "suspended": sum(m.status == "SUSPENDED" for m in self.markets.values())}


def parse_ts(ts: str) -> float:
    return datetime.fromisoformat(ts).timestamp()

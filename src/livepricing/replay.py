"""Replay recorded live-feed events onto a bus, in wall-clock order across a slate of games,
and run the pricing engine as a bus consumer."""

from __future__ import annotations

import time
from dataclasses import asdict

import pandas as pd

from .bus import TOPIC_ALERTS, TOPIC_EVENTS, TOPIC_PRICES
from .engine import PricingEngine
from .state import Event


def to_events(events: pd.DataFrame, games: pd.DataFrame) -> list[Event]:
    home = games.set_index("game_id")["home"]
    ev = events.sort_values(["ts", "game_id", "orderNumber"])
    return [Event.from_row(r, home[r.game_id]) for r in ev.itertuples()]


def publish(bus, events: list[Event], speed: float | None = None) -> int:
    """Publish events; speed=None is as fast as possible, speed=60 is 60x real time."""
    prev = None
    for e in events:
        if speed and prev is not None:
            gap = (pd.Timestamp(e.ts) - pd.Timestamp(prev)).total_seconds() / speed
            if gap > 0:
                time.sleep(min(gap, 5.0))
        bus.publish(TOPIC_EVENTS, str(e.game_id), asdict(e))
        prev = e.ts
    bus.flush()
    return len(events)


def run_engine(bus, engine: PricingEngine, tick_every: int = 50) -> int:
    """Consume events, publish prices and alerts. Returns events processed."""
    n = 0
    for _, value in bus.consume([TOPIC_EVENTS]):
        for msg in engine.on_event(Event(**value)):
            topic = TOPIC_PRICES if msg.pop("topic") == "prices" else TOPIC_ALERTS
            bus.publish(topic, str(msg["game_id"]), msg)
        n += 1
        if n % tick_every == 0:
            for msg in engine.on_tick():
                msg.pop("topic")
                bus.publish(TOPIC_PRICES, str(msg["game_id"]), msg)
    bus.flush()
    return n

import pytest

from livepricing.config import ModelConfig
from livepricing.state import Event


def make_event(seq, secs_left, home, away, action="2pt", side=1, poss=1, sub="", desc="",
               game_id=1, period=None):
    period = period or min(int((2880 - secs_left) // 720) + 1, 4)
    return Event(game_id, seq, f"2025-01-01T00:{seq // 60 % 60:02d}:{seq % 60:02d}+00:00",
                 period, float(secs_left), float(2880 - secs_left), home, away, side, poss,
                 action, sub, desc)


@pytest.fixture
def cfg():
    return ModelConfig.load()


@pytest.fixture
def scripted_game():
    """A home win: tied early, home pulls away late; one event per 30 game-seconds."""
    events, h, a = [], 0, 0
    for i, t in enumerate(range(2850, 0, -30)):
        if i % 2 == 0:
            h += 2 if t < 1200 else 1
            events.append(make_event(i + 1, t, h, a, side=1, poss=1))
        else:
            a += 1
            events.append(make_event(i + 1, t, h, a, side=-1, poss=-1))
    events.append(make_event(len(events) + 1, 0, h, a, action="game", side=0, poss=0, sub="end"))
    return events


@pytest.fixture
def pregame():
    return {1: {"home": "PHX", "away": "DEN", "mu0": 1.5, "total0": 228.0}}

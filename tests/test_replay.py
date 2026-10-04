from livepricing.bus import TOPIC_EVENTS, TOPIC_PRICES, InMemoryBus
from livepricing.engine import PricingEngine
from livepricing.replay import publish, run_engine


def test_bus_round_trip(cfg, pregame, scripted_game):
    bus = InMemoryBus()
    assert publish(bus, scripted_game + scripted_game[:5]) == len(scripted_game) + 5
    eng = PricingEngine(cfg, pregame)
    assert run_engine(bus, eng) == len(bus.log[TOPIC_EVENTS])
    prices = bus.log[TOPIC_PRICES]
    assert prices[-1]["status"] == "CLOSED"
    assert eng.counters["dropped_out_of_order"] == 5

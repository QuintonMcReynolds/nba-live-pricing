"""End-to-end through a real broker. Runs in CI against a Redpanda service container;
skipped locally unless KAFKA_BOOTSTRAP is set."""

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("KAFKA_BOOTSTRAP"), reason="no broker")


def test_events_in_prices_out(cfg, pregame, scripted_game):
    from livepricing import bus as b
    from livepricing.engine import PricingEngine
    from livepricing.replay import publish, run_engine

    kb = b.KafkaBus(os.environ["KAFKA_BOOTSTRAP"], group_id=f"test-{uuid.uuid4()}",
                    idle_timeout=5)
    publish(kb, scripted_game)
    eng = PricingEngine(cfg, pregame)
    n = run_engine(kb, eng)
    assert n == len(scripted_game)
    reader = b.KafkaBus(os.environ["KAFKA_BOOTSTRAP"], group_id=f"reader-{uuid.uuid4()}",
                        idle_timeout=5)
    prices = [v for _, v in reader.consume([b.TOPIC_PRICES])]
    assert prices[-1]["status"] == "CLOSED"

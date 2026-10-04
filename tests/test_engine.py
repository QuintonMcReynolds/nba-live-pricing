from dataclasses import replace

from livepricing.engine import PricingEngine
from livepricing.live import WinProbParams

from .conftest import make_event


def run(engine, events):
    msgs = []
    for e in events:
        msgs += engine.on_event(e)
    return msgs


def test_full_game_publishes_and_closes(cfg, pregame, scripted_game):
    eng = PricingEngine(cfg, pregame)
    msgs = run(eng, scripted_game)
    prices = [m for m in msgs if m["topic"] == "prices"]
    assert prices[-1]["status"] == "CLOSED" and prices[-1]["p_home"] == 1.0
    assert eng.metrics()["latency_us_p99"] < 5000


def test_only_material_moves_are_published(cfg, pregame):
    eng = PricingEngine(cfg, pregame)
    eng.on_event(make_event(1, 2000, 40, 38))
    # Substitutions during a dead ball: same score and clock - nothing to publish.
    subs = [make_event(i, 2000, 40, 38, action="substitution", side=0, poss=-1)
            for i in range(2, 12)]
    published = [m for e in subs for m in eng.on_event(e) if m["topic"] == "prices"]
    assert published == []


def test_unknown_game_and_duplicates_ignored(cfg, pregame, scripted_game):
    eng = PricingEngine(cfg, pregame)
    run(eng, scripted_game[:10])
    assert eng.on_event(scripted_game[5]) == []
    assert eng.counters["dropped_out_of_order"] == 1
    assert eng.on_event(make_event(1, 2800, 0, 0, game_id=99)) == []


def test_override_moves_price_and_decays(cfg, pregame, scripted_game):
    eng = PricingEngine(cfg, pregame)
    run(eng, scripted_game[:5])
    before = eng.markets[1].last_quote.p_home
    ok, _ = eng.apply_override(1, 3.0, "trader_a", "late scratch: DEN starter out")
    assert ok
    run(eng, scripted_game[5:7])
    q = eng.markets[1].last_quote
    assert q.p_home > before and q.override_pts > 2.5
    assert q.model_p_home < q.p_home              # shadow shows the untouched model
    run(eng, scripted_game[7:60])
    assert abs(eng.markets[1].last_quote.override_pts) < 1.0


def test_override_guardrails(cfg, pregame, scripted_game):
    eng = PricingEngine(cfg, pregame)
    run(eng, scripted_game[:3])
    assert not eng.apply_override(1, 10.0, "trader_a", "gut feel")[0]
    assert not eng.apply_override(1, 1.0, "", "no name")[0]
    assert len(eng.audit) == 2 and not any(a["accepted"] for a in eng.audit)


def test_large_jump_suspends_then_reopens(cfg, pregame):
    eng = PricingEngine(cfg, pregame)
    eng.on_event(make_event(1, 1500, 50, 50))
    msgs = eng.on_event(make_event(2, 1490, 75, 50))      # a 25-point swing in one event
    assert eng.markets[1].status == "SUSPENDED"
    assert any(m.get("status") == "SUSPENDED" for m in msgs)
    eng.on_event(make_event(3, 1480, 75, 52))
    eng.on_event(make_event(4, 1470, 77, 52))
    assert eng.markets[1].status == "OPEN"


def test_stale_feed_suspends_and_feed_resumption_reopens(cfg, pregame):
    now = [0.0]
    eng = PricingEngine(cfg, pregame, clock=lambda: now[0])
    eng.on_event(make_event(1, 1500, 50, 48))
    now[0] = 100.0
    eng.on_tick()
    assert eng.markets[1].reason.startswith("stale feed")
    eng.on_event(make_event(2, 1490, 52, 48))
    assert eng.markets[1].status == "OPEN"


def test_unreviewed_param_change_is_caught(cfg, pregame, scripted_game):
    eng = PricingEngine(cfg, pregame)
    run(eng, scripted_game[:20])
    hot = replace(cfg, version="hotfix", win=WinProbParams(sigma=6.0, poss_value=0.9))
    eng.set_params(hot, "someone")
    msgs = run(eng, scripted_game[20:22])
    assert any(m.get("type") == "unexplained_divergence" for m in msgs)
    assert eng.markets[1].status == "SUSPENDED"
    eng.set_params(cfg, "someone")                         # revert to the reviewed config
    run(eng, scripted_game[22:23])
    assert eng.markets[1].status == "OPEN"

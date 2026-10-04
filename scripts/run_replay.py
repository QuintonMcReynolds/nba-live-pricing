"""Replay the busiest night of 2025-26 through the bus and pricing engine, with incidents.

Incidents injected (simulation clock = the feed's own wall-clock timestamps):
  1. ~2% of events re-sent late (live feeds resend and correct actions)
  2. a 3-minute feed outage on one game in the third quarter
  3. a trader override (+3 points, "injury news") on another game in the second quarter
  4. an unreviewed hot edit of a model parameter for 10 minutes, then a revert

Also replays the whole 2025-26 season with no incidents to measure throughput and latency.
Writes artifacts/replay_report.json, artifacts/pregame_slate.json,
artifacts/slate_events.jsonl and docs/ figures.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from livepricing import data, plots
from livepricing.bus import TOPIC_ALERTS, TOPIC_EVENTS, TOPIC_PRICES, InMemoryBus
from livepricing.config import ModelConfig
from livepricing.engine import PricingEngine, parse_ts
from livepricing.live import WinProbParams
from livepricing.replay import to_events
from livepricing.state import Event

ROOT = Path(__file__).resolve().parents[1]
ART, DOCS = ROOT / "artifacts", ROOT / "docs"
SEASON = "2025-26"


def pregame_lookup(games: pd.DataFrame, pre: pd.DataFrame) -> dict[int, dict]:
    g = games.set_index("game_id").join(pre[["pred_margin", "pred_total"]], how="inner")
    return {int(i): {"home": r.home, "away": r.away, "mu0": round(float(r.pred_margin), 3),
                     "total0": round(float(r.pred_total), 2)} for i, r in g.iterrows()}


def slate_replay(games, events, pregame, cfg) -> dict:
    sg = games[games.season == SEASON]
    date = sg.groupby("date").size().idxmax()
    slate = sg[sg.date == date].sort_values("tipoff")
    ids = slate.game_id.tolist()
    evs = to_events(events[events.game_id.isin(ids)], slate)
    slate_pre = {i: pregame[i] for i in ids}
    (ART / "pregame_slate.json").write_text(json.dumps({str(k): v for k, v in slate_pre.items()},
                                                       indent=1))
    with open(ART / "slate_events.jsonl", "w") as fh:
        for e in evs:
            fh.write(json.dumps(asdict(e)) + "\n")

    rng = np.random.default_rng(7)
    # Override goes on the most competitive game at the end of Q1 (where it is visible);
    # the outage goes on a different game.
    from livepricing.live import win_prob
    q1 = {}
    for e in evs:
        if e.period == 1:
            q1[e.game_id] = float(win_prob(e.home_score - e.away_score, e.secs_left, 1,
                                           pregame[e.game_id]["mu0"], 0, cfg.win))
    override_game = min(q1, key=lambda g: abs(q1[g] - 0.5))
    outage_game = next(g for g in ids if g != override_game)
    teams = {i: f"{pregame[i]['away']} @ {pregame[i]['home']}" for i in ids}

    # Build the incident-laden stream.
    stream, resent, lost = [], 0, 0
    for e in evs:
        t = parse_ts(e.ts)
        if e.game_id == outage_game and e.period == 3 and 300 < e.secs_left - 720 < 480:
            lost += 1                                 # outage: these events never arrive
            continue
        stream.append((t, e))
        if rng.random() < 0.02:
            stream.append((t + rng.uniform(1, 20), e))  # resent late
            resent += 1
    stream.sort(key=lambda x: x[0])
    t_start = stream[0][0]
    hot_from = t_start + 75 * 60                      # 75 minutes into the night
    hot_to = hot_from + 10 * 60

    now = [t_start]
    eng = PricingEngine(cfg, slate_pre, clock=lambda: now[0])
    bus = InMemoryBus()
    for _, e in stream:
        bus.publish(TOPIC_EVENTS, str(e.game_id), asdict(e))

    hot_cfg = replace(cfg, version=cfg.version + "-hotedit",
                      win=WinProbParams(cfg.win.sigma * 0.75, cfg.win.poss_value))
    flags = {"override": False, "hot": False, "reverted": False}
    override_window = None
    i = 0
    for _, value in bus.consume([TOPIC_EVENTS]):
        e = Event(**value)
        now[0] = stream[i][0]
        i += 1
        if not flags["hot"] and now[0] >= hot_from:
            eng.set_params(hot_cfg, "ops_user (unreviewed)")
            flags["hot"] = True
        if flags["hot"] and not flags["reverted"] and now[0] >= hot_to:
            eng.set_params(cfg, "ops_user (revert)")
            flags["reverted"] = True
        for msg in eng.on_event(e):
            topic = TOPIC_PRICES if msg.pop("topic") == "prices" else TOPIC_ALERTS
            bus.publish(topic, str(msg["game_id"]), msg)
        if (not flags["override"] and e.game_id == override_game and e.period == 2
                and override_game in eng.markets):
            ok, why = eng.apply_override(override_game, 3.0, "trader_17",
                                         "injury news: away starter ruled out at half")
            flags["override"] = ok
            override_window = ((2880 - e.secs_left) / 60, 48)
        for msg in eng.on_tick():
            msg.pop("topic")
            bus.publish(TOPIC_PRICES, str(msg["game_id"]), msg)

    prices = pd.DataFrame(bus.log[TOPIC_PRICES])
    alerts = pd.DataFrame(bus.log[TOPIC_ALERTS])
    susp = prices[prices.status == "SUSPENDED"].drop_duplicates(["game_id", "reason"])
    reason_kind = susp["reason"].str.extract(
        r"^(stale feed|price jump|unexplained|guardrail|outcome decided)")[0]
    stale = susp[susp["reason"].str.startswith("stale feed")]
    stale_detail = [{"game": teams[int(r.game_id)], "period": int(4 - r.secs_left // 720)
                     if r.secs_left > 0 else 4, "clock_left": round(r.secs_left % 720),
                     "reason": r.reason} for r in stale.itertuples()]

    hot_games = alerts[alerts.type == "unexplained_divergence"].game_id.nunique() \
        if len(alerts) else 0
    report = {
        "date": str(date), "games": len(ids), "events_in_feed": len(evs),
        "events_delivered": len(stream), "resent_events": resent,
        "events_lost_in_outage": lost,
        "engine": eng.metrics(),
        "quotes_published": int((prices.status != "SUSPENDED").sum()),
        "suspension_episodes": {k: int(v) for k, v in Counter(reason_kind.fillna("other")).items()},
        "stale_feed_episodes": stale_detail,
        "alerts": {k: int(v) for k, v in Counter(alerts.get("type", pd.Series())).items()},
        "games_flagged_during_hot_edit": int(hot_games),
        "outage_game": teams[outage_game], "override_game": teams[override_game],
        "audit": eng.audit,
    }
    # Figures: the override game and the outage game.
    for gid, name, win in [(override_game, "override", override_window),
                           (outage_game, "outage", None)]:
        gp = prices[prices.game_id == gid]
        label = ("+3 pt trader override at Q2" if name == "override"
                 else "3-minute feed outage in Q3")
        plots.game_trace(gp, f"{teams[gid]}, {date}: {label}", DOCS / f"trace_{name}.png",
                         override_window=win)
    return report


def season_throughput(games, events, pregame, cfg) -> dict:
    sg = games[games.season == SEASON]
    evs = to_events(events[events.game_id.isin(sg.game_id)], sg)
    eng = PricingEngine(cfg, pregame)
    t = time.perf_counter()
    quotes = 0
    for e in evs:
        quotes += sum(m["topic"] == "prices" for m in eng.on_event(e))
    wall = time.perf_counter() - t
    m = eng.metrics()
    return {"games": len(sg), "events": len(evs), "quotes": quotes,
            "wall_seconds": round(wall, 2), "events_per_second": round(len(evs) / wall),
            "latency_us_p50": round(m["latency_us_p50"], 1),
            "latency_us_p99": round(m["latency_us_p99"], 1),
            "latency_us_max": round(m["latency_us_max"], 1)}


def main() -> None:
    games, events = data.build()
    pre = pd.read_parquet(ART / "pregame_predictions.parquet")
    pregame = pregame_lookup(games, pre)
    cfg = ModelConfig.load()
    report = {"slate": slate_replay(games, events, pregame, cfg),
              "season": season_throughput(games, events, pregame, cfg)}
    (ART / "replay_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report["slate"].items() if k != "audit"}, indent=2,
                     default=str))
    print(json.dumps(report["season"], indent=2))


if __name__ == "__main__":
    main()

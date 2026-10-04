"""Tune the ratings decay on 2023-24, then walk-forward test on 2024-25 and 2025-26.

Writes artifacts/pregame_predictions.parquet and artifacts/pregame_metrics.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from livepricing import backtest, data

ART = Path(__file__).resolve().parents[1] / "artifacts"
VALID, TEST = ["2023-24"], ["2024-25", "2025-26"]
HALF_LIVES = [45, 90, 180]


def main() -> None:
    ART.mkdir(exist_ok=True)
    games, _ = data.build()
    g = games.set_index("game_id")
    elo = backtest.elo_predictions(games)

    tuning = {}
    for hl in HALF_LIVES:
        t = time.time()
        pred = backtest.walk_forward(games, VALID, hl).set_index("game_id")
        pred = pred.rename(columns={"margin": "pred_margin", "total": "pred_total"}).join(g)
        tuning[hl] = {**backtest.score(pred, "p_home"),
                      **backtest.score_distribution(pred, "margin", "pred_margin", "margin_sd")}
        print(f"half-life {hl}: {tuning[hl]}  ({time.time() - t:.0f}s)", flush=True)
    best = min(tuning, key=lambda h: tuning[h]["log_loss"])

    pred = backtest.walk_forward(games, TEST, best).set_index("game_id")
    pred = pred.rename(columns={"margin": "pred_margin", "total": "pred_total"}).join(g)
    pred["p_elo"] = elo.reindex(pred.index)
    pred.to_parquet(ART / "pregame_predictions.parquet")

    metrics = {"tuning_2023_24": tuning, "chosen_half_life_days": best, "test": {}}
    for season, d in [("2024-25 + 2025-26", pred), *pred.groupby("season")]:
        metrics["test"][season] = {
            "bayes": backtest.score(d, "p_home"),
            "elo": backtest.score(d, "p_elo"),
            "home_rate_baseline": backtest.score(
                d.assign(p_base=games[games.season == "2022-23"].home_win.mean()), "p_base"),
            "margin": backtest.score_distribution(d, "margin", "pred_margin", "margin_sd"),
            "total": backtest.score_distribution(d, "total", "pred_total", "total_sd"),
        }
    (ART / "pregame_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()

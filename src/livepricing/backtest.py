"""Walk-forward evaluation of pregame forecasts: refit on a schedule, predict the next block."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from . import ratings


def walk_forward(games: pd.DataFrame, seasons: list[str], half_life_days: float,
                 refit_days: int = 7, **fit_kw) -> pd.DataFrame:
    """Predict every game in `seasons` from a model fit only on games before its block."""
    logging.getLogger("pymc").setLevel(logging.ERROR)
    games = games.assign(date=pd.to_datetime(games["date"]))
    target = games[games["season"].isin(seasons)]
    out = []
    for _season, sg in target.groupby("season"):
        start, end = sg["date"].min(), sg["date"].max()
        cuts = pd.date_range(start, end + pd.Timedelta(days=refit_days), freq=f"{refit_days}D")
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            block = sg[(sg["date"] >= lo) & (sg["date"] < hi)]
            if block.empty:
                continue
            post, _ = ratings.fit(games, lo, half_life_days=half_life_days, **fit_kw)
            for r in block.itertuples():
                p = post.predict(r.home, r.away)
                out.append({"game_id": r.game_id, "fit_asof": lo, **p})
    return pd.DataFrame(out)


def elo_predictions(games: pd.DataFrame) -> pd.Series:
    """Sequential Elo over all games (first season is warm-up); pre-game P(home win)."""
    elo = ratings.Elo()
    p = {}
    for r in games.sort_values(["tipoff", "game_id"]).itertuples():
        p[r.game_id] = elo.p_home(r.home, r.away)
        elo.update(r.home, r.away, r.margin, r.season)
    return pd.Series(p, name="p_elo")


def crps_normal(y, mu, sd):
    z = (y - mu) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def score(df: pd.DataFrame, p_col: str) -> dict[str, float]:
    p = np.clip(df[p_col].to_numpy(), 1e-6, 1 - 1e-6)
    y = df["home_win"].to_numpy()
    return {
        "games": len(df),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p > 0.5) == y)),
    }


def score_distribution(df: pd.DataFrame, target: str, mean: str, sd: str) -> dict[str, float]:
    y, m, s = df[target], df[mean], df[sd]
    lo, hi = stats.norm.ppf(0.1, m, s), stats.norm.ppf(0.9, m, s)
    return {
        "mae": float(np.mean(np.abs(y - m))),
        "rmse": float(np.sqrt(np.mean((y - m) ** 2))),
        "crps": float(np.mean(crps_normal(y, m, s))),
        "coverage_80": float(np.mean((y >= lo) & (y <= hi))),
    }

"""Load the NBA live-data (cdn.nba.com) play-by-play feed and derive games and event streams.

Source: public mirror at https://github.com/shufinskiy/nba_data (`cdnnba_<year>`; the year is
the year the season starts). Each row is one live-feed action with the running score and the
wall-clock time it happened, so a season can be replayed as a realistic event stream.
"""

from __future__ import annotations

import re
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
URL = "https://github.com/shufinskiy/nba_data/raw/main/datasets/cdnnba_{year}.tar.xz"
DEFAULT_YEARS = (2022, 2023, 2024, 2025)
COLUMNS = ["gameId", "actionNumber", "orderNumber", "period", "clock", "timeActual",
           "actionType", "subType", "teamId", "teamTricode", "possession", "scoreHome",
           "scoreAway", "description"]
REG_SECONDS = 48 * 60


def season_label(year: int) -> str:
    return f"{year}-{(year + 1) % 100:02d}"


def download(years=DEFAULT_YEARS, raw_dir: Path = RAW_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        if (raw_dir / f"cdnnba_{year}.csv").exists():
            continue
        archive = raw_dir / f"cdnnba_{year}.tar.xz"
        print(f"downloading {season_label(year)} ...")
        urllib.request.urlretrieve(URL.format(year=year), archive)
        with tarfile.open(archive) as tar:
            tar.extractall(raw_dir, filter="data")


_CLOCK = re.compile(r"PT(\d+)M([\d.]+)S")


def clock_seconds(clock: pd.Series) -> np.ndarray:
    parts = clock.str.extract(_CLOCK).astype(float)
    return (parts[0] * 60 + parts[1]).to_numpy()


def seconds_remaining(period: np.ndarray, clock_left: np.ndarray) -> np.ndarray:
    """Seconds left in regulation; in overtime, seconds left in the current OT period."""
    reg = np.maximum(4 - period, 0) * 720 + clock_left
    return np.where(period <= 4, reg, clock_left)


def load_events(year: int, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One row per live-feed action, sorted in feed order, with game state attached."""
    df = pd.read_csv(raw_dir / f"cdnnba_{year}.csv", usecols=COLUMNS, low_memory=False)
    df = df[df["gameId"].astype(str).str.startswith("2")]  # regular season only
    df = df.sort_values(["gameId", "orderNumber"]).reset_index(drop=True)
    df["season"] = season_label(year)
    df["clock_left"] = clock_seconds(df["clock"])
    df["secs_left"] = seconds_remaining(df["period"].to_numpy(), df["clock_left"].to_numpy())
    df["elapsed"] = np.where(
        df["period"] <= 4,
        (df["period"] - 1) * 720 + (720 - df["clock_left"]),
        REG_SECONDS + (df["period"] - 5) * 300 + (300 - df["clock_left"]),
    )
    df["ts"] = pd.to_datetime(df["timeActual"], utc=True, format="ISO8601", errors="coerce")
    df[["scoreHome", "scoreAway"]] = df.groupby("gameId")[["scoreHome", "scoreAway"]].ffill()
    df[["scoreHome", "scoreAway"]] = df[["scoreHome", "scoreAway"]].fillna(0).astype(int)
    df["margin"] = df["scoreHome"] - df["scoreAway"]
    return df.drop(columns=["clock", "timeActual"])


def home_away(events: pd.DataFrame) -> pd.DataFrame:
    """Infer home/away tricodes: the team credited when scoreHome increases is home."""
    e = events[events["teamTricode"].notna()].copy()
    g = e.groupby("gameId")
    e["d_home"] = g["scoreHome"].diff().fillna(e["scoreHome"])
    e["d_away"] = g["scoreAway"].diff().fillna(e["scoreAway"])
    home = e[(e["d_home"] > 0) & (e["d_away"] == 0)].groupby("gameId")["teamTricode"].agg(
        lambda s: s.value_counts().index[0])
    away = e[(e["d_away"] > 0) & (e["d_home"] == 0)].groupby("gameId")["teamTricode"].agg(
        lambda s: s.value_counts().index[0])
    out = pd.DataFrame({"home": home, "away": away}).dropna()
    return out[out["home"] != out["away"]]


def attach_possession(events: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """possession: +1 home has the ball, -1 away, 0 unknown (feed reports a team id)."""
    ids = events.dropna(subset=["teamId", "teamTricode"]).drop_duplicates("teamId")
    tricode = dict(zip(ids["teamId"].astype(int), ids["teamTricode"]))
    team = events["possession"].fillna(0).astype(int).map(tricode)
    home = events["game_id"].map(games.set_index("game_id")["home"])
    away = events["game_id"].map(games.set_index("game_id")["away"])
    events = events.copy()
    events["poss"] = np.select([team == home, team == away], [1, -1], 0)
    return events.drop(columns=["teamId", "possession"])


def games_table(events: pd.DataFrame) -> pd.DataFrame:
    """One row per game: teams, date, final score, margin, whether it went to overtime."""
    last = events.groupby("gameId").tail(1).set_index("gameId")
    first_ts = events.groupby("gameId")["ts"].min()
    g = home_away(events).join(last[["season", "scoreHome", "scoreAway", "period"]], how="inner")
    g["tipoff"] = first_ts.reindex(g.index)
    g["date"] = g["tipoff"].dt.tz_convert("America/New_York").dt.date
    g["margin"] = g["scoreHome"] - g["scoreAway"]
    g["total"] = g["scoreHome"] + g["scoreAway"]
    g["home_win"] = (g["margin"] > 0).astype(int)
    g["overtime"] = (g["period"] > 4).astype(int)
    return g.rename_axis("game_id").reset_index().sort_values(["tipoff", "game_id"])


def build(years=DEFAULT_YEARS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse all seasons and cache compact parquet files; returns (games, events)."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    gp, ep = PROC_DIR / "games.parquet", PROC_DIR / "events.parquet"
    if gp.exists() and ep.exists():
        return pd.read_parquet(gp), pd.read_parquet(ep)
    download(years)
    games, events = [], []
    for y in years:
        ev = load_events(y)
        games.append(games_table(ev))
        events.append(ev)
    games = pd.concat(games, ignore_index=True)
    events = pd.concat(events, ignore_index=True)
    events = events[events["gameId"].isin(games["game_id"])].rename(columns={"gameId": "game_id"})
    events = attach_possession(events, games)
    games.to_parquet(gp)
    events.to_parquet(ep)
    return games, events


def feed_silence_thresholds(events: pd.DataFrame, q: float = 0.999,
                            floor: float = 45.0) -> dict[str, float]:
    """How long the feed normally goes quiet after each kind of action (wall-clock seconds).

    A timeout or period break legitimately silences the feed for minutes; a missed shot
    does not. The q-quantile per action type, learned on a training season, becomes the
    stale-feed threshold, so a dead ball is not mistaken for an outage.
    """
    e = events.sort_values(["game_id", "orderNumber"])
    gap = e.groupby("game_id")["ts"].shift(-1).sub(e["ts"]).dt.total_seconds()
    t = gap.groupby(e["actionType"]).quantile(q)
    return {k: round(max(float(v), floor), 1) for k, v in t.dropna().items()}

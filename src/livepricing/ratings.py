"""Bayesian pregame team ratings, fit by MCMC (PyMC / NUTS).

    home_pts ~ mu + hca/2 + off[home] - def[away]
    away_pts ~ mu - hca/2 + off[away] - def[home]

with a bivariate-normal error (correlated through pace), partial pooling on team ratings,
and exponential time-decay weights so that recent games count more. Refitting on a schedule
with only games played before each date gives leak-free forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class RatingsPosterior:
    teams: list[str]
    mu: np.ndarray        # (draws,)
    hca: np.ndarray       # (draws,)
    off: np.ndarray       # (draws, teams)
    dfn: np.ndarray       # (draws, teams)
    sigma: np.ndarray     # (draws,)  per-team-score residual sd
    rho: np.ndarray       # (draws,)  correlation of the two scores

    def predict(self, home: str, away: str) -> dict[str, float]:
        """Posterior-predictive margin and total (home minus away) for one matchup."""
        h, a = self.teams.index(home), self.teams.index(away)
        hp = self.mu + self.hca / 2 + self.off[:, h] - self.dfn[:, a]
        ap = self.mu - self.hca / 2 + self.off[:, a] - self.dfn[:, h]
        mean_margin, mean_total = hp - ap, hp + ap
        var_margin = 2 * self.sigma**2 * (1 - self.rho)
        var_total = 2 * self.sigma**2 * (1 + self.rho)
        # Mixture over posterior draws: law of total variance.
        m, t = mean_margin.mean(), mean_total.mean()
        sd_m = float(np.sqrt(var_margin.mean() + mean_margin.var()))
        sd_t = float(np.sqrt(var_total.mean() + mean_total.var()))
        p_home = float(np.mean(stats.norm.sf(0, mean_margin, np.sqrt(var_margin))))
        return {"margin": float(m), "margin_sd": sd_m, "total": float(t), "total_sd": sd_t,
                "p_home": p_home}


def decay_weights(dates: pd.Series, asof: pd.Timestamp, half_life_days: float) -> np.ndarray:
    age = (asof - pd.to_datetime(dates)).dt.days.to_numpy()
    return 0.5 ** (age / half_life_days)


def fit(games: pd.DataFrame, asof, half_life_days: float = 75.0, draws: int = 500,
        tune: int = 500, chains: int = 2, seed: int = 0) -> RatingsPosterior:
    """Fit on games strictly before `asof` (a date)."""
    import pymc as pm

    asof = pd.Timestamp(asof)
    g = games[pd.to_datetime(games["date"]) < asof]
    teams = sorted(set(g["home"]) | set(g["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    h = g["home"].map(idx).to_numpy()
    a = g["away"].map(idx).to_numpy()
    y = np.column_stack([g["scoreHome"], g["scoreAway"]]).astype(float)
    w = decay_weights(g["date"], asof, half_life_days)

    with pm.Model():
        mu = pm.Normal("mu", 113, 10)
        hca = pm.Normal("hca", 2.5, 2)
        s_off = pm.HalfNormal("s_off", 5)
        s_def = pm.HalfNormal("s_def", 5)
        off = pm.ZeroSumNormal("off", sigma=s_off, shape=len(teams))
        dfn = pm.ZeroSumNormal("dfn", sigma=s_def, shape=len(teams))
        sigma = pm.HalfNormal("sigma", 15)
        rho = pm.Uniform("rho", -0.5, 0.9)
        hp = mu + hca / 2 + off[h] - dfn[a]
        ap = mu - hca / 2 + off[a] - dfn[h]
        cov = sigma**2 * pm.math.stack([[1.0, rho], [rho, 1.0]])
        ll = pm.logp(pm.MvNormal.dist(mu=pm.math.stack([hp, ap], axis=1), cov=cov), y)
        pm.Potential("weighted_ll", (w * ll).sum())
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1, random_seed=seed,
                          progressbar=False, compute_convergence_checks=False)
    post = idata["posterior"].to_dataset().stack(sample=("chain", "draw"))
    return RatingsPosterior(
        teams=teams,
        mu=post["mu"].values, hca=post["hca"].values,
        off=post["off"].values.T, dfn=post["dfn"].values.T,
        sigma=post["sigma"].values, rho=post["rho"].values,
    ), idata


class Elo:
    """FiveThirtyEight-style NBA Elo with margin-of-victory multiplier: the benchmark."""

    def __init__(self, k: float = 20, hca: float = 100, carryover: float = 0.75):
        self.k, self.hca, self.carry = k, hca, carryover
        self.r: dict[str, float] = {}
        self.season: str | None = None

    def p_home(self, home: str, away: str) -> float:
        d = self.r.get(home, 1505) + self.hca - self.r.get(away, 1505)
        return 1 / (1 + 10 ** (-d / 400))

    def update(self, home: str, away: str, margin: int, season: str) -> None:
        if season != self.season:
            self.r = {t: 1505 + self.carry * (v - 1505) for t, v in self.r.items()}
            self.season = season
        p = self.p_home(home, away)
        diff = self.r.get(home, 1505) + self.hca - self.r.get(away, 1505)
        win = 1.0 if margin > 0 else 0.0
        elo_diff = diff if margin > 0 else -diff
        mult = (abs(margin) + 3) ** 0.8 / (7.5 + 0.006 * elo_diff)
        delta = self.k * mult * (win - p)
        self.r[home] = self.r.get(home, 1505) + delta
        self.r[away] = self.r.get(away, 1505) - delta

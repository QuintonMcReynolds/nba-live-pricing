"""In-game models: live win probability (Brownian motion) and live total (Gamma-Poisson).

Win probability. Following Stern (1994), the home margin over the remaining fraction f of a
48-minute game moves like Brownian motion with drift equal to the pregame expected margin
mu0. Volatility rises late in games (fouling, faster possessions), so the variance scales
as f^gamma rather than f, with gamma fitted (gamma = 1 is pure Brownian motion):

    final margin ~ Normal(alpha(f) * margin + v * possession + beta * mu0 * f,  sigma^2 * f^gamma)

Leads also mean-revert (leaders ease off, trailers press), so only a fraction
alpha(f) = 1 - (a0 + a1 * f) * f of the current lead is expected to survive; a0 = a1 = 0
is a pure random walk. beta calibrates the pregame expected margin mu0.

v is the value of having the ball. Final margins are integers and a tie goes to overtime,
so P(win) = P(M > 0.5) + p_ot * P(-0.5 < M < 0.5).

Total. Scoring is a rate process. The pregame total sets a Gamma prior on points per
second worth k seconds of evidence; observed scoring updates it conjugately:

    rate | data ~ Gamma(k * r0 + points, k + elapsed)

The projected final total adds rate * remaining. The predictive variance has an
overdispersion factor d, because points arrive in clumps of 1-3. Close games late add
points through intentional fouling: within `close_margin` points and `close_window` seconds
of the end, a fitted `close_bonus` is added to the remaining expectation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import ndtr

from .state import Event, GameState

GAME_SECONDS = 2880.0
OT_SECONDS = 300.0


def remaining_fraction(secs_left, period):
    """Fraction of a 48-minute game left, counting only the current period's clock in OT."""
    secs_left = np.asarray(secs_left, dtype=float)
    return np.maximum(secs_left, 0.0) / GAME_SECONDS


@dataclass
class WinProbParams:
    sigma: float = 13.5       # sd of a full game's margin (points)
    poss_value: float = 0.9   # points the team with the ball is worth
    p_ot_home: float = 0.5    # P(home wins | overtime)
    gamma: float = 1.0        # variance scales as f^gamma (1 = Brownian motion)
    drift_scale: float = 1.0  # calibrates the pregame expected margin (beta * mu0)
    revert_a0: float = 0.0    # lead mean-reversion: alpha(f) = 1 - (a0 + a1 * f) * f
    revert_a1: float = 0.0


def margin_mean_sd(margin, f, mu0, poss, params: WinProbParams):
    """Mean and sd of the final home margin given the state."""
    f = np.asarray(f, float)
    alpha = 1.0 - (params.revert_a0 + params.revert_a1 * f) * f
    mean = (alpha * np.asarray(margin, float)
            + params.poss_value * np.asarray(poss, float) * (f > 0)
            + params.drift_scale * np.asarray(mu0, float) * f)
    sd = params.sigma * np.power(np.maximum(f, 0.0), params.gamma / 2) + 1e-6
    return mean, sd


def win_prob(margin, secs_left, period, mu0, poss, params: WinProbParams):
    f = remaining_fraction(secs_left, period)
    mean, sd = margin_mean_sd(margin, f, mu0, poss, params)
    p_win = ndtr((mean - 0.5) / sd)
    p_tie = ndtr((0.5 - mean) / sd) - ndtr((-0.5 - mean) / sd)
    p = p_win + params.p_ot_home * p_tie
    # At the horn the outcome is known (or it's overtime).
    done = f <= 0
    m = np.asarray(margin, float) * np.ones_like(f)
    return np.where(done, np.where(m > 0, 1.0, np.where(m < 0, 0.0, params.p_ot_home)), p)


_FREE = ("sigma", "poss_value", "gamma", "drift_scale", "revert_a0", "revert_a1")


def _unpack(theta, fixed: dict) -> WinProbParams:
    it = iter(theta)
    vals = {}
    for name in _FREE:
        if name in fixed:
            vals[name] = fixed[name]
        else:
            v = next(it)
            vals[name] = float(np.exp(v)) if name in ("sigma", "gamma") else float(v)
    return WinProbParams(vals["sigma"], vals["poss_value"], 0.5, vals["gamma"],
                         vals["drift_scale"], vals["revert_a0"], vals["revert_a1"])


def _x0(fixed: dict) -> list[float]:
    start = {"sigma": np.log(15.0), "poss_value": 0.4, "gamma": np.log(0.9),
             "drift_scale": 1.1, "revert_a0": 0.0, "revert_a1": 0.0}
    return [start[n] for n in _FREE if n not in fixed]


def fit_margin_model(states: pd.DataFrame, fixed: dict | None = None) -> WinProbParams:
    """Production fit: Gaussian likelihood of the FINAL MARGIN given each state.

    Fitting the whole distribution (not just who wins) is what keeps spread and moneyline
    prices coherent and calibrated. `fixed` pins parameters for ablations.
    """
    fixed = fixed or {}
    s = states[(states["overtime"] == 0) & (states["secs_left"] > 0)]
    f = s["secs_left"].to_numpy() / GAME_SECONDS

    def nll(theta):
        m, sd = margin_mean_sd(s["margin"], f, s["mu0"], s["poss"], _unpack(theta, fixed))
        return -np.mean(stats.norm.logpdf(s["final_margin"], m, sd))

    res = optimize.minimize(nll, x0=_x0(fixed), method="Nelder-Mead",
                            options={"maxiter": 8000, "xatol": 1e-7, "fatol": 1e-10})
    return _unpack(res.x, fixed)


def fit_win_prob(states: pd.DataFrame) -> WinProbParams:
    """Ablation: maximum likelihood on win/loss outcomes only (best moneyline log loss,
    but the implied margin distribution is too wide for spreads)."""
    y = states["home_win"].to_numpy()
    live = states["secs_left"].to_numpy() > 0

    def params(theta):
        return _unpack(theta, {})

    def nll(theta):
        p = win_prob(states["margin"], states["secs_left"], states["period"], states["mu0"],
                     states["poss"], params(theta))[live]
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.mean(y[live] * np.log(p) + (1 - y[live]) * np.log(1 - p))

    res = optimize.minimize(nll, x0=_x0({}), method="Nelder-Mead",
                            options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-8})
    return params(res.x)


@dataclass
class TotalParams:
    prior_seconds: float = 900.0   # k: pregame prior is worth this much observed game time
    dispersion: float = 2.0        # d: variance / mean of points scored per interval
    close_bonus: float = 0.0       # extra points expected from end-game fouling...
    close_margin: float = 10.0     # ...when the game is within this many points
    close_window: float = 300.0    # ...in the last this-many seconds


MAX_PRIOR_SECONDS = 1e6            # bound: "the pregame total alone" (~350 games of evidence)


def total_projection(points, elapsed, secs_left, period, total0, params: TotalParams, margin=0):
    """Posterior-predictive mean and sd of the final total (regulation or current OT)."""
    points = np.asarray(points, float)
    elapsed = np.asarray(elapsed, float)
    remaining = np.maximum(np.asarray(secs_left, float), 0.0)
    r0 = np.asarray(total0, float) / GAME_SECONDS
    k = params.prior_seconds
    a, b = k * r0 + points, k + elapsed
    rate = a / b
    close = (np.abs(np.asarray(margin, float)) <= params.close_margin) \
        & (remaining <= params.close_window) & (remaining > 0)
    extra = params.close_bonus * close
    mean = points + rate * remaining + extra
    var = params.dispersion * (rate * remaining + remaining**2 * a / b**2 + extra)
    return mean, np.sqrt(var) + 1e-6


def fit_total(states: pd.DataFrame) -> TotalParams:
    """Gaussian predictive likelihood for (k, d), on regulation-ending games only."""
    s = states[(states["overtime"] == 0) & (states["secs_left"] > 0)]

    def params(theta):
        k = float(min(np.exp(theta[0]), MAX_PRIOR_SECONDS))
        return TotalParams(k, float(np.exp(theta[1])), float(theta[2]))

    def nll(theta):
        mean, sd = total_projection(s["points"], s["elapsed"], s["secs_left"], s["period"],
                                    s["total0"], params(theta), s["margin"])
        return -np.mean(stats.norm.logpdf(s["total"], mean, sd))

    res = optimize.minimize(nll, x0=[np.log(900.0), np.log(2.0), 0.0], method="Nelder-Mead",
                            options={"maxiter": 2000})
    return params(res.x)


def params_dict(*params) -> dict:
    out = {}
    for p in params:
        out[type(p).__name__] = asdict(p)
    return out


def game_states(events: pd.DataFrame, games: pd.DataFrame, pregame: pd.DataFrame,
                every_seconds: int = 30) -> pd.DataFrame:
    """Snapshot each game on a game-clock grid (every 30s; every 5s in the last 3 minutes).

    States come from replaying events through the same GameState the live engine uses.
    mu0 / total0 are leak-free pregame forecasts.
    """
    g = games.set_index("game_id")
    pre = pregame.reindex(g.index)
    grid = np.r_[np.arange(0, 2700, every_seconds), np.arange(2700, 2880, 5)].astype(float)
    rows = []
    for gid, ge in events[events["game_id"].isin(pre.dropna(subset=["pred_margin"]).index)] \
            .groupby("game_id", sort=False):
        info = g.loc[gid]
        st = GameState(gid, info["home"], info["away"], 0.0, 0.0)
        el, mg, ps, pt = [0.0], [0], [0], [0]
        for r in ge.itertuples():
            if r.period > 4:
                break
            st.apply(Event.from_row(r, info["home"]))
            el.append(st.elapsed)
            mg.append(st.margin)
            ps.append(st.poss)
            pt.append(st.points)
        i = np.searchsorted(np.asarray(el), grid, side="right") - 1
        rows.append(pd.DataFrame({
            "game_id": gid, "season": info["season"], "secs_left": GAME_SECONDS - grid,
            "elapsed": grid, "period": np.minimum(grid // 720 + 1, 4).astype(int),
            "margin": np.asarray(mg)[i], "poss": np.asarray(ps)[i], "points": np.asarray(pt)[i],
            "mu0": pre.at[gid, "pred_margin"], "total0": pre.at[gid, "pred_total"],
            "home_win": info["home_win"], "final_margin": info["margin"], "total": info["total"],
            "overtime": info["overtime"],
        }))
    return pd.concat(rows, ignore_index=True)


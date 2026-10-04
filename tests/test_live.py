import numpy as np
import pandas as pd

from livepricing import live
from livepricing.live import TotalParams, WinProbParams

P = WinProbParams(sigma=13.0, poss_value=0.9)


def test_win_prob_monotone_in_margin_and_time():
    margins = np.arange(-20, 21)
    p = live.win_prob(margins, 600, 4, 0.0, 0, P)
    assert np.all(np.diff(p) > 0)
    # The same lead is worth more with less time left.
    assert live.win_prob(5, 60, 4, 0, 0, P) > live.win_prob(5, 1800, 2, 0, 0, P)


def test_win_prob_symmetry():
    for m, s, mu, poss in [(3, 900, 2.0, 1), (-7, 100, -1.0, -1), (0, 2880, 4.0, 0)]:
        a = live.win_prob(m, s, 1, mu, poss, P)
        b = live.win_prob(-m, s, 1, -mu, -poss, P)
        assert np.isclose(a + b, 1.0)


def test_win_prob_pregame_matches_normal():
    from scipy import stats

    p = live.win_prob(0, 2880, 1, 5.0, 0, P)
    expected = stats.norm.sf(0.5, 5, 13) + 0.5 * (stats.norm.cdf(0.5, 5, 13)
                                                  - stats.norm.cdf(-0.5, 5, 13))
    assert np.isclose(p, expected)


def test_win_prob_at_the_horn():
    assert live.win_prob(1, 0, 4, -10, 0, P) == 1.0
    assert live.win_prob(-1, 0, 4, 10, 0, P) == 0.0
    assert live.win_prob(0, 0, 4, 10, 0, P) == 0.5


def test_total_projection_limits():
    # Huge prior strength: projection is the pregame total pro-rated.
    m, _ = live.total_projection(60, 1440, 1440, 3, 228, TotalParams(1e9, 2))
    assert np.isclose(m, 60 + 228 / 2, atol=0.01)
    # No prior: pure pace extrapolation.
    m, _ = live.total_projection(130, 1440, 1440, 3, 228, TotalParams(1e-9, 2))
    assert np.isclose(m, 260, atol=0.01)
    # Uncertainty shrinks to zero at the end.
    _, sd = live.total_projection(220, 2880, 0, 4, 228, TotalParams(900, 2))
    assert sd < 1e-3


def test_margin_fit_recovers_known_random_walk():
    """Simulate a pure random walk (no reversion, gamma = 1) and recover its parameters."""
    rng = np.random.default_rng(0)
    n, sigma, v = 20000, 12.0, 0.6
    mu0 = rng.normal(0, 5, n)
    f = rng.uniform(0.02, 1.0, n)
    poss = rng.choice([-1, 1], n)
    path = rng.normal(mu0 * (1 - f), sigma * np.sqrt(1 - f))      # margin so far
    final = path + v * poss + rng.normal(mu0 * f, sigma * np.sqrt(f))
    states = pd.DataFrame({"margin": path, "secs_left": f * 2880, "period": 1, "mu0": mu0,
                           "poss": poss, "final_margin": final, "overtime": 0})
    fit = live.fit_margin_model(states)
    assert abs(fit.sigma - sigma) < 0.6
    assert abs(fit.gamma - 1.0) < 0.06
    assert abs(fit.drift_scale - 1.0) < 0.06
    assert abs(fit.poss_value - v) < 0.25
    alpha_half = 1 - (fit.revert_a0 + fit.revert_a1 * 0.5) * 0.5
    assert abs(alpha_half - 1.0) < 0.03


def test_mean_reversion_shrinks_leads():
    base = WinProbParams(sigma=14.0, poss_value=0.0)
    rev = WinProbParams(sigma=14.0, poss_value=0.0, revert_a0=0.6, revert_a1=-0.4)
    m0, _ = live.margin_mean_sd(10, 0.5, 0.0, 0, base)
    m1, _ = live.margin_mean_sd(10, 0.5, 0.0, 0, rev)
    assert m0 == 10 and 7 < m1 < 10
    # No time left: the lead is the result, whatever the reversion parameters.
    assert live.margin_mean_sd(10, 0.0, 0.0, 0, rev)[0] == 10

"""Fit the live models on 2024-25 game states and evaluate on 2025-26.

Writes config/model.json (fitted parameters, new version), config/model.lock,
artifacts/live_metrics.json and figures in docs/.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier

from livepricing import config, data, live, plots
from livepricing.backtest import crps_normal

ROOT = Path(__file__).resolve().parents[1]
ART, DOCS = ROOT / "artifacts", ROOT / "docs"
TRAIN, TEST = "2024-25", "2025-26"
BUCKETS = [(2880, 2160, "Q1"), (2160, 1440, "Q2"), (1440, 720, "Q3"),
           (720, 300, "Q4, >5 min left"), (300, 60, "Last 5 min"), (60, 0, "Last minute")]


def bucket(secs_left: pd.Series) -> pd.Series:
    out = pd.Series(index=secs_left.index, dtype=object)
    for hi, lo, name in BUCKETS:
        out[(secs_left <= hi) & (secs_left > lo)] = name
    return out


def log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def gbm_features(s: pd.DataFrame) -> np.ndarray:
    f = np.maximum(s["secs_left"] / live.GAME_SECONDS, 1e-3)
    return np.column_stack([s["margin"], s["secs_left"], s["mu0"], s["poss"],
                            s["margin"] / np.sqrt(f), s["mu0"] * f])


def main() -> None:
    games, events = data.build()
    pre = pd.read_parquet(ART / "pregame_predictions.parquet")
    ev = events[events["season"].isin([TRAIN, TEST])]
    states = live.game_states(ev, games, pre)
    train, test = states[states.season == TRAIN], states[states.season == TEST]
    train_live = train[train.secs_left > 0]
    test_live = test[test.secs_left > 0].copy()
    print(f"states: train {len(train):,}  test {len(test):,}")

    win = live.fit_margin_model(train)
    tot = live.fit_total(train)
    print(win, tot)

    # ---- win probability and margin distribution ------------------------------------------
    t = test_live
    t["bucket"] = bucket(t["secs_left"])
    param_variants = {
        "Production: fit to final margins": (win, t.mu0, t.poss),
        "Fit to win/loss only": (live.fit_win_prob(train), t.mu0, t.poss),
        "No lead mean-reversion": (
            live.fit_margin_model(train, {"revert_a0": 0.0, "revert_a1": 0.0}), t.mu0, t.poss),
        "No drift calibration (beta = 1)": (
            live.fit_margin_model(train, {"drift_scale": 1.0}), t.mu0, t.poss),
        "No late-volatility term (gamma = 1)": (
            live.fit_margin_model(train, {"gamma": 1.0}), t.mu0, t.poss),
        "No possession": (replace(win, poss_value=0.0), t.mu0, 0),
        "No pregame prior": (win, 0.0, t.poss),
    }
    y = t["home_win"].to_numpy()
    reg = (t["overtime"] == 0).to_numpy()
    win_metrics = {}
    preds = {}
    for name, (prm, mu0, poss) in param_variants.items():
        p = np.asarray(live.win_prob(t.margin, t.secs_left, t.period, mu0, poss, prm))
        f = t["secs_left"].to_numpy() / live.GAME_SECONDS
        m, sd = live.margin_mean_sd(t.margin, f, mu0, poss, prm)
        m, sd, fin = np.asarray(m)[reg], np.asarray(sd)[reg], t["final_margin"].to_numpy()[reg]
        lo, hi = stats.norm.ppf(0.1, m, sd), stats.norm.ppf(0.9, m, sd)
        preds[name] = p
        win_metrics[name] = {
            "params": {k: round(float(v), 4) for k, v in vars(prm).items()},
            "log_loss": log_loss(y, p), "brier": float(np.mean((p - y) ** 2)),
            "margin_crps": float(np.mean(crps_normal(fin, m, sd))),
            "margin_coverage_80": float(np.mean((fin >= lo) & (fin <= hi))),
        }
    gbm = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                         monotonic_cst=[1, 0, 1, 1, 1, 1], random_state=0)
    gbm.fit(gbm_features(train_live), train_live["home_win"])
    preds["Gradient boosting (win prob only)"] = gbm.predict_proba(gbm_features(t))[:, 1]
    win_metrics["Gradient boosting (win prob only)"] = {
        "log_loss": log_loss(y, preds["Gradient boosting (win prob only)"]),
        "brier": float(np.mean((preds["Gradient boosting (win prob only)"] - y) ** 2)),
        "margin_crps": None, "margin_coverage_80": None}
    for name, p in preds.items():
        win_metrics[name]["by_bucket"] = {
            b: log_loss(y[(t.bucket == b).to_numpy()], p[(t.bucket == b).to_numpy()])
            for _, _, b in BUCKETS}
        mm = win_metrics[name]
        crps = f"{mm['margin_crps']:.3f}" if mm["margin_crps"] else "  -  "
        cov = f"{mm['margin_coverage_80']:.3f}" if mm["margin_coverage_80"] else "  -  "
        print(f"{name:40s} logloss {mm['log_loss']:.4f}  margin CRPS {crps}  cov80 {cov}")
    main_p = preds["Production: fit to final margins"]

    # ---- live total -----------------------------------------------------------------------
    r = test_live[test_live.overtime == 0].copy()
    r["bucket"] = bucket(r["secs_left"])
    total_variants = {
        "Fitted model (pregame prior + close-game bonus)": tot,
        "Same, without close-game bonus": replace(tot, close_bonus=0.0),
        "Pace extrapolation only": replace(tot, prior_seconds=1.0, close_bonus=0.0),
    }
    total_metrics = {}
    for name, prm in total_variants.items():
        m, sd = live.total_projection(r.points, r.elapsed, r.secs_left, r.period, r.total0, prm,
                                      r.margin)
        crps = crps_normal(r.total.to_numpy(), m, sd)
        lo, hi = stats.norm.ppf(0.1, m, sd), stats.norm.ppf(0.9, m, sd)
        total_metrics[name] = {
            "mae": float(np.mean(np.abs(r.total - m))),
            "crps": float(np.mean(crps)),
            "coverage_80": float(np.mean((r.total >= lo) & (r.total <= hi))),
            "crps_by_bucket": {b: float(np.mean(crps[(r.bucket == b).to_numpy()]))
                               for _, _, b in BUCKETS},
        }
        print(f"{name:42s} CRPS {total_metrics[name]['crps']:.2f}")

    # ---- write the reviewed config and its lock -------------------------------------------
    cfg_path = config.CONFIG_DIR / "model.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["version"] = "1.0.0"
    cfg["automation"]["stale_seconds_by_action"] = data.feed_silence_thresholds(
        events[events.season == TRAIN])
    cfg["win"] = {"sigma": round(win.sigma, 4), "poss_value": round(win.poss_value, 4),
                  "p_ot_home": 0.5, "gamma": round(win.gamma, 4),
                  "drift_scale": round(win.drift_scale, 4),
                  "revert_a0": round(win.revert_a0, 4), "revert_a1": round(win.revert_a1, 4)}
    cfg["total"] = {"prior_seconds": round(tot.prior_seconds, 2),
                    "dispersion": round(tot.dispersion, 4),
                    "close_bonus": round(tot.close_bonus, 3), "close_margin": tot.close_margin,
                    "close_window": tot.close_window}
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    config.write_lock()

    metrics = {"train_season": TRAIN, "test_season": TEST, "test_states": len(test_live),
               "test_games": int(test_live.game_id.nunique()),
               "params": {"win": cfg["win"], "total": cfg["total"]},
               "win_prob": win_metrics, "total": total_metrics}
    (ART / "live_metrics.json").write_text(json.dumps(metrics, indent=2))

    plots.calibration(y, main_p, DOCS / "live_calibration.png")
    plots.loss_by_bucket(win_metrics, [b for _, _, b in BUCKETS], DOCS / "live_logloss.png",
                         "Production: fit to final margins",
                         ["No pregame prior", "Fit to win/loss only",
                          "Gradient boosting (win prob only)"])
    plots.total_crps({k: v for k, v in total_metrics.items() if "without" not in k},
                     [b for _, _, b in BUCKETS], DOCS / "live_total_crps.png")


if __name__ == "__main__":
    main()

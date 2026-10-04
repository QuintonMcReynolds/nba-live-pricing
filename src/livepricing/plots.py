"""Figures for the README (static PNG, light surface)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # blue, orange, aqua, yellow
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def calibration(y, p, path, bins: int = 20):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    t = pd.DataFrame({"b": idx, "p": p, "y": y}).groupby("b").agg(
        pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--", label="Perfect calibration")
    ax.plot(t["pred"], t["actual"], "o", ms=6, color=SERIES[0], mec=SURFACE, mew=1.2,
            label="Live win probability (20 bins)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(home win)")
    ax.set_ylabel("Actual home win rate")
    ax.set_title("Live calibration, held-out 2025-26")
    ax.legend(loc="upper left", labelcolor=INK_2)
    fig.savefig(path)
    plt.close(fig)


def loss_by_bucket(win_metrics: dict, buckets: list[str], path, base_name: str,
                   compare: list[str]):
    """Each compared variant's log loss minus the base model's, per game phase."""
    base = win_metrics[base_name]["by_bucket"]
    others = compare
    x = np.arange(len(buckets))
    w = 0.8 / len(others)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0, color=AXIS, lw=1)
    for i, (n, c) in enumerate(zip(others, SERIES[1:])):
        d = [(win_metrics[n]["by_bucket"][b] - base[b]) * 1000 for b in buckets]
        ax.bar(x + (i - (len(others) - 1) / 2) * w, d, w * 0.92, color=c, label=n)
    ax.set_xticks(x, buckets)
    ax.set_ylabel("Log loss minus production model (×1000)")
    ax.set_title("Win probability by game phase vs the production model")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", labelcolor=INK_2, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def total_crps(total_metrics: dict, buckets: list[str], path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(buckets))
    for (n, m), c in zip(total_metrics.items(), SERIES):
        ax.plot(x, [m["crps_by_bucket"][b] for b in buckets], "-o", color=c, lw=2, ms=6,
                mec=SURFACE, mew=1.2, label=n)
    ax.set_xticks(x, buckets)
    ax.set_ylabel("CRPS of final total (points, lower is better)")
    ax.set_title("Live total: forecast error through the game")
    ax.legend(loc="upper right", labelcolor=INK_2)
    fig.savefig(path)
    plt.close(fig)


def suspension_spans(prices: pd.DataFrame) -> list[tuple[float, float, str]]:
    """(start_minute, end_minute, reason) for each suspension, from the published stream."""
    p = prices.sort_values("seq")
    minute = ((2880 - p["secs_left"]) / 60).to_numpy()
    spans, start, reason = [], None, ""
    for m, status, r in zip(minute, p["status"], p["reason"].fillna("")):
        if status == "SUSPENDED" and start is None:
            start, reason = m, r
        elif status != "SUSPENDED" and start is not None:
            spans.append((start, m, reason))
            start = None
    if start is not None:
        spans.append((start, 48.0, reason))
    return spans


def game_trace(prices: pd.DataFrame, title: str, path, override_window=None):
    """Published live P(home win) over game time, suspensions shaded start-to-reopen."""
    from matplotlib.patches import Patch

    p = prices.sort_values("seq")
    open_q = p[p["status"] != "SUSPENDED"]
    t = (2880 - open_q["secs_left"]) / 60
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.set_ylim(0, 1)
    if override_window:
        ax.axvspan(*override_window, color=SERIES[3], alpha=0.14, lw=0)
    decided = False
    for a, b, reason in suspension_spans(p):
        if reason == "outcome decided":
            decided = True
            ax.axvspan(a, max(b, a + 0.2), color=MUTED, alpha=0.18, lw=0)
        else:
            ax.axvspan(a, max(b, a + 0.25), color=CRITICAL, alpha=0.22, lw=0)
    ax.plot(t, open_q["model_p_home"], color=MUTED, lw=1.2, ls="--", drawstyle="steps-post",
            label="Shadow model (no override)")
    ax.plot(t, open_q["p_home"], color=SERIES[0], lw=2, drawstyle="steps-post",
            label="Published")
    for q in (12, 24, 36):
        ax.axvline(q, color=GRID, lw=1)
    ax.set_xlim(0, 48)
    ax.set_xticks([0, 12, 24, 36, 48], ["Tip", "Q2", "Q3", "Q4", "End"])
    ax.set_ylabel("P(home win)")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(color=CRITICAL, alpha=0.35))
    labels.append("Suspended (incident)")
    if decided:
        handles.append(Patch(color=MUTED, alpha=0.3))
        labels.append("Closed: outcome decided")
    if override_window:
        handles.append(Patch(color=SERIES[3], alpha=0.3))
        labels.append("Trader override active")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncols=3,
              labelcolor=INK_2, fontsize=8)
    fig.savefig(path)
    plt.close(fig)

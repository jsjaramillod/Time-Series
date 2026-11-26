"""Build the report figures directly from the public estimate tables."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
RESULTS = Path(__file__).resolve().parents[1] / "results"
INK, TEAL, RUST = "#273847", "#21615b", "#a05539"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": "#999999",
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.titlecolor": INK,
        "legend.frameon": False,
        "axes.axisbelow": True,
    }
)


def finish(fig, name):
    fig.savefig(
        RESULTS / "figures" / f"{name}.png", dpi=175, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def forecasts():
    df = pd.read_csv(RESULTS / "forecasts.csv")
    fig, ax = plt.subplots(figsize=(7.25, 2.9), layout="constrained")
    x = df.target
    for col, label, color in [
        ("PCA 2", "Two components", RUST),
        ("Selected", "Validated ridge", TEAL),
    ]:
        rms = np.sqrt((df[col] - df.actual).pow(2).expanding().mean())
        ax.plot(x, rms, color=color, label=label, lw=1.65)
        ax.scatter(x.iloc[-1], rms.iloc[-1], s=17, color=color)
    ax.set(xlabel="Last target included, starting from observation 301", ylabel="Cumulative RMSE")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.23)
    finish(fig, "forecast_errors")


def factors():
    paths = pd.read_csv(RESULTS / "factor_paths.csv", parse_dates=["date"])
    fig, axes = plt.subplots(2, 1, figsize=(7.25, 4.3), layout="constrained")
    for ax, name, color in zip(axes, ["Named series", "Equities"], [TEAL, INK]):
        g = paths[paths.model == name]
        ax.fill_between(g.date, g.p10, g.p90, color=color, alpha=0.22, linewidth=0)
        ax.plot(g.date, g["median"], lw=0.55, color=color)
        ax.axhline(0, lw=0.6, color="#888888")
        ax.set_ylabel("Factor units")
        ax.set_title(
            {
                "Named series": "Monthly U.S. activity, February 1967–March 2026",
                "Equities": "Daily U.S. equity returns, January 2010–May 2026",
            }[name],
            loc="left",
            fontsize=10,
        )
    finish(fig, "factor_history")


def hours():
    irs = pd.read_csv(RESULTS / "hours_responses.csv").query("tightness==0.5")
    archive = pd.read_csv(RESULTS.parent / "data/coursework_hours.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.85), sharey=True, layout="constrained")
    for ax, (end, g) in zip(axes, irs.groupby("end")):
        ax.fill_between(
            g.quarter,
            100 * g.p10,
            100 * g.p90,
            alpha=0.19,
            color=TEAL,
            label="Central 80% interval",
        )
        ax.plot(g.quarter, 100 * g["median"], color=TEAL, marker=".", label="Refit median")
        if end == "1994Q3":
            ax.plot(
                archive.quarter,
                100 * archive["median"],
                color=RUST,
                lw=1.2,
                ls="--",
                label="Coursework median",
            )
        ax.axhline(0, lw=0.7, color=INK)
        ax.set(
            title=f"Through {end}", xlabel="Quarter after shock (1 = impact)", xticks=[1, 4, 8, 12]
        )
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("Hours response (100 × log change)")
    finish(fig, "hours_response")


if __name__ == "__main__":
    forecasts()
    factors()
    hours()

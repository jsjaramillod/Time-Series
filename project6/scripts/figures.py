"""Figures for the payroll report, built from the matched forecast records."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#777777", "#46778b", "#142e3a"]


def build():
    forecasts = pd.read_csv(ROOT / "results/forecasts.csv")
    base = forecasts[forecasts.intensity == 0.5].copy()
    base["month"] = pd.to_datetime(base.target)
    output = ROOT / "results/figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#303030",
            "axes.edgecolor": "#777777",
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(8.3, 2.7), layout="constrained")
    for m, c in zip(["AR", "Full"], [COLORS[0], COLORS[2]], strict=True):
        group = base[base.model == m]
        ax.plot(group.month, group.p_loss, lw=1.25, color=c, label=m)
    loss = group[group.first_release < 0]
    ax.scatter(
        loss.month,
        np.full(len(loss), -0.055),
        color="#a65831",
        marker="|",
        s=42,
        label="First release showed a loss",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(ylim=(-0.09, 1.04), yticks=[0, 0.25, 0.5, 0.75, 1], ylabel="Probability of job losses")
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.17)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.01), frameon=False, ncol=3, fontsize=8)
    fig.savefig(output / "loss_probabilities.png", dpi=175)
    plt.close(fig)

    stress = base[base.target.between("2020-03", "2020-06")]
    fig, ax = plt.subplots(figsize=(8.2, 3.0), layout="constrained")
    y = np.arange(4)
    for offset, key, color, marker in [
        (-0.18, "AR", COLORS[0], "s"),
        (0, "Full", COLORS[1], "o"),
        (0.18, "First release", COLORS[2], "D"),
    ]:
        vals = (
            stress[stress.model == key]["mean"]
            if key != "First release"
            else stress[stress.model == "AR"].first_release
        ).to_numpy() / 1000
        ax.hlines(y + offset, np.minimum(vals, 0), np.maximum(vals, 0), color=color, lw=1.2)
        ax.plot(vals, y + offset, linestyle="none", marker=marker, color=color, label=key, ms=5)
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set(
        yticks=y,
        yticklabels=["March", "April", "May", "June"],
        xlabel="Monthly payroll change (millions of jobs)",
        xlim=(-22, 7),
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.18)
    ax.legend(frameon=False, ncol=3, loc="lower left", fontsize=8)
    fig.savefig(output / "pandemic.png", dpi=175)
    plt.close(fig)

    ref = pd.read_csv(ROOT / "results/reference.csv").query("intensity == 0.5 and model == 'Full'")
    fig, ax = plt.subplots(figsize=(8.2, 2.25), layout="constrained")
    for i, row in enumerate(ref.itertuples()):
        ax.plot([row.p05, row.p95], [i, i], lw=2, color=COLORS[1])
        ax.plot([row.p05, row.p95], [i, i], linestyle="none", marker="|", ms=9, color=COLORS[1])
        ax.plot(row.median, i, "o", ms=5, color=COLORS[2])
    ax.axvline(115, linestyle="--", color="#965333", lw=1.3)
    ax.axvline(0, color="#aaaaaa", lw=0.6)
    ax.set(
        ylim=(-0.55, 1.55),
        yticks=[0, 1],
        yticklabels=["Observed history", "Interpolated history"],
        xlabel="April payroll change (thousands of jobs)",
        xlim=(-1750, 1500),
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)
    ax.legend(
        handles=[
            Line2D([], [], color=COLORS[1], marker="o", label="Median and 90% interval"),
            Line2D([], [], color="#965333", ls="--", label="First release (+115,000)"),
        ],
        frameon=False,
        ncol=2,
        loc="lower left",
        fontsize=8,
    )
    fig.savefig(output / "april_reference.png", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    build()

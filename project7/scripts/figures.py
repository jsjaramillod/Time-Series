"""Draw the forecast paths and the posterior scale in report units."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "figures"


def build():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "savefig.facecolor": "white",
        }
    )
    f = pd.read_csv(RES / "forecasts.csv")
    curves = f.pivot(index="target", columns="treatment", values="mean") / 1000
    x = pd.PeriodIndex(curves.index, freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(7.4, 3.35), layout="constrained")
    actual = f.drop_duplicates("target").set_index("target").loc[curves.index, "revised"] / 1000
    for name, color, style in [
        ("Observed", "#9f9c98", ":"),
        ("Interpolated", "#95653a", "--"),
        ("Temporary volatility", "#345d74", "-"),
    ]:
        ax.plot(x, curves[name], color=color, ls=style, lw=1.7, label=name)
    ax.plot(x, actual, color="#161616", lw=2.2, label="Realized")
    ax.set_ylabel("Six-month change (million jobs)")
    ax.set_xlabel("End of the six-month forecast window")
    ticks = [0, 6, 12, 18, 24, len(x) - 1]
    ax.set_xticks(x[ticks], [x[i].strftime("%b %y") for i in ticks])
    ax.grid(axis="y", lw=0.45, alpha=0.35)
    ax.legend(ncol=2, frameon=False)
    fig.savefig(OUT / "payroll_paths.png", dpi=180)
    plt.close(fig)
    g = pd.read_csv(RES / "volatility_percentiles.csv").set_index("percentile").multiplier
    m = np.arange(26)

    def path(q):
        return np.r_[g.loc[q] * 0.9 ** m[:-1], 1.0]

    fig, ax = plt.subplots(figsize=(7.4, 3.15), layout="constrained")
    ax.fill_between(m, path(2.5), path(97.5), color="#dce6eb", label="95% credible band")
    ax.plot(m, path(50), color="#345d74", lw=2, label="Portfolio median")
    archived = pd.read_csv(ROOT / "data/coursework.csv").set_index("quantity")
    ax.scatter(
        [0],
        [archived.at["march_variance_multiplier", "median"]],
        marker="x",
        color="#272727",
        label="Coursework, March 2020",
    )
    ax.axhline(1, color=".4", ls=":", lw=1)
    ax.set_xticks(
        [0, 6, 12, 18, 25],
        ["Mar 20", "Sep 20", "Mar 21", "Sep 21", "Apr 22"],
        rotation=16,
    )
    ax.set_ylabel("Innovation variance multiplier")
    ax.legend(frameon=False)
    fig.savefig(OUT / "volatility_posterior.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    build()

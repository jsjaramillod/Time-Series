"""Render publication figures from the recorded predictions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"


def finish(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", pad_inches=0.09)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    f = pd.read_csv(RESULTS / "forecasts.csv")
    a = f.query('training=="observed" and model=="AR" and horizon==4').dropna()
    t = pd.PeriodIndex(a.target, freq="Q").to_timestamp(how="end")
    fig, ax = plt.subplots(figsize=(8, 3.65), layout="constrained")
    ax.fill_between(t, a.p05, a.p95, color=".87", label="Central 90% forecast interval")
    ax.plot(t, a.p05, color=".42", lw=1, ls="--", label="Fifth percentile")
    ax.plot(t, a.actual, color="#172e44", lw=1.65, label="Observed growth")
    miss = a.actual < a.p05
    ax.scatter(
        t[miss], a.actual[miss], s=16, color="#8e4933", zorder=3, label="Below fifth percentile"
    )
    ax.axhline(0, lw=0.6, color=".4")
    ax.set(ylabel="Four-quarter log growth (%)", xlabel="Target quarter")
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="lower left")
    finish(fig, "historical_risk.png")
    q = pd.read_csv(RESULTS / "reference_quantiles.csv").query('model=="AR" and horizon==4')
    fig, ax = plt.subplots(figsize=(7.8, 3.25), layout="constrained")
    for sample, color, style, label in [
        ("observed", "#172e44", "-", "Observed GDP"),
        ("interpolated", "#8e4933", "--", "Pandemic interpolation"),
    ]:
        data = q[q.training == sample]
        ax.plot(data.growth, data.percentile, color=color, ls=style, lw=2, label=label)
        fifth = data.loc[data.percentile == 5, "growth"].iloc[0]
        ax.scatter(fifth, 5, color=color, s=30, zorder=4)
    ax.axvline(0, lw=0.7, color=".45")
    ax.axhline(5, lw=0.65, color=".5", ls=":")
    ax.set(
        xlabel="Four-quarter log growth to 2027Q1 (%)",
        ylabel="Cumulative probability (%)",
        yticks=[5, 25, 50, 75, 95],
        ylim=(0, 101),
    )
    ax.legend(frameon=False, loc="upper left")
    finish(fig, "reference_distribution.png")


if __name__ == "__main__":
    main()

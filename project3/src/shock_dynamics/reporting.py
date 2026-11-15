"""Tables and restrained scientific figures consume computed experiment results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


def build_figures(output, summary, paths, responses, means, precision, ma):
    plt.switch_backend("Agg")
    out = Path(output) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "normal",
            "figure.dpi": 130,
            "savefig.dpi": 190,
            "axes.prop_cycle": plt.cycler(color=["#222222", "#777777", "#aaaaaa"]),
        }
    )

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.1))
    ax.plot(paths["ar1_persistent"], label="Intercept 0; stationary mean 0", linewidth=1)
    ax.plot(
        paths["ar1_mean_shift"],
        label="Intercept 0.5; stationary mean 5",
        linestyle="--",
        linewidth=1,
    )
    ax.axhline(0, color="0.5", linewidth=0.5)
    ax.axhline(5, color="0.5", linewidth=0.5, linestyle="--")
    ax.set(
        xlabel="Retained period",
        ylabel="Model units",
        title="Shared innovations isolate a five-unit level shift",
    )
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, "mean_adjustment.png")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, name, title in zip(
        axes,
        ["ar3_stable", "ar3_unstable"],
        ["Stable: coefficients sum to 0.9", "Unstable: coefficients sum to 1.1"],
    ):
        ax.plot(paths[name], color="0.15", linewidth=1)
        ax.set(xlabel="Period", ylabel="Model units", title=title)
    save(fig, "stability_paths.png")

    fig, ax = plt.subplots(figsize=(5.7, 4.1))
    angle = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(angle), np.sin(angle), color="0.6", linewidth=0.8, label="Unit circle")
    for name, marker, label in [
        ("ar3_stable", "o", "Stable AR(3)"),
        ("ar3_unstable", "x", "Unstable AR(3)"),
        ("ar2_boundary", "D", "Boundary AR(2)"),
    ]:
        # Values are reconstructed from the recorded coefficient string for plotting only.
        phi = np.fromstring(summary.set_index("experiment").loc[name, "coefficients"], sep=";")
        eigenvalues = np.roots(np.r_[1.0, -phi])
        ax.plot(
            eigenvalues.real,
            eigenvalues.imag,
            marker=marker,
            linestyle="none",
            markerfacecolor="none" if marker in {"o", "D"} else "0.1",
            color="0.1",
            label=label,
        )
    ax.axhline(0, linewidth=0.4, color="0.7")
    ax.axvline(0, linewidth=0.4, color="0.7")
    ax.set(
        xlabel="Real part",
        ylabel="Imaginary part",
        xlim=(-1.25, 1.3),
        ylim=(-1.15, 1.15),
        aspect="equal",
    )
    ax.tick_params(labelsize=12)
    ax.set_xlabel("Real part", fontsize=12)
    ax.set_ylabel("Imaginary part", fontsize=12)
    ax.legend(frameon=False, fontsize=10.5, loc="lower left")
    save(fig, "companion_eigenvalues.png")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    labels = {
        "ar1_persistent": "AR(1): (0.9)",
        "ar2_response": "AR(2): (1.2, -0.3)",
        "ar3_stable": "AR(3): (0.5, 0.2, 0.2)",
    }
    for name, style in zip(labels, ["-", "--", ":"]):
        result = responses[name]
        axes[0].plot(
            result.horizons[:31], result.methods["recursion"][:31], style, label=labels[name]
        )
        axes[1].plot(result.horizons, result.cumulative, style, label=labels[name])
    axes[0].set(
        xlabel="Horizon after one-unit innovation",
        ylabel="Response",
        title="The timing of transmission differs",
    )
    axes[1].set(
        xlabel="Horizon after one-unit innovation",
        ylabel="Accumulated response",
        title="All three converge to ten",
    )
    axes[1].axhline(10, linewidth=0.5, color="0.6")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, "shock_propagation.png")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
    spread = np.sqrt(precision["exact_mean_variance"])
    x = np.linspace(-3.7 * spread, 3.7 * spread, 400)
    axes[0].hist(
        means,
        bins=45,
        density=True,
        color="0.83",
        edgecolor="white",
        label="10,000 replications" if len(means) == 10000 else f"{len(means):,} replications",
    )
    axes[0].plot(x, norm.pdf(x, scale=spread), color="0.1", label="Exact Gaussian")
    axes[0].plot(
        x,
        norm.pdf(x, scale=np.sqrt(precision["independence_mean_variance"])),
        "--",
        color="0.4",
        label="Independence",
    )
    axes[0].set(xlabel="Mean of 200 stationary observations", ylabel="Density")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    axes[1].bar(
        [0, 1],
        [precision["correlation_aware_coverage"], precision["independence_coverage"]],
        color=["0.25", "0.7"],
        width=0.55,
    )
    axes[1].axhline(0.95, color="0.1", linewidth=0.8, linestyle="--")
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Correlation-aware", "Independence assumption"],
        ylabel="Empirical coverage",
        ylim=(0, 1.06),
    )
    axes[1].tick_params(axis="x", labelsize=9)
    save(fig, "mean_precision.png")

    fig, ax = plt.subplots(figsize=(6, 2.6))
    values = ma["autocovariances_0_to_4"]
    ax.bar(np.arange(5), values, color="0.4", width=0.55)
    ax.set(
        xlabel="Lag",
        ylabel="Autocovariance",
        xticks=np.arange(5),
        title="Finite MA dependence ends after lag two",
    )
    save(fig, "ma_autocovariance.png")


def build_report(output):
    out = Path(output)
    summary = pd.read_csv(out / "dynamics_summary.csv").set_index("experiment")
    precision = json.loads((out / "mean_precision.json").read_text())
    ma = json.loads((out / "ma_control.json").read_text())
    validation = json.loads((out / "numerical_validation.json").read_text())
    columns = [
        "classification",
        "spectral_radius",
        "stationary_mean",
        "stationary_variance",
        "long_run_sum",
        "cumulative_at_20",
        "cumulative_at_100",
    ]
    # Avoid an optional Markdown rendering dependency in the numerical package.
    table = [
        "| Experiment | " + " | ".join(columns) + " |",
        "| --- | " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for name, row in summary.iterrows():
        cells = [
            str(row[c])
            if c == "classification"
            else (f"{row[c]:.8f}" if pd.notna(row[c]) else "Undefined")
            for c in columns
        ]
        table.append(f"| {name} | " + " | ".join(cells) + " |")
    text = f"""# Persistence and shock propagation: computed results

All parameters are fixed design inputs. The observations are synthetic, measured in generic model units and periods. No economic coefficients are estimated.

## Dynamic classifications and response totals

{chr(10).join(table)}

Stationary moments and convergent long-run sums are undefined for the unstable and unit-circle specifications. Finite paths and finite response sums remain computable; they do not establish stationarity.

## Persistence and precision

For the AR(1) coefficient 0.9 and innovation variance one, stationary variance is {precision["stationary_variance"]:.8f} and the exact shock half-life is {precision["ar1_shock_half_life_periods"]:.6f} periods. An intercept change from zero to 0.5 shifts the stationary mean from zero to five without changing centered dynamics.

For a stationary window of 200 observations, the exact variance of its mean is {precision["exact_mean_variance"]:.8f}, versus {precision["independence_mean_variance"]:.8f} under an independence assumption using the same marginal variance. Their ratio is {precision["variance_inflation_factor"]:.6f}, equivalent to {precision["effective_independent_observations"]:.6f} independent observations for estimating the mean.

Across {precision["replications"]:,} independent stationary replications, empirical 95% interval coverage is {100 * precision["correlation_aware_coverage"]:.2f}% when accounting for correlation and {100 * precision["independence_coverage"]:.2f}% under independence. Exact model-implied coverage of the independence interval is {100 * precision["exact_independence_coverage"]:.4f}%. Parameters and covariance are known; these are repeated-sample coverage statements, not posterior credible intervals.

## Moving-average control

The MA(2) polynomial is 1 + 1.5L + 1.3L^2. Its variance is {ma["stationary_variance"]:.2f}; autocovariances at lags 1 and 2 are {ma["autocovariances_0_to_4"][1]:.2f} and {ma["autocovariances_0_to_4"][2]:.2f}. Higher-lag autocovariances vanish. Its root moduli are {ma["root_moduli"][0]:.8f}; the supplied innovation representation is noninvertible despite covariance stationarity.

## Numerical verification and interpretation

All deterministic controls pass: **{validation["all_deterministic_controls_pass"]}**. Responses through horizon 100 agree across matrix powers, recursion, a controlled shock experiment, and an independent SciPy linear filter. Stationary covariance satisfies the Lyapunov equation and agrees with a 5,001-term squared-response sum. Detailed errors are recorded in `numerical_validation.json`.

The AR(3) path comparison uses exactly the same 200 innovations, zero lagged states and no burn-in. The stable coefficient vector is (0.5, 0.2, 0.2); the unstable vector is (0.5, 0.3, 0.3). AR(1) paired paths share 700 innovations and discard 500 warmup observations. All settings and source/output checksums are recorded. One simulated path cannot test an economic hypothesis or prove stationarity.

Read [methods](../docs/methods.md), the [executive summary](../EXECUTIVE_SUMMARY.md), and the [research report](../docs/persistence_shock_propagation.pdf). The executed notebook inspects these artifacts without implementing another simulator.
"""
    (out / "report.md").write_text(text)

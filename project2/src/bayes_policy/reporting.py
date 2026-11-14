"""Small publication artifacts; estimation code never lives in a notebook or figure."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from .results import PosteriorResult

COLORS = ["#176B87", "#D36B37", "#547557", "#8872A2"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "savefig.dpi": 160,
    }
)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def probit_figures(result: PosteriorResult, out: Path):
    table = result.summary()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for axis, name in zip(axes.ravel(), result.names):
        values = result.parameter(name)
        for chain in range(values.shape[0]):
            stride = max(1, values.shape[1] // 2000)
            axis.plot(
                np.arange(0, values.shape[1], stride),
                values[chain, ::stride],
                lw=0.35,
                alpha=0.6,
                color=COLORS[chain % len(COLORS)],
            )
        axis.set(title=name.replace("_", " "), xlabel="Retained iteration")
    fig.suptitle("Probit coefficients | February 2026 origin | four posterior chains", fontsize=12)
    fig.savefig(out / "probit_endpoint_traces.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
    for axis, name in zip(axes, result.names[1:]):
        flat = result.parameter(name).ravel()
        grid = np.linspace(*np.quantile(flat, [0.0025, 0.9975]), 300)
        axis.plot(grid, gaussian_kde(flat)(grid), color=COLORS[0], lw=2)
        axis.axvline(0, ls="--", color="#777777", lw=0.8)
        axis.set(title=name.replace("_", " "), xlabel="Coefficient value", ylabel="Density")
    fig.suptitle("Financial predictors | endpoint posterior | central 99.5% displayed", fontsize=12)
    fig.savefig(out / "probit_endpoint_densities.png")
    plt.close(fig)
    return table


def policy_figures(results: dict[str, PosteriorResult], out: Path):
    out.mkdir(parents=True, exist_ok=True)
    baseline = results["symmetric_v10"]
    asymmetric = results["asymmetric_v10"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for axis, result, title in zip(
        axes, [baseline, asymmetric], ["Single output-gap slope", "Separate signed-gap slopes"]
    ):
        table = result.summary().drop(index="sigma2")
        y = np.arange(len(table))
        axis.hlines(y, table.p05, table.p95, color=COLORS[0], lw=3)
        axis.scatter(table["median"], y, color=COLORS[0], s=30, zorder=3)
        axis.set_yticks(y, [s.replace("_", " ") for s in table.index])
        axis.invert_yaxis()
        axis.axvline(0, color="#999999", lw=0.7)
        axis.set(title=title, xlabel="Coefficient: median and central 90% credible interval")
    fig.suptitle("Monetary-policy responses | 1955Q1–2014Q3", fontsize=14)
    fig.savefig(out / "policy_coefficients.png")
    plt.close(fig)

    theta = baseline.parameter("inflation_gap") / (1 - baseline.parameter("lagged_rate"))
    difference = asymmetric.parameter("output_gap_positive") - asymmetric.parameter(
        "output_gap_negative"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, samples, title in zip(
        axes,
        [theta, difference],
        ["Long-run inflation response", "Positive minus negative gap slope"],
    ):
        flat = samples.ravel()
        # Display central 99%; every draw still enters probability calculations.
        lower, upper = np.quantile(flat, [0.005, 0.995])
        grid = np.linspace(lower, upper, 400)
        axis.plot(grid, gaussian_kde(flat)(grid), color=COLORS[0], lw=2)
        axis.axvline(0, ls="--", color="#777777", lw=0.9)
        axis.set(title=title, xlabel="Parameter value", ylabel="Posterior density")
    axes[0].axvline(1, color=COLORS[1], ls="--", label="Taylor-principle threshold")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Parameter uncertainty | central 99% displayed; all draws used in summaries", fontsize=11
    )
    fig.savefig(out / "policy_uncertainty.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, layout="constrained")
    for axis, name in zip(axes, ["inflation_gap", "lagged_rate", "sigma2"]):
        values = baseline.parameter(name)
        stride = max(1, values.shape[1] // 2500)
        for chain in range(values.shape[0]):
            color = COLORS[chain % len(COLORS)]
            axis.plot(
                np.arange(0, values.shape[1], stride),
                values[chain, ::stride],
                lw=0.35,
                alpha=0.65,
                color=color,
                label=f"Chain {chain + 1}",
            )
        axis.set_ylabel(name.replace("_", " "))
    axes[0].legend(ncol=4, frameon=False, fontsize=8)
    axes[0].set_title(f"Symmetric rule: {baseline.draws.shape[0]} independent chains after warmup")
    axes[-1].set_xlabel("Retained iteration (plot subsampled for readability)")
    fig.savefig(out / "policy_traces.png")
    plt.close(fig)


def recession_figure(frame, path: Path, synthetic: bool = False):
    """Keep scored history and the distant, unobserved endpoint in separate panels."""
    observed = frame.actual.notna()
    split = observed.any() and (~observed).any()
    if split:
        fig, axes = plt.subplots(
            1, 2, figsize=(11, 4.8), sharey=True, width_ratios=[3, 1.3], layout="constrained"
        )
        groups = [frame.loc[observed], frame.loc[~observed]]
    else:
        fig, axis = plt.subplots(figsize=(10, 4.5), layout="constrained")
        axes, groups = [axis], [frame]
    for index, (axis, group) in enumerate(zip(axes, groups)):
        dates = np.array(group.target_date, dtype="datetime64[D]")
        if len(group) > 1:
            axis.fill_between(
                dates,
                group.probability_p05,
                group.probability_p95,
                alpha=0.15,
                color=COLORS[0],
                label="90% probability interval",
            )
            axis.plot(dates, group.probability, "o-", color=COLORS[0], label="Probit probability")
            axis.plot(
                dates,
                group.benchmark_probability,
                "--",
                color=COLORS[1],
                label="Frequency benchmark",
            )
            positions = np.unique(np.linspace(0, len(group) - 1, min(5, len(group)), dtype=int))
            axis.set_xticks(
                dates[positions],
                [str(group.target_date.iloc[i])[:7] for i in positions],
                fontsize=8,
            )
        else:
            row = group.iloc[0]
            axis.errorbar(
                [0],
                [row.probability],
                yerr=[
                    [row.probability - row.probability_p05],
                    [row.probability_p95 - row.probability],
                ],
                fmt="o",
                capsize=5,
                color=COLORS[0],
                label="Probit probability",
            )
            axis.scatter(
                [0],
                [row.benchmark_probability],
                marker="_",
                s=200,
                color=COLORS[1],
                label="Frequency benchmark",
            )
            axis.set_xticks([0], [str(row.target_date)[:7]])
            axis.set_xlim(-0.8, 0.8)
        valid = group.actual.notna()
        if valid.any():
            axis.scatter(
                dates[valid],
                group.loc[valid, "actual"],
                marker="x",
                color="#333333",
                label="Observed outcome",
            )
        axis.set(ylim=(-0.03, 1.03), xlabel="Target month (origin + 12 months)")
        if split:
            axis.set_title(
                "Historical scored targets" if index == 0 else "Unobserved endpoint", fontsize=11
            )
        if index == 0:
            axis.set_ylabel("Recession probability")
            axis.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "SIMULATED DATA — software demonstration"
        if synthetic
        else "Twelve-month recession risk | revised-data pseudo-real-time",
        fontsize=13,
    )
    fig.savefig(path)
    plt.close(fig)


def build_report(output: str | Path):
    """Render checked numeric artifacts into a readable report without refitting."""
    import pandas as pd

    out = Path(output)
    summary = pd.read_csv(out / "policy_posterior.csv")
    evidence = pd.read_csv(out / "model_comparison.csv")
    derived = {row["model"]: row for row in json.loads((out / "policy_derived.json").read_text())}
    validation = json.loads((out / "validation.json").read_text())
    primary = derived["symmetric_v10"]
    lines = [
        "# Empirical results and validation",
        "",
        "Generated from the checked-in numeric artifacts. Source workbooks are external; "
        "their hashes and sample definitions are recorded in the manifests. "
        "Computational diagnostics assess simulation reliability, not economic identification.",
        "",
        "## Monetary-policy associations",
        "",
        "The effective sample is 239 quarterly observations, 1955Q1–2014Q3. "
        "Primary coefficients have independent N(0,10I) priors and precision has Gamma(shape=1.5, rate=1.5).",
        "",
        "| Model / parameter | OLS | Posterior median | SD | Central 90% interval | R-hat | Bulk ESS |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in summary[summary.model.isin(["symmetric_v10", "asymmetric_v10"])].itertuples():
        ols = "—" if np.isnan(row.ols) else f"{row.ols:.4f}"
        lines.append(
            f"| {row.model} / {row.parameter} | {ols} | {row.median:.4f} | {row.sd:.4f} "
            f"| [{row.p05:.4f}, {row.p95:.4f}] | {row.r_hat:.5f} | {row.ess_bulk:,.0f} |"
        )
    lines += [
        "",
        "![Policy coefficient intervals](figures/policy_coefficients.png)",
        "",
        "## Model comparison and sensitivity",
        "",
        "Posterior model probabilities below use equal prior model probabilities. "
        "Each pair explains identical observations; a preference is relative to these two specifications.",
        "",
        "| Model | Chib log evidence | MCSE | Quadrature log evidence | Difference | Posterior model probability |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in evidence.itertuples():
        lines.append(
            f"| {row.model} | {row.log_evidence_chib:.6f} | {row.mcse_log_evidence_chib:.6f} "
            f"| {row.log_evidence_quadrature:.6f} | {row.chib_minus_quadrature:.6f} "
            f"| {row.posterior_model_probability:.4%} |"
        )
    asym = evidence[evidence.model == "asymmetric_v10"].iloc[0]
    lines += [
        "",
        f"With the asymmetric model's prior probability set to 25% or 75%, its posterior "
        f"probability is {asym.probability_if_prior_asymmetric_0_25:.2%} or "
        f"{asym.probability_if_prior_asymmetric_0_75:.2%}, respectively. "
        "This dependence on prior assumptions is part of the result.",
        "",
        "## Long-run response and asymmetry",
        "",
        f"For the symmetric 10I model, the long-run inflation-response median is "
        f"{primary['theta_median']:.3f}, with central 90% interval "
        f"[{primary['theta_p05']:.3f}, {primary['theta_p95']:.3f}]. "
        f"P(theta>0) is {primary['long_run_positive']['probability']:.4%}; "
        f"P(theta>1) is {primary['long_run_above_one']['probability']:.4%} "
        f"(MCSE {primary['long_run_above_one']['mcse']:.4f}). "
        "These are distinct economic thresholds.",
        "",
        f"Unstable lag-coefficient draws account for {primary['unstable_draw_fraction']:.4%}; "
        f"{primary['near_unit_root_fraction_abs_one_minus_rho_below_0_01']:.4%} have "
        "abs(1-rho)<0.01. The explicitly stability-conditioned P(theta>1) is "
        f"{primary['theta_gt1_given_stable']:.4%}. Unconditional quantities retain all draws.",
        "",
        "In the asymmetric 10I model, P(positive-gap slope minus negative-gap slope > 0) is "
        f"{derived['asymmetric_v10']['positive_minus_negative_gap_above_zero']['probability']:.2%}. "
        "A lower slope during positive gaps is more probable within that model, but the "
        "marginal-likelihood comparison favors the simpler common-slope specification.",
        "",
        "![Derived posterior uncertainty](figures/policy_uncertainty.png)",
        "",
        "## Numerical reliability and model limitations",
        "",
        f"Sampler and derived-quantity diagnostic gates passed: "
        f"**{validation['sampler_diagnostics_passed']}**. Independent evidence controls passed: "
        f"**{validation['independent_evidence_controls_passed']}**. "
        "Thresholds are R-hat <=1.01 and bulk/tail ESS >=400. "
        "All retained draws enter summaries; only trace displays are subsampled.",
        "",
    ]
    check = validation["models"]["symmetric_v10"]["predictive_check"]
    lines += [
        f"The symmetric model's residual lag-one autocorrelation is "
        f"{check['residual_lag1_acf_at_posterior_mean']:.3f}. The conditional predictive check "
        f"assigns probability {check['ppp_abs_lag1_acf']:.4f} to simulated absolute "
        "autocorrelation at least as large as observed. This is evidence of an inadequate "
        "independent-error assumption, not a classical p-value. Historical regime changes, "
        "time-varying volatility and the lower bound are additional concerns. "
        "These estimates do not identify a causal policy rule.",
        "",
        "![Gaussian sampler traces](figures/policy_traces.png)",
        "",
    ]
    recession_dir = out / "recession"
    if (recession_dir / "recession_scores.json").is_file():
        scores = json.loads((recession_dir / "recession_scores.json").read_text())
        forecasts = pd.read_csv(recession_dir / "recession_forecasts.csv")
        lines += [
            "## Recession probabilities",
            "",
            "Revised-data **pseudo-real-time** evaluation. Primary training assumes retrospective "
            "labels available at their target month end. Historical NBER announcement availability "
            "and predictor vintages are not established. Forecasts made in 2007 target the "
            "corresponding months in 2008; a separate February 2026 origin targets February 2027.",
            "",
            "| Origin | Target | Training n | Last training target | Probability | 90% probability interval | Actual |",
            "| --- | --- | ---: | --- | ---: | --- | ---: |",
        ]
        for row in forecasts.itertuples():
            actual = "unobserved" if np.isnan(row.actual) else f"{row.actual:.0f}"
            lines.append(
                f"| {row.origin} | {row.target_date} | {row.training_n} | {row.last_training_target} "
                f"| {row.probability:.2%} | [{row.probability_p05:.2%}, {row.probability_p95:.2%}] | {actual} |"
            )
        lines += [
            "",
            "Intervals describe uncertainty about the latent recession probability, "
            "not intervals for a binary outcome. February 2027 is absent from the source "
            "workbook and excluded from scores. The February 2026-origin forecast is a historical "
            "exercise, not a live forecast as of this report.",
            "",
            "| Model | Observed targets | Mean Brier loss | Mean negative log score |",
            "| --- | ---: | ---: | ---: |",
        ]
        for key in ["probit", "benchmark"]:
            row = scores[key]
            lines.append(
                f"| {key} | {row['n']} | {row['brier']:.6f} | {row['negative_log_score']:.6f} |"
            )
        lines += [
            "",
            "The benchmark uses exactly the same training labels with Beta(1,1) smoothing. "
            "Smaller losses are better. All twelve 2008 outcomes are recessions in the input "
            "indicator: these scores cannot establish broad calibration or discrimination. "
            "No significance claim is made from this short, overlapping-horizon exercise.",
            "",
            f"All forecast coefficient diagnostic gates passed: **{scores['all_diagnostics_passed']}**.",
            "",
            "![Historical-origin recession probabilities](recession/recession_probabilities.png)",
            "",
        ]
        endpoint_diagnostics = recession_dir / "probit_2026-02-28_diagnostics.csv"
        if endpoint_diagnostics.is_file():
            endpoint = pd.read_csv(endpoint_diagnostics)
            lines += [
                "### Financial-predictor coefficients",
                "",
                "The February 2026-origin fit uses 779 matched observations with targets through "
                "February 2026. These latent-index coefficients describe predictive associations, "
                "not percentage-point changes in recession probability or causal effects.",
                "",
                "| Parameter | Posterior median | SD | Central 90% interval | R-hat | Bulk ESS |",
                "| --- | ---: | ---: | --- | ---: | ---: |",
            ]
            for row in endpoint.itertuples():
                lines.append(
                    f"| {row.parameter} | {row.median:.4f} | {row.sd:.4f} "
                    f"| [{row.p05:.4f}, {row.p95:.4f}] | {row.r_hat:.5f} | {row.ess_bulk:,.0f} |"
                )
            lines += [
                "",
                "![Endpoint probit coefficient densities](recession/probit_endpoint_densities.png)",
                "",
                "![Endpoint probit chain traces](recession/probit_endpoint_traces.png)",
                "",
            ]
        sensitivity_path = out / "recession_delay18" / "recession_scores.json"
        if sensitivity_path.is_file():
            sensitivity = json.loads(sensitivity_path.read_text())
            lines += [
                "### Label-delay sensitivity",
                "",
                "A separate run assumes an 18-month label delay. This is an illustrative "
                "availability assumption, not an empirically verified NBER publication calendar. "
                f"Its probit Brier loss is {sensitivity['probit']['brier']:.6f}, versus "
                f"{sensitivity['benchmark']['brier']:.6f} for its matched benchmark. "
                f"All diagnostic gates passed: **{sensitivity['all_diagnostics_passed']}**. "
                "See the separate CSV for every effective training window.",
                "",
            ]
    lines += [
        "## Reproduction and scope",
        "",
        "See [the project README](../README.md) for commands and "
        "[methods](../docs/methods.md) for assumptions and derivations. "
        "The [specification audit](../docs/reference_audit.md) records parameterization, transformation, and timing decisions. "
        "The committed tables and figures are reproduced results; original workbooks and private "
        "posterior draw archives are excluded. CI checks scientific controls using generated data "
        "and does not claim to rerun inaccessible empirical workbooks.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))

"""Restrained figures and numerical reporting from the recorded public tables."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


def build_figures(output):
    out = Path(output)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
            "savefig.dpi": 190,
        }
    )
    comparison = pd.read_csv(out / "model_comparison.csv")
    densities = pd.read_csv(out / "posterior_density_grid.csv")
    responses = pd.read_csv(out / "impulse_responses.csv")
    traces = pd.read_csv(out / "ar8_thinned_traces.csv")
    selected_order = next(
        g["map_order"]
        for g in json.loads((out / "study_summary.json").read_text())["groups"]
        if g["coefficient_variance"] == 0.5
    )

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / name, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    for offset, variance, color in [(-0.23, 0.25, "0.75"), (0, 0.5, "0.2"), (0.23, 1.0, "0.5")]:
        data = comparison[comparison.coefficient_variance == variance].sort_values("order")
        ax.bar(
            data.order + offset,
            data.posterior_model_probability,
            width=0.23,
            color=color,
            label=f"Coefficient variance {variance:g}",
        )
    ax.set(
        xlabel="Autoregressive order",
        ylabel="Posterior model probability",
        xticks=np.arange(1, 9),
        ylim=(0, 0.95),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, "lag_order_probabilities.png")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    primary = densities[densities.coefficient_variance == 0.5]
    for column, label, style, color in [
        ("bma_density", "BMA", "-", "0.1"),
        ("ar8_density", "AR(8)", "--", "0.45"),
        ("map_density", f"Selected AR({selected_order})", ":", "0.65"),
    ]:
        axes[0].plot(primary.coefficient_sum, primary[column], style, color=color, label=label)
    for variance, style, color in [(0.25, ":", "0.6"), (0.5, "-", "0.1"), (1.0, "--", "0.4")]:
        data = densities[densities.coefficient_variance == variance]
        axes[1].plot(
            data.coefficient_sum,
            data.bma_density,
            style,
            color=color,
            label=f"Variance {variance:g}",
        )
    for ax in axes:
        ax.set(xlabel="Sum of AR coefficients", ylabel="Posterior density")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    axes[0].set_title("Averaging and a fixed long specification")
    axes[1].set_title("Prior-concentration sensitivity")
    save(fig, "persistence_posteriors.png")
    fig, ax = plt.subplots(figsize=(8, 3.3))
    primary = responses[(responses.coefficient_variance == 0.5) & (responses.horizon > 0)]
    for target, label, style, color in [
        ("AR8", "AR(8)", "--", "0.55"),
        ("BMA", "Model average", "-", "0.1"),
        ("MAP", "Selected model", ":", "0.4"),
    ]:
        data = primary[primary.target == target].sort_values("horizon")
        ax.plot(data.horizon, data["median"], style, color=color, label=label, linewidth=1.3)
        if target in ["AR8", "BMA"]:
            ax.fill_between(
                data.horizon.to_numpy(),
                data.p05.to_numpy(),
                data.p95.to_numpy(),
                color=color,
                alpha=0.13,
            )
            ax.plot(data.horizon, data.p05, style, color=color, linewidth=0.5)
            ax.plot(data.horizon, data.p95, style, color=color, linewidth=0.5)
    ax.axhline(0, color="0.7", linewidth=0.5)
    ax.set(
        xlabel="Horizon after a one-unit innovation",
        ylabel="Response in supplied series units",
        xticks=np.arange(1, 11),
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, "response_uncertainty.png")
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    primary = traces[traces.coefficient_variance == 0.5]
    for chain, color in zip(sorted(primary.chain.unique()), ["0.1", "0.35", "0.6", "0.8"]):
        data = primary[primary.chain == chain]
        axes[0].plot(
            data.retained_draw,
            data.coefficient_sum,
            color=color,
            linewidth=0.55,
            label=f"Chain {chain}",
        )
        axes[1].plot(data.retained_draw, data.sigma2, color=color, linewidth=0.55)
    axes[0].set_ylabel("Coefficient sum")
    axes[1].set(
        xlabel="Retained draw index; every 50th draw displayed", ylabel="Innovation variance"
    )
    axes[0].legend(frameon=False, fontsize=9, ncol=4, loc="upper right")
    save(fig, "ar8_chain_traces.png")
    for filename, formula in [
        (
            "model_average_equation.png",
            r"$p(S\mid D)=\sum_{p=1}^{8}w_p\,p(S_p\mid D,M_p),\qquad S_p=\sum_{i=1}^{p}\phi_i$",
        ),
        (
            "variance_equation.png",
            r"$\mathrm{Var}(S\mid D)=\sum_p w_p\,\mathrm{Var}(S_p\mid D,M_p)+\sum_p w_p(\mu_p-\bar\mu)^2$",
        ),
    ]:
        fig = plt.figure(figsize=(8, 0.45))
        fig.text(0.5, 0.5, formula, ha="center", va="center", fontsize=14, math_fontfamily="stix")
        fig.savefig(figures / filename, bbox_inches="tight", dpi=220)
        plt.close(fig)


def build_report(output):
    out = Path(output)
    summary = json.loads((out / "study_summary.json").read_text())
    groups = summary["groups"]
    evidence = pd.read_csv(out / "model_comparison.csv")
    primary = next(g for g in groups if g["coefficient_variance"] == 0.5)
    bma, ar8, mapped = primary["bma"], primary["ar8"], primary["map"]
    lines = [
        "# Lag uncertainty and model-averaged persistence: computed results",
        "",
        "The supplied headerless file contains 200 ordered observations. All candidates condition on the same first eight and explain observations 9-200 (192 targets). The generating process, its parameters, seed, and any economic identity remain unverified.",
        "",
        "## Model comparison",
        "",
        "| Coefficient prior variance | AR order | Chib log evidence | Model probability | Log-evidence MCSE |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in evidence.iterrows():
        lines.append(
            f"| {row.coefficient_variance:g} | {int(row.order)} | {row.log_evidence_chib:.6f} | {row.posterior_model_probability:.8f} | {row.mcse_log_evidence:.6f} |"
        )
    lines += [
        "",
        f"The primary maximum-probability order is AR({primary['map_order']}), with posterior probability {primary['map_probability']:.6%}. Equal model prior probabilities and the declared independent parameter priors define these conditional comparisons.",
        "",
        "## Coefficient-sum inference",
        "",
        f"The model-averaged integrated-reference mean is {bma['integration_mean']:.6f}; its central 90% interval is [{bma['integration_p05']:.6f}, {bma['integration_p95']:.6f}]. The fixed AR(8) mean is {ar8['integration_mean']:.6f}, with interval [{ar8['integration_p05']:.6f}, {ar8['integration_p95']:.6f}]. These scalar reference calculations integrate precision after analytically integrating coefficients; Gibbs mixtures provide an independent simulation comparison.",
        "",
        f"Within-model posterior variance contributes {bma['integration_within_model_variance']:.8f} and between-model variance contributes {bma['integration_between_model_variance']:.8f}. The latter is {bma['integration_between_model_variance_share']:.3%} of the mixture variance. The selected-model interval is [{mapped['integration_p05']:.6f}, {mapped['integration_p95']:.6f}]. Model averaging need not widen inference relative to an arbitrary fixed high-order model.",
        "",
        f"The integrated-reference probability that the coefficient sum exceeds one is {bma['integration_probability_sum_above_one']:.10g}. Tail event counts in finite Gibbs samples can be zero; zero observed events are not reported as proof of zero posterior probability. A coefficient sum below one is not a sufficient stability condition.",
        "",
        "## Dynamics and stability",
        "",
        "Finite responses use a one-unit innovation, including a contemporaneous response of one. Horizons 1-10 have marginal 90% posterior intervals, not simultaneous bands. Both fixed AR(8) and model-averaged responses preserve all draws, including unstable draws. Model averaging is performed on the responses, not by padding and averaging coefficient vectors.",
        "",
        f"The model-weighted proportion of retained draws classified as stable is {primary['stable_draw_proportion_bma']:.8f}. This is a simulation proportion, not an exact probability of one. Separate weights conditional on the stability event are recorded and are not substituted for the primary weights.",
        "",
        "## Reproduction and limitations",
        "",
        "The main study runs 24 model/prior fits with the recorded independent chain streams. Additional intercept and Gamma-prior checks use deterministic integration and do not claim extra Gibbs fits. Diagnostic gates, evidence at three evaluation points, SVD and conditional-matrix integration agreement, scalar posterior controls, and companion-response checks are recorded in validation.json and the control tables.",
        "",
        "Data remain external, and private draw archives are ignored. Parameter inference is conditional on the supplied series and finite model set. No generating order is inferred from the file name, and no forecast superiority or empirical causal effect is claimed. See [methods](../docs/methods.md), the [executive summary](../EXECUTIVE_SUMMARY.md), and the [research report](../docs/lag_uncertainty_model_averaging.pdf).",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")

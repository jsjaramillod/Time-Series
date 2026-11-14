"""Reproducible analyses. Every artifact derives from an explicit dataset and seed."""

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from .comparison import ChibModelComparison, model_probabilities, quadrature_log_evidence
from .data import InterestRateData, RecessionData, file_digest
from .evaluation import ExpandingWindowEvaluator
from .models import GaussianRegression
from .priors import GammaPrecisionPrior, NormalPrior
from .reporting import policy_figures, probit_figures, recession_figure, write_json
from .results import PosteriorResult
from .samplers import GaussianGibbsSampler, SamplerConfig


def _run_metadata(config: SamplerConfig) -> dict:
    return {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "sampler": asdict(config),
        "software": {
            name: version(name)
            for name in ["bayes-policy", "numpy", "scipy", "pandas", "arviz", "matplotlib"]
        },
    }


def _probability_summary(draws: np.ndarray, threshold: float) -> dict:
    event = (draws > threshold).astype(float)
    probability = float(event.mean())
    mcse = 0.0 if probability in (0, 1) else float(az.mcse(event, method="mean").item())
    return {
        "probability": probability,
        "mcse": mcse,
        "no_observed_transitions": probability in (0, 1),
    }


def _derived(result: PosteriorResult):
    beta_pi, rho = result.parameter("inflation_gap"), result.parameter("lagged_rate")
    denominator = 1 - rho
    if np.any(denominator == 0):
        raise RuntimeError("Undefined long-run ratio: an exact unit-root draw occurred")
    theta = beta_pi / denominator
    stable = np.abs(rho) < 1
    derived = {
        "long_run_positive": _probability_summary(theta, 0),
        "long_run_above_one": _probability_summary(theta, 1),
        "unstable_draw_fraction": float((~stable).mean()),
        "near_unit_root_fraction_abs_one_minus_rho_below_0_01": float(
            (abs(denominator) < 0.01).mean()
        ),
        "theta_median": float(np.median(theta)),
        "theta_p05": float(np.quantile(theta, 0.05)),
        "theta_p95": float(np.quantile(theta, 0.95)),
        "theta_gt1_given_stable": float((theta[stable] > 1).mean()) if stable.any() else None,
    }
    values, names = [theta], ["long_run_inflation"]
    if "output_gap_positive" in result.names:
        difference = result.parameter("output_gap_positive") - result.parameter(
            "output_gap_negative"
        )
        derived["positive_minus_negative_gap_above_zero"] = _probability_summary(difference, 0)
        values.append(difference)
        names.append("positive_minus_negative_gap")
    derived_result = PosteriorResult(np.stack(values, axis=2), tuple(names), result.metadata)
    return derived, derived_result


def _conditional_predictive_check(model, result, seed):
    """Fixed-X conditional checks, not a dynamic policy simulation or causal assessment."""
    rng = np.random.default_rng(seed)
    flat = result.draws.reshape(-1, len(result.names))
    selected = flat[rng.choice(len(flat), size=min(2000, len(flat)), replace=False)]
    residuals = model.data.y[None, :] - selected[:, :-1] @ model.data.X.T
    simulated_errors = rng.normal(size=residuals.shape) * np.sqrt(selected[:, -1, None])

    def acf1(x):
        x = x - x.mean(axis=1, keepdims=True)
        return (x[:, :-1] * x[:, 1:]).sum(axis=1) / (x * x).sum(axis=1)

    residual = model.data.y - model.data.X @ flat[:, :-1].mean(axis=0)
    residual = residual - residual.mean()
    return {
        "type": "fixed-design conditional posterior predictive check",
        "residual_lag1_acf_at_posterior_mean": float(
            residual[:-1] @ residual[1:] / (residual @ residual)
        ),
        "ppp_abs_lag1_acf": float((abs(acf1(simulated_errors)) >= abs(acf1(residuals))).mean()),
        "draws_used": len(selected),
        "limitations": "Conditions on observed lagged rate; not a dynamic forecast",
    }


def run_policy(excel: str | Path, output: str | Path, config: SamplerConfig) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "draws").mkdir(exist_ok=True)
    dataset = InterestRateData.from_excel(excel)
    write_json(out / "data_manifest.json", dataset.manifest())
    metadata = _run_metadata(config)
    reference_dir = Path(excel).parent
    metadata["reference_code_sha256"] = {
        name: file_digest(reference_dir / name)
        for name in ["gen_h.m", "gen_beta.m", "interest_rate_rule_estimate_gibbs.m"]
        if (reference_dir / name).is_file()
    }
    write_json(out / "run_metadata.json", metadata)
    results, summaries, model_rows, derived_rows, checks = {}, [], [], [], {}
    all_gates, evidence_checks = [], []
    for variance in [10.0, 5.0]:
        models, fits = [], []
        for asymmetric in [False, True]:
            name = f"{'asymmetric' if asymmetric else 'symmetric'}_v{variance:g}"
            design = dataset.design(asymmetric)
            model = GaussianRegression(
                design,
                NormalPrior.isotropic(len(design.names), variance),
                GammaPrecisionPrior.from_reference(),
            )
            print(f"Sampling {name}: {len(design.y)} observations", flush=True)
            result = GaussianGibbsSampler(config).fit(model)
            results[name] = result
            result.save(out / "draws" / f"{name}.npz")
            summary = result.summary()
            summary["ols"] = np.r_[model.ols(), np.nan]
            summary["model"] = name
            summaries.append(summary.reset_index())
            derived, derived_result = _derived(result)
            derived_result.summary().to_csv(out / f"{name}_derived_diagnostics.csv")
            derived_rows.append({"model": name, **derived})
            gate = result.diagnostic_gate()
            derived_gate = derived_result.diagnostic_gate()
            all_gates += [gate["passed"], derived_gate["passed"]]
            checks[name] = {
                "sampler": gate,
                "derived": derived_gate,
                "predictive_check": _conditional_predictive_check(model, result, config.seed + 91),
            }
            models.append(model)
            fits.append(result)
        estimates, weights = ChibModelComparison.compare(models, fits)
        for model, result, estimate, weight in zip(models, fits, estimates, weights):
            control = quadrature_log_evidence(model)
            delta = estimate.log_evidence - control["log_evidence"]
            tolerance = max(0.01, 5 * estimate.mcse_log_evidence)
            matched = abs(delta) <= tolerance
            evidence_checks.append(matched)
            name = f"{'asymmetric' if len(model.data.names) == 5 else 'symmetric'}_v{variance:g}"
            model_rows.append(
                {
                    "model": name,
                    "prior_variance": variance,
                    "log_evidence_chib": estimate.log_evidence,
                    "mcse_log_evidence_chib": estimate.mcse_log_evidence,
                    "log_evidence_quadrature": control["log_evidence"],
                    "quadrature_relative_error": control["relative_quadrature_error"],
                    "chib_minus_quadrature": delta,
                    "agreement_tolerance": tolerance,
                    "evidence_control_passed": matched,
                    "posterior_model_probability": weight,
                    "probability_if_prior_asymmetric_0_25": model_probabilities(
                        [e.log_evidence for e in estimates], [0.75, 0.25]
                    )[int(len(model.data.names) == 5)],
                    "probability_if_prior_asymmetric_0_75": model_probabilities(
                        [e.log_evidence for e in estimates], [0.25, 0.75]
                    )[int(len(model.data.names) == 5)],
                }
            )
    pd.concat(summaries, ignore_index=True).to_csv(out / "policy_posterior.csv", index=False)
    pd.DataFrame(model_rows).to_csv(out / "model_comparison.csv", index=False)
    write_json(out / "policy_derived.json", derived_rows)
    validation = {
        "sampler_diagnostics_passed": all(all_gates),
        "independent_evidence_controls_passed": all(evidence_checks),
        "models": checks,
        "status": "validated computationally"
        if all(all_gates + evidence_checks)
        else "provisional; checks failed",
    }
    write_json(out / "validation.json", validation)
    policy_figures(results, out / "figures")
    return validation


def run_recession(
    excel: str | Path,
    output: str | Path,
    config: SamplerConfig,
    start="2007-01",
    end="2007-12",
    label_delay_months=0,
    label_calendar: pd.Series | None = None,
) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    data = RecessionData.from_excel(excel)
    origins = list(pd.period_range(start, end, freq="M"))
    # Include the requested endpoint forecast without using unavailable future outcomes.
    if pd.Period("2026-02", freq="M").to_timestamp(how="end").normalize() in data.features().index:
        if pd.Period("2026-02", freq="M") not in origins:
            origins.append(pd.Period("2026-02", freq="M"))
    evaluator = ExpandingWindowEvaluator(config, label_delay_months=label_delay_months)
    frame, scores, results = evaluator.evaluate(data, origins, label_calendar)
    frame.to_csv(out / "recession_forecasts.csv", index=False)
    write_json(out / "recession_scores.json", scores)
    write_json(
        out / "recession_manifest.json",
        {
            **data.manifest(),
            "horizon": 12,
            "prior_covariance": "10 I, independent normal coefficients",
            "latent_variance": 1,
            "metadata": _run_metadata(config),
        },
    )
    draws_dir = out / "draws"
    draws_dir.mkdir(exist_ok=True)
    for origin, result in results.items():
        result.save(draws_dir / f"probit_{origin}.npz")
        result.summary().to_csv(out / f"probit_{origin}_diagnostics.csv")
    recession_figure(frame, out / "recession_probabilities.png")
    if "2026-02-28" in results:
        probit_figures(results["2026-02-28"], out)
    return scores

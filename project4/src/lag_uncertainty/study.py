"""Common-window model comparison, posterior diagnostics, and scalar model averages."""

import hashlib
import importlib
import importlib.metadata
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from bayes_policy.comparison import model_probabilities, quadrature_log_evidence
from bayes_policy.results import PosteriorResult
from bayes_policy.samplers import GaussianGibbsSampler, SamplerConfig

from .data import ARDesignBuilder
from .dynamics import PosteriorDynamics
from .evidence import ChibOrdinateAudit
from .mixture import ModelAveragedInference
from .reference import IntegratedGaussianReference, integrated_mixture_summary
from .specification import PriorSpecification


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_hashes():
    modules = [
        "bayes_policy.data",
        "bayes_policy.models",
        "bayes_policy.priors",
        "bayes_policy.samplers",
        "bayes_policy.results",
        "bayes_policy.comparison",
        "shock_dynamics.analysis",
        "shock_dynamics.models",
        "shock_dynamics.simulation",
    ]
    modules += [f"lag_uncertainty.{p.stem}" for p in sorted(Path(__file__).parent.glob("*.py"))]
    return {
        name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
        for name in modules
    }


def _fit_signature():
    names = ["bayes_policy.models", "bayes_policy.priors", "bayes_policy.samplers"]
    return {
        name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
        for name in names
    }


def _same_fit(result, model, config):
    metadata = result.metadata
    return (
        all(metadata.get(k) == v for k, v in asdict(config).items())
        and metadata.get("data_fingerprint") == model.data.fingerprint()
        and metadata.get("coefficient_prior_mean") == model.prior.mean.tolist()
        and metadata.get("coefficient_prior_covariance") == model.prior.covariance.tolist()
        and metadata.get("precision_prior_shape") == model.precision_prior.shape
        and metadata.get("precision_prior_rate") == model.precision_prior.rate
        and metadata.get("fitting_source_sha256") == _fit_signature()
    )


def joint_batch_statistics(results, audits, dynamics):
    retained = results[0].draws.shape[1]
    batch_size = 1000 if retained % 1000 == 0 else retained
    evidence = np.column_stack([audit.batch_log_evidence(batch_size) for audit in audits])
    total_batches = len(evidence)
    sum_blocks = [d.coefficient_sum.reshape(-1, batch_size) for d in dynamics]
    probability_blocks = [
        (d.coefficient_sum > 1).reshape(-1, batch_size).mean(axis=1) for d in dynamics
    ]
    rows = []
    for index in range(total_batches):
        batch_weights = model_probabilities(evidence[index])
        mixture = ModelAveragedInference([values[index] for values in sum_blocks], batch_weights)
        quantiles = mixture.quantiles()
        rows.append(
            {
                "bma_mean": float(batch_weights @ np.array([v[index].mean() for v in sum_blocks])),
                "bma_probability_sum_above_one": float(
                    batch_weights @ np.array([v[index] for v in probability_blocks])
                ),
                "bma_p05": float(quantiles[0]),
                "bma_median": float(quantiles[1]),
                "bma_p95": float(quantiles[2]),
                **{f"weight_ar{p}": float(w) for p, w in enumerate(batch_weights, 1)},
            }
        )
    frame = pd.DataFrame(rows)
    mcse = (frame.std(ddof=1) / np.sqrt(total_batches)).to_dict()
    return frame, {
        "method": "joint contiguous batches preserving each chain; common pooled evaluation point",
        "batch_size": batch_size,
        "batches": total_batches,
        "mcse": mcse,
        "interpretation": "Numerical Monte Carlo uncertainty; distinct from posterior model uncertainty",
        "zero_event_note": "Zero sampled tail events do not establish zero posterior probability",
    }


class LagOrderStudy:
    def __init__(self, data, output="results", config=None):
        self.builder = data if isinstance(data, ARDesignBuilder) else ARDesignBuilder.from_csv(data)
        if self.builder.maximum_lag != 8:
            raise ValueError(
                "This study compares orders 1-8 on a common eight-observation conditioning window"
            )
        self.output = Path(output)
        self.config = SamplerConfig(seed=20261117) if config is None else config

    def _fit(self, model, config, archive, reuse_draws):
        if reuse_draws and archive.exists():
            result = PosteriorResult.load(archive)
            if not _same_fit(result, model, config):
                raise ValueError(
                    "Cached draws do not match current data, priors, settings or sampler source; use fresh output"
                )
            return result
        result = GaussianGibbsSampler(config).fit(model)
        result.metadata.update(
            {
                "fitting_source_sha256": _fit_signature(),
                "observation_index_kind": "one-based ordinal positions",
            }
        )
        result.save(archive)
        return result

    def run(self, reuse_draws=True):
        out = self.output
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "data_manifest.json", self.builder.manifest())
        evidence_rows, parameter_rows, derived_rows, point_rows = [], [], [], []
        persistence_rows, response_rows, stability_rows, density_rows, trace_rows = (
            [],
            [],
            [],
            [],
            [],
        )
        groups, gate_rows, reference_controls, batch_rows, batch_records = [], [], [], [], {}
        mixture_checks = []
        common_target_fingerprints = set()
        for scenario_index, variance in enumerate((0.5, 1.0, 0.25)):
            prior = PriorSpecification(variance)
            tag = f"variance_{str(variance).replace('.', '_')}"
            directory = out / "draws" / tag
            directory.mkdir(parents=True, exist_ok=True)
            models, results, audits, dynamic_results, references = [], [], [], [], []
            for order in range(1, 9):
                model = prior.model(self.builder.design(order))
                common_target_fingerprints.add(
                    hashlib.sha256(
                        model.data.y.tobytes() + model.data.observation_indices.tobytes()
                    ).hexdigest()
                )
                config = replace(
                    self.config, seed=self.config.seed + scenario_index * 10000 + order * 100
                )
                result = self._fit(model, config, directory / f"ar{order}.npz", reuse_draws)
                parameter_summary = result.summary()
                parameter_gate = result.diagnostic_gate()
                dynamics = PosteriorDynamics.from_coefficients(result.coefficients[..., 1:])
                transformed = PosteriorResult(
                    dynamics.continuous_draws(),
                    ("coefficient_sum", "spectral_radius")
                    + tuple(f"psi_{h}" for h in range(1, 11)),
                    {},
                )
                derived_summary = transformed.summary()
                derived_gate = transformed.diagnostic_gate()
                audit = ChibOrdinateAudit.run(model, result)
                reference = IntegratedGaussianReference(model)
                conditional_integral = quadrature_log_evidence(model)
                contrast = np.r_[0, np.ones(order)]
                exact = reference.contrast_summary(contrast)
                empirical_mean = float(dynamics.coefficient_sum.mean())
                empirical_mcse = float(derived_summary.loc["coefficient_sum", "mcse_mean"])
                evidence_ok = all(
                    abs(check["log_evidence"] - reference.log_evidence)
                    < max(0.003, 6 * check["mcse_log_evidence"])
                    for check in audit.point_checks
                )
                references_agree = (
                    abs(conditional_integral["log_evidence"] - reference.log_evidence) < 1e-8
                )
                sum_ok = abs(empirical_mean - exact["mean"]) < max(0.001, 6 * empirical_mcse)
                passed = (
                    parameter_gate["passed"]
                    and derived_gate["passed"]
                    and evidence_ok
                    and references_agree
                    and sum_ok
                    and dynamics.crosscheck_maximum_error < 1e-10
                )
                gate_rows.append(
                    {
                        "variance": variance,
                        "order": order,
                        "passed": bool(passed),
                        "parameters": parameter_gate,
                        "derived": derived_gate,
                        "evidence_point_checks_pass": bool(evidence_ok),
                        "independent_integrals_agree": bool(references_agree),
                        "coefficient_sum_control_pass": bool(sum_ok),
                    }
                )
                for kind, frame in [("parameter", parameter_summary), ("derived", derived_summary)]:
                    for name, row in frame.iterrows():
                        record = {
                            "coefficient_variance": variance,
                            "order": order,
                            "parameter": name,
                            **row.to_dict(),
                        }
                        (parameter_rows if kind == "parameter" else derived_rows).append(record)
                evidence_rows.append(
                    {
                        "coefficient_variance": variance,
                        "order": order,
                        "n": len(model.data.y),
                        "log_evidence_chib": audit.estimate.log_evidence,
                        "mcse_log_evidence": audit.estimate.mcse_log_evidence,
                        "log_evidence_svd_quadrature": reference.log_evidence,
                        "log_evidence_conditional_quadrature": conditional_integral["log_evidence"],
                        "absolute_chib_integration_error": abs(
                            audit.estimate.log_evidence - reference.log_evidence
                        ),
                        "svd_vs_conditional_integration_error": abs(
                            reference.log_evidence - conditional_integral["log_evidence"]
                        ),
                        "precision_node_mass_error": reference.node_mass_error,
                        "precision_quadrature_nodes": reference.node_count,
                        "seed": config.seed,
                    }
                )
                point_rows.extend(
                    {
                        "coefficient_variance": variance,
                        "order": order,
                        **row,
                        "absolute_error_vs_quadrature": abs(
                            row["log_evidence"] - reference.log_evidence
                        ),
                    }
                    for row in audit.point_checks
                )
                reference_controls.append(
                    {
                        "coefficient_variance": variance,
                        "order": order,
                        "mcmc_sum_mean": empirical_mean,
                        "mcmc_sum_mean_mcse": empirical_mcse,
                        **{"integration_" + k: v for k, v in exact.items()},
                        "mcmc_sum_probability_above_one": float(
                            np.mean(dynamics.coefficient_sum > 1)
                        ),
                    }
                )
                stability_rows.append(
                    {"coefficient_variance": variance, "order": order, **dynamics.summary()}
                )
                if order == 8:
                    for chain in range(self.config.chains):
                        for draw in range(0, self.config.draws, 50):
                            trace_rows.append(
                                {
                                    "coefficient_variance": variance,
                                    "chain": chain + 1,
                                    "retained_draw": draw + 1,
                                    "coefficient_sum": float(dynamics.coefficient_sum[chain, draw]),
                                    "sigma2": float(result.parameter("sigma2")[chain, draw]),
                                }
                            )
                models.append(model)
                results.append(result)
                audits.append(audit)
                dynamic_results.append(dynamics)
                references.append(reference)
                print(
                    f"variance={variance:g} AR({order}): log evidence={audit.estimate.log_evidence:.5f}; controls={bool(passed)}",
                    flush=True,
                )
                if not passed:
                    write_json(out / "validation.json", {"fits": gate_rows, "all_passed": False})
                    raise RuntimeError(
                        "Scientific control failed; inspect validation.json before interpreting results"
                    )
            weights = model_probabilities([a.estimate.log_evidence for a in audits])
            control_weights = model_probabilities([r.log_evidence for r in references])
            for row, w, w_ref in zip(evidence_rows[-8:], weights, control_weights):
                row.update(
                    {
                        "model_prior_probability": 0.125,
                        "posterior_model_probability": float(w),
                        "quadrature_model_probability": float(w_ref),
                    }
                )
            batch_frame, batch_record = joint_batch_statistics(results, audits, dynamic_results)
            batch_records[str(variance)] = batch_record
            for index, row in batch_frame.iterrows():
                batch_rows.append(
                    {"coefficient_variance": variance, "batch": index + 1, **row.to_dict()}
                )
            selected = int(np.argmax(weights))
            target_samples = [d.coefficient_sum for d in dynamic_results]
            resampling = ModelAveragedInference(target_samples, weights, tuple(range(1, 9)))
            resample_seed = self.config.seed + scenario_index * 10000 + 9999
            mixture_values, model_indices = resampling.draw(100000, resample_seed)
            np.savez_compressed(
                directory / "bma_scalar.npz",
                coefficient_sum=mixture_values,
                model_order=model_indices,
                seed=resample_seed,
            )
            mixture_checks.append(
                {
                    "coefficient_variance": variance,
                    "resamples": 100000,
                    "seed": resample_seed,
                    "weighted_empirical_mean": resampling.summary()["mean"],
                    "resample_mean": float(mixture_values.mean()),
                    "resampling_standard_error_of_mean": float(
                        mixture_values.std(ddof=1) / np.sqrt(len(mixture_values))
                    ),
                    "interpretation": "Resamples from finite model-specific Gibbs pools; not additional independent Gibbs information",
                }
            )
            for label, target_weights in [
                ("BMA", weights),
                ("AR8", np.eye(8)[7]),
                ("MAP", np.eye(8)[selected]),
            ]:
                scalar = ModelAveragedInference(target_samples, target_weights, tuple(range(1, 9)))
                exact = integrated_mixture_summary(references, target_weights)
                persistence_rows.append(
                    {
                        "coefficient_variance": variance,
                        "target": label,
                        "selected_order": selected + 1
                        if label == "MAP"
                        else (8 if label == "AR8" else 0),
                        **scalar.summary(),
                        **{"integration_" + k: v for k, v in exact.items()},
                        "mean_mcse_joint_batches": batch_record["mcse"]["bma_mean"]
                        if label == "BMA"
                        else float(
                            next(
                                row["mcse_mean"]
                                for row in derived_rows
                                if row["coefficient_variance"] == variance
                                and row["order"] == (8 if label == "AR8" else selected + 1)
                                and row["parameter"] == "coefficient_sum"
                            )
                        ),
                    }
                )
                for horizon in range(11):
                    mixture = ModelAveragedInference(
                        [d.responses[..., horizon] for d in dynamic_results], target_weights
                    )
                    s = mixture.summary()
                    response_rows.append(
                        {
                            "coefficient_variance": variance,
                            "target": label,
                            "horizon": horizon,
                            **{key: s[key] for key in ("mean", "median", "sd", "p05", "p95")},
                        }
                    )
            component_summaries = [
                r.contrast_summary(np.r_[0, np.ones(p)]) for p, r in enumerate(references, 1)
            ]
            lower = min(s["mean"] - 5 * s["sd"] for s in component_summaries)
            upper = max(s["mean"] + 5 * s["sd"] for s in component_summaries)
            grid = np.linspace(lower, upper, 501)
            component_densities = np.column_stack(
                [
                    r.contrast_density(grid, np.r_[0, np.ones(p)])
                    for p, r in enumerate(references, 1)
                ]
            )
            for x, density, ar8_density, map_density in zip(
                grid,
                component_densities @ weights,
                component_densities[:, 7],
                component_densities[:, selected],
            ):
                density_rows.append(
                    {
                        "coefficient_variance": variance,
                        "coefficient_sum": float(x),
                        "bma_density": float(density),
                        "ar8_density": float(ar8_density),
                        "map_density": float(map_density),
                    }
                )
            stable_probabilities = np.array([d.stable.mean() for d in dynamic_results])
            stable_total = float(weights @ stable_probabilities)
            stable_weights = (
                ModelAveragedInference.condition_weights(weights, stable_probabilities)
                if stable_total > 0
                else np.repeat(np.nan, 8)
            )
            for row, stable_weight in zip(stability_rows[-8:], stable_weights):
                row["model_weight_conditional_on_stability"] = float(stable_weight)
            groups.append(
                {
                    "coefficient_variance": variance,
                    "map_order": selected + 1,
                    "map_probability": float(weights[selected]),
                    "expected_order": float(weights @ np.arange(1, 9)),
                    "effective_number_of_models_inverse_squared_weights": float(
                        1 / (weights @ weights)
                    ),
                    "stable_draw_proportion_bma": stable_total,
                    "maximum_weight_difference_chib_vs_quadrature": float(
                        np.max(np.abs(weights - control_weights))
                    ),
                    "bma": next(row for row in persistence_rows[-3:] if row["target"] == "BMA"),
                    "ar8": next(row for row in persistence_rows[-3:] if row["target"] == "AR8"),
                    "map": next(row for row in persistence_rows[-3:] if row["target"] == "MAP"),
                }
            )
        sensitivity_rows = self._prior_sensitivity()
        if len(common_target_fingerprints) != 1:
            raise RuntimeError(
                "All model/prior cases must explain the identical target vector and positions"
            )
        tables = {
            "model_comparison": evidence_rows,
            "posterior_parameters": parameter_rows,
            "posterior_derived": derived_rows,
            "evidence_point_checks": point_rows,
            "scalar_integration_controls": reference_controls,
            "persistence_summary": persistence_rows,
            "impulse_responses": response_rows,
            "stability": stability_rows,
            "posterior_density_grid": density_rows,
            "ar8_thinned_traces": trace_rows,
            "joint_batches": batch_rows,
            "additional_prior_sensitivity": sensitivity_rows,
            "mixture_resampling_checks": mixture_checks,
        }
        for name, rows in tables.items():
            pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
        write_json(
            out / "study_summary.json",
            {
                "groups": groups,
                "interpretation": "Inference conditional on AR(1)-AR(8), stated priors, and the common initial conditioning observations; no forecasting evaluation",
                "primary_coefficient_variance": 0.5,
                "intercept_variance": 100,
                "precision_shape": 1,
                "precision_rate": 3,
                "scalar_tail_probability_note": "Integration controls resolve tails that may have zero events in retained Gibbs draws; neither is a root-stability test",
            },
        )
        write_json(out / "batch_mcse.json", batch_records)
        write_json(
            out / "validation.json",
            {
                "fits": gate_rows,
                "all_passed": all(r["passed"] for r in gate_rows),
                "fits_count": len(gate_rows),
                "common_target_fingerprints": sorted(common_target_fingerprints),
                "common_target_positions_inclusive": [9, len(self.builder.values)],
                "common_modeled_observations": len(self.builder.values) - 8,
                "evidence_absolute_tolerance_floor": 0.003,
                "evidence_mcse_multiplier": 6,
                "coefficient_sum_absolute_mean_tolerance_floor": 0.001,
                "sum_mcse_multiplier": 6,
                "stability_boundary_tolerance": 1e-10,
                "pointwise_response_horizons": [0, 10],
            },
        )
        write_json(
            out / "run_metadata.json",
            {
                "sampler": asdict(self.config),
                "seed_rule": "base seed + scenario index * 10000 + order * 100; separate spawned chain streams",
                "coefficient_variance_order": [0.5, 1.0, 0.25],
                "intercept_variance": 100.0,
                "precision_prior": {"shape": 1.0, "rate": 3.0},
                "model_prior": "equal probability 1/8 within each separate sensitivity analysis",
                "model_orders": list(range(1, 9)),
                "python": platform.python_version(),
                "platform": platform.system(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in [
                        "numpy",
                        "scipy",
                        "pandas",
                        "matplotlib",
                        "arviz",
                        "bayes-policy",
                        "shock-dynamics",
                    ]
                },
                "units": "unspecified series units and generic observation periods; no empirical calendar assigned",
            },
        )
        from .reporting import build_figures, build_report

        build_figures(out)
        build_report(out)
        manifest = {
            "source_sha256": source_hashes(),
            "artifact_sha256": {
                str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(out.rglob("*"))
                if p.is_file()
                and "draws" not in p.relative_to(out).parts
                and p.name
                not in {
                    "manifest.json",
                    "notebook_validation.json",
                    "report_validation.json",
                    "verification.json",
                }
            },
        }
        write_json(out / "manifest.json", manifest)
        return groups

    def _prior_sensitivity(self):
        rows = []
        alternatives = [
            ("primary", PriorSpecification()),
            ("intercept_variance_10", PriorSpecification(intercept_variance=10)),
            ("intercept_variance_1000", PriorSpecification(intercept_variance=1000)),
            (
                "precision_mean_degree_alternative",
                PriorSpecification(precision_shape=1.5, precision_rate=1.5),
            ),
        ]
        for label, prior in alternatives:
            refs = [
                IntegratedGaussianReference(prior.model(self.builder.design(order)))
                for order in range(1, 9)
            ]
            weights = model_probabilities([r.log_evidence for r in refs])
            summary = integrated_mixture_summary(refs, weights)
            for order, (r, w) in enumerate(zip(refs, weights), 1):
                rows.append(
                    {
                        "scenario": label,
                        **asdict(prior),
                        "order": order,
                        "log_evidence_quadrature": r.log_evidence,
                        "posterior_model_probability": float(w),
                        "bma_sum_mean": summary["mean"],
                        "bma_sum_sd": summary["sd"],
                        "bma_sum_p05": summary["p05"],
                        "bma_sum_p95": summary["p95"],
                        "method": "deterministic integration; no additional Gibbs fits claimed",
                    }
                )
        return rows

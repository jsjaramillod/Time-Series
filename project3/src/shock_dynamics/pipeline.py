"""Recorded, reproducible experiments and independent numerical controls."""

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import norm

from .analysis import DynamicsAnalyzer, MAAnalyzer
from .models import ARSpecification, MASpecification
from .simulation import ARSimulator, MASimulator, SimulationConfig


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def mean_precision_experiment(seed, replications=10000, length=200):
    if isinstance(replications, bool) or not isinstance(replications, int) or replications < 2:
        raise ValueError("Replications must be an integer >= 2")
    model = ARSpecification((0.9,))
    dynamics = DynamicsAnalyzer(model)
    variance = float(dynamics.stationary_covariance[0, 0])
    mean_variance = dynamics.sample_mean_variance(length)
    independent_variance = variance / length
    rng = np.random.default_rng(seed)
    state = rng.normal(scale=np.sqrt(variance), size=replications)
    totals = np.zeros(replications)
    for _ in range(length):
        state = 0.9 * state + rng.normal(size=replications)
        totals += state
    means = totals / length
    z = float(norm.ppf(0.975))

    def coverage(v):
        return float(np.mean(np.abs(means) <= z * np.sqrt(v)))

    summary = {
        "model": asdict(model),
        "seed": seed,
        "replications": replications,
        "length": length,
        "initialization": "Independent draws from the exact stationary Gaussian distribution",
        "stationary_variance": variance,
        "exact_mean_variance": mean_variance,
        "independence_mean_variance": independent_variance,
        "variance_inflation_factor": mean_variance / independent_variance,
        "effective_independent_observations": variance / mean_variance,
        "exact_mean_standard_error": np.sqrt(mean_variance),
        "independence_standard_error": np.sqrt(independent_variance),
        "empirical_mean": float(means.mean()),
        "empirical_mean_variance": float(means.var(ddof=1)),
        "mean_variance_mc_standard_error": mean_variance * np.sqrt(2 / (replications - 1)),
        "correlation_aware_coverage": coverage(mean_variance),
        "independence_coverage": coverage(independent_variance),
        "exact_correlation_aware_coverage": 0.95,
        "exact_independence_coverage": float(
            2 * norm.cdf(z * np.sqrt(independent_variance / mean_variance)) - 1
        ),
        "coverage_mc_standard_error_at_95_percent": np.sqrt(0.95 * 0.05 / replications),
        "ar1_shock_half_life_periods": np.log(0.5) / np.log(0.9),
        "interpretation": "Repeated-sample coverage under known parameters; not posterior intervals",
    }
    return summary, means


def run_experiments(output="results", seed=20261115, replications=10000):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    specifications = {
        "ar1_persistent": ARSpecification((0.9,)),
        "ar1_mean_shift": ARSpecification((0.9,), intercept=0.5),
        "ar3_stable": ARSpecification((0.5, 0.2, 0.2)),
        "ar3_unstable": ARSpecification((0.5, 0.3, 0.3)),
        "ar2_response": ARSpecification((1.2, -0.3)),
        "ar2_boundary": ARSpecification((1.5, -1.0)),
    }
    rows, response_rows, path_rows, configs, controls = [], [], [], {}, {}
    paths, responses = {}, {}
    for name, model in specifications.items():
        analyzer = DynamicsAnalyzer(model)
        stable = analyzer.classification == "stable"
        # AR(3) and boundary comparisons deliberately use zero-state, zero-burn paths.
        burn = 500 if name in {"ar1_persistent", "ar1_mean_shift", "ar2_response"} else 0
        stream = seed if name.startswith("ar1_") else seed + 1
        if name == "ar2_response":
            stream = seed + 2
        config = SimulationConfig(length=200, burn_in=burn, seed=stream)
        initial = np.repeat(analyzer.stationary_mean if stable else 0.0, model.order)
        simulation = ARSimulator(model).simulate(config, initial)
        paths[name] = simulation.values
        configs[name] = {
            "specification": asdict(model),
            "simulation": asdict(config),
            "initial_state_most_recent_first": initial.tolist(),
            "initial_state_role": "Lagged observations before generating burn-in and retained values",
            "innovation_sha256": hashlib.sha256(simulation.innovations.tobytes()).hexdigest(),
        }
        for t, (value, innovation) in enumerate(
            zip(simulation.values, simulation.retained_innovations)
        ):
            path_rows.append(
                {"experiment": name, "period": t, "value": value, "innovation": innovation}
            )
        irf = analyzer.impulse_response(100)
        responses[name] = irf
        control_noise = np.zeros(101)
        control_noise[0] = 1
        independent = lfilter([1.0], np.r_[1.0, -np.array(model.coefficients)], control_noise)
        controls[name] = {
            "maximum_three_method_disagreement": irf.maximum_disagreement,
            "scipy_filter_maximum_error": float(
                np.max(np.abs(independent - irf.methods["recursion"]))
            ),
            "response_agreement_pass": bool(
                all(
                    np.allclose(independent, v, rtol=1e-11, atol=1e-11)
                    for v in irf.methods.values()
                )
            ),
        }
        if stable:
            covariance = analyzer.stationary_covariance
            q = np.zeros_like(covariance)
            q[0, 0] = model.innovation_variance
            residual = analyzer.companion @ covariance @ analyzer.companion.T - covariance + q
            extended_noise = np.r_[1.0, np.zeros(5000)]
            extended_response = lfilter(
                [1.0], np.r_[1.0, -np.array(model.coefficients)], extended_noise
            )
            variance_error = abs(
                covariance[0, 0] - model.innovation_variance * extended_response @ extended_response
            )
            controls[name].update(
                {
                    "lyapunov_maximum_residual": float(np.abs(residual).max()),
                    "variance_vs_5001_response_terms_error": float(variance_error),
                    "stationary_moment_controls_pass": bool(
                        np.abs(residual).max() < 1e-10 and variance_error < 1e-10
                    ),
                }
            )
        rows.append(
            {
                "experiment": name,
                "coefficients": ";".join(map(str, model.coefficients)),
                "intercept": model.intercept,
                "innovation_variance": model.innovation_variance,
                "classification": analyzer.classification,
                "spectral_radius": analyzer.spectral_radius,
                "minimum_finite_root_modulus": float(np.min(np.abs(analyzer.characteristic_roots))),
                "stationary_mean": analyzer.stationary_mean if stable else None,
                "stationary_variance": float(analyzer.stationary_covariance[0, 0])
                if stable
                else None,
                "long_run_sum": analyzer.long_run_sum if stable else None,
                "response_at_20": irf.methods["recursion"][20],
                "cumulative_at_20": irf.cumulative[20],
                "cumulative_at_100": irf.cumulative[100],
                "remaining_sum_after_100": analyzer.long_run_sum - irf.cumulative[100]
                if stable
                else None,
                "path_sample_mean": float(simulation.values.mean()),
                "path_sample_variance_ddof1": float(simulation.values.var(ddof=1)),
                "burn_in": burn,
            }
        )
        for t in irf.horizons:
            response_rows.append(
                {
                    "experiment": name,
                    "horizon": int(t),
                    **{method: float(values[t]) for method, values in irf.methods.items()},
                    "cumulative": float(irf.cumulative[t]),
                }
            )
    ma = MASpecification((1.0, 1.5, 1.3))
    ma_analysis = MAAnalyzer(ma)
    ma_config = SimulationConfig(200, 500, seed + 3)
    ma_simulation = MASimulator(ma).simulate(ma_config, (0.0, 0.0))
    for t, (value, innovation) in enumerate(
        zip(ma_simulation.values, ma_simulation.retained_innovations)
    ):
        path_rows.append(
            {"experiment": "ma2_control", "period": t, "value": value, "innovation": innovation}
        )
    configs["ma2_control"] = {
        "specification": asdict(ma),
        "simulation": asdict(ma_config),
        "initial_innovation_state_most_recent_first": [0.0, 0.0],
        "innovation_sha256": hashlib.sha256(ma_simulation.innovations.tobytes()).hexdigest(),
    }
    ma_summary = {
        "specification": asdict(ma),
        "covariance_stationary": True,
        "invertible": ma_analysis.invertible,
        "root_moduli": np.abs(ma_analysis.roots).tolist(),
        "autocovariances_0_to_4": [ma_analysis.autocovariance(k) for k in range(5)],
        "stationary_mean": 0.0,
        "stationary_variance": ma_analysis.autocovariance(0),
    }
    precision, means = mean_precision_experiment(seed + 100, replications)
    controls["mean_shift"] = {
        "maximum_path_shift_error": float(
            np.max(np.abs(paths["ar1_mean_shift"] - paths["ar1_persistent"] - 5))
        )
    }
    controls["all_deterministic_controls_pass"] = bool(
        all(
            v["response_agreement_pass"] and v.get("stationary_moment_controls_pass", True)
            for k, v in controls.items()
            if k in specifications
        )
        and controls["mean_shift"]["maximum_path_shift_error"] < 1e-11
    )
    pd.DataFrame(rows).to_csv(out / "dynamics_summary.csv", index=False)
    pd.DataFrame(response_rows).to_csv(out / "impulse_responses.csv", index=False)
    pd.DataFrame(path_rows).to_csv(out / "simulation_paths.csv", index=False)
    pd.DataFrame({"replication": np.arange(replications), "sample_mean": means}).to_csv(
        out / "sample_mean_replications.csv", index=False
    )
    for name, value in {
        "configurations": configs,
        "ma_control": ma_summary,
        "mean_precision": precision,
        "numerical_validation": controls,
    }.items():
        write_json(out / f"{name}.json", value)
    environment = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pandas", "matplotlib")
        },
        "generator": "numpy.random.default_rng / PCG64",
        "seed": seed,
        "data_source": "Synthetic Gaussian innovations; parameters fixed by experiment design",
        "units": "Generic model units and periods; no empirical frequency assigned",
        "root_boundary_tolerance": 1e-10,
        "response_comparison_rtol": 1e-11,
        "response_comparison_atol": 1e-11,
    }
    write_json(out / "run_metadata.json", environment)
    from .reporting import build_figures, build_report

    build_figures(out, pd.DataFrame(rows), paths, responses, means, precision, ma_summary)
    build_report(out)
    source_root = Path(__file__).parent
    manifest = {
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source_root.glob("*.py"))
        },
        "artifact_sha256": {
            str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*"))
            if p.is_file()
            and p.name
            not in {
                "manifest.json",
                "report_validation.json",
                "notebook_validation.json",
                "verification.json",
            }
        },
    }
    write_json(out / "manifest.json", manifest)
    if not controls["all_deterministic_controls_pass"]:
        raise RuntimeError(
            "A deterministic scientific control failed; inspect numerical_validation.json"
        )
    return pd.DataFrame(rows), precision

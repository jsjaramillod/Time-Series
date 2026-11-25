"""Run matched six-month forecasts, the reference experiment and the signal control."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from bayes_policy.results import PosteriorResult
from payroll_risk.data import CHECKSUM, PayrollPanel

from .model import EMP, PandemicVAR, PayrollForecast, course_units, residual_scales
from .signal import signal_control

TREATMENTS = ("Observed", "Interpolated", "Temporary volatility")
BASE_SEED = 72026
REFERENCE = "2026-03"


def fit_origin(task):
    origin, treatment, observed, smoothed, feb2020, base_seed = task
    temporary = treatment == TREATMENTS[2]
    training = smoothed if treatment == "Interpolated" else observed
    months = pd.period_range("1962-01", origin, freq="M")
    model = PandemicVAR(training, months, residual_scales(observed), temporary)
    seed = int(
        np.random.SeedSequence(
            [base_seed, months[-1].ordinal, TREATMENTS.index(treatment)]
        ).generate_state(1)[0]
    )
    draws = 2048 if origin == REFERENCE else 512
    for _ in range(5):
        warmup = max(256, draws // 4)
        samples = model.sample(seed, warmup, draws)
        pred = PayrollForecast.predict(samples, observed, origin, seed + 1)
        traces = [pred.location, pred.sd, samples.coefficients[:, :, 1 + EMP, EMP]]
        names = ["six_month_location", "six_month_scale", "first_own_lag"]
        if temporary:
            traces.append(samples.gamma)
            names.append("march_multiplier")
        diagnosis = PosteriorResult(np.stack(traces, axis=-1), tuple(names), {}).summary()
        if (
            np.isfinite(diagnosis[["r_hat", "ess_bulk", "ess_tail"]]).all().all()
            and diagnosis.r_hat.max() <= 1.01
            and diagnosis.ess_bulk.min() >= 400
            and diagnosis.ess_tail.min() >= 400
        ):
            break
        draws *= 2
    else:
        raise ValueError(f"Nonconverged fit at {origin} / {treatment}")
    key = dict(origin=origin, target=str(months[-1] + 6), treatment=treatment)
    result = dict(
        **key,
        **pred.summary(),
        origin_payroll=pred.base,
        train_months=len(months),
        draws_per_chain=draws,
        warmup=warmup,
        seed=seed,
        max_rhat=diagnosis.r_hat.max(),
        min_bulk_ess=diagnosis.ess_bulk.min(),
        min_tail_ess=diagnosis.ess_tail.min(),
    )
    own_state = None
    if treatment == "Interpolated":
        own_state = dict(
            **key, **PayrollForecast.predict(samples, smoothed, origin, seed + 1).summary()
        )
    gamma = None
    if temporary and origin == REFERENCE:
        gamma = pd.DataFrame(
            {
                "percentile": np.arange(1, 1000) / 10,
                "multiplier": np.percentile(samples.gamma, np.arange(1, 1000) / 10),
            }
        )
    scenario = []
    if origin == "2022-03" and temporary:
        for elevated in [False, True]:
            p = PayrollForecast.predict(
                samples, feb2020, "2020-02", seed + 2, scale_future=elevated
            )
            scenario.append(
                dict(
                    origin="2020-02",
                    target="2020-08",
                    parameter_cutoff=origin,
                    future_covariance="Scaled" if elevated else "Constant",
                    **p.summary(),
                )
            )
    return result, own_state, gamma, scenario


def score_forecasts(frame):
    scored = frame.copy()
    scored["offset"] = (
        pd.PeriodIndex(scored.origin, freq="M").asi8 - pd.Period("2022-03", freq="M").ordinal
    ) % 6
    groups = [("All", scored)] + [(f"Offset {k}", g) for k, g in scored.groupby("offset")]
    scores = []
    for window, group in groups:
        for treatment, rows in group.groupby("treatment", sort=False):
            for key in ["revised", "first_change_sum"]:
                actual = rows[key]
                error = rows["mean"] - actual
                width = rows.p95 - rows.p05
                penalty = (rows.p05 - actual).clip(lower=0) + (actual - rows.p95).clip(lower=0)
                scores.append(
                    dict(
                        sample=window,
                        treatment=treatment,
                        outcome=key,
                        n=len(rows),
                        declines=int((actual < 0).sum()),
                        rmse=np.sqrt((error**2).mean()),
                        mae=error.abs().mean(),
                        bias=error.mean(),
                        covered=int((penalty == 0).sum()),
                        coverage=(penalty == 0).mean(),
                        width=width.mean(),
                        interval_score=(width + 20 * penalty).mean(),
                    )
                )
    return pd.DataFrame(scores)


def run(workbook, message, shared_data, project_data, output, workers=2, seed=BASE_SEED):
    panel = PayrollPanel.load(workbook)
    cal = pd.read_csv(shared_data / "release_calendar.csv", index_col="origin")
    first = pd.read_csv(shared_data / "first_releases.csv", index_col="target").first_release
    target_dates = cal.reset_index().set_index("target").target_release
    release = pd.read_csv(project_data / "september_release.csv")
    origins = list(pd.period_range("2022-03", "2024-06", freq="M").astype(str))
    ref_levels = release.set_index("month").seasonally_adjusted_payroll_thousands
    tasks, realized = [], []
    feb = course_units(panel, "2020-02")
    for origin in [*origins, REFERENCE]:
        o = pd.Period(origin, freq="M")
        observed = course_units(panel, origin)
        smoothed = course_units(panel, origin, smooth=True)
        for treatment in TREATMENTS:
            tasks.append((origin, treatment, observed, smoothed, feb, seed))
        if origin != REFERENCE:
            six = pd.period_range(o + 1, periods=6, freq="M").astype(str)
            actual = panel.raw.at[(o + 6).start_time, "EMP"] - panel.raw.at[o.start_time, "EMP"]
            realized.append(
                dict(
                    origin=origin,
                    issue_date=cal.at[origin, "issue_date"],
                    target_release=target_dates[str(o + 6)],
                    revised=actual,
                    first_change_sum=first.loc[six].sum(),
                )
            )
    estimates, sensitivities, scenarios = [], [], []
    output.mkdir(parents=True, exist_ok=True)
    for result, own, gamma, scenario in parallel_fits(tasks, workers):
        estimates.append(result)
        if own is not None:
            sensitivities.append(own)
        if gamma is not None:
            gamma.to_csv(output / "volatility_percentiles.csv", index=False)
        scenarios.extend(scenario)
    rows = pd.DataFrame(estimates)
    actuals = pd.DataFrame(realized)
    matched = rows.merge(actuals, on="origin", validate="many_to_one")
    ref = rows[rows.origin == REFERENCE].copy()
    ref["issue_date"] = cal.at[REFERENCE, "issue_date"]
    ref["target_release"] = release.release_date.iloc[0]
    ref["bls_gain"] = ref_levels["2026-09"] - ref_levels["2026-03"]
    ref["bls_log_growth"] = 100 * np.log(ref_levels["2026-09"] / ref_levels["2026-03"])
    sensitivity = pd.DataFrame(sensitivities).merge(actuals, on="origin", validate="one_to_one")
    stress = pd.DataFrame(scenarios)
    stress["observed_gain"] = (
        panel.raw.at[pd.Timestamp("2020-08-01"), "EMP"]
        - panel.raw.at[pd.Timestamp("2020-02-01"), "EMP"]
    )
    published_tables = [
        ("forecasts", matched),
        ("scores", score_forecasts(matched)),
        ("reference", ref),
        ("smoothed_state", sensitivity),
        ("smoothed_state_scores", score_forecasts(sensitivity)),
        ("retrospective_scenario", stress),
    ]
    for name, table in published_tables:
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.12g")
    (output / "signal.json").write_text(json.dumps(signal_control(message), indent=2) + "\n")
    meta = dict(
        seed=seed,
        chains=4,
        horizon=6,
        workbook_sha256=CHECKSUM,
        models=list(TREATMENTS),
        inputs={
            str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                shared_data / "release_calendar.csv",
                shared_data / "first_releases.csv",
                project_data / "september_release.csv",
                project_data / "coursework.csv",
                message,
            ]
        },
        packages={name: version(name) for name in ("numpy", "scipy", "pandas", "arviz")},
        outputs={
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in sorted(
                [f"{key}.csv" for key, _ in published_tables]
                + ["volatility_percentiles.csv", "signal.json"]
            )
        },
    )
    (output / "run.json").write_text(json.dumps(meta, indent=2) + "\n")


def parallel_fits(tasks, workers):
    if workers == 1:
        yield from map(fit_origin, tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(fit_origin, tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workbook", "message", "shared-data", "project-data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    args = parser.parse_args()
    run(
        args.workbook,
        args.message,
        args.shared_data,
        args.project_data,
        args.output,
        args.workers,
        args.seed,
    )


if __name__ == "__main__":
    main()

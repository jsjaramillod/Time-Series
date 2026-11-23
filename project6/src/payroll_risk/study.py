"""Matched expanding-window forecasts and first-release/revised payroll scores."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from bayes_policy.results import PosteriorResult
from scipy.special import ndtr

from .data import MODELS, PayrollPanel
from .model import BayesianVAR, MinnesotaPrior

BASE_SEED = 61027


def forecast(task, seed_base=BASE_SEED):
    origin, model, intensity, training, values, reference = task
    index = MODELS[model].index("EMP")
    seed = int(
        np.random.SeedSequence(
            [
                seed_base,
                pd.Period(origin, "M").ordinal,
                list(MODELS).index(model),
                int(10 * intensity),
                int(training == "interpolated"),
            ]
        ).generate_state(1)[0]
    )
    estimator = BayesianVAR(values, index, MinnesotaPrior(intensity))
    track = reference and model == "Full"
    draws = 2048 if reference else (1024 if seed_base != BASE_SEED else 512)
    for attempt in range(6):
        warmup = max(256, draws // 2)
        sample = estimator.sample(seed, warmup, draws, track_lags=track)
        names = ["payroll_location", "payroll_scale"]
        traces = np.stack([sample.location, sample.scale], axis=-1)
        if track:
            traces = np.concatenate([traces, sample.cpi_lags], axis=-1)
            names.extend([f"cpi_lag{k}" for k in range(1, 4)])
        diagnostic = PosteriorResult(traces, tuple(names), {}).summary()
        if (
            np.isfinite(diagnostic[["r_hat", "ess_bulk", "ess_tail"]]).all().all()
            and diagnostic.r_hat.max() <= 1.01
            and diagnostic.ess_bulk.min() >= 400
            and diagnostic.ess_tail.min() >= 400
        ):
            break
        if attempt == 5:
            raise RuntimeError(f"Convergence check failed at {origin} / {model}")
        draws *= 2
    key = dict(origin=origin, model=model, intensity=intensity, training=training)
    row = dict(
        **key,
        **sample.summarize(),
        train_months=len(values),
        seed=seed,
        draws_per_chain=draws,
        warmup=warmup,
        max_rhat=diagnostic.r_hat.max(),
        min_ess_bulk=diagnostic.ess_bulk.min(),
        min_ess_tail=diagnostic.ess_tail.min(),
    )
    if reference:
        for jobs in [63, 115]:
            row[f"cdf_{jobs}k"] = ndtr(
                (np.log1p(jobs / sample.employment) - sample.location) / sample.scale
            ).mean()
    lags = []
    if track:
        for k in range(1, 4):
            stats = diagnostic.loc[f"cpi_lag{k}"]
            lags.append(dict(**key, lag=k, **stats[["median", "p05", "p95", "r_hat"]].to_dict()))
    return row, lags


def evaluate(frame):
    rows = []
    episodes = {
        "All": np.ones(len(frame), dtype=bool),
        "2008-09": frame.target.between("2008-01", "2009-12"),
        "Mar-Jun 2020": frame.target.between("2020-03", "2020-06"),
    }
    episodes["Other months"] = ~(episodes["2008-09"] | episodes["Mar-Jun 2020"])
    for label, mask in episodes.items():
        for (model, intensity), group in frame[mask].groupby(["model", "intensity"], sort=False):
            for vintage in ["first_release", "revised"]:
                actual = group[vintage]
                rows.append(
                    dict(
                        episode=label,
                        model=model,
                        intensity=intensity,
                        vintage=vintage,
                        n=len(group),
                        declines=(actual < 0).sum(),
                        rmse=np.mean((group["mean"] - actual) ** 2) ** 0.5,
                        mae=(group["mean"] - actual).abs().mean(),
                        coverage=((group.p05 <= actual) & (actual <= group.p95)).mean(),
                        width=(group.p95 - group.p05).mean(),
                        brier=((group.p_loss - (actual < 0).astype(float)) ** 2).mean(),
                    )
                )
    return pd.DataFrame(rows)


def run(workbook, root, workers):
    panel = PayrollPanel.load(workbook)
    calendar = pd.read_csv(root / "data/release_calendar.csv").merge(
        pd.read_csv(root / "data/first_releases.csv"), on="target", validate="one_to_one"
    )
    changes = panel.raw.EMP.diff().rename("revised")
    changes.index = changes.index.to_period("M").astype(str)
    calendar = calendar.join(changes, on="target")
    tasks = []
    for origin in calendar.origin:
        ref = origin == "2026-03"
        for training in ["observed", "interpolated"] if ref else ["observed"]:
            for model in MODELS:
                values = panel.design(origin, model, training == "interpolated")
                for intensity in [0.5, 0.1]:
                    tasks.append((origin, model, intensity, training, values, ref))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(forecast, tasks, chunksize=6))
        repeat = [task for task in tasks if task[2] == 0.5 and not task[-1]]
        replay = list(pool.map(partial(forecast, seed_base=61111), repeat, chunksize=6))
    rows = pd.DataFrame([r for r, _ in records]).merge(
        calendar, on="origin", validate="many_to_one"
    )
    history = rows[rows.origin != "2026-03"].drop(columns=["cdf_63k", "cdf_115k"])
    ref = rows[rows.origin == "2026-03"].drop(columns="revised")
    dest = root / "results"
    dest.mkdir(exist_ok=True)
    replay_frame = pd.DataFrame([r for r, _ in replay]).merge(
        calendar, on="origin", validate="many_to_one"
    )
    tables = dict(
        replay_scores=evaluate(replay_frame),
        forecasts=history,
        reference=ref,
        scores=evaluate(history),
        cpi_coefficients=pd.DataFrame([x for _, lags in records for x in lags]),
    )
    for name, frame in tables.items():
        frame.to_csv(dest / f"{name}.csv", index=False, float_format="%.12g")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.workbook, args.root, args.workers)


if __name__ == "__main__":
    main()

"""Run the frozen GDP forecasting exercise and publish auditable forecast records."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from bayes_policy.samplers import GaussianGibbsSampler, SamplerConfig

from .models import GrowthModel, PredictiveLaw, QuarterlyPanel, empirical

FIRST = pd.Period("2003Q4", freq="Q")
LAST = pd.Period("2026Q1", freq="Q")
PATHS = 65536
SEED = 20261118
KINDS = ("Drift", "AR", "ARDL")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scores(frame):
    error = frame.actual - frame["mean"]
    tail = frame.actual - frame.p05
    return dict(
        n=len(frame),
        rmse=float(np.sqrt(np.mean(error**2))),
        coverage=float(((frame.actual >= frame.p05) & (frame.actual <= frame.p95)).mean()),
        width=float((frame.p95 - frame.p05).mean()),
        breaches=int((tail < 0).sum()),
        breach_rate=float((tail < 0).mean()),
        loss05=float(np.maximum(0.05 * tail, -0.95 * tail).mean()),
    )


def score_table(rows):
    forecast = rows.dropna(subset=["actual"])
    eligible = forecast.origin >= "2022Q1"
    groups = {
        "full": forecast[forecast.training == "observed"],
        "post_endpoint": forecast[eligible],
    }
    four = groups["full"].query("horizon==4")
    for p in range(4):
        groups[f"phase_{p}"] = four[
            four.origin.map(lambda q: (pd.Period(q, freq="Q").ordinal - FIRST.ordinal) % 4) == p
        ]
    out = []
    for sample, frame in groups.items():
        for (kind, h, tr), part in frame.groupby(["model", "horizon", "training"]):
            out.append(dict(sample=sample, model=kind, horizon=h, training=tr, **scores(part)))
    return pd.DataFrame(out)


def gibbs_check(spec, horizon, result, seed, integrated):
    """Independent conditional sampler and independent recursive paths for tail resolution."""
    beta = result.coefficients.reshape(-1, len(result.names) - 1)
    sigma = result.parameter("sigma2").ravel()
    paths = spec.paths(beta, sigma, horizon, np.random.default_rng(seed))
    means = np.array([empirical(part)["p05"] for part in np.split(paths, 32)])
    batch_se = means.std(ddof=1) / np.sqrt(len(means))
    return dict(
        gibbs_mean=paths.mean(),
        gibbs_p05=np.quantile(paths, 0.05),
        batch_se=batch_se,
        delta_p05=np.quantile(paths, 0.05) - integrated["p05"],
    )


def run(gdp, spread, calendar, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    for path, expected in [
        (gdp, "2b949b886e26c571d736763f4b730976c909e2b99eed513183f503ce8d55fc04"),
        (spread, "84b1acf52995d10513963afaf4c3bb7f622c2675ba1928c3fb97c71ed112f1b0"),
        (calendar, "b37a44047bc1a51d7ca817fdd4e821b7e57d25004281d681f8b6df7a67f5c036"),
    ]:
        if digest(path) != expected:
            raise ValueError(f"{Path(path).name} differs from the fixed study snapshot")
    panel = QuarterlyPanel.from_files(gdp, spread)
    if not panel.level.index.equals(pd.period_range("1984Q1", LAST, freq="Q")):
        raise ValueError("The recorded protocol requires the 1984Q1–2026Q1 sample")
    dates = pd.read_csv(calendar, index_col="quarter", parse_dates=["issue_date"])
    origins = pd.period_range(FIRST, LAST, freq="Q")
    if (
        list(dates.index) != list(origins.astype(str))
        or not dates.issue_date.is_monotonic_increasing
        or dates.issue_date.isna().any()
        or (dates.issue_date.to_numpy() <= origins.to_timestamp(how="end").to_numpy()).any()
    ):
        raise ValueError("Missing or inconsistent GDP releases")
    rows, checks, distributions = [], [], []
    for origin in origins:
        for adjusted in [False, True] if origin >= pd.Period("2022Q1", freq="Q") else [False]:
            y, x = panel.window(origin, adjusted)
            training = "interpolated" if adjusted else "observed"
            for k, kind in enumerate(KINDS):
                spec = GrowthModel(y, x, kind)
                law = PredictiveLaw(spec)
                seed = SEED + origin.ordinal * 10 + k
                rng = np.random.default_rng(seed)
                beta, var = (
                    law.parameters(PATHS, rng) if kind == "AR" or origin == LAST else (None, None)
                )
                if origin == LAST:
                    cfg = SamplerConfig(draws=8000, warmup=1500, chains=4, seed=seed + 70000)
                    fit = GaussianGibbsSampler(cfg).fit(spec.model)
                    gate = fit.diagnostic_gate()
                    if not gate["passed"]:
                        raise ArithmeticError(gate)
                for h in [1] if kind == "ARDL" else [1, 4]:
                    target = origin + h
                    if origin != LAST and target > LAST:
                        continue
                    mcse = 0.0
                    if kind == "AR" and h == 4:
                        paths = spec.paths(beta, var, h, rng)
                        pred = empirical(paths)
                        mcse = np.std(
                            [np.quantile(chunk, 0.05) for chunk in np.split(paths, 32)], ddof=1
                        ) / np.sqrt(32)
                    else:
                        pred = law.normal_mixture(h)
                    actual = panel.level[target] - panel.level[origin] if target <= LAST else np.nan
                    record = dict(
                        origin=str(origin),
                        issue_date=dates.loc[str(origin), "issue_date"].date(),
                        target=str(target),
                        horizon=h,
                        training=training,
                        model=kind,
                        n_fit=len(y) - 4,
                        actual=actual,
                        p05_mcse=mcse,
                        **pred,
                    )
                    rows.append(record)
                    if origin == LAST:
                        check = gibbs_check(spec, h, fit, seed + 100000 + h, pred)
                        if abs(check["delta_p05"]) > 6 * np.hypot(mcse, check["batch_se"]):
                            raise ArithmeticError((record, check))
                        checks.append(
                            dict(
                                training=training,
                                model=kind,
                                horizon=h,
                                max_rhat=gate["max_rhat"],
                                min_bulk_ess=gate["min_bulk_ess"],
                                min_tail_ess=gate["min_tail_ess"],
                                **check,
                            )
                        )
                        if not (kind == "AR" and h == 4):
                            paths = spec.paths(beta, var, h, rng)
                        for p, q in zip(np.arange(1, 100), np.percentile(paths, np.arange(1, 100))):
                            distributions.append(
                                dict(
                                    training=training, model=kind, horizon=h, percentile=p, growth=q
                                )
                            )
    records = pd.DataFrame(rows)
    frames = dict(
        forecasts=records,
        scores=score_table(records),
        reference=records[records.origin == str(LAST)],
        gibbs_checks=pd.DataFrame(checks),
        reference_quantiles=pd.DataFrame(distributions),
    )
    for name, frame in frames.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    metadata = dict(
        inputs={
            k: digest(v)
            for k, v in dict(GDPC1=gdp, T10Y2Y=spread, release_calendar=calendar).items()
        },
        seed=SEED,
        ar_paths=PATHS,
        gibbs_chains=4,
        gibbs_retained_per_chain=8000,
        gibbs_warmup=1500,
        first_origin=str(FIRST),
        last_origin=str(LAST),
        python=platform.python_version(),
        packages={
            pkg: version(pkg)
            for pkg in ("numpy", "pandas", "scipy", "bayes-policy", "lag-uncertainty", "arviz")
        },
        outputs={name: digest(out / f"{name}.csv") for name in frames},
    )
    (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gdp", "spread"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--calendar", type=Path, default=Path("data/release_calendar.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()

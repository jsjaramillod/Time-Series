"""Produce the public estimates from fingerprinted input snapshots."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bayes_policy.results import PosteriorResult

from .factor import DynamicFactor, activity
from .forecast import WidePanel
from .structural import TechnologyVAR

HASHES = {
    "ci_data.xlsx": "0cde5c41c9df45701e7cdc31e4e6af33c1e0beeea371ea56a733737ead65e46e",
    "Wide_Data.xlsx": "81b91820aee0fb19653e0a155abc3b75eebd8cb1e0e4e7797ad38f69a2ed7a38",
    "OPHNFB.csv": "1a2155ebb301ccce6bf334af6a389e995e9edb3718cb78186f74eb13cdceb643",
    "HOANBS.csv": "091b9258d764ddb263659ffa54a640abe10bcb2e3ab7abdd6c89cf139591fdf6",
    "adjusted_closes.csv": "3f989ca2409e67926003c6cc4ec56abb5f12fc986253197d2000c504b762a366",
}


def save(table, path):
    table.to_csv(path, index=False, float_format="%.12g")


def paired_bootstrap(test):
    losses = (test[["PCA 2", "Selected"]].sub(test.actual, axis=0) ** 2).to_numpy()
    rng = np.random.default_rng(8831)
    records = []
    for block in [5, 10]:
        starts = rng.integers(0, len(test) - block + 1, size=(5000, len(test) // block, 1))
        indices = (starts + np.arange(block)).reshape(5000, -1)
        rms = np.sqrt(losses[indices].mean(axis=1))
        diff = rms[:, 0] - rms[:, 1]
        records.append(
            dict(
                block=block,
                replications=len(diff),
                improvement=np.sqrt(losses.mean(axis=0))[0] - np.sqrt(losses.mean(axis=0))[1],
                p025=np.quantile(diff, 0.025),
                p975=np.quantile(diff, 0.975),
            )
        )
    return pd.DataFrame(records)


def forecast_stage(data, out):
    tables = WidePanel(pd.read_excel(data / "Wide_Data.xlsx")).evaluate()
    for n, t in zip(["validation_forecasts", "validation_scores", "forecasts", "scores"], tables):
        save(t, out / f"{n}.csv")
    save(paired_bootstrap(tables[2]), out / "paired_uncertainty.csv")


def factor_stage(data, equities, out):
    macro = pd.read_excel(data / "ci_data.xlsx")
    etf = pd.read_csv(equities, index_col="date", parse_dates=True)
    if (
        list(etf) != ["SPY", "QQQ", "IWM", "DIA"]
        or not etf.index.is_unique
        or not etf.index.is_monotonic_increasing
        or etf.index[0] != pd.Timestamp("2009-12-30")
        or etf.index[-1] != pd.Timestamp("2026-05-29")
        or len(etf) != 4128
        or not np.isfinite(etf.to_numpy()).all()
        or (etf <= 0).any().any()
    ):
        raise ValueError("Unexpected adjusted-price panel")
    growth = (100 * np.log(etf).diff()).loc["2010-01":"2026-05"]
    fits = [(n, activity(macro, n)) for n in ["Coursework", "Named series", "Observed"]]
    fits.append(("Equities", (growth - growth.mean()) / growth.std(ddof=1)))
    bridge = []
    for treatment, frame in fits[:2]:
        p, _ = DynamicFactor(frame).sample(1234, 200, 400, legacy=True)
        bridge.append(
            dict(
                treatment=treatment,
                phi_mean=p[:, 0].mean(),
                **dict(zip(frame.columns, p[:, 1:5].mean(axis=0))),
            )
        )
    save(pd.DataFrame(bridge), out / "column_audit.csv")
    parameters = []
    paths = []
    checks = []
    distributions = []
    for label, frame in fits:
        estimator = DynamicFactor(frame)
        names = ["phi", *[f"loading_{v}" for v in frame], *[f"variance_{v}" for v in frame]]
        if label == "Coursework":
            p, h = estimator.sample(1234, 200, 400, legacy=True)
            p, h = p[None], h[None]
        else:
            n = 1536
            for _ in range(4):
                runs = [
                    estimator.sample(8300 + c, n // 2, n, chain=c, scale_move=label == "Equities")
                    for c in range(4)
                ]
                p = np.stack([v[0] for v in runs])
                h = np.concatenate([v[1] for v in runs])
                del runs
                stats = PosteriorResult(p, tuple(names), {}).summary()
                ok = (
                    stats.r_hat.max() <= 1.01
                    and stats.ess_bulk.min() >= 400
                    and stats.ess_tail.min() >= 400
                )
                print(
                    label, n, round(stats.r_hat.max(), 4), round(stats.ess_bulk.min()), flush=True
                )
                if ok:
                    break
                del h
                n *= 2
            else:
                raise ValueError(f"Insufficient chain precision: {label}")
            checks.append(
                dict(
                    model=label,
                    draws_per_chain=n,
                    chains=4,
                    max_rhat=stats.r_hat.max(),
                    min_bulk=stats.ess_bulk.min(),
                    min_tail=stats.ess_tail.min(),
                )
            )
        h = h.reshape(-1, len(frame))
        flat = p.reshape(-1, len(names))
        for i, name in enumerate(names):
            v = flat[:, i]
            parameters.append(
                dict(
                    model=label,
                    parameter=name,
                    mean=v.mean(),
                    p10=np.quantile(v, 0.1),
                    median=np.median(v),
                    p90=np.quantile(v, 0.9),
                )
            )
        probabilities = np.arange(1, 100) / 100
        for i, series in enumerate(frame, 1):
            distributions.append(
                pd.DataFrame(
                    dict(
                        model=label,
                        series=series,
                        probability=probabilities,
                        loading=np.quantile(flat[:, i], probabilities),
                    )
                )
            )
        low, med, high = np.quantile(h, [0.1, 0.5, 0.9], axis=0)
        paths.append(
            pd.DataFrame(
                dict(
                    model=label,
                    date=frame.index.strftime("%Y-%m-%d"),
                    p10=low,
                    median=med,
                    p90=high,
                )
            )
        )
        print(label, parameters[-9]["mean"], flush=True)
        del h
    save(pd.DataFrame(parameters), out / "factor_parameters.csv")
    save(pd.concat(distributions), out / "factor_loadings.csv")
    save(pd.concat(paths), out / "factor_paths.csv")
    save(pd.DataFrame(checks), out / "factor_diagnostics.csv")


def structural_stage(data, out):
    levels = pd.concat(
        [
            pd.read_csv(data / f"{n}.csv", index_col="observation_date", parse_dates=True)
            for n in ["OPHNFB", "HOANBS"]
        ],
        axis=1,
    )
    levels.index = levels.index.to_period("Q")
    coefficients = []
    irs = []
    checks = []
    for end in ["1994Q3", "2019Q4"]:
        for tight in [0.5, 0.25]:
            model = TechnologyVAR(levels, end, tight)
            paths, controls, beta = model.sample(8500)
            stable = np.isfinite(paths[:, :, 0])
            stats = PosteriorResult(beta, tuple(f"b{i}" for i in range(18)), {}).summary()
            terms = [
                "intercept",
                *[f"{v}_lag{k}" for k in range(1, 5) for v in ("productivity", "hours")],
            ]
            posterior_mean = beta.mean(axis=(0, 1)).reshape(2, 9).T
            for term, (productivity, hours) in zip(terms, posterior_mean):
                coefficients.append(
                    dict(
                        end=end, tightness=tight, term=term, productivity=productivity, hours=hours
                    )
                )
            for j in range(12):
                v = paths[:, :, j][stable]
                irs.append(
                    dict(
                        end=end,
                        tightness=tight,
                        quarter=j + 1,
                        mean=v.mean(),
                        p10=np.quantile(v, 0.1),
                        median=np.median(v),
                        p90=np.quantile(v, 0.9),
                        p_negative=np.mean(v < 0),
                    )
                )
            checks.append(
                dict(
                    end=end,
                    start=model.start,
                    n=model.n,
                    tightness=tight,
                    draws=stable.size,
                    stable=int(stable.sum()),
                    max_radius=controls[:, 0].max(),
                    max_condition=controls[:, 1].max(),
                    max_cov_error=np.nanmax(controls[:, 2]),
                    max_longrun_zero=np.nanmax(controls[:, 3]),
                    max_rhat=stats.r_hat.max(),
                    min_bulk=stats.ess_bulk.min(),
                    min_tail=stats.ess_tail.min(),
                )
            )
    save(pd.DataFrame(coefficients), out / "var_coefficients.csv")
    save(pd.DataFrame(irs), out / "hours_responses.csv")
    save(pd.DataFrame(checks), out / "hours_diagnostics.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("equities", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--stage", choices=["factor", "forecast", "structural", "all"], default="all")
    p.add_argument(
        "--new-equity-vintage",
        action="store_true",
        help="Use and record a newly retrieved adjusted-price snapshot.",
    )
    args = p.parse_args()
    required = {
        "factor": ["ci_data.xlsx", "adjusted_closes.csv"],
        "structural": ["OPHNFB.csv", "HOANBS.csv"],
        "forecast": ["Wide_Data.xlsx"],
    }
    files = list(HASHES) if args.stage == "all" else required[args.stage]
    manifest_path = args.output / "inputs.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for n in files:
        path = args.equities if n == "adjusted_closes.csv" else args.data / n
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != HASHES[n] and not (n == "adjusted_closes.csv" and args.new_equity_vintage):
            raise ValueError(f"Snapshot differs from the recorded input: {n}")
        if manifest.get(n, digest) != digest:
            raise ValueError("Use a separate output directory for a different data vintage")
        manifest[n] = digest
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in ("forecast", "all"):
        forecast_stage(args.data, args.output)
    if args.stage in ("structural", "all"):
        structural_stage(args.data, args.output)
    if args.stage in ("factor", "all"):
        factor_stage(args.data, args.equities, args.output)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

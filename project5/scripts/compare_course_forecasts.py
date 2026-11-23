"""Reconstruct the doctoral trend AR(4) at each eligible four-quarter origin."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import invgamma

ROOT = Path(__file__).resolve().parents[1]


def coursework_forecast(levels):
    """Use the original diffuse-prior regression and simulation seeds in log GDP levels."""
    y = levels.to_numpy()
    t = np.arange(1, len(y) + 1)
    x = np.column_stack([np.ones(len(y) - 4), t[4:], *[y[4 - j : len(y) - j] for j in range(1, 5)]])
    target = y[4:]
    inverse = np.linalg.inv(x.T @ x)
    beta = inverse @ (x.T @ target)
    residual = target - x @ beta
    rng = np.random.default_rng(123)
    draws = 20000
    sd = np.sqrt(
        invgamma.rvs(
            (len(x) - x.shape[1]) / 2, scale=residual @ residual / 2, size=draws, random_state=rng
        )
    )
    coefficients = (
        beta
        + (rng.standard_normal((draws, x.shape[1])) @ np.linalg.cholesky(inverse).T) * sd[:, None]
    )
    innovations = np.random.default_rng(456).standard_normal((draws, 4)) * sd[:, None]
    lags = np.tile(y[-4:][::-1], (draws, 1))
    for step in range(4):
        next_value = (
            coefficients[:, 0]
            + (len(y) + step + 1) * coefficients[:, 1]
            + np.sum(coefficients[:, 2:] * lags, axis=1)
            + innovations[:, step]
        )
        lags = np.column_stack([next_value, lags[:, :3]])
    growth = lags[:, 0] - y[-1]
    return float(growth.mean()), float(np.quantile(growth, 0.05))


def eligible_origins(forecasts):
    annual = forecasts.query("model == 'AR' and horizon == 4")
    before = (annual.origin <= "2019Q4") & (annual.training == "observed")
    after = (annual.origin >= "2022Q1") & (annual.training == "interpolated")
    return annual[before | after].sort_values("origin")


def compare(gdp_file):
    inputs = json.loads((ROOT / "results/run.json").read_text())["inputs"]
    if hashlib.sha256(gdp_file.read_bytes()).hexdigest() != inputs["GDPC1"]:
        raise ValueError("GDP data differ from the frozen input")
    source = pd.read_csv(gdp_file, parse_dates=["observation_date"]).set_index("observation_date")
    source.index = source.index.to_period("Q")
    log_gdp = 100 * np.log(source.GDPC1)
    records = []
    for row in eligible_origins(pd.read_csv(ROOT / "results/forecasts.csv")).itertuples():
        levels = log_gdp.loc[: row.origin].copy()
        if row.training == "interpolated":
            levels.loc["2020Q1":"2021Q4"] = np.nan
            levels = levels.interpolate()
        forecast_mean, p05 = coursework_forecast(levels)
        records.append(
            dict(
                origin=row.origin,
                target=row.target,
                training=row.training,
                actual=row.actual,
                course_mean=forecast_mean,
                project_mean=row.mean,
                course_p05=p05,
                project_p05=row.p05,
            )
        )
    return pd.DataFrame(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gdp_file", type=Path)
    args = parser.parse_args()
    compare(args.gdp_file).to_csv(
        ROOT / "results/course_comparison.csv", index=False, float_format="%.12g"
    )

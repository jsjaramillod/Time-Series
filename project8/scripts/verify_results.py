"""Audit the released estimates using only the public forecast and posterior records."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common_signals.study import HASHES, paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def read(name):
    return pd.read_csv(OUT / f"{name}.csv")


def verify():
    archive = pd.read_csv(ROOT / "data/coursework.csv").set_index("quantity").value
    validation = read("validation_forecasts")
    final = read("forecasts")
    for records, start in [(validation, 201), (final, 301)]:
        np.testing.assert_array_equal(records.target, np.arange(start, start + 100))
        np.testing.assert_array_equal(records.target - 1, records.origin)
    scores = read("validation_scores").set_index("candidate")
    actual = (validation[scores.index].sub(validation.actual, axis=0) ** 2).mean() ** 0.5
    np.testing.assert_allclose(scores.validation_rmse, actual)
    winner = actual.idxmin()
    assert winner == "Ridge 500"
    np.testing.assert_array_equal(final[winner], final.Selected)
    for row in read("scores").itertuples():
        sample = {"All": final, "301-350": final[:50], "351-400": final[50:]}[row.window]
        loss = sample[row.method] - sample.actual
        assert row.n == len(sample)
        np.testing.assert_allclose(
            [row.rmse, row.mae, row.bias], [(loss**2).mean() ** 0.5, loss.abs().mean(), loss.mean()]
        )
        if row.window == "All" and f"{row.method} RMSE" in archive:
            assert abs(row.rmse - archive[f"{row.method} RMSE"]) < 0.0005
    np.testing.assert_allclose(read("paired_uncertainty"), paired_bootstrap(final), atol=1e-8)
    audit = read("column_audit").set_index("treatment")
    np.testing.assert_allclose(audit.loc["Coursework"], archive.iloc[:5], atol=1e-8)
    for row in read("factor_diagnostics").itertuples():
        assert row.chains == 4 and row.max_rhat <= 1.01 and min(row.min_bulk, row.min_tail) >= 400
    paths = read("factor_paths")
    assert (paths.p10 <= paths["median"]).all() and (paths["median"] <= paths.p90).all()
    for name, g in paths.groupby("model"):
        assert g.date.is_unique and g.date.is_monotonic_increasing
        expected = (
            (4126, "2010-01-04", "2026-05-29")
            if name == "Equities"
            else (710, "1967-02-01", "2026-03-01")
        )
        assert (len(g), g.date.iloc[0], g.date.iloc[-1]) == expected
    parameters = read("factor_parameters").set_index(["model", "parameter"])
    for (name, series), g in read("factor_loadings").groupby(["model", "series"]):
        assert g.loading.is_monotonic_increasing and len(g) == 99
        row = parameters.loc[name, f"loading_{series}"]
        np.testing.assert_allclose(
            g.set_index("probability").loading.loc[[0.1, 0.5, 0.9]], row[["p10", "median", "p90"]]
        )
    coefficients = read("var_coefficients")
    course_lags = pd.read_csv(ROOT / "data/coursework_var.csv").set_index("term")
    for _, sample in coefficients.groupby(["end", "tightness"]):
        assert sample.term.tolist() == course_lags.index.tolist()
        assert np.isfinite(sample[["productivity", "hours"]]).all().all()
    for row in read("hours_diagnostics").itertuples():
        assert row.start == "1948Q2" and row.n == {"1994Q3": 186, "2019Q4": 287}[row.end]
        assert row.draws == row.stable == 4800 and row.max_radius < 1
        assert max(row.max_cov_error, row.max_longrun_zero) < 1e-12
        assert row.max_rhat <= 1.01 and min(row.min_bulk, row.min_tail) > 400
    for _, g in read("hours_responses").groupby(["end", "tightness"]):
        np.testing.assert_array_equal(g.quarter, np.arange(1, 13))
        assert g.p90.iloc[0] < 0 and g.p10.iloc[-1] < 0 < g.p90.iloc[-1]
    assert json.loads((OUT / "inputs.json").read_text()) == HASHES
    print(
        "Verified 200 forecast origins, archived benchmarks, four factor paths and both structural samples."
    )


if __name__ == "__main__":
    verify()

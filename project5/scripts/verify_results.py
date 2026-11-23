"""Verify the published estimates using only the tracked records."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gdp_risk.study import score_table


def verify_course_comparison(root, forecasts):
    matched = pd.read_csv(root / "results/course_comparison.csv")
    annual = forecasts.query('model == "AR" and horizon == 4')
    eligible = (
        annual.loc[
            ((annual.origin <= "2019Q4") & (annual.training == "observed"))
            | ((annual.origin >= "2022Q1") & (annual.training == "interpolated"))
        ]
        .sort_values("origin")
        .reset_index(drop=True)
    )
    columns = ["origin", "target", "training", "actual", "project_mean", "project_p05"]
    pd.testing.assert_frame_equal(
        matched[columns],
        eligible.rename(columns={"mean": "project_mean", "p05": "project_p05"})[columns],
    )
    assert matched.origin.is_unique
    assert np.isfinite(matched[["course_mean", "course_p05"]]).all().all()
    endpoint = matched.loc[matched.actual.isna()]
    assert endpoint.target.tolist() == ["2027Q1"]
    saved = pd.read_csv(root.parent / "coursework_results.csv").query("project == 5")
    assert saved.target.tolist() == endpoint.target.tolist()
    assert abs(saved.saved_value.item() - endpoint.course_p05.item()) < 0.001
    realized = matched.dropna(subset=["actual"])
    for records, count, expected_rmse, expected_tail in [
        (realized, 78, [2.19146, 2.06927], [0.44239, 0.40508]),
        (realized.query('training == "interpolated"'), 13, [0.57756, 0.62535], [0.13099, 0.13778]),
    ]:
        assert len(records) == count
        for i, method in enumerate(["course", "project"]):
            errors = records.actual - records[f"{method}_mean"]
            gaps = records.actual - records[f"{method}_p05"]
            assert abs(np.sqrt((errors**2).mean()) - expected_rmse[i]) < 0.00002
            loss = np.maximum(0.05 * gaps, -0.95 * gaps).mean()
            assert abs(loss - expected_tail[i]) < 0.00002


def verify(root):
    results = root / "results"
    m = json.loads((results / "run.json").read_text())
    assert (
        hashlib.sha256((root / "data/release_calendar.csv").read_bytes()).hexdigest()
        == m["inputs"]["release_calendar"]
    )
    for name, digest in m["outputs"].items():
        assert hashlib.sha256((results / f"{name}.csv").read_bytes()).hexdigest() == digest, name
    rows = pd.read_csv(results / "forecasts.csv")
    dates = pd.read_csv(root / "data/release_calendar.csv", index_col="quarter")
    assert rows.issue_date.equals(rows.origin.map(dates.issue_date).rename("issue_date"))
    assert not rows.duplicated(["model", "origin", "training", "horizon"]).any()
    realized = rows.dropna(subset=["actual"])
    assert (
        pd.PeriodIndex(realized.origin, freq="Q") + realized.horizon.to_numpy()
        == pd.PeriodIndex(realized.target, freq="Q")
    ).all()
    assert (
        pd.to_datetime(realized.issue_date) < pd.to_datetime(realized.target.map(dates.issue_date))
    ).all()
    assert rows.loc[rows.actual.isna(), "origin"].eq("2026Q1").all()
    assert (rows.query('training=="interpolated"').origin >= "2022Q1").all()
    assert realized.groupby(["origin", "horizon"]).actual.nunique().eq(1).all()
    assert realized.groupby(["origin", "horizon"]).n_fit.nunique().eq(1).all()
    assert (rows.p05 < rows["median"]).all() and (rows["median"] < rows.p95).all()
    expected = score_table(rows)
    pd.testing.assert_frame_equal(expected, pd.read_csv(results / "scores.csv"), atol=1e-11)
    checks = pd.read_csv(results / "gibbs_checks.csv")
    reference = pd.read_csv(results / "reference.csv")
    pd.testing.assert_frame_equal(reference, rows[rows.origin == "2026Q1"].reset_index(drop=True))
    joined = checks.merge(reference, on=["model", "training", "horizon"], validate="one_to_one")
    assert len(joined) == 10 and len(realized) == 513
    assert (joined.max_rhat <= 1.01).all()
    assert (joined[["min_tail_ess", "min_bulk_ess"]] >= 400).all().all()
    assert (joined.delta_p05.abs() < 6 * np.hypot(joined.batch_se, joined.p05_mcse)).all()
    verify_course_comparison(root, rows)
    return "Verified coursework comparison, input calendar, result hashes, release timing, matched targets, scores and independent Gibbs checks."


if __name__ == "__main__":
    print(verify(Path(__file__).resolve().parents[1]))

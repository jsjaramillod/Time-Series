"""Validate the released records without the private workbook or message series."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from payroll_volatility.study import TREATMENTS, score_forecasts

ROOT = Path(__file__).resolve().parents[1]


def verify():
    r = ROOT / "results"
    meta = json.loads((r / "run.json").read_text())
    for name, sha in meta["outputs"].items():
        assert hashlib.sha256((r / name).read_bytes()).hexdigest() == sha, name
    shared = ROOT.parent / "project6/data"
    for file in [
        shared / "release_calendar.csv",
        shared / "first_releases.csv",
        ROOT / "data/coursework.csv",
        ROOT / "data/september_release.csv",
    ]:
        assert hashlib.sha256(file.read_bytes()).hexdigest() == meta["inputs"][file.name]
    f = pd.read_csv(r / "forecasts.csv")
    reference = pd.read_csv(r / "reference.csv")
    combined = pd.concat([f, reference])
    cal = pd.read_csv(shared / "release_calendar.csv")
    assert not combined.duplicated(["origin", "treatment"]).any()
    assert set(f.treatment) == set(TREATMENTS)
    assert f.groupby("origin").size().eq(3).all() and len(f) == 84 and len(reference) == 3
    assert set(f.origin) == set(pd.period_range("2022-03", "2024-06", freq="M").astype(str))
    assert (
        pd.PeriodIndex(combined.origin, freq="M") + 6 == pd.PeriodIndex(combined.target, freq="M")
    ).all()
    assert (combined.issue_date == combined.origin.map(cal.set_index("origin").issue_date)).all()
    assert (f.target_release == f.target.map(cal.set_index("target").target_release)).all()
    assert (combined.issue_date < combined.target_release).all()
    assert (
        combined.groupby("origin")[["issue_date", "target", "train_months", "origin_payroll"]]
        .nunique()
        .eq(1)
        .all()
        .all()
    )
    assert f.groupby("origin")[["revised", "first_change_sum"]].nunique().eq(1).all().all()
    monthly = pd.read_csv(shared / "first_releases.csv", index_col="target").first_release
    for row in f.drop_duplicates("origin").itertuples():
        six = pd.period_range(pd.Period(row.origin, "M") + 1, periods=6, freq="M").astype(str)
        assert monthly.loc[six].sum() == row.first_change_sum
    for name, records in [
        ("scores", f),
        ("smoothed_state_scores", pd.read_csv(r / "smoothed_state.csv")),
    ]:
        pd.testing.assert_frame_equal(
            score_forecasts(records), pd.read_csv(r / f"{name}.csv"), rtol=1e-9, check_dtype=False
        )
    assert (
        (combined[["p025", "p05", "median", "p95", "p975"]].diff(axis=1).iloc[:, 1:] > 0)
        .all()
        .all()
    )
    assert (
        combined.max_rhat.max() <= 1.01
        and combined.min_bulk_ess.min() >= 400
        and combined.min_tail_ess.min() >= 400
    )
    levels = pd.read_csv(ROOT / "data/september_release.csv").seasonally_adjusted_payroll_thousands
    assert (reference.bls_gain == levels.iloc[1] - levels.iloc[0]).all()
    np.testing.assert_allclose(
        reference.bls_log_growth, 100 * np.log(levels.iloc[1] / levels.iloc[0])
    )
    sig = json.loads((r / "signal.json").read_text())
    np.testing.assert_allclose(
        [sig["q25"], sig["median"], sig["q75"]], [-2.71955629, -2.14684671, -1.57413713], atol=1e-8
    )
    assert sig["max_conditioning_difference"] < 1e-9
    stress = pd.read_csv(r / "retrospective_scenario.csv").set_index("future_covariance")
    assert stress.parameter_cutoff.gt(stress.target).all()
    assert stress.loc["Scaled", "p05"] < stress.loc["Constant", "p05"]
    np.testing.assert_allclose(stress.mean_log.iloc[0], stress.mean_log.iloc[1])
    simple = pd.read_csv(r / "persistence.csv")
    assert simple.origin.tolist() == sorted(combined.origin.unique())
    np.testing.assert_allclose(simple.prediction, simple.origin_level - simple.previous_level)
    joined = f.drop_duplicates("origin").merge(simple, on="origin", validate="one_to_one")
    for col in ["revised", "first_change_sum", "origin_level"]:
        target_col = "origin_payroll" if col == "origin_level" else col + "_x"
        other_col = col if col == "origin_level" else col + "_y"
        np.testing.assert_allclose(joined[target_col], joined[other_col])
    assert simple.reference_gain.dropna().tolist() == [levels.iloc[1] - levels.iloc[0]]
    replay = pd.read_csv(r / "replay_scores.csv").query('sample=="All"')
    primary = pd.read_csv(r / "scores.csv").query('sample=="All"')
    for outcome in primary.outcome.unique():

        def rank(t):
            return t[t.outcome == outcome].sort_values("rmse").treatment.tolist()

        assert rank(primary) == rank(replay)
    g = pd.read_csv(r / "volatility_percentiles.csv")
    assert g.percentile.is_monotonic_increasing and g.multiplier.is_monotonic_increasing
    print(
        "Verified 84 matched forecasts, three reference fits, scorecards, second-seed rankings, release timing, volatility records and the signal control."
    )


if __name__ == "__main__":
    verify()

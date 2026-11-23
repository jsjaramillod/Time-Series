"""Verify checked-in results without the private workbook."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from payroll_risk.study import evaluate

ROOT = Path(__file__).resolve().parents[1]


def verify():
    data = ROOT / "data"
    res = ROOT / "results"
    f = pd.read_csv(res / "forecasts.csv")
    ref = pd.read_csv(res / "reference.csv")
    scores = pd.read_csv(res / "scores.csv")
    sources = pd.read_csv(data / "release_calendar.csv").merge(
        pd.read_csv(data / "first_releases.csv"), on="target", validate="one_to_one"
    )
    assert len(sources) == 217
    assert len(ref) == 12 and len(f) == 216 * 3 * 2
    assert not f.duplicated(["target", "model", "intensity"]).any()
    assert not ref.duplicated(["target", "model", "intensity", "training"]).any()
    expected = pd.period_range("2007-01", "2024-12", freq="M").astype(str).tolist()
    assert sorted(f.target.unique()) == expected
    for frame in [f, ref]:
        pd.testing.assert_frame_equal(
            frame[sources.columns].drop_duplicates().reset_index(drop=True),
            sources[sources.target.isin(frame.target)].reset_index(drop=True),
            check_dtype=False,
        )
        assert ((pd.PeriodIndex(frame.target, freq="M") - 1).astype(str) == frame.origin).all()
        releases = frame[["payroll_release", "cpi_release", "ip_release"]]
        assert frame.issue_date.equals(releases.max(axis=1).rename("issue_date"))
        lead = pd.to_datetime(frame.target_release) - pd.to_datetime(frame.issue_date)
        np.testing.assert_equal(lead.dt.days.to_numpy(), frame.lead_days.to_numpy())
        assert lead.dt.days.min() > 0 and frame.p_loss.between(0, 1).all()
        assert (frame.p05 < frame["median"]).all() and (frame["median"] < frame.p95).all()
        assert (frame.max_rhat <= 1.01).all() and frame.min_ess_bulk.min() >= 400
        assert frame.min_ess_tail.min() >= 400
    keys = ["episode", "model", "intensity", "vintage"]
    pd.testing.assert_frame_equal(
        evaluate(f).sort_values(keys).reset_index(drop=True),
        scores.sort_values(keys).reset_index(drop=True),
        rtol=1e-8,
    )
    assert (ref.first_release == 115).all()
    original = pd.read_csv(data / "original_april_forecast.csv").iloc[0]
    assert original.target == ref.target.iloc[0] and original.training == "interpolated"
    assert np.isclose(ref.first_release.iloc[0] - original["mean"], 53.86)
    matched = pd.read_csv(res / "matched_histories.csv")
    base = f.query("model == 'Full' and intensity == 0.5 and origin >= '2022-02'")
    actual = matched.query("training == 'observed'")
    assert len(matched) == 68 and len(base) == 34
    assert not matched.duplicated(["training", "origin"]).any()
    columns = ["origin", "target", "issue_date", "first_release"]
    pd.testing.assert_frame_equal(
        base[columns].reset_index(drop=True), actual[columns].reset_index(drop=True)
    )
    np.testing.assert_allclose(
        base[["mean", "p05", "p95", "p_loss"]], actual[["mean", "p05", "p95", "p_loss"]]
    )
    adjusted = matched.query("training == 'interpolated'")
    pd.testing.assert_frame_equal(
        adjusted[columns].reset_index(drop=True), actual[columns].reset_index(drop=True)
    )
    assert adjusted.first_release.between(adjusted.p05, adjusted.p95).sum() == 31
    assert (adjusted.first_release > 0).all()
    assert actual.first_release.between(actual.p05, actual.p95).all()
    assert round((adjusted["mean"] - adjusted.first_release).abs().mean() * 1000) == 145702
    assert round((actual["mean"] - actual.first_release).abs().mean() * 1000) == 226414
    manifest = json.loads((res / "checksums.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    print(
        f"Verified {len(f)} historical fits, {len(ref)} reference fits and {len(scores)} score rows."
    )


if __name__ == "__main__":
    verify()

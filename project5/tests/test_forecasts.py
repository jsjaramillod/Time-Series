import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from gdp_risk.models import GrowthModel, PredictiveLaw, QuarterlyPanel, read_quarters
from gdp_risk.study import scores


@pytest.fixture
def panel():
    dates = pd.period_range("1984Q1", "2026Q1", freq="Q")
    rng = np.random.default_rng(45)
    return QuarterlyPanel(
        pd.Series(900 + np.cumsum(rng.normal(0.5, 0.8, len(dates))), index=dates),
        pd.Series(rng.normal(0, 1, len(dates)), index=dates),
    )


def test_csv_missing_quarter(tmp_path):
    path = tmp_path / "gdp.csv"
    path.write_text("observation_date,GDPC1\n1984-01-01,4\n1984-07-01,5\n")
    with pytest.raises(ValueError, match="chronological"):
        read_quarters(path, "GDPC1")


def test_interpolation_information_boundary(panel):
    with pytest.raises(ValueError, match="endpoint"):
        panel.window(pd.Period("2021Q4"), True)
    raw, x = panel.window(pd.Period("2022Q1"))
    adj, _ = panel.window(pd.Period("2022Q1"), True)
    assert np.flatnonzero(raw != adj).size == 8
    np.testing.assert_allclose(
        np.diff(adj.loc["2019Q4":"2022Q1"]), np.repeat((raw.iloc[-1] - raw.loc["2019Q4"]) / 9, 9)
    )
    assert panel.level.loc["2020Q2"] == raw.loc["2020Q2"]
    before = GrowthModel(raw, x, "ARDL")
    panel.level.loc["2022Q2":] += 500
    after = GrowthModel(*panel.window(pd.Period("2022Q1")), "ARDL")
    np.testing.assert_array_equal(before.next_row(), after.next_row())
    assert before.model.data.fingerprint() == after.model.data.fingerprint()


@pytest.mark.parametrize("kind", ["Drift", "AR", "ARDL"])
def test_next_row_is_next_training_observation(panel, kind):
    first = GrowthModel(*panel.window(pd.Period("2019Q4")), kind)
    next_fit = GrowthModel(*panel.window(pd.Period("2020Q1")), kind)
    np.testing.assert_allclose(first.next_row(), next_fit.model.data.X[-1])
    assert first.model.data.dates[0] == pd.Timestamp("1985-01-01")
    assert first.model.data.y[-1] == pytest.approx(
        panel.level.loc["2019Q4"] - panel.level.loc["2019Q3"]
    )


def test_recursion_matches_level_equation(panel):
    spec = GrowthModel(*panel.window(pd.Period("2010Q4")), "AR")
    b = np.array([0.4, 0.2, -1.1, 0.25, -0.15, 0.13])
    y = list(spec.level)
    # Expansion of the change equation into coefficients of the four lagged levels.
    ar = [1 + b[2] / 100 + b[3], b[4] - b[3], b[5] - b[4], -b[5]]
    for step in range(4):
        y.append(b[0] - b[2] * y[0] / 100 + b[1] * (spec.n + step) / 100 + np.dot(ar, y[-4:][::-1]))
    answer = spec.paths(b[None, :], np.zeros(1), 4, np.random.default_rng(27))
    assert answer[0] == pytest.approx(y[-1] - spec.level[-1], abs=1e-10)


def test_fixed_drift_innovation_width(panel):
    spec = GrowthModel(*panel.window(pd.Period("2010Q4")), "Drift")
    n = 80000
    draws = spec.paths(np.full((n, 1), 0.5), np.full(n, 0.25), 4, np.random.default_rng(9))
    assert np.mean(draws) == pytest.approx(2, abs=0.015)
    assert np.quantile(draws, 0.05) == pytest.approx(2 + norm.ppf(0.05), abs=0.025)


def test_integrated_tail_against_full_predictive_draws(panel):
    spec = GrowthModel(*panel.window(pd.Period("2008Q4")), "ARDL")
    law = PredictiveLaw(spec)
    rng = np.random.default_rng(47)
    beta, var = law.parameters(50000, rng)
    samples = spec.paths(beta, var, 1, rng)
    assert np.quantile(samples, 0.05) == pytest.approx(law.normal_mixture(1)["p05"], abs=0.025)
    with pytest.raises(ValueError, match="future spread"):
        spec.paths(beta, var, 4, rng)


def test_scores_cover_and_breach_different_events():
    frame = pd.DataFrame(dict(actual=[-2, 1, 5], mean=[0, 1, 1], p05=[-1, 0, -2], p95=[2, 2, 2]))
    s = scores(frame)
    assert s["breaches"] == 1 and s["coverage"] == pytest.approx(1 / 3)
    assert s["loss05"] == pytest.approx((0.95 + 0.05 + 0.35) / 3)
    assert s["rmse"] == pytest.approx(np.sqrt(20 / 3))

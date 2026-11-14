import numpy as np
import pandas as pd
import pytest

from bayes_policy.data import InterestRateData, RecessionData, RegressionData


def test_signed_gaps_and_common_sample(policy):
    symmetric, asymmetric = policy.design(), policy.design(True)
    np.testing.assert_array_equal(symmetric.y, asymmetric.y)
    assert symmetric.dates.equals(asymmetric.dates)
    np.testing.assert_allclose(symmetric.X[:, 2], asymmetric.X[:, 2] + asymmetric.X[:, 3])
    assert (asymmetric.X[:, 2] >= 0).all() and (asymmetric.X[:, 3] <= 0).all()
    np.testing.assert_allclose(symmetric.X[:, -1], np.linspace(1, 5, 16)[:-1])
    assert symmetric.dates[0] == pd.Timestamp("2000-04-01")


@pytest.mark.parametrize("kind", ["missing_period", "duplicate", "negative_gdp", "nan"])
def test_policy_rejects_invalid_input(policy, kind):
    frame = policy._frame.copy()
    if kind == "missing_period":
        frame = frame.drop(index=3)
    elif kind == "duplicate":
        frame.loc[1, "DATE"] = frame.loc[0, "DATE"]
    elif kind == "negative_gdp":
        frame.loc[3, "GDP"] = -1
    else:
        frame.loc[3, "FF"] = np.nan
    with pytest.raises(ValueError):
        InterestRateData(frame)


def test_simple_returns_and_final_feature_survives(recession):
    features = recession.features()
    np.testing.assert_allclose(features.equity_return_3m, 100 * (np.exp(0.03) - 1))
    assert features.index[0] == pd.Timestamp("1990-04-30")
    pairs = recession.pairs()
    assert pairs.target.tail(12).isna().all()
    assert features.index[-1] == pd.Timestamp("2009-12-31")


def test_twelve_month_alignment_and_no_future_labels(recession):
    pairs = recession.pairs()
    np.testing.assert_array_equal(pairs.target.iloc[:-12], recession._frame.NBER.iloc[15:])
    train, metadata = recession.training_at("2007-01")
    assert train.dates[-1] == pd.Timestamp("2006-01-31")
    assert metadata["last_training_target"] == "2007-01-31"
    train_delayed, delayed = recession.training_at("2007-01", label_delay_months=18)
    assert delayed["last_training_target"] == "2005-07-31"
    assert train_delayed.dates[-1] == pd.Timestamp("2004-07-31")
    assert metadata["evaluation_type"] == "pseudo-real-time"


def test_explicit_release_calendar_controls_training(recession):
    target_dates = pd.DatetimeIndex(recession.pairs().target_date)
    calendar = pd.Series(target_dates + pd.DateOffset(months=15), index=target_dates)
    training, timing = recession.training_at("2007-01", label_calendar=calendar)
    assert pd.Timestamp(timing["latest_label_availability"]) <= pd.Timestamp("2007-01-31")
    assert training.dates[-1] <= pd.Timestamp("2004-10-31")
    with pytest.raises(ValueError, match="cover"):
        recession.training_at("2007-01", label_calendar=calendar.iloc[:5])


def test_bad_binary_and_design():
    dates = pd.date_range("2000-01-01", periods=20, freq="MS")
    frame = pd.DataFrame(
        {
            "Date": dates,
            "NBER": 2,
            "FederalFundsRate": 1,
            "SP500": 100,
            "TenYearBondYield": 3,
            "ThreeMonthBillYield": 2,
        }
    )
    with pytest.raises(ValueError, match="binary"):
        RecessionData(frame)
    with pytest.raises(ValueError, match="finite"):
        RegressionData(np.array([1.0, np.nan]), np.ones((2, 1)), ("x",), dates[:2])

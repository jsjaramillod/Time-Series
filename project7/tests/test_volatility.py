import numpy as np
import pandas as pd
import pytest
from payroll_risk.data import lag_design

from payroll_volatility import model
from payroll_volatility.model import (
    EMP,
    LAGS,
    CrossProducts,
    PandemicVAR,
    PayrollForecast,
    Samples,
    exponents,
    residual_scales,
)
from payroll_volatility.study import score_forecasts


def test_event_window_and_reset():
    dates = pd.period_range("2020-02", "2022-04", freq="M")
    assert exponents(dates).tolist() == [-1, *range(25), -1]


def test_qr_sufficient_statistics_match_weighting_entire_rows():
    rng = np.random.default_rng(3)
    months = pd.period_range("2010-01", "2022-03", freq="M")
    values = rng.normal(size=(len(months), 5)).cumsum(axis=0)
    x, y, _ = lag_design(values, LAGS)
    fit = PandemicVAR(values, months, residual_scales(values), True)
    theta = rng.normal(size=fit.prior_mean.shape)
    coef = fit.prior_mean + fit.inv_r @ theta
    offsets = exponents(months[LAGS:])
    w = np.ones(len(y))
    event = offsets >= 0
    w[event] = 1 / (17 * 0.9 ** offsets[event])
    residual = y - x @ coef
    np.testing.assert_allclose(
        fit.normal.residual_ss(theta) + fit.event.residual_ss(theta) / 17,
        residual.T @ (w[:, None] * residual),
        atol=1e-9,
    )
    q, _ = np.linalg.qr(x, mode="reduced")
    np.testing.assert_allclose(
        fit.normal.xx + fit.event.xx / 17, q.T @ (w[:, None] * q), atol=1e-12
    )


def test_covariance_draw_is_independent_of_factor_orientation(monkeypatch):
    rng = np.random.default_rng(10)
    months = pd.period_range("2012-01", "2022-04", freq="M")
    v = rng.normal(size=(len(months), 5))
    fit = PandemicVAR(v, months, residual_scales(v), True)
    baseline = fit.sample(7, 2, 3, 2)
    original = model.covariance_draw
    rotation = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    monkeypatch.setattr(model, "covariance_draw", lambda *args: original(*args) @ rotation)
    rotated = fit.sample(7, 2, 3, 2)
    np.testing.assert_allclose(baseline.coefficients, rotated.coefficients, atol=1e-10)
    np.testing.assert_allclose(baseline.gamma, rotated.gamma, atol=1e-10)


def test_future_variance_and_level_transformation():
    n = 4000
    beta = np.zeros((2, n, 61, 5))
    beta[:, :, 1:6, :] = np.eye(5)
    beta[:, :, 0, EMP] = 0.1
    factor = np.broadcast_to(np.eye(5), (2, n, 5, 5))
    s = Samples(beta, factor, np.full((2, n), 100.0))
    data = np.zeros((12, 5))
    data[:, EMP] = 100 * np.log(150000.0)
    regular = PayrollForecast.predict(s, data, "2020-02", 11)
    np.testing.assert_allclose(regular.location, 0.6)
    np.testing.assert_allclose(regular.sd, np.sqrt(6))
    shifted = PayrollForecast.predict(s, data, "2020-02", 11, scale_future=True)
    np.testing.assert_allclose(shifted.sd, np.sqrt(100 * np.sum(0.9 ** np.arange(6))))
    score = regular.summary()
    assert score["mean"] == pytest.approx(150000 * np.expm1(0.006 + 0.5 * 6 / 10000))
    assert score["median"] == pytest.approx(150000 * np.expm1(0.006))
    assert abs(regular.simulated_growth.mean() - 0.6) < 0.1


def test_interval_score_penalizes_misses_and_uses_six_offsets():
    records = pd.DataFrame(
        dict(
            origin=pd.period_range("2022-03", periods=6, freq="M").astype(str),
            treatment="Observed",
            mean=5.0,
            p05=0.0,
            p95=10.0,
            revised=15.0,
            first_change_sum=5.0,
        )
    )
    scores = score_forecasts(records)
    assert len(scores) == 14
    a = scores.query('sample=="All" and outcome=="revised"').iloc[0]
    assert (a.rmse, a.coverage, a.interval_score) == (10, 0, 110)


def test_forecast_requires_complete_event():
    months = pd.period_range("2010-01", "2021-12", freq="M")
    values = np.random.default_rng(6).normal(size=(len(months), 5))
    with pytest.raises(ValueError, match="complete"):
        PandemicVAR(values, months, residual_scales(values), True)


def test_cross_products_with_no_event_are_zero():
    cp = CrossProducts.from_rows(np.zeros((0, 3)), np.zeros((0, 2)))
    np.testing.assert_array_equal(cp.residual_ss(np.ones((3, 2))), np.zeros((2, 2)))


def test_conditional_forecast_matches_dense_companion_recursion():
    rng = np.random.default_rng(58)
    d = 5
    beta = rng.normal(scale=0.04, size=(1 + LAGS * d, d))
    factor = rng.normal(size=(d, d))
    history = rng.normal(size=(LAGS, d))
    draws = Samples(
        np.broadcast_to(beta, (2, 3, *beta.shape)),
        np.broadcast_to(factor, (2, 3, d, d)),
        np.full((2, 3), 12.0),
    )
    forecast = PayrollForecast.predict(draws, history, "2019-12", 10, scale_future=True)
    companion = np.zeros((LAGS * d, LAGS * d))
    companion[:d] = beta[1:].T
    companion[d:, :-d] = np.eye((LAGS - 1) * d)
    state = history[::-1].ravel()
    cov = np.zeros_like(companion)
    for h in range(6):
        state = companion @ state
        state[:d] += beta[0]
        cov = companion @ cov @ companion.T
        scale = 1 if h < 2 else 12 * 0.9 ** (h - 2)
        cov[:d, :d] += scale * (factor @ factor.T)
    np.testing.assert_allclose(forecast.location, state[EMP] - history[-1, EMP], atol=1e-12)
    np.testing.assert_allclose(forecast.sd, np.sqrt(cov[EMP, EMP]), atol=1e-12)


def test_single_message_matches_normal_update():
    from payroll_volatility.signal import filter_messages

    observed = 2.0
    conditional = filter_messages([observed])
    signal_var = (1 - 0.4) / ((1 + 0.4) * ((1 - 0.4) ** 2 - 0.5**2))
    noise_var = 0.5 / (1 - 0.7**2)
    expected = signal_var / (signal_var + noise_var)
    assert conditional["median"] == pytest.approx(expected * observed)
    assert conditional["variance"] == pytest.approx(expected * noise_var)
    assert conditional["q25"] < conditional["median"] < conditional["q75"]


@pytest.mark.parametrize("values", [[], [np.nan], [[1.0, 2.0]]])
def test_message_input_contract(values):
    from payroll_volatility.signal import filter_messages

    with pytest.raises(ValueError):
        filter_messages(values)

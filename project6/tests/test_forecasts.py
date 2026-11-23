import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr
from scipy.stats import invwishart

from payroll_risk.data import COLUMNS, PayrollPanel, lag_design
from payroll_risk.model import BayesianVAR, ForecastDraws, MinnesotaPrior, covariance_draw


def test_lag_order_and_forecast_date():
    values = np.arange(16).reshape(-1, 2)
    x, y, next_x = lag_design(values, 2)
    np.testing.assert_equal(x[0], [1, 2, 3, 0, 1])
    np.testing.assert_equal(y[0], [4, 5])
    np.testing.assert_equal(next_x, [1, 14, 15, 12, 13])


def test_prior_scale_and_direction():
    mean, var = MinnesotaPrior().moments(np.array([1, 4.0]), 2)
    np.testing.assert_equal(mean[1:3], np.eye(2))
    np.testing.assert_allclose(var[1:3], [[0.5, 0.02], [0.00125, 0.5]])
    np.testing.assert_allclose(var[3:], var[1:3] / 4)


def test_wishart_orientation():
    scale = np.array([[0.006, -0.001], [-0.001, 0.002]])
    factors = covariance_draw(np.tile(scale, (45000, 1, 1)), 12, np.random.default_rng(34))
    cov = factors @ factors.transpose(0, 2, 1)
    np.testing.assert_allclose(cov.mean(axis=0), scale / 9, rtol=0.016)
    np.testing.assert_allclose(
        np.linalg.inv(cov).mean(axis=0), 12 * np.linalg.inv(scale), rtol=0.016
    )


def test_joint_sampler_against_dense_independent_implementation():
    # Native coefficient coordinates, a single dense Gaussian draw, SciPy's IW.
    rng = np.random.default_rng(92)
    series = np.cumsum(rng.normal(0, 0.015, (120, 2)) @ [[1, 0.7], [0, 0.8]], axis=0)
    prior = MinnesotaPrior()
    x, y, new_x = lag_design(series, 1)
    fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
    scales = np.sqrt(np.mean((y - fitted) ** 2, axis=0))
    m, var = prior.moments(scales, 1)
    v_inv = np.diag(1 / var.T.ravel())
    b = m.copy()
    locations, deviations = [], []
    for step in range(3100):
        e = y - x @ b
        sigma = invwishart.rvs(df=len(y) + 4, scale=np.diag(scales**2) + e.T @ e, random_state=rng)
        w = np.linalg.inv(sigma)
        prec = v_inv + np.kron(w, x.T @ x)
        rhs = v_inv @ m.T.ravel() + (x.T @ y @ w).T.ravel()
        coef = np.linalg.solve(prec, rhs) + np.linalg.solve(
            np.linalg.cholesky(prec).T, rng.standard_normal(6)
        )
        b = coef.reshape(2, 3).T
        if step >= 300:
            locations.append((new_x @ b)[0] - series[-1, 0])
            deviations.append(np.sqrt(sigma[0, 0]))
    draws = BayesianVAR(series, 0, prior, lags=1).sample(8, 300, 1200)
    assert abs(draws.location.mean() - np.mean(locations)) < 0.00025
    assert abs(draws.scale.mean() - np.mean(deviations)) < 0.00012
    assert abs(draws.summarize()["p_loss"] - ndtr(-np.array(locations) / deviations).mean()) < 0.008


def test_employment_transformation_and_joint_innovations():
    loc, scale = np.full((4, 50), 0.001), np.full((4, 50), 0.03)
    sample = ForecastDraws(loc, scale, np.zeros_like(loc), None, 150000, 0)
    stats = sample.summarize()
    assert stats["mean"] == pytest.approx(150000 * np.expm1(0.001 + 0.03**2 / 2))
    assert stats["median"] == pytest.approx(150000 * np.expm1(0.001))
    assert stats["p_loss"] == pytest.approx(ndtr(-0.001 / 0.03))


def test_transformations_respect_available_endpoints():
    months = pd.date_range("2019-01", "2026-03", freq="MS")
    raw = pd.DataFrame(
        {c: np.arange(len(months), dtype=float) ** 2 + 20 for c in COLUMNS}, index=months
    )
    raw.loc["2025-10-01", "CPI"] = np.nan
    panel = PayrollPanel(raw)
    with pytest.raises(ValueError, match="November CPI"):
        panel.design("2025-10", "Full")
    with pytest.raises(ValueError, match="Endpoint"):
        panel.design("2021-02", "Full", True)
    values = panel.design("2026-03", "Full", True)
    i = months.get_loc("2025-10-01")
    assert values[i, -1] == pytest.approx(np.log(raw.CPI.iloc[[i - 1, i + 1]]).mean())
    i = months.get_loc("2022-01-01")
    a, b = months.get_loc("2019-11-01"), months.get_loc("2022-02-01")
    weight = (months[i] - months[a]) / (months[b] - months[a])
    assert np.exp(values[i, 2]) == pytest.approx(
        (1 - weight) * raw.EMP.iloc[a] + weight * raw.EMP.iloc[b]
    )
    assert np.exp(values[b, 2]) == pytest.approx(raw.EMP.iloc[b])

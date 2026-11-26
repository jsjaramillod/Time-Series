import numpy as np
import pandas as pd
import pytest

from common_signals.factor import NAMES, SPECIAL, DynamicFactor, activity
from common_signals.forecast import WidePanel
from common_signals.structural import identify


def test_forecast_origin_and_ridge_scale():
    rng = np.random.default_rng(15)
    x = rng.normal(size=(400, 100))
    y = rng.normal(size=400)
    frame = pd.DataFrame(x, columns=[f"X{i}" for i in range(1, 101)])
    frame.insert(0, "Y", y)
    frame.insert(0, "DATE", np.arange(1, 401))
    origin = 201
    first = WidePanel(frame).predict(origin)
    # Independent normal-equation calculation, using the printed target row numbers.
    design = np.stack(
        [np.r_[y[j - 1 : j - 5 : -1] if j > 4 else y[:4][::-1], x[j - 1]] for j in range(4, origin)]
    )
    last = np.r_[y[origin - 4 : origin][::-1], x[origin - 1]]
    target = y[4:origin]
    for name, scale in [("Ridge 15 raw", np.ones(104)), ("Ridge 15", design.std(axis=0, ddof=1))]:
        z = (design - design.mean(axis=0)) / scale
        beta = np.linalg.solve(z.T @ z + 15 * np.eye(104), z.T @ (target - target.mean()))
        expected = target.mean() + (last - design.mean(axis=0)) / scale @ beta
        assert first[name] == pytest.approx(expected)
    frame.iloc[origin:, 1:] += 1000
    assert WidePanel(frame).predict(origin) == pytest.approx(first)


def test_joint_path_matches_gaussian_conditioning():
    phi = 0.64
    loadings = np.array([0.2, 0.4, 0.7, 0.8])
    variance = np.array([0.5, 0.8, 0.45, 1.3])
    rng = np.random.default_rng(161)
    values = rng.normal(size=(5, 4))
    index = np.arange(len(values))
    prior = phi ** np.abs(index[:, None] - index) / (1 - phi**2)
    covariance = np.linalg.inv(
        np.linalg.inv(prior) + np.eye(len(values)) * np.sum(loadings**2 / variance)
    )
    mean = covariance @ (values @ (loadings / variance))
    draws = np.stack(
        [DynamicFactor.path(values, phi, loadings, variance, rng) for _ in range(4200)]
    )
    np.testing.assert_allclose(draws.mean(axis=0), mean, atol=0.042)
    np.testing.assert_allclose(np.cov(draws, rowvar=False), covariance, atol=0.029)


def test_header_assignment_preserves_other_series():
    rng = np.random.default_rng(22)
    raw = pd.DataFrame(np.exp(rng.normal(0, 0.01, (711, 4)).cumsum(axis=0)), columns=NAMES)
    raw.insert(0, "OBSERVATION_MONTH", pd.date_range("1967-01-01", periods=711, freq="MS"))
    old, new = [activity(raw, n) for n in ["Coursework", "Named series"]]
    np.testing.assert_allclose(old[["CMRMTSPL", "PAYEMS"]], new[["CMRMTSPL", "PAYEMS"]])
    assert new.W875RX1.iloc[SPECIAL].nunique() == 1 and old.W875RX1.iloc[SPECIAL].nunique() == 8
    assert new.INDPRO.iloc[SPECIAL].nunique() == 8 and old.INDPRO.iloc[SPECIAL].nunique() == 1
    assert (new.iloc[637:662].nunique() == 1).all()


def test_long_run_identification_and_impact_index():
    beta = np.zeros((9, 2))
    beta[1:3] = 0.5 * np.eye(2)
    sigma = np.array([[1, -0.2], [-0.2, 0.8]])
    (responses, impact, longrun), radius, _ = identify(beta, sigma)
    np.testing.assert_allclose(impact @ impact.T, sigma, atol=1e-12)
    assert radius == pytest.approx(0.5)
    assert longrun[0, 0] > 0 and abs(longrun[0, 1]) < 1e-12
    np.testing.assert_allclose(responses, impact[1, 0] * np.cumsum(0.5 ** np.arange(12)))
    beta[1, 0] = 1.1
    assert identify(beta, sigma)[0] is None

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.special import gammaln, ndtr

from bayes_policy.comparison import (
    ChibModelComparison,
    model_probabilities,
    quadrature_log_evidence,
)
from bayes_policy.data import RegressionData
from bayes_policy.models import BayesianProbit, GaussianRegression
from bayes_policy.priors import GammaPrecisionPrior, NormalPrior
from bayes_policy.results import PosteriorResult
from bayes_policy.samplers import (
    GaussianGibbsSampler,
    ProbitGibbsSampler,
    SamplerConfig,
    _normal_draw,
)


def test_gamma_reference_and_conditionals(gaussian_model):
    prior = GammaPrecisionPrior.from_reference(1, 3)
    assert prior.shape == prior.rate == 1.5
    assert prior.shape / prior.rate == 1
    shape, rate = gaussian_model.precision_conditional(np.array([0.5, -0.8]))
    residual = gaussian_model.data.y - gaussian_model.data.X @ [0.5, -0.8]
    assert shape == 51.5
    assert rate == pytest.approx(1.5 + residual @ residual / 2)
    mean, chol = gaussian_model.coefficient_conditional(2)
    P = np.eye(2) / 10 + 2 * gaussian_model.data.X.T @ gaussian_model.data.X
    np.testing.assert_allclose(chol @ chol.T, P)
    np.testing.assert_allclose(
        mean, np.linalg.solve(P, 2 * gaussian_model.data.X.T @ gaussian_model.data.y)
    )


def test_normal_precision_factor_orientation():
    P = np.array([[4.0, 1.5], [1.5, 2.0]])
    rng = np.random.default_rng(32)
    samples = np.array(
        [_normal_draw(np.zeros(2), np.linalg.cholesky(P), rng) for _ in range(20000)]
    )
    np.testing.assert_allclose(np.cov(samples.T), np.linalg.inv(P), atol=0.015)


def test_coin_control_and_extreme_log_weights():
    fair = Fraction(1, 2) ** 6
    weighted = (
        Fraction(1, 3) ** 4 * Fraction(2, 3) ** 2 + Fraction(2, 3) ** 4 * Fraction(1, 3) ** 2
    ) / 2
    assert fair / (fair + weighted) == Fraction(729, 1369)
    weights = model_probabilities(np.log([float(fair), float(weighted)]))
    np.testing.assert_allclose(weights, [729 / 1369, 640 / 1369])
    np.testing.assert_allclose(
        model_probabilities([-100000, -100001], [2, 1]),
        [2 / (2 + np.exp(-1)), np.exp(-1) / (2 + np.exp(-1))],
    )
    with pytest.raises(ValueError):
        model_probabilities([1, 2], [0, 1])


def test_quadrature_against_exact_integrated_gamma_noise():
    y = np.array([0.5, -0.8, 1.2])
    data = RegressionData(y, np.zeros((3, 1)), ("unused",), pd.date_range("2000-01-01", periods=3))
    prior = GammaPrecisionPrior(2, 3)
    model = GaussianRegression(data, NormalPrior.isotropic(1), prior)
    exact = (
        gammaln(3.5)
        - gammaln(2)
        + 2 * np.log(3)
        - 1.5 * np.log(2 * np.pi)
        - 3.5 * np.log(3 + y @ y / 2)
    )
    assert quadrature_log_evidence(model)["log_evidence"] == pytest.approx(exact, abs=1e-9)


def test_gaussian_recovery_evidence_and_serialization(gaussian_model, tmp_path):
    config = SamplerConfig(draws=2500, warmup=500, chains=4, seed=72)
    result = GaussianGibbsSampler(config).fit(gaussian_model)
    summary = result.summary()
    assert result.draws.shape == (4, 2500, 3)
    assert (result.parameter("sigma2") > 0).all()
    assert abs(summary.loc["intercept", "mean"] - 0.5) < 0.2
    assert abs(summary.loc["slope", "mean"] + 0.8) < 0.2
    assert result.diagnostic_gate()["passed"]
    chib = ChibModelComparison.estimate(gaussian_model, result)
    control = quadrature_log_evidence(gaussian_model)
    assert abs(chib.log_evidence - control["log_evidence"]) < max(0.015, 5 * chib.mcse_log_evidence)
    changed_prior = GaussianRegression(
        gaussian_model.data, NormalPrior.isotropic(2, 5), gaussian_model.precision_prior
    )
    with pytest.raises(ValueError, match="different priors"):
        ChibModelComparison.estimate(changed_prior, result)
    path = tmp_path / "posterior.npz"
    result.save(path)
    loaded = PosteriorResult.load(path)
    np.testing.assert_array_equal(result.draws, loaded.draws)
    np.testing.assert_array_equal(
        result.draws, GaussianGibbsSampler(config).fit(gaussian_model).draws
    )


def test_comparison_rejects_changed_sample(gaussian_model):
    data = gaussian_model.data
    other = GaussianRegression(
        RegressionData(data.y + 1, data.X, data.names, data.dates),
        gaussian_model.prior,
        gaussian_model.precision_prior,
    )
    with pytest.raises(ValueError, match="identical"):
        ChibModelComparison.compare([gaussian_model, other], [None, None])


def test_latent_signs_even_in_extreme_tails():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    X = np.array([[1, 40], [1, -40], [1, -40], [1, 40]])
    latent = ProbitGibbsSampler.draw_latent(y, X, np.array([0.0, 1.0]), np.random.default_rng(4))
    assert np.isfinite(latent).all()
    np.testing.assert_array_equal(latent > 0, y.astype(bool))


def test_probit_intercept_against_independent_quadrature():
    # Known one-dimensional posterior; a stronger control than checking draw shapes.
    y = np.array([1.0] * 18 + [0.0] * 42)
    data = RegressionData(
        y, np.ones((60, 1)), ("intercept",), pd.date_range("2000-01-01", periods=60)
    )
    model = BayesianProbit(data, NormalPrior.isotropic(1, 4))

    def kernel(b):
        return np.exp(model.log_likelihood(np.array([b])) - b * b / 8)

    denominator = quad(kernel, -8, 8, epsabs=1e-22)[0]
    exact_mean = quad(lambda b: b * kernel(b), -8, 8, epsabs=1e-22)[0] / denominator
    result = ProbitGibbsSampler(SamplerConfig(2500, 600, 4, 89)).fit(model)
    summary = result.summary()
    assert abs(summary.loc["intercept", "mean"] - exact_mean) < max(
        0.02, 5 * summary.loc["intercept", "mcse_mean"]
    )
    assert result.diagnostic_gate()["passed"]
    probabilities = result.probability_draws(np.array([[1.0]]))
    assert probabilities.shape == (4, 2500, 1)
    assert abs(probabilities.mean() - 0.3) < 0.05
    assert np.isfinite(model.log_likelihood(np.array([40.0])))
    assert ndtr(-40) == 0  # log likelihood above must remain finite despite CDF underflow.


@pytest.mark.parametrize(
    "factory",
    [
        lambda: NormalPrior.isotropic(2, -1),
        lambda: GammaPrecisionPrior(0, 1),
        lambda: SamplerConfig(chains=1),
    ],
)
def test_invalid_configuration(factory):
    with pytest.raises(ValueError):
        factory()

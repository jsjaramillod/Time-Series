from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from bayes_policy.comparison import ChibModelComparison, quadrature_log_evidence
from bayes_policy.models import GaussianRegression
from bayes_policy.priors import GammaPrecisionPrior, NormalPrior
from bayes_policy.samplers import GaussianGibbsSampler, SamplerConfig
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import norm

from lag_uncertainty import (
    ARDesignBuilder,
    IndexedARData,
    ModelAveragedInference,
    PriorSpecification,
)
from lag_uncertainty.dynamics import PosteriorDynamics
from lag_uncertainty.evidence import ChibOrdinateAudit
from lag_uncertainty.reference import IntegratedGaussianReference, integrated_mixture_summary
from lag_uncertainty.study import joint_batch_statistics


def test_headerless_first_observation_and_common_alignment(tmp_path):
    values = np.arange(200, dtype=float) + 2.1
    path = tmp_path / "series.csv"
    np.savetxt(path, values, fmt="%.2f")
    builder = ARDesignBuilder.from_csv(path)
    assert len(builder.values) == 200 and builder.values[0] == 2.1
    common = builder.design(8)
    for order in range(1, 9):
        design = builder.design(order)
        np.testing.assert_array_equal(design.y, values[8:])
        np.testing.assert_array_equal(design.observation_indices, np.arange(9, 201))
        np.testing.assert_array_equal(design.X[:, 0], 1)
        for lag in range(1, order + 1):
            np.testing.assert_array_equal(design.X[:, lag], values[8 - lag : 200 - lag])
        assert design.dates.equals(common.dates)
    assert builder.manifest()["modeled_observations"] == 192
    assert builder.manifest()["header"] is None
    assert not isinstance(common.dates, pd.DatetimeIndex)


@pytest.mark.parametrize(
    "text", ["value\n1\n2\n", "1\n\n2\n", "1,2\n3,4\n", "1\nnan\n", "1\ninf\n"]
)
def test_csv_contract_rejects_headers_blanks_multiple_columns_and_missing_values(tmp_path, text):
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(ValueError):
        ARDesignBuilder.from_csv(path, expected_observations=2, maximum_lag=1)


def test_common_window_comparison_rejects_different_target_data():
    first = PriorSpecification().model(ARDesignBuilder(np.arange(40.0)).design(1))
    second = PriorSpecification().model(ARDesignBuilder(np.arange(40.0) + 1).design(2))
    with pytest.raises(ValueError, match="identical"):
        ChibModelComparison.compare([first, second], [None, None])
    assert first.data.fingerprint() != second.data.fingerprint()


@pytest.mark.parametrize("order", [0, 9, True, 1.5])
def test_invalid_orders_rejected(order):
    with pytest.raises(ValueError):
        ARDesignBuilder(np.arange(30.0)).design(order)


def test_data_copy_is_immutable_and_fingerprint_binds_design():
    values = np.arange(40.0)
    builder = ARDesignBuilder(values)
    values[10] = 999
    assert builder.values[10] == 10
    assert builder.design(1).fingerprint() != builder.design(2).fingerprint()
    with pytest.raises(ValueError):
        builder.design(1).X[0, 0] = 7
    with pytest.raises(ValueError):
        IndexedARData(np.arange(3.0), np.ones((3, 1)), ("c",), np.array([1, 2, 4]))


def test_prior_variance_and_gamma_rate_are_explicit():
    model = PriorSpecification().model(ARDesignBuilder(np.arange(40.0)).design(3))
    np.testing.assert_array_equal(model.prior.covariance, np.diag([100, 0.5, 0.5, 0.5]))
    assert model.precision_prior.shape == 1 and model.precision_prior.rate == 3
    assert model.precision_prior.shape / model.precision_prior.rate == pytest.approx(1 / 3)
    alternate = PriorSpecification(precision_shape=1.5, precision_rate=1.5)
    assert alternate.precision_shape / alternate.precision_rate == 1
    with pytest.raises(ValueError):
        PriorSpecification(coefficient_variance=0)


def zero_design_model(mean=(0.0,), variance=(4.0,)):
    y = np.array([0.5, -0.8, 1.2])
    data = IndexedARData(
        y, np.zeros((3, len(mean))), tuple(f"b{k}" for k in range(len(mean))), np.arange(1, 4)
    )
    return GaussianRegression(
        data, NormalPrior(np.array(mean), np.diag(variance)), GammaPrecisionPrior(2, 3)
    )


def test_svd_evidence_matches_closed_gamma_integral_and_unused_coefficient_prior():
    model = zero_design_model()
    reference = IntegratedGaussianReference(model)
    y = model.data.y
    exact = (
        gammaln(3.5)
        - gammaln(2)
        + 2 * np.log(3)
        - 1.5 * np.log(2 * np.pi)
        - 3.5 * np.log(3 + y @ y / 2)
    )
    assert reference.log_evidence == pytest.approx(exact, abs=1e-9)
    assert reference.log_evidence == pytest.approx(
        quadrature_log_evidence(model)["log_evidence"], abs=1e-9
    )
    summary = reference.contrast_summary(np.array([1.0]))
    assert summary["mean"] == pytest.approx(0, abs=1e-12)
    assert summary["sd"] == pytest.approx(2)
    assert summary["p05"] == pytest.approx(norm.ppf(0.05) * 2)
    assert summary["probability_sum_above_one"] == pytest.approx(norm.sf(0.5))
    assert reference.node_mass_error < 1e-8


def test_scalar_survival_function_preserves_extremely_rare_probability():
    reference = IntegratedGaussianReference(zero_design_model(variance=(0.01,)))
    tail = reference.contrast_survival(1, np.array([1.0]))
    assert tail > 0
    assert tail == pytest.approx(norm.sf(10), rel=1e-8, abs=0)
    assert 1 - reference.contrast_cdf(1, np.array([1.0])) == 0


def test_coefficient_and_precision_integration_order_agree():
    y = np.array([-0.2, 0.4, 0.9])
    data = IndexedARData(y, np.ones((3, 1)), ("intercept",), np.arange(1, 4))
    model = GaussianRegression(data, NormalPrior.isotropic(1, 2), GammaPrecisionPrior(2, 3))
    reference = IntegratedGaussianReference(model)
    constant = np.exp(gammaln(3.5) - gammaln(2) + 2 * np.log(3) - 1.5 * np.log(2 * np.pi))

    def density(beta):
        return (
            constant
            * (3 + np.sum((y - beta) ** 2) / 2) ** (-3.5)
            * norm.pdf(beta, scale=np.sqrt(2))
        )

    evidence = quad(density, -15, 15, epsabs=1e-13)[0]
    mean = quad(lambda b: b * density(b), -15, 15, epsabs=1e-13)[0] / evidence
    second = quad(lambda b: b * b * density(b), -15, 15, epsabs=1e-13)[0] / evidence
    tail = quad(density, 1, 15, epsabs=1e-13)[0] / evidence
    summary = reference.contrast_summary(np.ones(1))
    assert reference.log_evidence == pytest.approx(np.log(evidence), abs=1e-9)
    assert summary["mean"] == pytest.approx(mean, abs=1e-9)
    assert summary["sd"] ** 2 == pytest.approx(second - mean**2, abs=1e-9)
    assert summary["probability_sum_above_one"] == pytest.approx(tail, abs=1e-9)


@pytest.fixture(scope="module")
def fitted_ar():
    rng = np.random.default_rng(83)
    y = np.zeros(220)
    for t in range(1, len(y)):
        y[t] = 0.2 + 0.6 * y[t - 1] + rng.normal()
    model = PriorSpecification().model(ARDesignBuilder(y).design(1))
    result = GaussianGibbsSampler(SamplerConfig(2000, 500, 4, 531)).fit(model)
    return model, result


def test_shared_sampler_uses_ordinal_labels_and_matches_integrated_posterior(fitted_ar):
    model, result = fitted_ar
    assert result.metadata["start"] == "9" and result.metadata["end"] == "220"
    assert result.metadata["n"] == 212
    sums = result.coefficients[..., 1]
    exact = IntegratedGaussianReference(model).contrast_summary(np.array([0, 1]))
    assert abs(sums.mean() - exact["mean"]) < 0.005
    assert abs(sums.std() - exact["sd"]) < 0.005
    assert result.diagnostic_gate()["passed"]


def test_chib_evaluation_point_audit_and_joint_batches(fitted_ar):
    model, result = fitted_ar
    audit = ChibOrdinateAudit.run(model, result)
    exact = IntegratedGaussianReference(model).log_evidence
    for point in audit.point_checks:
        assert abs(point["log_evidence"] - exact) < max(0.01, 6 * point["mcse_log_evidence"])
    assert audit.batch_log_evidence(500).shape == (16,)
    with pytest.raises(ValueError):
        audit.batch_log_evidence(333)


def test_weighted_distribution_moments_cdf_and_inverted_empirical_quantiles():
    mixture = ModelAveragedInference([[0, 2], [4, 6]], [0.25, 0.75], labels=[1, 2])
    summary = mixture.summary()
    assert summary["mean"] == 4
    assert summary["within_model_variance"] == 1
    assert summary["between_model_variance"] == 3
    assert summary["sd"] == 2
    assert summary["probability_sum_above_one"] == 0.875
    assert mixture.cdf(1) == 0.125
    np.testing.assert_array_equal(mixture.quantiles(), [0, 4, 6])
    draws, labels = mixture.draw(50000, 91)
    assert abs(draws.mean() - 4) < 0.04
    assert abs(np.mean(labels == 2) - 0.75) < 0.01
    np.testing.assert_array_equal(draws, mixture.draw(50000, 91)[0])


def test_single_model_limit_and_weight_renormalization():
    mixture = ModelAveragedInference([[0, 1], [3, 5, 8]], [0, 9], labels=["first", "second"])
    assert mixture.summary()["between_model_variance"] == 0
    np.testing.assert_array_equal(mixture.quantiles(), [3, 5, 8])
    draws, labels = mixture.draw(100, 8)
    assert set(draws) <= set([3, 5, 8]) and (labels == "second").all()


@pytest.mark.parametrize(
    "samples,weights",
    [
        ([[0], [1]], [0, 0]),
        ([[0], [1]], [-1, 2]),
        ([[0], [1]], [np.nan, 1]),
        ([[np.nan], [1]], [1, 1]),
        ([[], [1]], [1, 1]),
        ([[0], [1]], [1]),
    ],
)
def test_invalid_mixtures_rejected(samples, weights):
    with pytest.raises(ValueError):
        ModelAveragedInference(samples, weights)


def test_conditioning_an_event_changes_model_weights():
    np.testing.assert_allclose(
        ModelAveragedInference.condition_weights([0.5, 0.5], [0.9, 0.1]), [0.9, 0.1]
    )
    with pytest.raises(ValueError):
        ModelAveragedInference.condition_weights([0.5, 0.5], [0, 0])
    with pytest.raises(ValueError):
        ModelAveragedInference.condition_weights([0.5, 0.5], [1, 2])


def test_posterior_dynamics_preserves_unstable_draws_and_crosschecks_companion():
    phi = np.array([[[1.2, -0.3], [1.5, -1.0], [-1.5, 0.5]]] * 2)
    dynamics = PosteriorDynamics.from_coefficients(phi)
    np.testing.assert_array_equal(dynamics.stable[0], [True, False, False])
    assert dynamics.coefficient_sum[0, 2] < 1 and not dynamics.stable[0, 2]
    assert dynamics.responses.shape == (2, 3, 11)
    np.testing.assert_allclose(dynamics.responses[0, 0, :4], [1, 1.2, 1.14, 1.008])
    assert dynamics.crosscheck_maximum_error < 1e-10
    assert dynamics.summary()["draws_removed_for_stability"] == 0
    assert dynamics.summary()["draws_retained"] == 6


def test_integrated_mixture_obeys_total_variance_and_correct_scalar_definition():
    first = IntegratedGaussianReference(zero_design_model((0, 1), (5, 1)))
    second = IntegratedGaussianReference(zero_design_model((0, 5), (5, 1)))
    result = integrated_mixture_summary([first, second], [0.25, 0.75])
    assert result["mean"] == pytest.approx(4)
    assert result["within_model_variance"] == pytest.approx(1)
    assert result["between_model_variance"] == pytest.approx(3)
    assert result["sd"] == pytest.approx(2)


def test_joint_batches_include_weight_uncertainty_with_zero_within_model_uncertainty():
    class Audit:
        def __init__(self, values):
            self.values = values

        def batch_log_evidence(self, size):
            return np.log(self.values)

    results = [SimpleNamespace(draws=np.zeros((2, 40, 2)))] * 2
    dynamics = [
        SimpleNamespace(coefficient_sum=np.zeros((2, 40))),
        SimpleNamespace(coefficient_sum=np.full((2, 40), 2.0)),
    ]
    frame, record = joint_batch_statistics(
        results, [Audit([0.25, 0.75]), Audit([0.75, 0.25])], dynamics
    )
    np.testing.assert_allclose(frame.bma_mean, [1.5, 0.5])
    assert record["mcse"]["bma_mean"] == pytest.approx(0.5)
    assert record["mcse"]["weight_ar1"] == pytest.approx(0.25)

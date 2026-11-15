import numpy as np
import pytest
from scipy.linalg import toeplitz
from scipy.signal import lfilter

from shock_dynamics import (
    ARSimulator,
    ARSpecification,
    DynamicsAnalyzer,
    MAAnalyzer,
    MASpecification,
    SimulationConfig,
)
from shock_dynamics.pipeline import mean_precision_experiment
from shock_dynamics.simulation import MASimulator


def test_stability_has_independent_algebraic_controls():
    stable = DynamicsAnalyzer(ARSpecification((0.5, 0.2, 0.2)))
    # A submultiplicative norm below one for F^3 implies spectral radius(F) < 1.
    assert np.linalg.norm(np.linalg.matrix_power(stable.companion, 3), ord=np.inf) < 1
    # The unstable characteristic polynomial changes sign above one.
    assert np.polyval([1, -0.5, -0.3, -0.3], 1) < 0
    assert np.polyval([1, -0.5, -0.3, -0.3], 2) > 0
    ar2 = DynamicsAnalyzer(ARSpecification((1.2, -0.3)))
    np.testing.assert_allclose(np.sort(ar2.eigenvalues), [0.6 - np.sqrt(0.06), 0.6 + np.sqrt(0.06)])
    # Boundary roots are 3/4 +/- i*sqrt(7)/4, with modulus exactly one.
    assert (3 / 4) ** 2 + (np.sqrt(7) / 4) ** 2 == pytest.approx(1)


@pytest.mark.parametrize(
    "phi,classification",
    [
        ((0.9,), "stable"),
        ((0.5, 0.2, 0.2), "stable"),
        ((0.5, 0.3, 0.3), "unstable"),
        ((1.2, -0.3), "stable"),
        ((1.5, -1), "unit-circle boundary or numerically near boundary"),
        ((-0.9,), "stable"),
        ((1.0,), "unit-circle boundary or numerically near boundary"),
    ],
)
def test_classifications_and_reciprocal_roots(phi, classification):
    analyzer = DynamicsAnalyzer(ARSpecification(phi))
    assert analyzer.classification == classification
    np.testing.assert_allclose(
        np.sort_complex(analyzer.characteristic_roots),
        np.sort_complex(1 / analyzer.eigenvalues),
        atol=1e-12,
    )


def test_zero_final_coefficient_and_white_noise_have_finite_roots_only():
    analyzer = DynamicsAnalyzer(ARSpecification((0.5, 0.0)))
    np.testing.assert_allclose(analyzer.characteristic_roots, [2.0])
    white = DynamicsAnalyzer(ARSpecification((0.0, 0.0)))
    assert white.characteristic_roots.size == 0
    assert white.classification == "stable"
    np.testing.assert_allclose(white.stationary_covariance, np.eye(2))


@pytest.mark.parametrize("phi", [(0.9,), (-0.8,), (1.2, -0.3), (0.5, 0.2, 0.2)])
def test_covariance_satisfies_lyapunov_and_infinite_ma_variance(phi):
    analyzer = DynamicsAnalyzer(ARSpecification(phi, innovation_variance=4.0))
    f = analyzer.companion
    covariance = analyzer.stationary_covariance
    q = np.zeros_like(f)
    q[0, 0] = 4
    np.testing.assert_allclose(covariance - f @ covariance @ f.T, q, atol=1e-12)
    response = lfilter([1], np.r_[1, -np.array(phi)], np.r_[1, np.zeros(5000)])
    np.testing.assert_allclose(covariance[0, 0], 4 * response @ response, rtol=1e-12)


def test_ar1_analytical_moments_and_mean_variance_with_toeplitz():
    analyzer = DynamicsAnalyzer(ARSpecification((0.9,), intercept=0.5, innovation_variance=4))
    assert analyzer.stationary_mean == pytest.approx(5)
    assert analyzer.stationary_covariance[0, 0] == pytest.approx(4 / (1 - 0.9**2))
    gamma = 4 / (1 - 0.9**2) * 0.9 ** np.arange(200)
    independent_calculation = toeplitz(gamma).sum() / 200**2
    assert analyzer.sample_mean_variance(200) == pytest.approx(independent_calculation)
    assert analyzer.sample_mean_variance(1) == pytest.approx(gamma[0])


@pytest.mark.parametrize("phi", [(1.1,), (0.5, 0.3, 0.3), (1.5, -1), (1 - 1e-12,)])
def test_stationary_quantities_reject_unstable_and_boundary_cases(phi):
    analyzer = DynamicsAnalyzer(ARSpecification(phi))
    for operation in [
        lambda: analyzer.stationary_mean,
        lambda: analyzer.stationary_covariance,
        lambda: analyzer.long_run_sum,
    ]:
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize(
    "phi", [(0.9,), (-0.7,), (1.2, -0.3), (0.5, 0.2, 0.2), (1.5, -1), (0.5, 0.3, 0.3)]
)
def test_three_irfs_match_independent_filter_for_nonzero_intercept(phi):
    analyzer = DynamicsAnalyzer(ARSpecification(phi, intercept=0.3))
    response = analyzer.impulse_response(100)
    expected = lfilter([1], np.r_[1, -np.array(phi)], np.r_[1, np.zeros(100)])
    for values in response.methods.values():
        np.testing.assert_allclose(values, expected, rtol=1e-10, atol=1e-10)
    assert expected[0] == 1
    assert analyzer.impulse_response(0).cumulative[0] == 1


def test_ar2_first_responses_peak_and_long_run_sum():
    analyzer = DynamicsAnalyzer(ARSpecification((1.2, -0.3)))
    response = analyzer.impulse_response(100)
    np.testing.assert_allclose(response.methods["recursion"][:4], [1, 1.2, 1.14, 1.008])
    assert np.argmax(response.methods["recursion"]) == 1
    assert analyzer.long_run_sum == pytest.approx(10)
    assert response.cumulative[-1] == pytest.approx(10, abs=2e-6)


def test_initial_lag_order_matches_hand_calculation_without_negative_wraparound():
    model = ARSpecification((0.5, 0.2, 0.2), intercept=0.1)
    values = ARSimulator(model).path([1, 0], [3, 2, 1])
    first = 0.1 + 0.5 * 3 + 0.2 * 2 + 0.2 * 1 + 1
    second = 0.1 + 0.5 * first + 0.2 * 3 + 0.2 * 2
    np.testing.assert_allclose(values, [first, second])


def test_reproducibility_burn_in_and_variance_to_standard_deviation():
    model = ARSpecification((0.0,), innovation_variance=9)
    config = SimulationConfig(5, 3, 83)
    result = ARSimulator(model).simulate(config, [7])
    expected = np.random.default_rng(83).normal(scale=3, size=8)
    np.testing.assert_array_equal(result.innovations, expected)
    np.testing.assert_array_equal(result.values, expected[3:])
    np.testing.assert_array_equal(result.values, ARSimulator(model).simulate(config, [7]).values)
    assert result.initial_state == (7,)


def test_common_innovations_produce_exact_mean_shift():
    config = SimulationConfig(200, 500, 20261115)
    first = ARSimulator(ARSpecification((0.9,))).simulate(config, [0])
    shifted = ARSimulator(ARSpecification((0.9,), intercept=0.5)).simulate(config, [5])
    np.testing.assert_array_equal(first.innovations, shifted.innovations)
    np.testing.assert_allclose(shifted.values - first.values, 5, atol=1e-12)


def test_ma_stationarity_does_not_require_invertibility():
    analysis = MAAnalyzer(MASpecification((1, 1.5, 1.3)))
    assert not analysis.invertible
    np.testing.assert_allclose(np.abs(analysis.roots), np.sqrt(1 / 1.3))
    np.testing.assert_allclose(
        [analysis.autocovariance(k) for k in range(5)], [4.94, 3.45, 1.3, 0, 0]
    )
    assert MAAnalyzer(MASpecification((1, 0.2))).invertible
    assert MAAnalyzer(MASpecification((1, 0))).invertible


def test_ma_simulation_matches_independent_filter_and_retains_full_noise():
    specification = MASpecification((1, 1.5, 1.3), innovation_variance=4)
    config = SimulationConfig(80, 20, 41)
    result = MASimulator(specification).simulate(config, [0, 0])
    expected = lfilter([1, 1.5, 1.3], [1], result.innovations)[20:]
    np.testing.assert_allclose(result.values, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coefficients": ()},
        {"coefficients": (np.nan,)},
        {"coefficients": (0.9,), "innovation_variance": -1},
        {"coefficients": (0.9,), "intercept": np.inf},
    ],
)
def test_invalid_models_rejected(kwargs):
    with pytest.raises(ValueError):
        ARSpecification(**kwargs)


@pytest.mark.parametrize("kwargs", [{"length": 0}, {"length": 2.5}, {"burn_in": -1}, {"seed": -1}])
def test_invalid_experiment_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        SimulationConfig(**kwargs)


def test_malformed_noise_initial_states_and_horizons_rejected():
    simulator = ARSimulator(ARSpecification((0.5, 0.2)))
    with pytest.raises(ValueError):
        simulator.path([0], [0])
    with pytest.raises(ValueError):
        simulator.path([np.nan], [0, 0])
    with pytest.raises(ValueError):
        simulator.simulate(SimulationConfig(2), [0, 0], innovations=[0])
    with pytest.raises(ValueError):
        DynamicsAnalyzer(simulator.specification).impulse_response(-1)
    with pytest.raises(ValueError):
        MASpecification((2, 0.1))


def test_mean_precision_uses_exact_stationary_initialization_and_reproduces():
    summary, first = mean_precision_experiment(20261215, replications=300)
    second_summary, second = mean_precision_experiment(20261215, replications=300)
    np.testing.assert_array_equal(first, second)
    assert summary == second_summary
    assert summary["variance_inflation_factor"] > 18
    assert 10 < summary["effective_independent_observations"] < 12
    assert summary["exact_independence_coverage"] < 0.4
    with pytest.raises(ValueError):
        mean_precision_experiment(1, replications=1)

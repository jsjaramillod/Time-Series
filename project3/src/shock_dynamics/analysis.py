"""Deterministic stability, covariance, and response calculations."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from .models import ARSpecification, MASpecification
from .simulation import ARSimulator


def _horizon(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("Horizon must be a nonnegative integer")


@dataclass
class ImpulseResponseResult:
    horizons: np.ndarray
    methods: dict[str, np.ndarray]

    @property
    def cumulative(self):
        return np.cumsum(self.methods["recursion"])

    @property
    def maximum_disagreement(self):
        baseline = self.methods["recursion"]
        return max(float(np.max(np.abs(value - baseline))) for value in self.methods.values())


class DynamicsAnalyzer:
    def __init__(self, specification: ARSpecification, tolerance: float = 1e-10):
        if not np.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError("Tolerance must be finite and between zero and one")
        self.specification = specification
        self.tolerance = tolerance

    @property
    def companion(self):
        p = self.specification.order
        matrix = np.zeros((p, p))
        matrix[0] = self.specification.coefficients
        if p > 1:
            matrix[1:, :-1] = np.eye(p - 1)
        return matrix

    @property
    def eigenvalues(self):
        return np.linalg.eigvals(self.companion)

    @property
    def characteristic_roots(self):
        polynomial = np.polynomial.polynomial.polytrim(
            np.r_[1.0, -np.asarray(self.specification.coefficients)]
        )
        return np.polynomial.polynomial.polyroots(polynomial)

    @property
    def spectral_radius(self):
        return float(np.max(np.abs(self.eigenvalues)))

    @property
    def classification(self):
        radius = self.spectral_radius
        if radius < 1 - self.tolerance:
            return "stable"
        if radius > 1 + self.tolerance:
            return "unstable"
        return "unit-circle boundary or numerically near boundary"

    def _require_stable(self):
        if self.classification != "stable":
            raise ValueError(
                "Stationary moments and convergent long-run sums require strict stability"
            )

    @property
    def stationary_mean(self):
        self._require_stable()
        return self.specification.intercept / (1 - sum(self.specification.coefficients))

    @property
    def stationary_covariance(self):
        self._require_stable()
        q = np.zeros_like(self.companion)
        q[0, 0] = self.specification.innovation_variance
        return solve_discrete_lyapunov(self.companion, q)

    @property
    def long_run_sum(self):
        self._require_stable()
        return 1 / (1 - sum(self.specification.coefficients))

    def autocovariances(self, horizon):
        _horizon(horizon)
        covariance = self.stationary_covariance
        return np.array(
            [
                (np.linalg.matrix_power(self.companion, k) @ covariance)[0, 0]
                for k in range(horizon + 1)
            ]
        )

    def sample_mean_variance(self, length):
        if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 1:
            raise ValueError("Length must be a positive integer")
        gamma = self.autocovariances(length - 1)
        return float((length * gamma[0] + 2 * np.arange(length - 1, 0, -1) @ gamma[1:]) / length**2)

    def impulse_response(self, horizon):
        _horizon(horizon)
        phi = self.specification.coefficients
        recursive = np.zeros(horizon + 1)
        recursive[0] = 1
        for j in range(1, horizon + 1):
            recursive[j] = sum(
                phi[i - 1] * recursive[j - i] for i in range(1, min(j, len(phi)) + 1)
            )
        powers = np.array(
            [np.linalg.matrix_power(self.companion, j)[0, 0] for j in range(horizon + 1)]
        )
        noise = np.zeros(horizon + 1)
        simulator = ARSimulator(self.specification)
        baseline = simulator.path(noise, np.zeros(self.specification.order))
        noise[0] = 1
        shock = simulator.path(noise, np.zeros(self.specification.order)) - baseline
        return ImpulseResponseResult(
            np.arange(horizon + 1), {"recursion": recursive, "matrix": powers, "shock": shock}
        )


class MAAnalyzer:
    def __init__(self, specification: MASpecification, tolerance=1e-10):
        if not np.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError("Tolerance must be finite and between zero and one")
        self.specification = specification
        self.tolerance = tolerance

    @property
    def roots(self):
        coefficients = np.polynomial.polynomial.polytrim(self.specification.coefficients)
        return np.polynomial.polynomial.polyroots(coefficients)

    @property
    def invertible(self):
        return bool(np.all(np.abs(self.roots) > 1 + self.tolerance))

    def autocovariance(self, lag):
        _horizon(lag)
        theta = np.asarray(self.specification.coefficients)
        if lag >= theta.size:
            return 0.0
        return float(
            self.specification.innovation_variance * theta[: theta.size - lag] @ theta[lag:]
        )

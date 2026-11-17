"""Draw-wise finite responses and companion stability without removing unstable draws."""

from dataclasses import dataclass

import numpy as np
from shock_dynamics import ARSpecification, DynamicsAnalyzer


@dataclass
class PosteriorDynamics:
    coefficient_sum: np.ndarray
    spectral_radius: np.ndarray
    stable: np.ndarray
    responses: np.ndarray
    crosscheck_maximum_error: float

    @classmethod
    def from_coefficients(cls, coefficients, horizon=10, tolerance=1e-10):
        phi = np.asarray(coefficients, dtype=float)
        if phi.ndim != 3 or phi.shape[-1] < 1 or not np.isfinite(phi).all():
            raise ValueError("AR coefficients must have chain, draw, and lag dimensions")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
            raise ValueError("Horizon must be a nonnegative integer")
        if not np.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError("Boundary tolerance must be in (0, 1)")
        responses = np.zeros(phi.shape[:2] + (horizon + 1,))
        responses[..., 0] = 1
        for j in range(1, horizon + 1):
            for lag in range(1, min(j, phi.shape[-1]) + 1):
                responses[..., j] += phi[..., lag - 1] * responses[..., j - lag]
        if not np.isfinite(responses).all():
            raise FloatingPointError("Finite-horizon responses exceeded numerical range")
        flat = phi.reshape(-1, phi.shape[-1])
        radii = np.empty(len(flat))
        order = phi.shape[-1]
        for start in range(0, len(flat), 4000):
            block = flat[start : start + 4000]
            matrices = np.zeros((len(block), order, order))
            matrices[:, 0, :] = block
            if order > 1:
                matrices[:, 1:, :-1] = np.eye(order - 1)
            radii[start : start + len(block)] = np.max(np.abs(np.linalg.eigvals(matrices)), axis=1)
        radii = radii.reshape(phi.shape[:2])
        maximum = 0.0
        for index in np.linspace(0, len(flat) - 1, min(8, len(flat)), dtype=int):
            reference = DynamicsAnalyzer(ARSpecification(tuple(flat[index])), tolerance)
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(
                            reference.impulse_response(horizon).methods["matrix"]
                            - responses.reshape(-1, horizon + 1)[index]
                        )
                    )
                ),
                abs(reference.spectral_radius - radii.reshape(-1)[index]),
            )
        return cls(phi.sum(axis=-1), radii, radii < 1 - tolerance, responses, maximum)

    def continuous_draws(self):
        return np.concatenate(
            [
                self.coefficient_sum[..., None],
                self.spectral_radius[..., None],
                self.responses[..., 1:],
            ],
            axis=-1,
        )

    def summary(self):
        total = self.coefficient_sum
        return {
            "stable_draw_proportion": float(self.stable.mean()),
            "unstable_or_boundary_draw_proportion": float(1 - self.stable.mean()),
            "sum_below_one_but_not_stable_proportion": float(np.mean((total < 1) & ~self.stable)),
            "maximum_vectorized_vs_companion_error": self.crosscheck_maximum_error,
            "draws_retained": int(self.stable.size),
            "draws_removed_for_stability": 0,
        }

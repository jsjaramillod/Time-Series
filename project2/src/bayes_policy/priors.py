"""Explicit prior parameterizations; coefficient covariance is independent of precision."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NormalPrior:
    mean: NDArray
    covariance: NDArray

    def __post_init__(self):
        mean = np.array(self.mean, dtype=float, copy=True)
        covariance = np.array(self.covariance, dtype=float, copy=True)
        if mean.ndim != 1 or covariance.shape != (len(mean), len(mean)):
            raise ValueError("Normal prior mean and covariance dimensions disagree")
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ValueError("Prior parameters must be finite")
        if not np.allclose(covariance, covariance.T):
            raise ValueError("Prior covariance must be symmetric")
        np.linalg.cholesky(covariance)
        mean.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)

    @classmethod
    def isotropic(cls, dimension: int, variance: float = 10.0):
        if dimension < 1 or not np.isfinite(variance) or variance <= 0:
            raise ValueError("Dimension and variance must be positive")
        return cls(np.zeros(dimension), np.eye(dimension) * variance)


@dataclass(frozen=True)
class GammaPrecisionPrior:
    """Gamma(shape, rate), with E[h] = shape/rate; never Gamma(shape, scale)."""

    shape: float = 1.5
    rate: float = 1.5

    def __post_init__(self):
        if not np.isfinite([self.shape, self.rate]).all() or min(self.shape, self.rate) <= 0:
            raise ValueError("Gamma shape and rate must be finite and positive")

    @classmethod
    def from_reference(cls, m: float = 1.0, v: float = 3.0):
        """gen_h.m: E[h]=m, Var[h]=2m²/v => shape=v/2, rate=v/(2m)."""
        if m <= 0 or v <= 0:
            raise ValueError("Reference m and v must be positive")
        return cls(shape=v / 2, rate=v / (2 * m))

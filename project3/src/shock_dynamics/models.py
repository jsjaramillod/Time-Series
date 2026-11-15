"""Immutable model parameters, separate from experiment settings."""

from dataclasses import dataclass

import numpy as np


def _coefficients(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("Coefficients must be a nonempty finite one-dimensional sequence")
    return tuple(float(v) for v in array)


@dataclass(frozen=True)
class ARSpecification:
    coefficients: tuple[float, ...]
    intercept: float = 0.0
    innovation_variance: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "coefficients", _coefficients(self.coefficients))
        if not np.isfinite(self.intercept):
            raise ValueError("Intercept must be finite")
        if not np.isfinite(self.innovation_variance) or self.innovation_variance <= 0:
            raise ValueError("Innovation variance must be finite and positive")

    @property
    def order(self):
        return len(self.coefficients)


@dataclass(frozen=True)
class MASpecification:
    """Finite MA polynomial, including a normalized contemporaneous coefficient of one."""

    coefficients: tuple[float, ...]
    innovation_variance: float = 1.0

    def __post_init__(self):
        coefficients = _coefficients(self.coefficients)
        if coefficients[0] != 1.0:
            raise ValueError("The contemporaneous innovation coefficient must equal one")
        object.__setattr__(self, "coefficients", coefficients)
        if not np.isfinite(self.innovation_variance) or self.innovation_variance <= 0:
            raise ValueError("Innovation variance must be finite and positive")

    @property
    def order(self):
        return len(self.coefficients) - 1

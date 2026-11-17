from dataclasses import dataclass

import numpy as np
from bayes_policy.models import GaussianRegression
from bayes_policy.priors import GammaPrecisionPrior, NormalPrior


@dataclass(frozen=True)
class PriorSpecification:
    coefficient_variance: float = 0.5
    intercept_variance: float = 100.0
    precision_shape: float = 1.0
    precision_rate: float = 3.0

    def __post_init__(self):
        values = [
            self.coefficient_variance,
            self.intercept_variance,
            self.precision_shape,
            self.precision_rate,
        ]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("All prior variances, shape and rate must be finite and positive")

    def model(self, data):
        covariance = np.diag(
            [self.intercept_variance] + [self.coefficient_variance] * (len(data.names) - 1)
        )
        return GaussianRegression(
            data,
            NormalPrior(np.zeros(len(data.names)), covariance),
            GammaPrecisionPrior(self.precision_shape, self.precision_rate),
        )

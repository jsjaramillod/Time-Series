"""Statistical specifications independent of sampler and reporting choices."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve
from scipy.special import log_ndtr
from scipy.stats import gamma, multivariate_normal

from .data import RegressionData
from .priors import GammaPrecisionPrior, NormalPrior


@dataclass
class GaussianRegression:
    data: RegressionData
    prior: NormalPrior
    precision_prior: GammaPrecisionPrior

    def __post_init__(self):
        if len(self.prior.mean) != self.data.X.shape[1]:
            raise ValueError("Prior dimension does not match design")
        self.xtx = self.data.X.T @ self.data.X
        self.xty = self.data.X.T @ self.data.y
        self.prior_precision = np.linalg.solve(self.prior.covariance, np.eye(len(self.prior.mean)))

    def coefficient_conditional(self, precision: float) -> tuple[np.ndarray, np.ndarray]:
        """Return conditional mean and lower Cholesky factor of posterior precision."""
        if not np.isfinite(precision) or precision <= 0:
            raise ValueError("Precision must be positive and finite")
        chol = np.linalg.cholesky(self.prior_precision + precision * self.xtx)
        mean = cho_solve(
            (chol, True),
            self.prior_precision @ self.prior.mean + precision * self.xty,
            check_finite=False,
        )
        return mean, chol

    def precision_conditional(self, beta: np.ndarray) -> tuple[float, float]:
        residual = self.data.y - self.data.X @ beta
        return (
            self.precision_prior.shape + len(residual) / 2,
            self.precision_prior.rate + residual @ residual / 2,
        )

    def log_likelihood(self, beta: np.ndarray, precision: float) -> float:
        residual = self.data.y - self.data.X @ beta
        n = len(residual)
        return float(
            n / 2 * (np.log(precision) - np.log(2 * np.pi)) - precision / 2 * residual @ residual
        )

    def log_prior(self, beta: np.ndarray, precision: float) -> float:
        return float(
            multivariate_normal.logpdf(beta, self.prior.mean, self.prior.covariance)
            + gamma.logpdf(
                precision, self.precision_prior.shape, scale=1 / self.precision_prior.rate
            )
        )

    def ols(self) -> np.ndarray:
        return np.linalg.lstsq(self.data.X, self.data.y, rcond=None)[0]


@dataclass
class BayesianProbit:
    data: RegressionData
    prior: NormalPrior

    def __post_init__(self):
        if len(self.prior.mean) != self.data.X.shape[1]:
            raise ValueError("Prior dimension does not match design")
        if not np.isin(self.data.y, [0, 1]).all():
            raise ValueError("Probit responses must be binary")
        self.prior_precision = np.linalg.solve(self.prior.covariance, np.eye(len(self.prior.mean)))
        self.conditional_chol = np.linalg.cholesky(
            self.prior_precision + self.data.X.T @ self.data.X
        )

    def coefficient_conditional(self, latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = cho_solve(
            (self.conditional_chol, True),
            self.prior_precision @ self.prior.mean + self.data.X.T @ latent,
            check_finite=False,
        )
        return mean, self.conditional_chol

    def log_likelihood(self, beta: np.ndarray) -> float:
        # log Phi(signed index) is stable even for rare events / extreme predictors.
        return float(log_ndtr((2 * self.data.y - 1) * (self.data.X @ beta)).sum())

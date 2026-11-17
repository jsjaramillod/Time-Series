"""Audit Chib's posterior ordinate at several points and preserve joint batches."""

from dataclasses import dataclass

import arviz as az
import numpy as np
from bayes_policy.comparison import ChibModelComparison
from scipy.linalg import cho_solve
from scipy.special import logsumexp
from scipy.stats import gamma, multivariate_normal


@dataclass
class ChibOrdinateAudit:
    estimate: object
    point_checks: list[dict]
    log_conditional_ordinates: np.ndarray
    log_numerator_minus_coefficient_ordinate: float

    @classmethod
    def run(cls, model, result):
        estimate = ChibModelComparison.estimate(model, result)
        beta = result.coefficients.reshape(-1, len(model.prior.mean))
        rates = np.empty(len(beta))
        for start in range(0, len(beta), 4000):
            residual = model.data.y[None, :] - beta[start : start + 4000] @ model.data.X.T
            rates[start : start + 4000] = (
                model.precision_prior.rate + np.einsum("ij,ij->i", residual, residual) / 2
            )
        shape = model.precision_prior.shape + len(model.data.y) / 2
        point_checks = []
        for multiplier in (0.8, 1.0, 1.2):
            precision = estimate.evaluation_precision * multiplier
            mean, chol = model.coefficient_conditional(precision)
            covariance = cho_solve((chol, True), np.eye(len(mean)))
            beta_ordinate = multivariate_normal.logpdf(estimate.evaluation_beta, mean, covariance)
            log_ordinates = gamma.logpdf(precision, shape, scale=1 / rates).reshape(
                result.draws.shape[:2]
            )
            constant = (
                model.log_likelihood(np.array(estimate.evaluation_beta), precision)
                + model.log_prior(np.array(estimate.evaluation_beta), precision)
                - beta_ordinate
            )
            log_evidence = constant - logsumexp(log_ordinates) + np.log(log_ordinates.size)
            normalized = np.exp(log_ordinates - log_ordinates.max())
            mcse = float(az.mcse(normalized, method="mean").item() / normalized.mean())
            point_checks.append(
                {
                    "precision_multiplier": multiplier,
                    "log_evidence": float(log_evidence),
                    "mcse_log_evidence": mcse,
                }
            )
            if multiplier == 1.0:
                baseline_ordinates = log_ordinates
                baseline_constant = float(constant)
        if abs(point_checks[1]["log_evidence"] - estimate.log_evidence) > 1e-10:
            raise RuntimeError("Chib audit and shared estimator disagree at the same point")
        return cls(estimate, point_checks, baseline_ordinates, baseline_constant)

    def batch_log_evidence(self, batch_size):
        chains, draws = self.log_conditional_ordinates.shape
        if draws % batch_size or batch_size < 20:
            raise ValueError("Batch size must divide retained draws and be at least twenty")
        batches = self.log_conditional_ordinates.reshape(chains, draws // batch_size, batch_size)
        return (
            self.log_numerator_minus_coefficient_ordinate
            - logsumexp(batches, axis=-1)
            + np.log(batch_size)
        ).reshape(-1)

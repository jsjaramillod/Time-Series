"""Chib's identity, an independent one-dimensional evidence control, and model weights."""

from dataclasses import dataclass

import arviz as az
import numpy as np
from scipy.integrate import quad
from scipy.linalg import cho_solve
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import gamma, multivariate_normal

from .models import GaussianRegression
from .results import PosteriorResult


def model_probabilities(log_evidence, prior_weights=None) -> np.ndarray:
    values = np.asarray(log_evidence, dtype=float)
    weights = (
        np.ones(len(values)) if prior_weights is None else np.asarray(prior_weights, dtype=float)
    )
    if (
        values.ndim != 1
        or len(values) < 2
        or weights.shape != values.shape
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("Need finite log evidence and matching positive model prior weights")
    joint = values + np.log(weights / weights.sum())
    return np.exp(joint - logsumexp(joint))


@dataclass(frozen=True)
class EvidenceEstimate:
    log_evidence: float
    mcse_log_evidence: float
    evaluation_precision: float
    evaluation_beta: list[float]


class ChibModelComparison:
    @staticmethod
    def estimate(model: GaussianRegression, result: PosteriorResult) -> EvidenceEstimate:
        if result.names != model.data.names + ("sigma2",):
            raise ValueError("Posterior names do not match the model")
        if result.metadata.get("n") != len(model.data.y):
            raise ValueError("Posterior sample size does not match model")
        if result.metadata.get("data_fingerprint") != model.data.fingerprint():
            raise ValueError("Posterior was fitted to a different response or design")
        if (
            result.metadata.get("coefficient_prior_mean") != model.prior.mean.tolist()
            or result.metadata.get("coefficient_prior_covariance")
            != model.prior.covariance.tolist()
            or result.metadata.get("precision_prior_shape") != model.precision_prior.shape
            or result.metadata.get("precision_prior_rate") != model.precision_prior.rate
        ):
            raise ValueError("Posterior was fitted under different priors")
        flat = result.coefficients.reshape(-1, len(model.prior.mean))
        beta_star = flat.mean(axis=0)
        h_star = float((1 / result.parameter("sigma2")).mean())
        mean, chol = model.coefficient_conditional(h_star)
        covariance = cho_solve((chol, True), np.eye(len(mean)))
        log_beta_ordinate = multivariate_normal.logpdf(beta_star, mean, covariance)
        shape = model.precision_prior.shape + len(model.data.y) / 2
        log_ordinates = np.empty(len(flat))
        for start in range(0, len(flat), 2048):
            block = flat[start : start + 2048]
            residuals = model.data.y[None, :] - block @ model.data.X.T
            rates = model.precision_prior.rate + np.einsum("ij,ij->i", residuals, residuals) / 2
            log_ordinates[start : start + len(block)] = gamma.logpdf(h_star, shape, scale=1 / rates)
        log_h_ordinate = logsumexp(log_ordinates) - np.log(len(log_ordinates))
        # Delta-method MCSE for log(mean conditional density), preserving chain order.
        shifted = np.exp(log_ordinates - log_ordinates.max()).reshape(result.draws.shape[:2])
        mcse = float(az.mcse(shifted, method="mean").item() / shifted.mean())
        log_evidence = (
            model.log_likelihood(beta_star, h_star)
            + model.log_prior(beta_star, h_star)
            - log_beta_ordinate
            - log_h_ordinate
        )
        return EvidenceEstimate(float(log_evidence), mcse, h_star, beta_star.tolist())

    @staticmethod
    def compare(
        models: list[GaussianRegression], results: list[PosteriorResult], prior_weights=None
    ):
        if len(models) != len(results) or len(models) < 2:
            raise ValueError("Need matching model/result lists")
        baseline = models[0].data
        for model in models[1:]:
            if not baseline.dates.equals(model.data.dates) or not np.array_equal(
                baseline.y, model.data.y
            ):
                raise ValueError("Model comparison requires identical observed responses and dates")
        estimates = [
            ChibModelComparison.estimate(model, result) for model, result in zip(models, results)
        ]
        weights = model_probabilities([e.log_evidence for e in estimates], prior_weights)
        return estimates, weights


def quadrature_log_evidence(model: GaussianRegression) -> dict:
    """Integrate beta analytically, then integrate h numerically on log(h).

    This independent deterministic control does not reuse Chib's posterior ordinate.
    Bounds expand until both log-kernel tails are 60 below the located maximum.
    """
    n = len(model.data.y)
    logdet_v = np.linalg.slogdet(model.prior.covariance)[1]

    def log_kernel(z):
        h = np.exp(z)
        mean, chol = model.coefficient_conditional(h)
        residual = model.data.y - model.data.X @ mean
        delta = mean - model.prior.mean
        quadratic = h * residual @ residual + delta @ model.prior_precision @ delta
        logdet = -n * z + logdet_v + 2 * np.log(np.diag(chol)).sum()
        log_marginal_y = -0.5 * (n * np.log(2 * np.pi) + logdet + quadratic)
        return float(
            log_marginal_y
            + gamma.logpdf(h, model.precision_prior.shape, scale=1 / model.precision_prior.rate)
            + z
        )

    mode = minimize_scalar(lambda z: -log_kernel(z), bounds=(-30, 30), method="bounded")
    if not mode.success:
        raise RuntimeError("Could not locate evidence integration mode")
    center, peak = float(mode.x), float(-mode.fun)
    lower, upper = center - 2, center + 2
    while log_kernel(lower) > peak - 60:
        lower -= 2
        if lower < -100:
            raise RuntimeError("Evidence lower tail did not decay")
    while log_kernel(upper) > peak - 60:
        upper += 2
        if upper > 100:
            raise RuntimeError("Evidence upper tail did not decay")
    integral, error = quad(
        lambda z: np.exp(log_kernel(z) - peak), lower, upper, epsabs=1e-11, epsrel=1e-10, limit=200
    )
    if integral <= 0 or not np.isfinite(integral):
        raise RuntimeError("Evidence quadrature failed")
    return {
        "log_evidence": float(peak + np.log(integral)),
        "relative_quadrature_error": float(error / integral),
        "log_precision_bounds": [lower, upper],
    }

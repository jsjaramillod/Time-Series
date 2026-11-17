"""Gibbs sampling by composition: models supply conditionals, samplers own RNG state."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import truncnorm

from .models import BayesianProbit, GaussianRegression
from .results import PosteriorResult


def _observation_label(value):
    """Preserve dated labels while allowing explicitly ordinal observation indices."""
    return str(value.date()) if hasattr(value, "date") else str(value)


@dataclass(frozen=True)
class SamplerConfig:
    draws: int = 10000
    warmup: int = 3000
    chains: int = 4
    seed: int = 20261114

    def __post_init__(self):
        if any(not isinstance(v, int) for v in asdict(self).values()):
            raise ValueError("Sampler settings must be integers")
        if self.draws < 20 or self.warmup < 0 or self.chains < 2 or self.seed < 0:
            raise ValueError("Need >=20 draws, nonnegative warmup/seed, and >=2 chains")


def _normal_draw(mean, precision_chol, rng):
    # If P=LL', L'^(-1) z has covariance P^(-1). L^(-1) z is incorrect.
    return mean + solve_triangular(
        precision_chol.T, rng.normal(size=len(mean)), lower=False, check_finite=False
    )


class GaussianGibbsSampler:
    def __init__(self, config: SamplerConfig):
        self.config = config

    def fit(self, model: GaussianRegression) -> PosteriorResult:
        cfg = self.config
        k = len(model.prior.mean)
        draws = np.empty((cfg.chains, cfg.draws, k + 1))
        for chain, stream in enumerate(np.random.SeedSequence(cfg.seed).spawn(cfg.chains)):
            rng = np.random.default_rng(stream)
            beta = model.prior.mean + 3 * np.linalg.cholesky(model.prior.covariance) @ rng.normal(
                size=k
            )
            for iteration in range(cfg.warmup + cfg.draws):
                shape, rate = model.precision_conditional(beta)
                precision = rng.gamma(shape, 1 / rate)
                beta = _normal_draw(*model.coefficient_conditional(precision), rng)
                if iteration >= cfg.warmup:
                    draws[chain, iteration - cfg.warmup, :-1] = beta
                    draws[chain, iteration - cfg.warmup, -1] = 1 / precision
        return PosteriorResult(
            draws,
            model.data.names + ("sigma2",),
            {
                "sampler": "Gaussian Gibbs; independent Normal/Gamma priors",
                **asdict(cfg),
                "coefficient_prior_mean": model.prior.mean.tolist(),
                "coefficient_prior_covariance": model.prior.covariance.tolist(),
                "precision_prior_shape": model.precision_prior.shape,
                "precision_prior_rate": model.precision_prior.rate,
                "n": len(model.data.y),
                "start": _observation_label(model.data.dates[0]),
                "end": _observation_label(model.data.dates[-1]),
                "data_fingerprint": model.data.fingerprint(),
            },
        )


class ProbitGibbsSampler:
    def __init__(self, config: SamplerConfig):
        self.config = config

    @staticmethod
    def draw_latent(y, X, beta, rng):
        mean = X @ beta
        # SciPy's truncated-normal sampler handles extreme tails without rejection loops.
        lower = np.where(y == 1, -mean, -np.inf)
        upper = np.where(y == 1, np.inf, -mean)
        return truncnorm.rvs(lower, upper, loc=mean, scale=1, random_state=rng)

    def fit(self, model: BayesianProbit) -> PosteriorResult:
        cfg = self.config
        k = len(model.prior.mean)
        draws = np.empty((cfg.chains, cfg.draws, k))
        for chain, stream in enumerate(np.random.SeedSequence(cfg.seed).spawn(cfg.chains)):
            rng = np.random.default_rng(stream)
            # Modestly dispersed starts avoid pathological rare-event initial predictions.
            beta = model.prior.mean + rng.normal(scale=0.5, size=k)
            for iteration in range(cfg.warmup + cfg.draws):
                latent = self.draw_latent(model.data.y, model.data.X, beta, rng)
                beta = _normal_draw(*model.coefficient_conditional(latent), rng)
                if iteration >= cfg.warmup:
                    draws[chain, iteration - cfg.warmup] = beta
        return PosteriorResult(
            draws,
            model.data.names,
            {
                "sampler": "Albert-Chib latent-normal probit",
                **asdict(cfg),
                "coefficient_prior_mean": model.prior.mean.tolist(),
                "coefficient_prior_covariance": model.prior.covariance.tolist(),
                "latent_variance": 1.0,
                "n": len(model.data.y),
                "data_fingerprint": model.data.fingerprint(),
            },
        )

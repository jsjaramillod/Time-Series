"""Independent-prior Bayesian VAR with equation-block Gibbs updates."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from .data import lag_design


@dataclass
class MinnesotaPrior:
    intensity: float = 0.5
    cross: float = 0.01
    intercept_variance: float = 1000.0

    def moments(self, scales, lags):
        d = len(scales)
        mean = np.zeros((1 + d * lags, d))
        mean[1 : 1 + d] = np.eye(d)
        relative = self.cross * scales[None, :] / scales[:, None]
        np.fill_diagonal(relative, 1.0)
        variance = np.vstack(
            [
                np.full(d, self.intercept_variance),
                *[self.intensity * relative / k**2 for k in range(1, lags + 1)],
            ]
        )
        return mean, variance


def covariance_draw(scale, df, rng):
    """Bartlett factor of IW(df, scale), with independent draws for each chain."""
    c, d, _ = scale.shape
    a = np.zeros_like(scale)
    idx = np.arange(d)
    a[:, idx, idx] = np.sqrt(rng.chisquare(df - idx, size=(c, d)))
    i, j = np.tril_indices(d, -1)
    a[:, i, j] = rng.normal(size=(c, len(i)))
    return np.linalg.cholesky(scale) @ np.linalg.inv(a).transpose(0, 2, 1)


@dataclass
class ForecastDraws:
    location: np.ndarray  # log growth, chains x draws
    scale: np.ndarray
    joint_payroll: np.ndarray  # simulated employment changes (thousands)
    cpi_lags: np.ndarray | None
    employment: float
    rejected: int

    def summarize(self):
        loc, sd = self.location, self.scale

        def quantile(p):
            value = brentq(
                lambda z: ndtr((z - loc) / sd).mean() - p,
                float((loc - 10 * sd).min()),
                float((loc + 10 * sd).max()),
            )
            return self.employment * np.expm1(value)

        return dict(
            mean=self.employment * np.expm1(loc + 0.5 * sd**2).mean(),
            median=quantile(0.5),
            p05=quantile(0.05),
            p95=quantile(0.95),
            p_loss=ndtr(-loc / sd).mean(),
            joint_mean=self.joint_payroll.mean(),
            joint_p_loss=(self.joint_payroll < 0).mean(),
            covariance_rejections=self.rejected,
        )


class BayesianVAR:
    def __init__(self, values, payroll_index, prior, lags=12):
        x, y, self.next = lag_design(values, lags)
        self.d = y.shape[1]
        self.payroll = payroll_index
        self.last = values[-1]
        self.df = self.d + 2 + len(y)
        q, r = np.linalg.qr(x, mode="reduced")
        reduced = q.T @ y
        residual = y - q @ reduced
        orthogonal = residual.T @ residual
        scales = np.sqrt(np.diag(orthogonal) / len(y))
        if np.any(scales <= 0) or np.linalg.matrix_rank(r) < x.shape[1]:
            raise ValueError("Singular training design")
        self.base_scale = orthogonal + np.diag(scales**2)
        mean, variance = prior.moments(scales, lags)
        self.base = reduced - r @ mean
        self.next_mean = self.next @ mean
        self.rotations = []
        for j in range(self.d):
            sd = np.sqrt(variance[:, j])
            u, singular, vt = np.linalg.svd(r * sd, full_matrices=False)
            basis = sd[:, None] * vt.T
            self.rotations.append((u, singular, self.next @ basis, basis))

    def sample(self, seed, warmup=180, draws=384, chains=4, track_lags=False):
        rng = np.random.default_rng(seed)
        d, j_emp = self.d, self.payroll
        # Dispersed starts around the OLS fitted values in reduced coordinates.
        err = rng.normal(size=(chains, *self.base.shape)) * np.sqrt(
            np.diag(self.base_scale) / self.df
        )
        location, sd, jobs = np.empty((3, chains, draws))
        cpi_lags = np.empty((chains, draws, 3)) if track_lags else None
        idx = np.arange(chains)
        conditional = np.empty((chains, d))
        rejected = 0
        for step in range(warmup + draws):
            scale = self.base_scale + err.transpose(0, 2, 1) @ err
            factor = covariance_draw(scale, self.df, rng)
            bad = np.linalg.norm(factor[:, j_emp], axis=1) > 0.25
            while bad.any():
                rejected += int(bad.sum())
                factor[bad] = covariance_draw(scale[bad], self.df, rng)
                bad = np.linalg.norm(factor[:, j_emp], axis=1) > 0.25
            inverse = np.linalg.inv(factor)
            precision = inverse.transpose(0, 2, 1) @ inverse
            for j, (u, singular, next_vector, basis) in enumerate(self.rotations):
                w = precision[:, j, j, None]
                weights = precision[:, :, j].copy()
                weights[:, j] = 0
                target = self.base[:, j] + (err @ weights[:, :, None])[:, :, 0] / w
                denom = 1 + w * singular**2
                eta = (target @ u) * (w * singular / denom)
                eta += rng.standard_normal(eta.shape) / np.sqrt(denom)
                err[:, :, j] = self.base[:, j] - (eta * singular) @ u.T
                conditional[:, j] = self.next_mean[j] + eta @ next_vector
                if track_lags and j == d - 1 and step >= warmup:
                    rows = 1 + j_emp + d * np.arange(3)
                    cpi_lags[:, step - warmup, :] = eta @ basis[rows].T
            if step >= warmup:
                k = step - warmup
                location[:, k] = conditional[:, j_emp] - self.last[j_emp]
                sd[:, k] = np.linalg.norm(factor[:, j_emp], axis=1)
                joint_innovation = (factor @ rng.standard_normal((chains, d, 1)))[:, :, 0]
                jobs[:, k] = np.exp(self.last[j_emp]) * np.expm1(
                    location[idx, k] + joint_innovation[:, j_emp]
                )
        return ForecastDraws(location, sd, jobs, cpi_lags, np.exp(self.last[j_emp]), rejected)

"""A five-equation BVAR with a temporary scale on the innovation covariance."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from payroll_risk.data import lag_design
from payroll_risk.model import MinnesotaPrior, covariance_draw
from scipy.linalg import block_diag, cho_solve, cholesky, solve_triangular
from scipy.optimize import brentq
from scipy.special import ndtr

LAGS = 12
EMP = 2
FIRST = pd.Period("2020-03", freq="M")
LAST = pd.Period("2022-03", freq="M")
DECAY = 0.9


def course_units(panel, origin, smooth=False):
    values = panel.design(origin, "Full", smooth=smooth)
    values[:, 1:] *= 100
    return values


def exponents(months):
    offsets = pd.PeriodIndex(months, freq="M").asi8 - FIRST.ordinal
    return np.where((offsets >= 0) & (offsets <= LAST.ordinal - FIRST.ordinal), offsets, -1)


def residual_scales(values):
    x, y, _ = lag_design(values, LAGS)
    q, _ = np.linalg.qr(x, mode="reduced")
    return np.sqrt(np.mean((y - q @ (q.T @ y)) ** 2, axis=0))


@dataclass
class CrossProducts:
    xx: np.ndarray
    xy: np.ndarray
    yy: np.ndarray

    @classmethod
    def from_rows(cls, x, y):
        return cls(x.T @ x, x.T @ y, y.T @ y)

    def residual_ss(self, theta):
        cross = self.xy.T @ theta
        return self.yy - cross - cross.T + theta.T @ self.xx @ theta


@dataclass
class Samples:
    coefficients: np.ndarray
    factor: np.ndarray
    gamma: np.ndarray


class PandemicVAR:
    """Joint Gaussian coefficient draw in QR coordinates, with unchanged prior."""

    def __init__(self, values, months, scales, temporary):
        x, y, _ = lag_design(values, LAGS)
        self.n, self.d = y.shape
        if len(months) != len(values) or (scales <= 0).any():
            raise ValueError("Dates and prior scales must match the training panel")
        offsets = exponents(months[LAGS:])
        event = (offsets >= 0) & temporary
        self.count = int(event.sum())
        if temporary and self.count != 25:
            raise ValueError("The fitted volatility window must be complete")
        q, r = np.linalg.qr(x, mode="reduced")
        self.inv_r = solve_triangular(r, np.eye(len(r)), check_finite=False)
        self.prior_mean, variances = MinnesotaPrior(intensity=0.1).moments(scales, LAGS)
        self.prior_precision = block_diag(
            *[self.inv_r.T @ (self.inv_r / var[:, None]) for var in variances.T]
        )
        # Match the inverse residual-variance diagonal used in the archived sampler.
        self.iw_scale = np.diag(1 / scales**2)
        y = y - x @ self.prior_mean
        self.normal = CrossProducts.from_rows(q[~event], y[~event])
        weights = DECAY ** (-offsets[event, None] / 2)
        self.event = CrossProducts.from_rows(q[event] * weights, y[event] * weights)

    def sample(self, seed, warmup, draws, chains=4):
        k, d = self.prior_mean.shape
        beta = np.empty((chains, draws, k, d))
        factors = np.empty((chains, draws, d, d))
        gamma = np.ones((chains, draws))
        for c, child in enumerate(np.random.SeedSequence(seed).spawn(chains)):
            rng = np.random.default_rng(child)
            theta = rng.standard_normal((k, d)) * 0.1
            inverse_scale = 1 / (10**c) if self.count else 1.0
            for step in range(warmup + draws):
                ss = self.normal.residual_ss(theta) + self.iw_scale
                if self.count:
                    ss += inverse_scale * self.event.residual_ss(theta)
                factor = covariance_draw(ss[None, :, :], self.n + d + 2, rng)[0]
                inverse = np.linalg.inv(factor)
                precision = inverse.T @ inverse
                gram = self.normal.xx + inverse_scale * self.event.xx
                xy = self.normal.xy + inverse_scale * self.event.xy
                chol = cholesky(
                    self.prior_precision + np.kron(precision, gram), lower=True, check_finite=False
                )
                loc = cho_solve((chol, True), (xy @ precision).T.ravel(), check_finite=False)
                noise = solve_triangular(chol.T, rng.standard_normal(k * d), check_finite=False)
                theta = (loc + noise).reshape(d, k).T
                if self.count:
                    weighted_ss = self.event.residual_ss(theta)
                    shape = 1 + self.count * d / 2
                    rate = 1 + 0.5 * np.sum(precision * weighted_ss)
                    inverse_scale = rng.gamma(shape, 1 / rate)
                if step >= warmup:
                    t = step - warmup
                    beta[c, t] = self.prior_mean + self.inv_r @ theta
                    factors[c, t] = factor
                    gamma[c, t] = 1 / inverse_scale
        return Samples(beta, factors, gamma)


@dataclass
class PayrollForecast:
    """Six-step conditional Gaussian mixture in 100 log points."""

    location: np.ndarray
    sd: np.ndarray
    base: float
    simulated_growth: np.ndarray

    @classmethod
    def predict(cls, samples, history, origin, seed, scale_future=False):
        chains, draws, k, d = samples.coefficients.shape
        beta = samples.coefficients.reshape(-1, k, d)
        factor = samples.factor.reshape(-1, d, d)
        rng = np.random.default_rng(seed)
        m = len(beta)
        lags = np.broadcast_to(history[-LAGS:][::-1], (m, LAGS, d)).copy()
        paths = lags.copy()
        matrices = beta[:, 1:].reshape(m, LAGS, d, d)
        horizon = 6
        offsets = exponents(pd.period_range(pd.Period(origin, "M") + 1, periods=horizon, freq="M"))
        variance = np.ones((m, horizon))
        if scale_future:
            for j, offset in enumerate(offsets):
                if offset >= 0:
                    variance[:, j] = samples.gamma.ravel() * DECAY**offset
        impulse = [np.broadcast_to(np.eye(d), (m, d, d))]
        for h in range(horizon):
            for state in (lags, paths):
                next_value = beta[:, 0] + np.einsum("mli,mlij->mj", state, matrices)
                if state is paths:
                    shock = np.einsum("mij,mj->mi", factor, rng.standard_normal((m, d)))
                    next_value += np.sqrt(variance[:, h, None]) * shock
                state[:, 1:] = state[:, :-1]
                state[:, 0] = next_value
            if h + 1 < horizon:
                impulse.append(sum(impulse[h - j] @ matrices[:, j] for j in range(h + 1)))
        var = np.zeros(m)
        for j, response in enumerate(impulse):
            loading = np.einsum("mi,mij->mj", response[:, :, EMP], factor)
            var += variance[:, horizon - 1 - j] * np.sum(loading**2, axis=1)

        def matrix(x):
            return x.reshape(chains, draws)

        return cls(
            matrix(lags[:, 0, EMP] - history[-1, EMP]),
            matrix(np.sqrt(var)),
            float(np.exp(history[-1, EMP] / 100)),
            matrix(paths[:, 0, EMP] - history[-1, EMP]),
        )

    def summary(self):
        def quantile(q):
            return brentq(
                lambda x: ndtr((x - self.location) / self.sd).mean() - q,
                np.min(self.location - 12 * self.sd),
                np.max(self.location + 12 * self.sd),
            )

        quantiles = {
            label: quantile(q)
            for label, q in [
                ("p025", 0.025),
                ("p05", 0.05),
                ("median", 0.5),
                ("p95", 0.95),
                ("p975", 0.975),
            ]
        }
        mean_jobs = self.base * np.expm1(self.location / 100 + 0.5 * (self.sd / 100) ** 2)
        return dict(
            mean=float(mean_jobs.mean()),
            **{k: self.base * np.expm1(v / 100) for k, v in quantiles.items()},
            median_log=quantiles["median"],
            mean_log=self.location.mean(),
            p05_log=quantiles["p05"],
            p95_log=quantiles["p95"],
            probability_decline=ndtr(-self.location / self.sd).mean(),
            path_mean=self.base * np.expm1(self.simulated_growth / 100).mean(),
            chain_mean_se=mean_jobs.mean(axis=1).std(ddof=1) / np.sqrt(len(mean_jobs)),
        )

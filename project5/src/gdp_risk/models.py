"""Quarterly information sets and finite-horizon predictive distributions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from bayes_policy.data import RegressionData
from bayes_policy.models import GaussianRegression
from bayes_policy.priors import GammaPrecisionPrior, NormalPrior
from lag_uncertainty.reference import IntegratedGaussianReference
from scipy.optimize import brentq
from scipy.special import ndtr


def read_quarters(path, name):
    frame = pd.read_csv(path)
    if list(frame) != ["observation_date", name]:
        raise ValueError(f"Expected observation_date and {name}")
    dates = pd.DatetimeIndex(pd.to_datetime(frame.observation_date))
    index = dates.to_period("Q")
    if (
        not index.equals(pd.period_range(index[0], index[-1], freq="Q"))
        or not (dates == index.to_timestamp()).all()
    ):
        raise ValueError("Expected one chronological observation per quarter start")
    return pd.Series(pd.to_numeric(frame[name]).to_numpy(), index=index, name=name)


@dataclass
class QuarterlyPanel:
    level: pd.Series
    spread: pd.Series

    @classmethod
    def from_files(cls, gdp, spread):
        level = read_quarters(gdp, "GDPC1")
        x = read_quarters(spread, "T10Y2Y").reindex(level.index)
        if not np.isfinite(level).all() or (level <= 0).any() or not np.isfinite(x).all():
            raise ValueError("Need positive GDP and finite spreads on every GDP quarter")
        return cls(100 * np.log(level), x)

    def window(self, origin, adjusted=False):
        if origin not in self.level.index:
            raise ValueError("Origin outside panel")
        y = self.level.loc[:origin].copy()
        if adjusted:
            if origin < pd.Period("2022Q1", freq="Q"):
                raise ValueError("Interpolation endpoint unavailable at origin")
            y.loc["2020Q1":"2021Q4"] = np.linspace(y.loc["2019Q4"], y.loc["2022Q1"], 10)[1:-1]
        return y, self.spread.loc[:origin]


class GrowthModel:
    """An unrestricted level AR(4) written in changes for interpretable priors."""

    def __init__(self, y, spread, kind):
        if kind not in {"AR", "ARDL", "Drift"}:
            raise ValueError("Unknown forecast specification")
        if not y.index.equals(spread.index) or len(y) < 16:
            raise ValueError("Need aligned GDP and spread history")
        self.kind = kind
        self.level = y.to_numpy()
        self.spread = spread.to_numpy()
        self.n = len(y)
        change = y.diff()
        features = {"intercept": np.ones(self.n)}
        if kind != "Drift":
            features.update(trend=np.arange(self.n) / 100, level=(y.shift(1) - y.iloc[0]) / 100)
            features.update({f"change_{j}": change.shift(j) for j in range(1, 4)})
        if kind == "ARDL":
            features.update({f"spread_{j}": spread.shift(j) for j in range(1, 5)})
        x = pd.DataFrame(features, index=y.index).iloc[4:]
        data = RegressionData(
            change.iloc[4:].to_numpy(), x.to_numpy(), tuple(x), x.index.to_timestamp()
        )
        sd = np.array([2 if name in {"intercept", "trend", "level"} else 0.5 for name in x])
        self.model = GaussianRegression(
            data, NormalPrior(np.zeros(len(x.columns)), np.diag(sd**2)), GammaPrecisionPrior(2, 1)
        )

    def next_row(self):
        if self.kind == "Drift":
            return np.ones(1)
        row = np.r_[
            1, self.n / 100, (self.level[-1] - self.level[0]) / 100, np.diff(self.level)[-3:][::-1]
        ]
        return np.r_[row, self.spread[-4:][::-1]] if self.kind == "ARDL" else row

    def paths(self, beta, variance, horizon, rng):
        """Keep each parameter draw fixed and generate independent step innovations."""
        if horizon < 1 or (self.kind == "ARDL" and horizon != 1):
            raise ValueError("ARDL needs an explicit future spread path beyond one quarter")
        lag = np.tile(np.diff(self.level)[-3:][::-1], (len(beta), 1))
        growth = np.zeros(len(beta))
        for step in range(horizon):
            mean = beta[:, 0].copy()
            if self.kind != "Drift":
                mean += beta[:, 1] * (self.n + step) / 100
                mean += beta[:, 2] * (self.level[-1] + growth - self.level[0]) / 100
                mean += np.einsum("ij,ij->i", beta[:, 3:6], lag)
            if self.kind == "ARDL":
                mean += beta[:, 6:] @ self.spread[-4:][::-1]
            change = mean + rng.standard_normal(len(beta)) * np.sqrt(variance)
            growth += change
            lag = np.column_stack((change, lag[:, :2]))
        return growth


class PredictiveLaw:
    def __init__(self, spec):
        self.spec = spec
        self.reference = IntegratedGaussianReference(spec.model)

    def parameters(self, n, rng):
        """Sample normalized log-precision nodes, then Gaussian coefficients by SVD."""
        r = self.reference
        h = np.exp(rng.choice(r.z, size=n, p=r.masses))[:, None]
        denom = 1 + h * r.singular_values**2
        white = h * r.singular_values * r.projected / denom + rng.standard_normal(
            (n, len(r.singular_values))
        ) / np.sqrt(denom)
        return r.model.prior.mean + (white @ r.right_vectors) @ r.prior_cholesky.T, 1 / h[:, 0]

    def normal_mixture(self, horizon):
        if horizon != 1 and self.spec.kind != "Drift":
            raise ValueError("Analytic mixture available at one step or for the drift model")
        r = self.reference
        mu, var = r.contrast_parameters(self.spec.next_row())
        mu = mu * horizon
        sd = np.sqrt(var * horizon**2 + horizon * np.exp(-r.z))

        def cdf(x):
            return float(r.masses @ ndtr((x - mu) / sd))

        low, high = (mu - 12 * sd).min(), (mu + 12 * sd).max()
        q = [brentq(lambda x: cdf(x) - p, low, high) for p in (0.05, 0.5, 0.95)]
        return dict(mean=float(r.masses @ mu), p05=q[0], median=q[1], p95=q[2], p_negative=cdf(0))


def empirical(values):
    lo, mid, hi = np.quantile(values, [0.05, 0.5, 0.95])
    return dict(
        mean=float(values.mean()), p05=lo, median=mid, p95=hi, p_negative=float((values < 0).mean())
    )

"""Bayesian VAR draws and long-run technology identification in quarterly growth."""

import numpy as np
import pandas as pd
from payroll_risk.data import lag_design
from payroll_risk.model import covariance_draw
from scipy.linalg import cho_solve, cholesky, solve_triangular


def identify(beta, sigma):
    """Return the signed impact matrix, long-run matrix, stability and conditioning."""
    lag = beta[1:].reshape(4, 2, 2).transpose(0, 2, 1)
    companion = np.zeros((8, 8))
    companion[:2] = np.concatenate(lag, axis=1)
    companion[2:, :-2] = np.eye(6)
    radius = max(abs(np.linalg.eigvals(companion)))
    a = np.eye(2) - lag.sum(axis=0)
    condition = np.linalg.cond(a)
    if radius >= 1 or condition > 1e8:
        return None, radius, condition
    psi = np.linalg.inv(a)
    longrun = np.linalg.cholesky(psi @ sigma @ psi.T)
    impact = a @ longrun
    impulse = np.vstack([impact, np.zeros((6, 2))])
    responses = []
    total = np.zeros((2, 2))
    for _ in range(12):
        total = total + impulse[:2]
        responses.append(total[1, 0])
        impulse = companion @ impulse
    return (np.asarray(responses), impact, psi @ impact), radius, condition


class TechnologyVAR:
    def __init__(self, levels, end, tightness=0.5):
        q = levels.loc[:end]
        growth = np.log(q).diff().dropna()
        x, y, _ = lag_design(growth.to_numpy(), 4)
        first = growth.index[4:]
        selected = first >= pd.Period("1948Q1", "Q")
        x, y = x[selected], y[selected]
        self.start = str(first[selected][0])
        self.end = end
        self.n = len(y)
        sd = np.sqrt(np.mean((y - x @ np.linalg.lstsq(x, y, rcond=None)[0]) ** 2, axis=0))
        # Independent normal coefficients; intercept variance is ten times residual variance.
        variance = np.vstack(
            [
                10 * sd**2,
                *[(tightness / k) ** 2 * (sd[None, :] / sd[:, None]) ** 2 for k in range(1, 5)],
            ]
        )
        self.prior = np.diag(1 / variance.T.ravel())
        self.prior_scale = np.diag(sd**2)
        self.x = x
        self.y = y
        self.xx = x.T @ x
        self.xy = x.T @ y

    def sample(self, seed, warmup=400, draws=1200, chains=4):
        summaries = []
        diagnostics = []
        parameters = []
        for child in np.random.SeedSequence(seed).spawn(chains):
            rng = np.random.default_rng(child)
            beta = np.zeros((9, 2))
            outcomes = np.full((draws, 12), np.nan)
            reduced = np.empty((draws, 18))
            stats = []
            for j in range(draws + warmup):
                resid = self.y - self.x @ beta
                L = covariance_draw((self.prior_scale + resid.T @ resid)[None], self.n + 4, rng)[0]
                sigma = L @ L.T
                prec = np.linalg.inv(sigma)
                ch = cholesky(self.prior + np.kron(prec, self.xx), lower=True, check_finite=False)
                mu = cho_solve((ch, True), (self.xy @ prec).T.ravel(), check_finite=False)
                beta = (
                    (mu + solve_triangular(ch.T, rng.normal(size=18), check_finite=False))
                    .reshape(2, 9)
                    .T
                )
                if j < warmup:
                    continue
                t = j - warmup
                reduced[t] = beta.T.ravel()
                out, radius, cond = identify(beta, sigma)
                if out is None:
                    stats.append([radius, cond, np.nan, np.nan])
                    continue
                ir, impact, longrun = out
                outcomes[t] = ir
                stats.append(
                    [radius, cond, np.max(abs(impact @ impact.T - sigma)), abs(longrun[0, 1])]
                )
            summaries.append(outcomes)
            parameters.append(reduced)
            diagnostics.extend(stats)
        return np.stack(summaries), np.array(diagnostics), np.stack(parameters)

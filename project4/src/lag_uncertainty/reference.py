"""SVD-based coefficient integration, adaptive precision quadrature, and scalar controls."""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ndtr
from scipy.stats import gamma


class IntegratedGaussianReference:
    def __init__(self, model, node_count=160):
        self.model = model
        self.n = len(model.data.y)
        self.prior_cholesky = np.linalg.cholesky(model.prior.covariance)
        a = model.data.X @ self.prior_cholesky
        u, self.singular_values, self.right_vectors = np.linalg.svd(a, full_matrices=False)
        residual = model.data.y - model.data.X @ model.prior.mean
        self.projected = u.T @ residual
        self.orthogonal_squared = max(
            0.0, float(residual @ residual - self.projected @ self.projected)
        )
        mode = minimize_scalar(lambda z: -self.log_kernel(z), bounds=(-25, 25), method="bounded")
        if not mode.success:
            raise RuntimeError("Could not locate the integrated precision mode")
        self.center, self.peak = float(mode.x), float(-mode.fun)
        self.lower, self.upper = self.center - 2, self.center + 2
        while self.log_kernel(self.lower) > self.peak - 80:
            self.lower -= 2
            if self.lower < -100:
                raise RuntimeError("Precision lower tail did not decay")
        while self.log_kernel(self.upper) > self.peak - 80:
            self.upper += 2
            if self.upper > 100:
                raise RuntimeError("Precision upper tail did not decay")
        integral, error = quad(
            lambda z: np.exp(self.log_kernel(z) - self.peak),
            self.lower,
            self.upper,
            epsabs=1e-12,
            epsrel=1e-11,
            limit=200,
        )
        if integral <= 0 or not np.isfinite(integral):
            raise RuntimeError("Integrated evidence failed")
        self.log_evidence = float(self.peak + np.log(integral))
        self.relative_quadrature_error = float(error / integral)
        self._set_nodes(node_count)
        if self.node_mass_error > 1e-8:
            self._set_nodes(node_count * 2)
        if self.node_mass_error > 1e-8:
            raise RuntimeError("Posterior precision nodes failed normalization control")

    def log_kernel(self, z):
        h = np.exp(z)
        denominator = 1 + h * self.singular_values**2
        determinant = -self.n * z + np.log(denominator).sum()
        quadratic = h * (self.orthogonal_squared + np.sum(self.projected**2 / denominator))
        prior = self.model.precision_prior
        return float(
            -0.5 * (self.n * np.log(2 * np.pi) + determinant + quadratic)
            + gamma.logpdf(h, prior.shape, scale=1 / prior.rate)
            + z
        )

    def _set_nodes(self, node_count):
        x, weights = leggauss(node_count)
        self.z = (self.upper - self.lower) * x / 2 + (self.upper + self.lower) / 2
        masses = (
            weights
            * (self.upper - self.lower)
            / 2
            * np.exp(np.array([self.log_kernel(z) for z in self.z]) - self.log_evidence)
        )
        self.node_mass_error = abs(float(masses.sum()) - 1)
        self.masses = masses / masses.sum()
        self.node_count = node_count

    def contrast_parameters(self, contrast):
        contrast = np.asarray(contrast, dtype=float)
        if contrast.shape != self.model.prior.mean.shape or not np.isfinite(contrast).all():
            raise ValueError("Contrast dimension must match coefficients")
        h = np.exp(self.z)[:, None]
        denominator = 1 + h * self.singular_values**2
        q = self.right_vectors @ (self.prior_cholesky.T @ contrast)
        means = contrast @ self.model.prior.mean + np.sum(
            q * (h * self.singular_values * self.projected / denominator), axis=1
        )
        variances = np.sum(q * q / denominator, axis=1)
        return means, variances

    def contrast_cdf(self, value, contrast):
        means, variances = self.contrast_parameters(contrast)
        return float(self.masses @ ndtr((value - means) / np.sqrt(variances)))

    def contrast_survival(self, value, contrast):
        means, variances = self.contrast_parameters(contrast)
        return float(self.masses @ ndtr((means - value) / np.sqrt(variances)))

    def contrast_density(self, grid, contrast):
        means, variances = self.contrast_parameters(contrast)
        centered = np.asarray(grid)[:, None] - means
        densities = np.exp(-0.5 * centered**2 / variances) / np.sqrt(2 * np.pi * variances)
        return densities @ self.masses

    def contrast_summary(self, contrast):
        means, variances = self.contrast_parameters(contrast)
        mean = float(self.masses @ means)
        variance = float(self.masses @ (variances + (means - mean) ** 2))
        lower = float(means.min() - 12 * np.sqrt(variances.max()))
        upper = float(means.max() + 12 * np.sqrt(variances.max()))
        quantiles = [
            brentq(lambda x: self.contrast_cdf(x, contrast) - p, lower, upper)
            for p in (0.05, 0.5, 0.95)
        ]
        return {
            "mean": mean,
            "sd": np.sqrt(variance),
            "p05": quantiles[0],
            "median": quantiles[1],
            "p95": quantiles[2],
            "probability_sum_above_one": self.contrast_survival(1.0, contrast),
        }


def integrated_mixture_summary(references, weights):
    weights = np.asarray(weights, dtype=float)
    if (
        weights.shape != (len(references),)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("Need finite nonnegative weights matching references")
    weights = weights / weights.sum()
    contrasts = [np.r_[0.0, np.ones(len(r.model.prior.mean) - 1)] for r in references]
    components = [r.contrast_summary(c) for r, c in zip(references, contrasts)]
    means = np.array([s["mean"] for s in components])
    variances = np.array([s["sd"] ** 2 for s in components])
    mean = float(weights @ means)
    within = float(weights @ variances)
    between = float(weights @ (means - mean) ** 2)
    lower = min(s["mean"] - 15 * s["sd"] for s in components)
    upper = max(s["mean"] + 15 * s["sd"] for s in components)

    def cdf(value):
        return sum(w * r.contrast_cdf(value, c) for w, r, c in zip(weights, references, contrasts))

    quantiles = [brentq(lambda value: cdf(value) - p, lower, upper) for p in (0.05, 0.5, 0.95)]
    return {
        "mean": mean,
        "sd": np.sqrt(within + between),
        "p05": quantiles[0],
        "median": quantiles[1],
        "p95": quantiles[2],
        "within_model_variance": within,
        "between_model_variance": between,
        "between_model_variance_share": between / (within + between),
        "probability_sum_above_one": float(
            weights @ np.array([s["probability_sum_above_one"] for s in components])
        ),
    }

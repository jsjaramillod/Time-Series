"""Explicit scalar mixtures, preserving model uncertainty and zero-weight cases."""

import numpy as np


class ModelAveragedInference:
    def __init__(self, samples, weights, labels=None):
        self.samples = tuple(np.asarray(s, dtype=float).reshape(-1) for s in samples)
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or len(weights) != len(self.samples) or not len(weights):
            raise ValueError("Need one weight per component")
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Weights must be finite, nonnegative, and have positive total")
        if any(not len(s) or not np.isfinite(s).all() for s in self.samples):
            raise ValueError("Components must contain finite scalar draws")
        self.weights = weights / weights.sum()
        self.labels = tuple(range(len(weights))) if labels is None else tuple(labels)
        if len(self.labels) != len(weights) or len(set(self.labels)) != len(weights):
            raise ValueError("Labels must identify components uniquely")

    def cdf(self, value):
        if not np.isfinite(value):
            raise ValueError("CDF evaluation point must be finite")
        return float(sum(w * np.mean(s <= value) for s, w in zip(self.samples, self.weights)))

    @staticmethod
    def condition_weights(weights, event_probabilities):
        weights = np.asarray(weights, dtype=float)
        probabilities = np.asarray(event_probabilities, dtype=float)
        if weights.ndim != 1 or probabilities.shape != weights.shape or not len(weights):
            raise ValueError("One event probability is required per model weight")
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Model weights must be finite and nonnegative with positive total")
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError("Event probabilities must lie in [0, 1]")
        joint = weights * probabilities
        if joint.sum() <= 0:
            raise ValueError("Conditioning event has zero estimated probability")
        return joint / joint.sum()

    def quantiles(self, probabilities=(0.05, 0.5, 0.95)):
        probabilities = np.asarray(probabilities, dtype=float)
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError("Quantile probabilities must be in [0, 1]")
        active = [(s, w) for s, w in zip(self.samples, self.weights) if w > 0]
        values = np.concatenate([s for s, _ in active])
        masses = np.concatenate([np.repeat(w / len(s), len(s)) for s, w in active])
        order = np.argsort(values, kind="stable")
        cumulative = np.cumsum(masses[order])
        cumulative /= cumulative[-1]
        positions = np.searchsorted(cumulative, probabilities, side="left")
        return values[order[np.minimum(positions, len(order) - 1)]]

    def summary(self):
        means = np.array([s.mean() for s in self.samples])
        variances = np.array([s.var(ddof=0) for s in self.samples])
        mean = float(self.weights @ means)
        within = float(self.weights @ variances)
        between = float(self.weights @ (means - mean) ** 2)
        lower, median, upper = self.quantiles()
        return {
            "mean": mean,
            "median": float(median),
            "sd": np.sqrt(within + between),
            "p05": float(lower),
            "p95": float(upper),
            "within_model_variance": within,
            "between_model_variance": between,
            "between_model_variance_share": between / (within + between)
            if within + between > 0
            else 0.0,
            "probability_sum_above_one": float(
                sum(w * np.mean(s > 1) for s, w in zip(self.samples, self.weights))
            ),
        }

    def draw(self, size, seed):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("Size must be a positive integer")
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.weights), p=self.weights, size=size)
        draws = np.empty(size)
        for component, values in enumerate(self.samples):
            mask = indices == component
            draws[mask] = values[rng.integers(len(values), size=int(mask.sum()))]
        return draws, np.array(self.labels)[indices]

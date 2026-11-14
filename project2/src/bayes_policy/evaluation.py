"""Forecast origins, future targets, and label availability are separate objects."""

from dataclasses import replace

import arviz as az
import numpy as np
import pandas as pd

from .data import RecessionData
from .models import BayesianProbit
from .priors import NormalPrior
from .samplers import ProbitGibbsSampler, SamplerConfig


def probability_scores(y, probability) -> dict:
    y, p = np.asarray(y, dtype=float), np.asarray(probability, dtype=float)
    if y.shape != p.shape or y.size == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("Scores need matching, nonempty binary outcomes and probabilities")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must lie in [0, 1]")
    safe = np.clip(p, 1e-12, 1 - 1e-12)
    return {
        "n": int(y.size),
        "brier": float(np.mean((y - p) ** 2)),
        "negative_log_score": float(-np.mean(y * np.log(safe) + (1 - y) * np.log1p(-safe))),
        "log_score_clip": 1e-12,
    }


class ExpandingWindowEvaluator:
    def __init__(
        self,
        config: SamplerConfig,
        prior_variance: float = 10,
        horizon: int = 12,
        label_delay_months: int = 0,
    ):
        if horizon < 1 or label_delay_months < 0:
            raise ValueError("Horizon must be positive and assumed label delay nonnegative")
        self.config = config
        self.prior_variance = prior_variance
        self.horizon = horizon
        self.label_delay_months = label_delay_months

    def evaluate(self, data: RecessionData, origins, label_calendar: pd.Series | None = None):
        rows, results = [], {}
        pairs, features = data.pairs(self.horizon), data.features()
        for offset, requested_origin in enumerate(origins):
            origin = pd.Period(requested_origin, freq="M").to_timestamp(how="end").normalize()
            if origin not in features.index:
                raise ValueError(f"No predictor row at requested origin {origin.date()}")
            training, timing = data.training_at(
                origin, self.horizon, self.label_delay_months, label_calendar
            )
            model = BayesianProbit(
                training, NormalPrior.isotropic(len(training.names), self.prior_variance)
            )
            result = ProbitGibbsSampler(replace(self.config, seed=self.config.seed + offset)).fit(
                model
            )
            p = result.probability_draws(features.loc[origin].to_numpy())[:, :, 0]
            target_date = (origin.to_period("M") + self.horizon).to_timestamp(how="end").normalize()
            # Beta(1,1) posterior-predictive frequency benchmark on the SAME training labels.
            benchmark = (training.y.sum() + 1) / (len(training.y) + 2)
            gate = result.diagnostic_gate()
            rows.append(
                {
                    **timing,
                    "target_date": str(target_date.date()),
                    "actual": float(pairs.loc[origin, "target"]),
                    "probability": float(p.mean()),
                    "probability_p05": float(np.quantile(p, 0.05)),
                    "probability_p95": float(np.quantile(p, 0.95)),
                    "probability_mcse": float(az.mcse(p, method="mean").item()),
                    "benchmark_probability": float(benchmark),
                    "diagnostics_passed": gate["passed"],
                    "max_rhat": gate["max_rhat"],
                    "min_bulk_ess": gate["min_bulk_ess"],
                    "min_tail_ess": gate["min_tail_ess"],
                }
            )
            results[str(origin.date())] = result
        frame = pd.DataFrame(rows)
        observed = frame.actual.notna()
        scores = {
            "evaluation_type": "pseudo-real-time",
            "observed_targets": int(observed.sum()),
            "all_diagnostics_passed": bool(frame.diagnostics_passed.all()),
            "label_delay_months": self.label_delay_months,
            "limitations": [
                "revised predictor data; historical vintages unavailable",
                "label availability assumed unless an explicit calendar was supplied",
                "overlapping horizons and a short window limit inference",
            ],
        }
        if observed.any():
            scores["probit"] = probability_scores(
                frame.loc[observed, "actual"], frame.loc[observed, "probability"]
            )
            scores["benchmark"] = probability_scores(
                frame.loc[observed, "actual"], frame.loc[observed, "benchmark_probability"]
            )
        return frame, scores, results

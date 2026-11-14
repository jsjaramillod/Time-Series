import numpy as np
import pandas as pd
import pytest

from bayes_policy.evaluation import ExpandingWindowEvaluator, probability_scores
from bayes_policy.results import PosteriorResult
from bayes_policy.samplers import ProbitGibbsSampler, SamplerConfig


def test_proper_scores_and_bounds():
    scores = probability_scores([0, 1], [0.2, 0.8])
    assert scores["brier"] == pytest.approx(0.04)
    assert scores["negative_log_score"] == pytest.approx(-np.log(0.8))
    assert np.isfinite(probability_scores([1], [0])["negative_log_score"])
    with pytest.raises(ValueError):
        probability_scores([1], [1.1])


def test_evaluator_uses_origin_predictor_and_only_mature_labels(recession, monkeypatch):
    seen = []
    forecast_rows = []
    original_prediction = PosteriorResult.probability_draws

    def prediction(self, X):
        forecast_rows.append(X.copy())
        return original_prediction(self, X)

    monkeypatch.setattr(PosteriorResult, "probability_draws", prediction)

    def fit(self, model):
        seen.append(model.data)
        draws = np.random.default_rng(99).normal(scale=0.01, size=(4, 1000, 4))
        return PosteriorResult(draws, model.data.names, {})

    monkeypatch.setattr(ProbitGibbsSampler, "fit", fit)
    evaluator = ExpandingWindowEvaluator(SamplerConfig(draws=1000), label_delay_months=6)
    frame, scores, _ = evaluator.evaluate(recession, ["2007-01", "2009-12"])
    assert frame.loc[0, "last_training_target"] == "2006-07-31"
    assert seen[0].dates[-1] == pd.Timestamp("2005-07-31")
    assert frame.loc[1, "target_date"] == "2010-12-31"
    assert np.isnan(frame.loc[1, "actual"])
    assert scores["observed_targets"] == 1
    assert scores["evaluation_type"] == "pseudo-real-time"
    np.testing.assert_allclose(forecast_rows[0], recession.features().loc["2007-01-31"].to_numpy())
    np.testing.assert_allclose(forecast_rows[1], recession.features().loc["2009-12-31"].to_numpy())

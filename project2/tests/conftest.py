import numpy as np
import pandas as pd
import pytest

from bayes_policy.data import InterestRateData, RecessionData, RegressionData
from bayes_policy.models import GaussianRegression
from bayes_policy.priors import GammaPrecisionPrior, NormalPrior


@pytest.fixture
def policy():
    n = 16
    return InterestRateData(
        pd.DataFrame(
            {
                "DATE": pd.date_range("2000-01-01", periods=n, freq="QS"),
                "FF": np.linspace(1, 5, n),
                "PI": np.linspace(1, 3, n),
                "PI_Target": 2.0,
                "GDP": np.exp(np.linspace(-0.04, 0.04, n)) * 100,
                "GDP_POT": 100.0,
            }
        )
    )


@pytest.fixture
def recession():
    dates = pd.date_range("1990-01-01", periods=240, freq="MS")
    t = np.arange(len(dates))
    return RecessionData(
        pd.DataFrame(
            {
                "Date": dates.strftime("%Y%m").astype(int),
                "NBER": (t % 31 < 6).astype(int),
                "FederalFundsRate": 3 + np.sin(t / 12),
                "SP500": 100 * np.exp(t * 0.01),
                "TenYearBondYield": 5 + np.cos(t / 6),
                "ThreeMonthBillYield": 3.0,
            }
        )
    )


@pytest.fixture
def gaussian_model():
    rng = np.random.default_rng(100)
    X = np.column_stack([np.ones(100), rng.normal(size=100)])
    y = X @ np.array([0.5, -0.8]) + rng.normal(scale=0.6, size=100)
    data = RegressionData(y, X, ("intercept", "slope"), pd.date_range("2000-01-01", periods=100))
    return GaussianRegression(
        data, NormalPrior.isotropic(2, 10), GammaPrecisionPrior.from_reference()
    )

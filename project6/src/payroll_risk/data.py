"""Workbook contract, training-only transformations and nested information sets."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

CHECKSUM = "daf1da34b9f1da01d184b6867c6f37d62ee8b849e8d9abc613fc731199a95093"
COLUMNS = ["EMP", "IP", "CPI", "SP500", "TenYearBond", "ThreeMonthBill"]
MODELS = {
    "AR": ["EMP"],
    "Macro": ["EMP", "IP", "CPI"],
    "Full": ["spread", "SP500", "EMP", "IP", "CPI"],
}


@dataclass
class PayrollPanel:
    raw: pd.DataFrame

    @classmethod
    def load(cls, filename):
        path = Path(filename)
        if sha256(path.read_bytes()).hexdigest() != CHECKSUM:
            raise ValueError("Workbook differs from the frozen study input")
        frame = pd.read_excel(path, index_col="Date")[COLUMNS]
        expected = pd.date_range("1962-01-01", "2026-03-01", freq="MS")
        gaps = frame.isna().stack()
        if not frame.index.equals(expected) or list(gaps[gaps].index) != [
            (pd.Timestamp("2025-10-01"), "CPI")
        ]:
            raise ValueError("Unexpected monthly coverage or missing cells")
        return cls(frame)

    def design(self, origin, model, smooth=False):
        raw = self.raw.loc[: pd.Period(origin, "M").start_time].copy()
        if raw.index[-1] != pd.Period(origin, "M").start_time:
            raise ValueError("Origin is outside the available panel")
        gap = pd.Timestamp("2025-10-01")
        if gap in raw.index and pd.isna(raw.at[gap, "CPI"]) and model != "AR":
            if raw.index[-1] < pd.Timestamp("2025-11-01"):
                raise ValueError("November CPI is unavailable at this origin")
            raw.at[gap, "CPI"] = np.sqrt(raw.at["2025-09-01", "CPI"] * raw.at["2025-11-01", "CPI"])
        if smooth:
            if raw.index[-1] < pd.Timestamp("2022-02-01"):
                raise ValueError("Endpoint of the retrospective interpolation is unavailable")
            raw.loc["2019-12":"2022-01", :] = np.nan
            raw = raw.interpolate(method="time")
        raw["spread"] = raw["TenYearBond"] - raw["ThreeMonthBill"]
        for key in ["EMP", "IP", "CPI", "SP500"]:
            raw[key] = np.log(raw[key])
        values = raw[MODELS[model]].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("Non-finite model input")
        return values


def lag_design(values, lags):
    n = len(values)
    if n <= lags:
        raise ValueError("Insufficient observations")
    x = np.column_stack(
        [np.ones(n - lags), *[values[lags - j : n - j] for j in range(1, lags + 1)]]
    )
    following = np.r_[1.0, values[-lags:][::-1].ravel()]
    return x, values[lags:], following

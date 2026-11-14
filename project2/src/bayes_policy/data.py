"""Validated data adapters. Input workbooks remain external to the public package."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


def file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class RegressionData:
    y: np.ndarray
    X: np.ndarray
    names: tuple[str, ...]
    dates: pd.DatetimeIndex

    def __post_init__(self):
        y, X = np.array(self.y, dtype=float, copy=True), np.array(self.X, dtype=float, copy=True)
        dates = pd.DatetimeIndex(self.dates)
        if y.ndim != 1 or X.ndim != 2 or X.shape != (len(y), len(self.names)):
            raise ValueError("Incompatible response, design, or parameter names")
        if len(y) < 2 or len(dates) != len(y) or len(set(self.names)) != len(self.names):
            raise ValueError("Need at least two observations, matching dates, and unique names")
        if not np.isfinite(y).all() or not np.isfinite(X).all():
            raise ValueError("Response and design must be finite")
        if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError("Dates must be unique, chronological, and nonmissing")
        y.setflags(write=False)
        X.setflags(write=False)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "dates", dates)

    def fingerprint(self) -> str:
        """Bind posterior draws to the exact response, design, dates, and column order."""
        digest = sha256()
        for values in [self.y, self.X, self.dates.asi8]:
            digest.update(np.ascontiguousarray(values).tobytes())
        digest.update("|".join(self.names).encode())
        return digest.hexdigest()


def _regular_dates(values, frequency: str) -> pd.DatetimeIndex:
    if len(values) < 2:
        raise ValueError("Need at least two dated observations")
    # YYYYMM integers must not be interpreted as nanoseconds since the Unix epoch.
    if pd.api.types.is_numeric_dtype(values):
        dates = pd.DatetimeIndex(pd.to_datetime(values.astype("Int64").astype(str), format="%Y%m"))
    else:
        text = values.astype(str)
        if text.str.fullmatch(r"\d{6}").all():
            dates = pd.DatetimeIndex(pd.to_datetime(text, format="%Y%m"))
        else:
            dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    periods = dates.to_period(frequency)
    if dates.hasnans or not periods.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be chronological with exactly one row per period")
    if not np.array_equal(periods.asi8, np.arange(periods.asi8[0], periods.asi8[-1] + 1)):
        raise ValueError("Missing periods: positional lags would be invalid")
    return dates


def _numeric(frame: pd.DataFrame, columns: list[str]):
    absent = set(columns) - set(frame)
    if absent:
        raise ValueError(f"Missing columns: {sorted(absent)}")
    values = frame[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Input contains missing or nonfinite values; no silent imputation")
    return values


class InterestRateData:
    """Quarterly policy data with explicit inflation and output-gap transformations."""

    def __init__(self, frame: pd.DataFrame, source_hash: str | None = None):
        required = ["FF", "PI", "PI_Target", "GDP", "GDP_POT"]
        numeric = _numeric(frame, required)
        self.dates = _regular_dates(frame["DATE"], "Q")
        if (numeric[["GDP", "GDP_POT"]].to_numpy() <= 0).any():
            raise ValueError("GDP levels must be strictly positive before logs")
        self._frame = frame.copy(deep=True)
        self._frame[required] = numeric
        self.source_hash = source_hash

    @classmethod
    def from_excel(cls, path: str | Path):
        path = Path(path)
        return cls(pd.read_excel(path), file_digest(path))

    def design(self, asymmetric: bool = False) -> RegressionData:
        d = self._frame
        inflation = (d.PI - d.PI_Target).to_numpy()[1:]
        gap = (100 * (np.log(d.GDP) - np.log(d.GDP_POT))).to_numpy()[1:]
        columns = [np.ones(len(gap)), inflation]
        names = ["intercept", "inflation_gap"]
        if asymmetric:
            columns += [np.maximum(gap, 0), np.minimum(gap, 0)]
            names += ["output_gap_positive", "output_gap_negative"]
        else:
            columns += [gap]
            names += ["output_gap"]
        columns += [d.FF.to_numpy()[:-1]]
        names += ["lagged_rate"]
        return RegressionData(
            d.FF.to_numpy()[1:], np.column_stack(columns), tuple(names), self.dates[1:]
        )

    def manifest(self) -> dict:
        design = self.design()
        return {
            "input": "int_rate_rule_data.xlsx",
            "sha256": self.source_hash,
            "raw_rows": len(self._frame),
            "raw_start": str(self.dates[0].date()),
            "raw_end": str(self.dates[-1].date()),
            "effective_n": len(design.y),
            "effective_start": str(design.dates[0].date()),
            "effective_end": str(design.dates[-1].date()),
            "frequency": "quarterly",
            "inflation_gap": "PI - PI_Target, percentage points as supplied",
            "output_gap": "100 * (log(GDP) - log(GDP_POT))",
            "rate": "FF as supplied, percentage points; one quarterly lag",
            "missing_handling": "reject; no interpolation or imputation",
            "vintage": "unknown; local teaching workbook, not a historical-vintage dataset",
            "trend_inflation": "PI_Target supplied; construction not independently reproduced",
            "redistribution": "not established; source workbook excluded from public files",
        }


class RecessionData:
    """Monthly features kept apart from future targets, including the final forecast row."""

    names = ("intercept", "federal_funds_rate", "equity_return_3m", "term_spread")

    def __init__(self, frame: pd.DataFrame, source_hash: str | None = None):
        columns = ["NBER", "FederalFundsRate", "SP500", "TenYearBondYield", "ThreeMonthBillYield"]
        values = _numeric(frame, columns)
        dates = _regular_dates(frame["Date"], "M")
        if len(frame) < 16 or not np.isin(values.NBER, [0, 1]).all():
            raise ValueError("Need at least 16 monthly rows and binary NBER labels")
        if (values.SP500 <= 0).any():
            raise ValueError("Equity index must be positive")
        self._frame = values.set_axis(dates.to_period("M").to_timestamp(how="end").normalize())
        self.source_hash = source_hash

    @classmethod
    def from_excel(cls, path: str | Path):
        path = Path(path)
        return cls(pd.read_excel(path), file_digest(path))

    def features(self) -> pd.DataFrame:
        d = self._frame
        return pd.DataFrame(
            {
                "intercept": 1.0,
                "federal_funds_rate": d.FederalFundsRate,
                "equity_return_3m": 100 * (d.SP500 / d.SP500.shift(3) - 1),
                "term_spread": d.TenYearBondYield - d.ThreeMonthBillYield,
            }
        ).iloc[3:]

    def pairs(self, horizon: int = 12) -> pd.DataFrame:
        if horizon < 1:
            raise ValueError("Horizon must be positive")
        features = self.features()
        target_periods = features.index.to_period("M") + horizon
        target_dates = target_periods.to_timestamp(how="end").normalize()
        pairs = features.copy()
        pairs["target_date"] = target_dates
        pairs["target"] = self._frame.NBER.reindex(target_dates).to_numpy()
        return pairs

    def manifest(self) -> dict:
        return {
            "input": "Recession_Forecast_Data.xlsx",
            "sha256": self.source_hash,
            "raw_rows": len(self._frame),
            "start": str(self._frame.index[0].date()),
            "end": str(self._frame.index[-1].date()),
            "frequency": "monthly",
            "equity_return": "100*(SP500_t/SP500_(t-3)-1), simple percent return",
            "spread": "TenYearBondYield - ThreeMonthBillYield, percentage points",
            "missing_handling": "reject; no imputation",
            "vintage": "unknown",
            "redistribution": "source workbook excluded; rights not established",
        }

    def training_at(
        self,
        origin,
        horizon: int = 12,
        label_delay_months: int = 0,
        label_calendar: pd.Series | None = None,
    ) -> tuple[RegressionData, dict]:
        """Revised-data pseudo-real-time unless a verified availability calendar is supplied.

        A fixed delay is a declared sensitivity assumption, never an NBER release calendar.
        Predictor vintages/publication times remain unverified even with a label calendar.
        """
        if label_delay_months < 0:
            raise ValueError("Label delay cannot be negative")
        origin = pd.Period(origin, freq="M").to_timestamp(how="end").normalize()
        pairs = self.pairs(horizon)
        if label_calendar is None:
            periods = pd.DatetimeIndex(pairs.target_date).to_period("M") + label_delay_months
            available = pd.Series(periods.to_timestamp(how="end").normalize(), index=pairs.index)
            policy = f"assumed {label_delay_months}-month label delay; retrospective labels"
        else:
            if not label_calendar.index.is_unique:
                raise ValueError("Label calendar must have unique target dates")
            available = pd.Series(
                pd.to_datetime(label_calendar.reindex(pairs.target_date).to_numpy()),
                index=pairs.index,
            )
            if available[pairs.target.notna()].isna().any():
                raise ValueError("Label calendar does not cover all observed targets")
            if (available < pairs.target_date).any():
                raise ValueError("A target cannot be available before its observation month")
            policy = "provided label-availability calendar; predictor vintages still unverified"
        keep = pairs.target.notna() & (pairs.target_date <= origin) & (available <= origin)
        train = pairs.loc[keep]
        data = RegressionData(
            train.target.to_numpy(), train[list(self.names)].to_numpy(), self.names, train.index
        )
        return data, {
            "origin": str(origin.date()),
            "last_training_predictor": str(train.index[-1].date()),
            "last_training_target": str(train.target_date.iloc[-1].date()),
            "latest_label_availability": str(available.loc[keep].max().date()),
            "training_n": len(train),
            "horizon_months": horizon,
            "label_policy": policy,
            "evaluation_type": "pseudo-real-time",
        }

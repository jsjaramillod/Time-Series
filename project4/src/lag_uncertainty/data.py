"""Headerless series and common conditional target windows; no invented calendar."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IndexedARData:
    y: np.ndarray
    X: np.ndarray
    names: tuple[str, ...]
    observation_indices: np.ndarray

    def __post_init__(self):
        y, x = np.array(self.y, dtype=float, copy=True), np.array(self.X, dtype=float, copy=True)
        indices = np.array(self.observation_indices, copy=True)
        if y.ndim != 1 or x.shape != (len(y), len(self.names)) or len(y) < 2:
            raise ValueError("Response and design dimensions disagree")
        if indices.shape != y.shape or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Target labels must be integer observation positions")
        if not np.all(np.diff(indices) == 1) or len(set(self.names)) != len(self.names):
            raise ValueError("Target positions must be consecutive and names unique")
        if not np.isfinite(y).all() or not np.isfinite(x).all():
            raise ValueError("Data must be finite; no silent imputation")
        for name, value in (("y", y), ("X", x), ("observation_indices", indices)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def dates(self):
        """Index compatibility with the shared regression engine; labels are ordinal."""
        return pd.Index(self.observation_indices, name="observation")

    def fingerprint(self):
        digest = sha256()
        for value in (self.y, self.X, self.observation_indices):
            digest.update(np.ascontiguousarray(value).tobytes())
        digest.update("|".join(self.names).encode())
        return digest.hexdigest()


class ARDesignBuilder:
    def __init__(self, values, maximum_lag=8, source_name=None, source_hash=None):
        y = np.asarray(values, dtype=float)
        if isinstance(maximum_lag, bool) or not isinstance(maximum_lag, int) or maximum_lag < 1:
            raise ValueError("Maximum lag must be a positive integer")
        if y.ndim != 1 or len(y) <= maximum_lag + 1 or not np.isfinite(y).all():
            raise ValueError("Need a finite one-dimensional series and at least two targets")
        self.values = y.copy()
        self.values.setflags(write=False)
        self.maximum_lag = maximum_lag
        self.source_name, self.source_hash = source_name, source_hash

    @classmethod
    def from_csv(cls, path, expected_observations=200, maximum_lag=8):
        path = Path(path)
        if any(not line.strip() for line in path.read_text().splitlines()):
            raise ValueError("Blank observations are not silently removed")
        frame = pd.read_csv(path, header=None, skip_blank_lines=False)
        if frame.shape != (expected_observations, 1):
            raise ValueError(f"Expected {expected_observations} rows in one headerless column")
        return cls(
            pd.to_numeric(frame.iloc[:, 0], errors="raise").to_numpy(),
            maximum_lag,
            path.name,
            sha256(path.read_bytes()).hexdigest(),
        )

    def design(self, order):
        if (
            isinstance(order, bool)
            or not isinstance(order, int)
            or not 1 <= order <= self.maximum_lag
        ):
            raise ValueError("Order must be between one and the common conditioning lag")
        n, start = len(self.values), self.maximum_lag
        columns = [np.ones(n - start)] + [
            self.values[start - j : n - j] for j in range(1, order + 1)
        ]
        return IndexedARData(
            self.values[start:],
            np.column_stack(columns),
            ("intercept",) + tuple(f"lag_{j}" for j in range(1, order + 1)),
            np.arange(start + 1, n + 1),
        )

    def manifest(self):
        return {
            "source_file": self.source_name,
            "source_sha256": self.source_hash,
            "header": None,
            "observations": len(self.values),
            "conditioning_observations": self.maximum_lag,
            "modeled_observations": len(self.values) - self.maximum_lag,
            "target_positions_inclusive": [self.maximum_lag + 1, len(self.values)],
            "index_type": "one-based ordinal observation positions; no calendar assigned",
            "sample_mean": float(self.values.mean()),
            "sample_sd_ddof1": float(self.values.std(ddof=1)),
            "data_generation_parameters": "unverified",
            "generation_seed": "unverified",
            "provenance": "user-supplied series; original generation and redistribution rights unverified",
        }

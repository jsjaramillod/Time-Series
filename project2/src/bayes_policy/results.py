"""Named, chain-preserving draws with diagnostics supplied by ArviZ."""

import json
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import ndtr


@dataclass(frozen=True)
class PosteriorResult:
    draws: np.ndarray
    names: tuple[str, ...]
    metadata: dict

    def __post_init__(self):
        draws = np.array(self.draws, dtype=float, copy=True)
        if draws.ndim != 3 or draws.shape[2] != len(self.names) or draws.shape[0] < 2:
            raise ValueError("Draws must have shape (at least 2 chains, draws, parameters)")
        if not np.isfinite(draws).all() or len(set(self.names)) != len(self.names):
            raise ValueError("Draws must be finite and parameter names unique")
        draws.setflags(write=False)
        object.__setattr__(self, "draws", draws)

    @property
    def coefficients(self):
        return self.draws[:, :, : len(self.names) - int("sigma2" in self.names)]

    def parameter(self, name: str) -> np.ndarray:
        return self.draws[:, :, self.names.index(name)]

    def summary(self) -> pd.DataFrame:
        idata = az.from_dict(posterior={name: self.parameter(name) for name in self.names})
        diagnostics = az.summary(idata, kind="diagnostics", round_to="none")
        flat = self.draws.reshape(-1, len(self.names))
        summary = pd.DataFrame(
            {
                "mean": flat.mean(axis=0),
                "median": np.median(flat, axis=0),
                "sd": flat.std(axis=0, ddof=1),
                "p05": np.quantile(flat, 0.05, axis=0),
                "p95": np.quantile(flat, 0.95, axis=0),
            },
            index=pd.Index(self.names, name="parameter"),
        )
        return summary.join(diagnostics)

    def diagnostic_gate(self, max_rhat: float = 1.01, min_ess: float = 400) -> dict:
        summary = self.summary()
        passed = (
            np.isfinite(summary[["r_hat", "ess_bulk", "ess_tail"]]).all().all()
            and (summary.r_hat <= max_rhat).all()
            and (summary.ess_bulk >= min_ess).all()
            and (summary.ess_tail >= min_ess).all()
        )
        return {
            "passed": bool(passed),
            "max_rhat": float(summary.r_hat.max()),
            "min_bulk_ess": float(summary.ess_bulk.min()),
            "min_tail_ess": float(summary.ess_tail.min()),
            "thresholds": {"rhat_max": max_rhat, "ess_min": min_ess},
        }

    def probability_draws(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.coefficients.shape[2] or not np.isfinite(X).all():
            raise ValueError("Forecast design must be finite and match coefficient dimension")
        return ndtr(np.einsum("cdk,nk->cdn", self.coefficients, X))

    def save(self, path: str | Path):
        np.savez_compressed(
            path, draws=self.draws, names=np.array(self.names), metadata=json.dumps(self.metadata)
        )

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["draws"], tuple(data["names"].tolist()), json.loads(str(data["metadata"]))
            )

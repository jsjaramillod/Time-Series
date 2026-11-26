"""A one-factor stationary state-space model with joint latent-path simulation."""

import numpy as np
from scipy.linalg import solve_banded

NAMES = ["CMRMTSPL", "INDPRO", "PAYEMS", "W875RX1"]
SPECIAL = np.array([311, 312, 323, 324, 455, 456, 551, 552]) - 1


def activity(frame, treatment):
    levels = frame.set_index("OBSERVATION_MONTH")[NAMES]
    growth = 100 * np.log(levels).diff().iloc[1:].copy()
    if not np.isfinite(growth).all().all() or len(growth) != 710:
        raise ValueError("Unexpected macro input")
    if treatment not in ("Coursework", "Named series", "Observed"):
        raise ValueError(treatment)
    if treatment != "Observed":
        column = "INDPRO" if treatment == "Coursework" else "W875RX1"
        growth.loc[growth.index[SPECIAL], column] = growth[column].mean()
        growth.iloc[637:662] = np.broadcast_to(growth.mean().to_numpy(), (25, 4))
    return (growth - growth.mean()) / growth.std(ddof=1)


class DynamicFactor:
    def __init__(self, frame):
        self.values = frame.to_numpy()
        self.dates = frame.index
        self.names = list(frame)
        if self.values.shape[1] != 4 or not np.isfinite(self.values).all():
            raise ValueError("Four finite series are required")

    @staticmethod
    def path(values, phi, loading, var, rng):
        """Scalar Kalman filtering and Carter–Kohn backward sampling in banded form."""
        n = len(values)
        info = np.sum(loading**2 / var)
        signal = values @ (loading / var)
        if n == 1:
            variance = 1 / (1 - phi**2 + info)
            return variance * signal + np.sqrt(variance) * rng.standard_normal(1)
        filtered = np.empty(n)
        p = 1 / (1 - phi**2)
        for i in range(n):
            filtered[i] = p / (1 + p * info)
            if i and filtered[i] == filtered[i - 1]:
                filtered[i:] = filtered[i]
                break
            p = 1 + phi**2 * filtered[i]
        pred = np.r_[1 / (1 - phi**2), 1 + phi**2 * filtered[:-1]]
        lower = np.ones((2, n))
        lower[1, :-1] = -phi * filtered[1:] / pred[1:]
        m = solve_banded((1, 0), lower, filtered * signal, check_finite=False)
        gain = phi * filtered[:-1] / pred[1:]
        offsets = m.copy()
        offsets[:-1] *= 1 - gain * phi
        std = np.sqrt(np.r_[filtered[:-1] / pred[1:], filtered[-1]])
        upper = np.ones((2, n))
        upper[0, 1:] = -gain
        return solve_banded(
            (0, 1), upper, offsets + std * rng.standard_normal(n)[::-1], check_finite=False
        )

    def sample(self, seed, warmup, draws, legacy=False, chain=0, scale_move=False):
        rng = np.random.default_rng(seed)
        y = self.values
        phi = 0.0 if legacy else (-0.55, 0.15, 0.5, 0.8)[chain]
        loading = np.full(4, 0.5)
        var = np.full(4, 0.2)
        params = np.empty((draws, 9))
        paths = np.empty((draws, len(y)), dtype=np.float32)
        for i in range(warmup + draws):
            f = self.path(y, phi, loading, var, rng)
            v = 1 / (1 + (f @ f) / var)
            loading = rng.normal(v * (f @ y) / var, np.sqrt(v))
            if loading[0] < 0:
                loading = -loading
                f = -f
            residual = y - f[:, None] * loading
            var = 1 / rng.gamma((len(y) + 2) / 2, 2 / (2 + (residual**2).sum(axis=0)))
            v = 1 / (1 + f[:-1] @ f[:-1])
            loc = v * (f[:-1] @ f[1:])
            new = rng.normal(loc, np.sqrt(v))
            while abs(new) >= 1:
                new = rng.normal(loc, np.sqrt(v))
            # The stationary density of the first state also depends on phi.
            if legacy or np.log(rng.uniform()) < 0.5 * (
                np.log1p(-(new**2)) - np.log1p(-(phi**2)) + (new**2 - phi**2) * f[0] ** 2
            ):
                phi = new
            if scale_move:
                # A joint scale move preserves every fitted observation. Include the
                # stationary path density, loading prior and transformation Jacobian.
                step = rng.normal(0, 1.4 / np.sqrt(len(f)))
                innovations = f[1:] - phi * f[:-1]
                energy = (1 - phi**2) * f[0] ** 2 + innovations @ innovations
                ratio = (len(f) - len(loading)) * step - 0.5 * (
                    np.expm1(2 * step) * energy + np.expm1(-2 * step) * (loading @ loading)
                )
                if np.log(rng.uniform()) < ratio:
                    f *= np.exp(step)
                    loading *= np.exp(-step)
            if i >= warmup:
                params[i - warmup] = np.r_[phi, loading, var]
                paths[i - warmup] = f
        return params, paths

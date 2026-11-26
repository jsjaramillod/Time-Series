"""Origin-by-origin forecast comparison on a 400-row panel."""

import numpy as np
import pandas as pd

COMPONENTS = (0, 1, 2, 4, 8, 16)
PENALTIES = (1, 5, 15, 50, 150, 500, 1500, 5000)


class WidePanel:
    def __init__(self, frame):
        cols = ["DATE", "Y", *[f"X{i}" for i in range(1, 101)]]
        if (
            list(frame) != cols
            or frame.DATE.tolist() != list(range(1, 401))
            or not np.isfinite(frame.to_numpy()).all()
        ):
            raise ValueError("Expected 400 indexed rows with a target and 100 finite predictors")
        self.y = frame.Y.to_numpy()
        self.x = frame.iloc[:, 2:].to_numpy()
        self.lags = np.column_stack([self.y[3 - lag : len(frame) - 1 - lag] for lag in range(4)])

    def predict(self, t):
        """t observations available; the first unavailable target has index t."""
        if not 106 <= t < len(self.y):
            raise ValueError("Invalid expanding forecast origin")
        y = self.y[4:t]
        lags = self.lags[: t - 4]
        future = self.lags[t - 4]
        history = self.x[:t]
        centered = history - history.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        factors = centered @ vt[: max(COMPONENTS)].T
        ans = {}
        for k in COMPONENTS:
            z = np.column_stack([np.ones(len(y)), lags, factors[3:-1, :k]])
            name = f"PCA {k}" if k else "AR(4)"
            ans[name] = np.r_[1, future, factors[-1, :k]] @ np.linalg.lstsq(z, y, rcond=None)[0]
        x = np.column_stack([lags, history[3:-1]])
        row = np.r_[future, history[-1]]
        avg = x.mean(axis=0)
        sd = x.std(axis=0, ddof=1)
        u, s, v = np.linalg.svd((x - avg) / sd, full_matrices=False)
        basis = ((row - avg) / sd @ v.T) * (u.T @ (y - y.mean()))
        raw = x - avg
        raw_beta = np.linalg.solve(raw.T @ raw + 15 * np.eye(raw.shape[1]), raw.T @ (y - y.mean()))
        ans["Ridge 15 raw"] = y.mean() + (row - avg) @ raw_beta
        ans["OLS"] = y.mean() + np.sum(basis / s)
        for penalty in PENALTIES:
            ans[f"Ridge {penalty}"] = y.mean() + np.sum(basis * s / (s * s + penalty))
        return ans

    def evaluate(self):
        records = []
        for t in range(200, 400):
            records.append(dict(origin=t, target=t + 1, actual=self.y[t], **self.predict(t)))
        all_rows = pd.DataFrame(records)
        candidates = [
            "AR(4)",
            *[f"PCA {k}" for k in COMPONENTS if k],
            *[f"Ridge {p}" for p in PENALTIES],
        ]
        validation = all_rows.iloc[:100]
        losses = (
            (validation[candidates].sub(validation.actual, axis=0) ** 2)
            .mean()
            .pow(0.5)
            .rename("validation_rmse")
        )
        chosen = losses.idxmin()
        scores = []
        test = all_rows.iloc[100:].copy()
        test["Selected"] = test[chosen]
        methods = ["OLS", "Ridge 15 raw", "Ridge 15", "PCA 2", "AR(4)", "Selected"]
        for label, sample in [
            ("All", test),
            ("301-350", test.iloc[:50]),
            ("351-400", test.iloc[50:]),
        ]:
            for m in methods:
                e = sample[m] - sample.actual
                scores.append(
                    dict(
                        window=label,
                        method=m,
                        n=len(sample),
                        rmse=np.sqrt(np.mean(e**2)),
                        mae=np.mean(abs(e)),
                        bias=e.mean(),
                    )
                )
        return (
            all_rows.iloc[:100],
            losses.rename_axis("candidate").reset_index(),
            test[list(dict.fromkeys(["origin", "target", "actual", *methods, chosen]))],
            pd.DataFrame(scores),
        )

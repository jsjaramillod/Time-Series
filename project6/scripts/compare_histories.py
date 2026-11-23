"""Score the original smoothing choice at every eligible historical payroll origin."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from payroll_risk.data import PayrollPanel
from payroll_risk.study import forecast


def compare(workbook, root):
    panel = PayrollPanel.load(workbook)
    history = pd.read_csv(root / "results/forecasts.csv")
    matched = history.loc[
        (history.model == "Full") & (history.intensity == 0.5) & (history.origin >= "2022-02"),
        ["origin", "target", "issue_date", "first_release", "mean", "p05", "p95", "p_loss"],
    ].copy()
    frames = [matched.assign(training="observed")]
    rows = []
    for row in matched.itertuples():
        values = panel.design(row.origin, "Full", smooth=True)
        estimate, _ = forecast((row.origin, "Full", 0.5, "interpolated", values, False))
        rows.append(
            dict(
                origin=row.origin,
                target=row.target,
                issue_date=row.issue_date,
                first_release=row.first_release,
                training="interpolated",
                **{k: estimate[k] for k in ["mean", "p05", "p95", "p_loss"]},
            )
        )
    frames.append(pd.DataFrame(rows))
    forecasts = pd.concat(frames, ignore_index=True)
    forecasts.to_csv(root / "results/matched_histories.csv", index=False, float_format="%.12g")
    return forecasts


def scores(forecasts):
    rows = []
    for treatment, group in forecasts.groupby("training", sort=False):
        errors = group["mean"] - group.first_release
        rows.append(
            dict(
                training=treatment,
                months=len(group),
                declines=int((group.first_release < 0).sum()),
                mae=errors.abs().mean(),
                rmse=np.sqrt(np.mean(errors**2)),
                coverage=group.first_release.between(group.p05, group.p95).mean(),
                width=(group.p95 - group.p05).mean(),
                brier=((group.p_loss - (group.first_release < 0).astype(float)) ** 2).mean(),
            )
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(scores(compare(args.workbook, args.root)).to_string(index=False))

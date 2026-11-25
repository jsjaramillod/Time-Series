"""Supplementary point forecast: repeat the gain of the previous six months."""

import argparse
from pathlib import Path

import pandas as pd
from payroll_risk.data import PayrollPanel

ROOT = Path(__file__).resolve().parents[1]


def calculate(workbook):
    payroll = PayrollPanel.load(workbook).raw.EMP
    payroll.index = payroll.index.to_period("M")
    files = ROOT / "results"
    origins = pd.concat(
        [
            pd.read_csv(files / name).drop_duplicates("origin")
            for name in ("forecasts.csv", "reference.csv")
        ],
        ignore_index=True,
    )
    records = []
    for row in origins.itertuples():
        origin = pd.Period(row.origin, "M")
        old = payroll[origin - 6]
        current = payroll[origin]
        records.append(
            dict(
                origin=row.origin,
                target=row.target,
                issue_date=row.issue_date,
                previous_level=old,
                origin_level=current,
                prediction=current - old,
                revised=row.revised,
                first_change_sum=row.first_change_sum,
                reference_gain=row.bls_gain,
            )
        )
    return pd.DataFrame(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    args = parser.parse_args()
    calculate(args.workbook).to_csv(
        ROOT / "results/persistence.csv", index=False, float_format="%.12g"
    )

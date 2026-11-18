"""Check the supplied quarterly spread against FRED's available daily observations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("daily", type=Path)
    parser.add_argument("quarterly", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    def series(path):
        return pd.read_csv(path, index_col="observation_date", parse_dates=True)["T10Y2Y"]

    daily = series(args.daily)
    quarterly = series(args.quarterly).dropna()
    quarterly.index = quarterly.index.to_period("Q")
    mean = daily.groupby(daily.index.to_period("Q")).mean().reindex(quarterly.index)
    difference = (mean - quarterly).abs()
    if not np.isfinite(difference).all() or difference.max() > 0.00500001:
        raise ValueError("Quarterly spread disagrees with rounded daily mean")
    record = dict(
        daily_url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y&cosd=1984-01-01&coed=2026-03-31",
        daily_sha256=hashlib.sha256(args.daily.read_bytes()).hexdigest(),
        quarterly_sha256=hashlib.sha256(args.quarterly.read_bytes()).hexdigest(),
        quarters=len(mean),
        max_abs_difference_pp=round(float(difference.max()), 10),
        permitted_rounding_pp=0.005,
    )
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Align regular G.17 releases and the BLS employment and CPI release calendars."""

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


def release_dates(path):
    return pd.to_datetime(re.findall(r"^(\d{4}-\d{2}-\d{2})$", path.read_text(), re.M))


def build(source):
    payroll = release_dates(source / "release-50.txt")
    cpi = release_dates(source / "release-10.txt")
    ip = {}
    for month, release in re.findall(
        r"^(\w+ \d{4})\s+(\d{1,2}-[A-Za-z]+-\d{4})\s*$", (source / "fed.txt").read_text(), re.M
    ):
        ip[pd.Period(month, freq="M")] = pd.Timestamp(datetime.strptime(release, "%d-%B-%Y"))
    records = []
    targets = [*pd.period_range("2007-01", "2024-12", freq="M"), pd.Period("2026-04", "M")]
    for target in targets:

        def monthly(dates, month):
            return dates[dates.to_period("M") == month]

        origin_release = monthly(payroll, target).min()
        target_release = monthly(payroll, target + 1).min()
        # Extra February dates are annual seasonal-factor updates before January CPI.
        cpi_release = monthly(cpi, target).max()
        issue = max(origin_release, cpi_release, ip[target])
        lead = (target_release - issue).days
        if not 0 < lead < 45:
            raise ValueError(f"Invalid lead for {target}")
        records.append(
            [
                str(target - 1),
                str(target),
                *[
                    d.date()
                    for d in [origin_release, cpi_release, ip[target], issue, target_release]
                ],
                lead,
            ]
        )
    return pd.DataFrame(
        records,
        columns=[
            "origin",
            "target",
            "payroll_release",
            "cpi_release",
            "ip_release",
            "issue_date",
            "target_release",
            "lead_days",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source).to_csv(args.output, index=False)

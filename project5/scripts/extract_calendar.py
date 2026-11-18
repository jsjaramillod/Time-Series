"""Extract the earliest GDP publication within each quarter's BEA vintage table."""

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import openpyxl


def extract(workbook, output):
    sheet = openpyxl.load_workbook(workbook, read_only=True, data_only=True)["Vintage History"]
    by_quarter = {}
    quarter = None
    for title, vintage, _, _, growth, _, release in sheet.values:
        if re.fullmatch(r"\d{4}Q[1-4]", str(title)):
            quarter = title
        if not quarter or not vintage or not release or not re.search(r"\d", str(growth)):
            continue
        match = re.search(r"[A-Za-z]{3,9} \d{1,2},? +\d{4}", str(release))
        if not match:
            raise ValueError(f"Unrecognized BEA date: {release}")
        date_text = match[0].replace(",", "").split()
        date_text[0] = date_text[0][:3]
        date = datetime.strptime(" ".join(date_text), "%b %d %Y").date()
        by_quarter.setdefault(quarter, []).append((date.isoformat(), vintage))
    with Path(output).open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["quarter", "issue_date", "release_label"])
        for quarter, values in sorted(by_quarter.items()):
            if "2003Q4" <= quarter <= "2026Q1":
                writer.writerow([quarter, *min(values)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook")
    parser.add_argument("output")
    args = parser.parse_args()
    extract(args.workbook, args.output)

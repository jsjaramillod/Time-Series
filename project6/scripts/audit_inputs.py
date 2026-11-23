"""Cross-check the private equity series and the two public monthly Treasury series."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import pandas as pd

from payroll_risk.data import CHECKSUM, PayrollPanel


def audit(panel_file, equity_file, yields_file):
    panel = PayrollPanel.load(panel_file).raw
    equity = pd.read_excel(equity_file, index_col=0).iloc[:, 0].rename("SP500")
    yields = pd.read_csv(yields_file, index_col=0, parse_dates=True)
    external = yields.rename(columns={"GS10": "TenYearBond", "TB3MS": "ThreeMonthBill"}).join(
        equity
    )
    cols = ["SP500", "TenYearBond", "ThreeMonthBill"]
    aligned = external.reindex(panel.index)[cols]
    if aligned.isna().any().any():
        raise ValueError("Reference files do not cover the monthly sample")
    delta = (panel[cols] - aligned).abs().max().to_dict()
    if max(delta.values()) > 1e-9:
        raise ValueError(delta)
    return dict(
        workbook_sha256=CHECKSUM,
        equity_workbook_sha256=sha256(Path(equity_file).read_bytes()).hexdigest(),
        matched_months=len(panel),
        maximum_absolute_differences=delta,
        october_2025_imputed_cpi=float(
            (panel.at["2025-09-01", "CPI"] * panel.at["2025-11-01", "CPI"]) ** 0.5
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["workbook", "equity", "yields", "output"]:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(audit(args.workbook, args.equity, args.yields), indent=2) + "\n"
    )

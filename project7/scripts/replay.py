"""Repeat all three-way comparisons with new seeds and retain independent scorecards."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from payroll_volatility.study import run

ROOT = Path(__file__).resolve().parents[1]
REPLAY_SEED = 73191


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("message", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    with TemporaryDirectory() as stage:
        out = Path(stage)
        run(
            args.workbook,
            args.message,
            ROOT.parent / "project6/data",
            ROOT / "data",
            out,
            args.workers,
            REPLAY_SEED,
        )
        for name in ["scores", "reference"]:
            table = pd.read_csv(out / f"{name}.csv")
            table["base_seed"] = REPLAY_SEED
            table.to_csv(ROOT / f"results/replay_{name}.csv", index=False, float_format="%.12g")


if __name__ == "__main__":
    main()

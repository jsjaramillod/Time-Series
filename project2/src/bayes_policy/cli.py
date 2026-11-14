"""Command-line entry point with explicit external data paths and deterministic seeds."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .pipeline import run_policy, run_recession
from .samplers import SamplerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", choices=["policy", "recession"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--draws", type=int, default=10000, help="Retained draws per chain")
    parser.add_argument("--warmup", type=int, default=3000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--start", default="2007-01")
    parser.add_argument("--end", default="2007-12")
    parser.add_argument("--label-delay-months", type=int, default=0)
    parser.add_argument("--label-calendar", type=Path, help="CSV: target_date,label_available_date")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Input workbook does not exist: {args.data}")
    config = SamplerConfig(args.draws, args.warmup, args.chains, args.seed)
    if args.analysis == "policy":
        result = run_policy(args.data, args.output, config)
        passed = (
            result["sampler_diagnostics_passed"] and result["independent_evidence_controls_passed"]
        )
    else:
        calendar = None
        if args.label_calendar:
            frame = pd.read_csv(
                args.label_calendar, parse_dates=["target_date", "label_available_date"]
            )
            calendar = frame.set_index("target_date").label_available_date
        result = run_recession(
            args.data, args.output, config, args.start, args.end, args.label_delay_months, calendar
        )
        passed = result["all_diagnostics_passed"]
    print(json.dumps(result, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

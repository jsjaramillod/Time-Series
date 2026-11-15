import argparse

from .pipeline import run_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Known-parameter persistence and shock experiments"
    )
    parser.add_argument("--output", default="results")
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--replications", type=int, default=10000)
    args = parser.parse_args()
    run_experiments(args.output, args.seed, args.replications)


if __name__ == "__main__":
    main()

import argparse

from .study import LagOrderStudy


def main():
    parser = argparse.ArgumentParser(
        description="Common-window Bayesian AR lag comparison and model averaging"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=3000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    from bayes_policy.samplers import SamplerConfig

    LagOrderStudy(
        args.data, args.output, SamplerConfig(args.draws, args.warmup, args.chains, args.seed)
    ).run()


if __name__ == "__main__":
    main()

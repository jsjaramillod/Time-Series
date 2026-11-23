"""Read recorded public estimates and render the English research report."""

import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path

from report_style import ResearchReport


def build_report(root, report_date):
    results = root / "results"

    def load(name):
        return json.loads((results / name).read_text())

    def rows(name):
        with (results / name).open() as stream:
            return list(csv.DictReader(stream))

    def f(value, n=4):
        return f"{float(value):.{n}f}"

    def pct(value, n=2):
        return f"{100 * float(value):.{n}f}%"

    summary = load("study_summary.json")
    groups = summary["groups"]
    primary = next(g for g in groups if g["coefficient_variance"] == 0.5)
    bma, ar8, selected = primary["bma"], primary["ar8"], primary["map"]
    comparison = rows("model_comparison.csv")
    response = rows("impulse_responses.csv")
    parameter = rows("posterior_parameters.csv")
    derived = rows("posterior_derived.csv")
    controls = rows("scalar_integration_controls.csv")
    stability = rows("stability.csv")
    points = rows("evidence_point_checks.csv")
    sensitivity = rows("additional_prior_sensitivity.csv")
    metadata = load("run_metadata.json")
    verification = load("verification.json")
    validation = load("validation.json")
    assert validation["all_passed"] and validation["fits_count"] == 24
    assert (
        verification["replayed_fits"] == 24 and verification["all_retained_draw_arrays_exact_match"]
    )
    maximum_evidence_error = max(float(r["absolute_chib_integration_error"]) for r in comparison)
    maximum_integral_error = max(
        float(r["svd_vs_conditional_integration_error"]) for r in comparison
    )
    maximum_point_error = max(float(r["absolute_error_vs_quadrature"]) for r in points)
    maximum_rhat = max(float(r["r_hat"]) for r in parameter + derived)
    minimum_bulk = min(float(r["ess_bulk"]) for r in parameter + derived)
    minimum_tail = min(float(r["ess_tail"]) for r in parameter + derived)

    def primary_response(target, horizon):
        return next(
            r
            for r in response
            if float(r["coefficient_variance"]) == 0.5
            and r["target"] == target
            and int(r["horizon"]) == horizon
        )

    def prior_weight(variance, order):
        return next(
            r
            for r in comparison
            if float(r["coefficient_variance"]) == variance and int(r["order"]) == order
        )

    report = ResearchReport(
        "Lag Uncertainty and Bayesian Model Averaging",
        "Persistence and Shock Transmission under Uncertain AR Order",
        report_date,
        results / "figures",
    )
    p, h, table, figure, page = report.p, report.h, report.table, report.figure, report.page

    h("Executive summary")
    p(
        "My doctoral assignment averaged eight lag orders. The two-lag model received 71.54% posterior probability, a result the portfolio reproduces. The eight-lag coefficient sum averaged 0.8006 in class and 0.8008 under numerical integration. How strongly do the priors affect that conclusion? Across the tested specifications, the two-lag weight ranges from 60.01% to 80.96%, while model-averaged persistence stays between 0.8430 and 0.8504."
    )
    h("Model preference changes with prior concentration", 2)
    p(
        f"Under the primary coefficient prior, AR(2) receives {pct(primary['map_probability'])} posterior probability. AR(4), AR(3), and AR(5) receive {pct(prior_weight(0.5, 4)['posterior_model_probability'])}, {pct(prior_weight(0.5, 3)['posterior_model_probability'])}, and {pct(prior_weight(0.5, 5)['posterior_model_probability'])}. AR(8) receives only {pct(prior_weight(0.5, 8)['posterior_model_probability'], 4)}. AR(2) remains the leading order when the coefficient-prior variance changes from 0.25 to 1, but its weight ranges from {pct(next(g['map_probability'] for g in groups if g['coefficient_variance'] == 0.25))} to {pct(next(g['map_probability'] for g in groups if g['coefficient_variance'] == 1.0))}."
    )
    h("Averaging alters the persistence summary", 2)
    p(
        f"The primary model-averaged coefficient-sum mean is {f(bma['integration_mean'])}, with central 90% interval [{f(bma['integration_p05'])}, {f(bma['integration_p95'])}]. Fixed AR(8) gives {f(ar8['integration_mean'])}, with interval [{f(ar8['integration_p05'])}, {f(ar8['integration_p95'])}]. Between-model uncertainty contributes {pct(bma['integration_between_model_variance_share'])} of the mixture variance. The mixture is wider than the selected AR(2) distribution, yet narrower than AR(8). Accounting for specification uncertainty need not widen an interval relative to every fixed model."
    )
    h("Transmission timing depends on the specification", 2)
    first, last = primary_response("BMA", 1), primary_response("BMA", 10)
    last8 = primary_response("AR8", 10)
    p(
        f"The model-averaged one-period response to a unit innovation has median {f(first['median'], 3)} and pointwise 90% interval [{f(first['p05'], 3)}, {f(first['p95'], 3)}]. At horizon ten, its median is {f(last['median'], 3)}, compared with {f(last8['median'], 3)} under AR(8). Both horizon-ten intervals include zero. Differences in central responses therefore do not establish a robust sign reversal. These are model-implied responses in unspecified series units, not identified macroeconomic causal effects."
    )
    h("Reliable computation includes recognizing tail limits", 2)
    p(
        f"All 24 Gibbs fits pass their parameter and derived-quantity diagnostics. Baseline Chib log evidences agree with independent integration within {maximum_evidence_error:.6f} log units, and an independent replay reproduces all retained arrays in the recorded environment. Nevertheless, finite event counts do not resolve every tail. Integration assigns approximately {bma['integration_probability_sum_above_one']:.3g} probability to a coefficient sum above one; zero sampled events cannot be interpreted as zero probability."
    )

    page()
    h("1. Comparable lag orders and explicit prior choices")
    p(
        "The headerless input contains 200 ordered observations. Every candidate conditions on the initial eight observations and explains the same 192 targets, positions 9-200. Reading the first numeric row as a header would discard data. No calendar, original generation parameters, or generation seed is assigned. The comparison concerns conditional Gaussian AR(1)-AR(8) likelihoods, with the intercept included in every model."
    )
    p(
        "The primary independent priors are normal with mean zero and variance 100 for the intercept, normal with variance 0.5 for each AR coefficient, and Gamma(shape=1, rate=3) for precision. Normal coefficients are independent of precision. Equal model prior probabilities are 1/8. Variances 0.25 and 1 define separate coefficient-prior sensitivities; they are not pooled into one average."
    )
    figure(
        "lag_order_probabilities.png",
        "Figure 1. Posterior model probabilities from Chib evidence under three coefficient-prior variances. All models explain identical targets and use proper normalized parameter priors.",
        175,
    )
    table(
        "Table 1. Posterior model probabilities, in percent. Small entries are rounded.",
        ["AR order", "Variance 0.25", "Variance 0.5", "Variance 1"],
        [
            [
                str(order),
                *[
                    pct(prior_weight(v, order)["posterior_model_probability"], 4)
                    for v in (0.25, 0.5, 1.0)
                ],
            ]
            for order in range(1, 9)
        ],
        [0.16, 0.28, 0.28, 0.28],
    )
    p(
        "A more diffuse coefficient prior can reduce the evidence for larger models through the prior normalization and integrated fit. It is not a neutral change. The leading order remains AR(2), but the retained probability on other orders is substantial. Posterior model probability is a relative statement within these alternatives, not proof that an AR(2) generated the series."
    )

    page()
    h("2. Selecting a model and averaging a shared quantity")
    p(
        "The common scalar is the sum of AR coefficients, excluding the intercept. Its posterior mixture preserves within-model parameter uncertainty and uncertainty about the order. Model-specific distributions are averaged; coefficient vectors of unequal dimensions are not padded and averaged into a new fitted model [2]."
    )
    figure("model_average_equation.png", maximum_height=25)
    table(
        "Table 2. Primary coefficient-sum posteriors: integrated scalar references using Chib model weights.",
        ["Inference target", "Mean", "Posterior SD", "Central 90% interval"],
        [
            [
                label,
                f(row["integration_mean"]),
                f(row["integration_sd"]),
                f"[{f(row['integration_p05'])}, {f(row['integration_p95'])}]",
            ]
            for label, row in [
                ("Selected AR(2)", selected),
                ("Model average", bma),
                ("Fixed AR(8)", ar8),
            ]
        ],
        [0.3, 0.17, 0.18, 0.35],
    )
    figure(
        "persistence_posteriors.png",
        "Figure 2. Coefficient-sum posterior densities from scalar integration. Left: selected AR(2), model average, and fixed AR(8) under the primary prior. Right: separate prior-concentration sensitivities. Numerical summaries use the full distributions.",
        185,
    )
    figure("variance_equation.png", maximum_height=25)
    p(
        f"The primary within-model variance is {f(bma['integration_within_model_variance'], 7)} and between-model variance is {f(bma['integration_between_model_variance'], 7)}. The latter accounts for {pct(bma['integration_between_model_variance_share'])} of the total. Relative to selected AR(2), model uncertainty increases dispersion; relative to fixed AR(8), the lower-order component distributions produce a more concentrated mixture."
    )
    p(
        "Across coefficient-prior variances 0.25, 0.5 and 1, the mixture means are "
        + ", ".join(
            f(next(g["bma"]["integration_mean"] for g in groups if g["coefficient_variance"] == v))
            for v in (0.25, 0.5, 1.0)
        )
        + ". Model weights are visibly more sensitive than this shared posterior summary. The sum is useful for comparison but is not a sufficient stability diagnostic or a complete description of shock persistence."
    )

    page()
    h("3. Finite shock responses and uncertainty about transmission")
    p(
        "A one-unit innovation occurs at horizon zero, with subsequent innovations set to zero. Each retained AR coefficient draw defines a response through the coefficient recursion. Horizon-zero response is one; horizons 1-10 have marginal 90% posterior intervals. Model-averaged response distributions retain all models and draws, including unstable draws."
    )
    figure(
        "response_uncertainty.png",
        "Figure 3. Primary finite-horizon responses. Lines are posterior medians. Shading and thin boundary lines show pointwise 90% intervals for the model average and AR(8); the selected-model line is shown for comparison. The intervals are not a simultaneous band.",
        195,
    )
    table(
        "Table 3. Selected response horizons under the primary prior.",
        ["Horizon", "Target", "Median", "Pointwise 90% interval"],
        [
            [str(j), label, f(r["median"]), f"[{f(r['p05'])}, {f(r['p95'])}]"]
            for j in (1, 5, 10)
            for label, target in [("Model average", "BMA"), ("AR(8)", "AR8")]
            for r in [primary_response(target, j)]
        ],
        [0.13, 0.24, 0.18, 0.45],
    )
    p(
        f"At horizon one, the model-averaged interval [{f(first['p05'], 3)}, {f(first['p95'], 3)}] lies above the initial unit impact, supporting short-run amplification within this model set. By horizon five, its median is {f(primary_response('BMA', 5)['median'], 3)}, compared with {f(primary_response('AR8', 5)['median'], 3)} under AR(8). The difference at horizon ten is also material in the medians, while both intervals retain zero."
    )
    p(
        "Averaging a nonlinear response is different from evaluating it at an averaged coefficient vector. The reported mixture operates on the model-specific response distributions at each horizon. The exercise describes conditional dynamics in the supplied series, without a verified economic shock identification or an out-of-sample forecast comparison."
    )

    page()
    h("4. Stability and the resolution of rare posterior events")
    p(
        "Companion eigenvalues are inspected for every retained coefficient vector. Strict stability is classified using spectral radius below 1 minus a numerical margin of 1e-10. A coefficient sum below one is necessary but not sufficient in the causal real-coefficient AR setting. One primary AR(8) draw illustrates that distinction: its sum is below one while its companion representation is not classified as stable."
    )
    sample_size = metadata["sampler"]["chains"] * metadata["sampler"]["draws"]
    primary_control = {
        int(r["order"]): r for r in controls if float(r["coefficient_variance"]) == 0.5
    }
    primary_stability = {
        int(r["order"]): r for r in stability if float(r["coefficient_variance"]) == 0.5
    }
    table(
        "Table 4. Primary rare-event counts. Each model retains 40,000 draws; events in the two final columns differ.",
        [
            "Order",
            "Observed S > 1",
            "Expected S > 1 from integration",
            "Root-unstable / boundary draws",
        ],
        [
            [
                str(order),
                str(
                    round(
                        sample_size
                        * float(primary_control[order]["mcmc_sum_probability_above_one"])
                    )
                ),
                f(
                    sample_size
                    * float(primary_control[order]["integration_probability_sum_above_one"]),
                    3,
                ),
                str(
                    round(
                        sample_size
                        * float(primary_stability[order]["unstable_or_boundary_draw_proportion"])
                    )
                ),
            ]
            for order in range(1, 9)
        ],
        [0.12, 0.23, 0.33, 0.32],
    )
    ar2_tail = float(primary_control[2]["integration_probability_sum_above_one"])
    p(
        f"For AR(2), the integrated probability of S above one is approximately {ar2_tail:.3g}, implying only {f(sample_size * ar2_tail, 2)} expected events in 40,000 posterior draws. No such events occurred in the retained sample. This is compatible with a positive small probability. Increasing resampling from an empirical pool with no observed events cannot restore its missing tail information."
    )
    p(
        f"The model-averaged scalar reference probability is {bma['integration_probability_sum_above_one']:.3g}. It is calculated by integrating normal survival functions over precision, avoiding subtraction of two nearly equal probabilities. Estimated batch errors from zero event counts can appear reassuringly small, so they are not used to certify this rare probability. Mean agreement and convergence checks address different numerical questions."
    )
    p(
        "Observed stable-draw shares are extremely high, but they do not establish exact posterior stability probabilities of one. Model averaging conditional on stability would also change model probabilities: each weight must be multiplied by its model-specific stability probability and renormalized. Separate diagnostic weights implement that rule. The primary scalar and finite-response analysis remains unrestricted and removes no draws."
    )
    p(
        "The study does not report an unconditional infinite-response multiplier. Such a target would require a clearly specified stability rule and treatment of near-unit roots, distinct from the finite responses and coefficient sums considered here."
    )

    page()
    h("5. Numerical evidence and sensitivity to normalized priors")
    p(
        "Chib evidence [1] uses the likelihood, proper prior, and posterior ordinate at the same point. A Rao-Blackwell average of conditional Gamma densities estimates the precision ordinate; the conditional coefficient density is analytic. The audit also evaluates precision at 0.8 and 1.2 times the pooled point. Independent coefficient integration via SVD and adaptive integration over log precision provide controls."
    )
    table(
        "Table 5. Numerical controls across the 24 main fits.",
        ["Control", "Recorded value"],
        [
            [
                "Largest baseline Chib / integration discrepancy",
                f"{maximum_evidence_error:.6f} log units",
            ],
            [
                "Largest discrepancy across all evaluation-point checks",
                f"{maximum_point_error:.6f} log units",
            ],
            [
                "Largest SVD / conditional-matrix integral discrepancy",
                f"{maximum_integral_error:.3g} log units",
            ],
            ["Maximum rank-normalized split R-hat", f(maximum_rhat, 5)],
            [
                "Minimum bulk / tail effective sample size",
                f"{minimum_bulk:,.0f} / {minimum_tail:,.0f}",
            ],
            [
                "Scientific tests / shared-engine regression tests",
                f"{verification['project4_tests_passed']} / {verification['project2_tests_passed']}",
            ],
            [
                "Independently replayed retained arrays",
                str(verification["replayed_fits"]) + " fits: exact matches",
            ],
        ],
        [0.69, 0.31],
    )
    figure(
        "ar8_chain_traces.png",
        "Figure 4. Primary AR(8) chains for the coefficient sum and error variance. Every 50th retained draw is plotted; diagnostics and summaries use all 40,000 retained draws. Warmup comprises 3,000 iterations per chain.",
        175,
    )
    extra = [r for r in sensitivity if int(r["order"]) == 2]
    labels = {
        "primary": "Primary",
        "intercept_variance_10": "Intercept variance 10",
        "intercept_variance_1000": "Intercept variance 1000",
        "precision_mean_degree_alternative": "Gamma shape 1.5, rate 1.5",
    }
    table(
        "Table 6. Additional integration-only sensitivities at coefficient variance 0.5.",
        ["Prior setting", "AR(2) probability", "BMA sum mean", "Central 90% interval"],
        [
            [
                labels[r["scenario"]],
                pct(r["posterior_model_probability"]),
                f(r["bma_sum_mean"]),
                f"[{f(r['bma_sum_p05'])}, {f(r['bma_sum_p95'])}]",
            ]
            for r in extra
        ],
        [0.35, 0.2, 0.17, 0.28],
        compact=True,
    )
    p(
        "Proper-intercept sensitivities barely change relative model weights in this sample. The alternative Gamma convention changes the precision prior and modestly changes the weights and scalar summaries. It is kept separate from the primary shape-rate specification. Numerical convergence does not establish economic adequacy of the finite AR candidate set."
    )

    page()
    h("6. What the inference supports and what remains uncertain")
    p(
        "The main substantive result is a distinction between model preference and inference about a common quantity. The leading order remains AR(2), while its posterior probability changes materially with coefficient-prior concentration. The model-averaged coefficient-sum distribution changes less. Selecting a model and fixing a long lag order answer different conditional questions; neither substitutes for carrying specification uncertainty into the stated target."
    )
    p(
        "The model average shifts the coefficient-sum center upward relative to AR(8), adds a between-model component relative to the selected specification, and changes response timing. Both horizon-ten intervals include zero. These findings support reporting model sensitivity and uncertainty together, rather than treating a preferred order as known or treating a median response as a sign certainty."
    )
    table(
        "Table 7. Primary AR(8) parameter posterior summaries from Gibbs draws.",
        ["Parameter", "Median", "Posterior SD", "Central 90% interval"],
        [
            [
                r["parameter"]
                .replace("lag_", "Lag ")
                .replace("intercept", "Intercept")
                .replace("sigma2", "Error variance"),
                f(r["median"]),
                f(r["sd"]),
                f"[{f(r['p05'])}, {f(r['p95'])}]",
            ]
            for r in parameter
            if float(r["coefficient_variance"]) == 0.5 and int(r["order"]) == 8
        ],
        [0.3, 0.17, 0.18, 0.35],
        compact=True,
    )
    p(
        f"The Gibbs-based BMA sum mean is {f(bma['mean'], 5)}, with joint-batch MCSE {f(bma['mean_mcse_joint_batches'], 5)}. The scalar integration reference is {f(bma['integration_mean'], 5)}. These estimates target the same quantity but use different numerical representations. The batch MCSE describes simulation error in the Gibbs mixture, not the posterior SD and not the remaining error of the integrated reference."
    )
    p(
        "The supplied series has no verified generating parameters, calendar, or economic identity. The analysis conditions on eight initial observations and a restricted Gaussian AR model set. Computational checks do not demonstrate that these assumptions describe an observed economy, identify a causal effect, or improve future forecasts. Source and input hashes, explicit seeds, an executed notebook, and independent replay preserve a reproducible account of the reported calculation."
    )
    h("References", 2)
    for text in [
        "[1] Chib, S. (1995). Marginal Likelihood from the Gibbs Output. Journal of the American Statistical Association, 90, 1313-1321. DOI: 10.1080/01621459.1995.10476635.",
        "[2] Hoeting, J. A., Madigan, D., Raftery, A. E., and Volinsky, C. T. (1999). Bayesian Model Averaging: A Tutorial (with Discussion), corrected version. Statistical Science, 14, 382-401. Correction: 15, 193-195. DOI: 10.1214/ss/1009212519.",
        "[3] Vehtari, A. et al. (2021). Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC. Bayesian Analysis, 16, 667-718. DOI: 10.1214/20-BA1221.",
    ]:
        report.reference(text)

    page()
    h("Appendix. Model evidence and horizon-level response estimates")
    table(
        "Table A1. All main evidence comparisons. Chib and SVD integration use identical likelihoods and normalized priors.",
        [
            "Prior variance",
            "AR order",
            "Chib log evidence",
            "Integrated log evidence",
            "Absolute error",
            "MCSE",
        ],
        [
            [
                f(r["coefficient_variance"], 2),
                r["order"],
                f(r["log_evidence_chib"], 5),
                f(r["log_evidence_svd_quadrature"], 5),
                f(r["absolute_chib_integration_error"], 6),
                f(r["mcse_log_evidence"], 6),
            ]
            for r in comparison
        ],
        [0.12, 0.09, 0.23, 0.23, 0.17, 0.16],
        compact=True,
    )
    table(
        "Table A2. Primary pointwise response estimates for a one-unit innovation.",
        ["Horizon", "AR(8) median", "AR(8) 90% interval", "BMA median", "BMA 90% interval"],
        [
            [
                str(j),
                f(primary_response("AR8", j)["median"]),
                f"[{f(primary_response('AR8', j)['p05'])}, {f(primary_response('AR8', j)['p95'])}]",
                f(primary_response("BMA", j)["median"]),
                f"[{f(primary_response('BMA', j)['p05'])}, {f(primary_response('BMA', j)['p95'])}]",
            ]
            for j in range(1, 11)
        ],
        [0.1, 0.17, 0.28, 0.17, 0.28],
        compact=True,
    )
    p(
        "Horizon-zero response is one by normalization. Intervals are marginal at each horizon and include all draws. The CSV artifacts retain full precision, model weights, settings, diagnostics, and the separate scalar integration controls."
    )
    output = root / "docs/lag_uncertainty_model_averaging.pdf"
    report.save(output)
    inputs = [
        "study_summary.json",
        "model_comparison.csv",
        "impulse_responses.csv",
        "posterior_parameters.csv",
        "posterior_derived.csv",
        "scalar_integration_controls.csv",
        "stability.csv",
        "evidence_point_checks.csv",
        "additional_prior_sensitivity.csv",
        "run_metadata.json",
        "verification.json",
        "validation.json",
    ]
    source_paths = [results / name for name in inputs] + list((results / "figures").glob("*.png"))
    record = {
        "report_date": report_date.isoformat(),
        "output": str(output.relative_to(root)),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        "builder_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(__file__).with_name("report_style.py")]
        },
        "pdf_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (results / "report_validation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-11-23")
    args = parser.parse_args()
    build_report(Path(__file__).resolve().parents[1], date.fromisoformat(args.date))


if __name__ == "__main__":
    main()

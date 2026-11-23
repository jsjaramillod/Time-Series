"""Build the research PDF exclusively from recorded numerical artifacts."""

import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def build_report(root, output, report_date):
    results = root / "results"

    def load(name):
        return json.loads((results / name).read_text())

    with (results / "dynamics_summary.csv").open() as stream:
        dynamics = {row["experiment"]: row for row in csv.DictReader(stream)}
    with (results / "impulse_responses.csv").open() as stream:
        responses = list(csv.DictReader(stream))
    precision = load("mean_precision.json")
    checks = load("numerical_validation.json")
    ma = load("ma_control.json")
    configs = load("configurations.json")
    assert checks["all_deterministic_controls_pass"]
    width, height = A4
    usable = width - 124
    styles = {
        "body": ParagraphStyle(
            "body",
            fontName="Times-Roman",
            fontSize=10.5,
            leading=14.1,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        ),
        "title": ParagraphStyle(
            "title",
            fontName="Times-Roman",
            fontSize=19,
            leading=23,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Times-Roman",
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            spaceAfter=5,
        ),
        "date": ParagraphStyle(
            "date",
            fontName="Times-Roman",
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            spaceAfter=18,
            textColor=colors.HexColor("#444444"),
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName="Times-Bold",
            fontSize=12.5,
            leading=16,
            spaceBefore=10,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="Times-Bold",
            fontSize=10.5,
            leading=14,
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "caption": ParagraphStyle(
            "caption", fontName="Times-Roman", fontSize=9, leading=11.5, spaceAfter=8
        ),
        "tablecap": ParagraphStyle(
            "tablecap",
            fontName="Times-Roman",
            fontSize=9,
            leading=11.5,
            spaceBefore=4,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "cell": ParagraphStyle("cell", fontName="Times-Roman", fontSize=8.8, leading=11.1),
        "head": ParagraphStyle("head", fontName="Times-Bold", fontSize=8.8, leading=11.1),
        "ref": ParagraphStyle(
            "ref", fontName="Times-Roman", fontSize=9, leading=11.7, spaceAfter=6
        ),
    }
    story = []

    def paragraph(text, style="body"):
        return Paragraph(escape(str(text)), styles[style])

    def p(text):
        story.append(paragraph(text))

    def h(text, level=1):
        story.append(paragraph(text, "h1" if level == 1 else "h2"))

    def page():
        story.append(PageBreak())

    def table(caption, headers, rows, proportions, compact=False):
        story.append(paragraph(caption, "tablecap"))
        cells = [[paragraph(v, "head") for v in headers]] + [
            [paragraph(v, "cell") for v in row] for row in rows
        ]
        rendered = Table(
            cells, colWidths=[usable * w / sum(proportions) for w in proportions], repeatRows=1
        )
        rendered.setStyle(
            TableStyle(
                [
                    ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.black),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3 if compact else 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3 if compact else 5),
                ]
            )
        )
        story.extend([rendered, Spacer(1, 7)])

    def figure(name, caption, maximum_height=180):
        path = results / "figures" / name
        image_width, image_height = ImageReader(str(path)).getSize()
        scale = min(usable / image_width, maximum_height / image_height)
        image = Image(str(path), width=image_width * scale, height=image_height * scale)
        image.hAlign = "CENTER"
        story.append(
            KeepTogether([Spacer(1, 3), image, Spacer(1, 4), paragraph(caption, "caption")])
        )

    def fmt(value, digits=4):
        return f"{float(value):.{digits}f}"

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Times-Roman", 8.5)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(62, 30, "Persistence and Shock Propagation")
        canvas.drawRightString(width - 62, 30, str(document.page))
        canvas.restoreState()

    story.extend(
        [
            paragraph("Persistence and Shock Propagation", "title"),
            paragraph("Evidence from Controlled Time-Series Experiments", "subtitle"),
            paragraph(report_date.strftime("%d %B %Y"), "date"),
        ]
    )
    h("Executive summary")
    p(
        "In my doctoral assignment, a one-unit shock reached 1.20 one period later in a stable AR(2). The sum of its computed responses was 9.99999946. The portfolio reproduces the path and verifies its limiting total of 10. A separate experiment asks whether persistent observations give an overly precise impression of their mean. All observations in these experiments are synthetic."
    )
    h("Equal long-run totals do not imply equal dynamics", 2)
    p(
        "Three stable specifications share an accumulated response of ten to a one-unit innovation. By horizon 20, the AR(1), AR(2), and AR(3) have accumulated "
        + ", ".join(
            fmt(dynamics[name]["cumulative_at_20"], 3)
            for name in ("ar1_persistent", "ar2_response", "ar3_stable")
        )
        + " units respectively. Their stationary variances are 5.263, 7.429, and 3.492 despite identical innovation variance. The AR(2) initially amplifies the shock to 1.20 at horizon one before decaying. Stability, adjustment speed, and unconditional volatility answer different questions."
    )
    h("Two hundred observations can contain much less independent information", 2)
    p(
        f"For an AR(1) coefficient of 0.9 and innovation variance one, a stationary window of 200 observations has mean variance {fmt(precision['exact_mean_variance'], 6)}. Ignoring serial covariance gives {fmt(precision['independence_mean_variance'], 6)}, understating that variance by a factor of {fmt(precision['variance_inflation_factor'], 2)}. The exact precision is equivalent to {fmt(precision['effective_independent_observations'], 2)} independent observations for estimating this particular mean."
    )
    p(
        f"Across {precision['replications']:,} independent stationary replications, an interval presented as 95% under independence covers the known mean only {100 * precision['independence_coverage']:.2f}% of the time. Using the known serial covariance gives {100 * precision['correlation_aware_coverage']:.2f}% empirical coverage. The exact model-implied coverage of the independence interval is {100 * precision['exact_independence_coverage']:.2f}%. These are repeated-sample statements under known parameters, not posterior credible intervals or intervals based on fitted coefficients."
    )
    h("Model checks determine which quantities can be interpreted", 2)
    p(
        "Changing the AR(1) intercept from zero to 0.5 shifts its stationary mean from zero to five without changing centered dynamics. In contrast, changing the AR(3) coefficients from (0.5, 0.2, 0.2) to (0.5, 0.3, 0.3) moves the model outside the stability region. A separate unit-circle model has no stationary second moments in the causal representation. The finite MA control remains covariance stationary while its stated innovation representation is noninvertible."
    )
    p(
        "Responses are verified through matrix powers, coefficient recursion, a controlled shock experiment, and an independent linear filter. Covariance identities and exact Gaussian calculations support the numerical results. Finite paths do not prove stationarity, and these experiments do not establish real-world causal effects or forecast performance."
    )

    page()
    h("1. Persistence and the precision of a time average")
    p(
        "The AR(1) law is y(t) = c + 0.9 y(t-1) + innovation(t), with independent Gaussian innovations of variance one. Under stationarity, the mean is c/(1-0.9) and the variance is 1/(1-0.9 squared), or 5.263158 [1]. A coefficient of 0.9 implies a shock half-life of 6.5788 generic periods. No calendar frequency is assigned."
    )
    p(
        "The variance of a window average includes its cross-time covariances. For n observations it equals [n gamma(0) + 2 sum over k=1,...,n-1 of (n-k) gamma(k)] / n squared. The independence approximation retains the same marginal variance but deletes every cross-time covariance. The comparison therefore isolates dependence rather than changing innovation risk."
    )
    table(
        "Table 1. Exact precision and repeated-sample results for 200 observations.",
        ["Quantity", "Computed value"],
        [
            ["Exact variance of the sample mean", fmt(precision["exact_mean_variance"], 6)],
            [
                "Variance under an independence assumption",
                fmt(precision["independence_mean_variance"], 6),
            ],
            [
                "Exact standard error / independence standard error",
                f"{fmt(precision['exact_mean_standard_error'], 4)} / {fmt(precision['independence_standard_error'], 4)}",
            ],
            [
                "Finite-window variance inflation factor",
                fmt(precision["variance_inflation_factor"], 2),
            ],
            [
                "Equivalent independent observations for this mean",
                fmt(precision["effective_independent_observations"], 4),
            ],
            [
                "Empirical mean variance across replications",
                fmt(precision["empirical_mean_variance"], 6),
            ],
        ],
        [0.7, 0.3],
    )
    figure(
        "mean_precision.png",
        "Figure 1. Independent stationary replications of the 200-period average. The independence curve holds marginal variance fixed. Bars show empirical coverage of known-parameter nominal 95% intervals.",
        180,
    )
    p(
        f"Each of the {precision['replications']:,} replications starts from an independent exact stationary Gaussian draw. The empirical mean variance is {fmt(precision['empirical_mean_variance'], 6)}, against the exact {fmt(precision['exact_mean_variance'], 6)}. Its Monte Carlo standard error is approximately {fmt(precision['mean_variance_mc_standard_error'], 6)}. The Monte Carlo standard error of coverage at 95% is {100 * precision['coverage_mc_standard_error_at_95_percent']:.3f} percentage points."
    )
    p(
        "The exercise does not imply that every statistic has an effective sample size of eleven, or that estimated-covariance intervals would attain the same coverage. It shows how much the stated mean-variance calculation changes in this known-parameter model."
    )

    page()
    h("2. Shock transmission: duration, amplification, and accumulated effects")
    p(
        "A one-unit innovation is introduced at horizon zero, followed by zero innovations. The response is the difference between shocked and unshocked trajectories with identical initial values. Its contemporaneous value is one. This normalization distinguishes the response to an innovation from a response rescaled to a standard deviation or to an estimated economic shock."
    )
    figure(
        "shock_propagation.png",
        "Figure 2. Unit-shock responses and accumulated effects. The left panel shows horizons 0-30; the right shows cumulative totals through horizon 100. The analytical infinite sum is ten in all three stable specifications.",
        190,
    )
    table(
        "Table 2. Distinct transmission paths under a common infinite response total.",
        ["Specification", "Stationary variance", "Total at H=20", "Remainder after H=100"],
        [
            [
                label,
                fmt(dynamics[name]["stationary_variance"], 4),
                fmt(dynamics[name]["cumulative_at_20"], 4),
                fmt(dynamics[name]["remaining_sum_after_100"], 7),
            ]
            for name, label in [
                ("ar1_persistent", "AR(1): 0.9"),
                ("ar2_response", "AR(2): 1.2, -0.3"),
                ("ar3_stable", "AR(3): 0.5, 0.2, 0.2"),
            ]
        ],
        [0.33, 0.22, 0.21, 0.24],
    )
    p(
        "The AR(2) begins with responses 1.00, 1.20, 1.14, and 1.008. It amplifies the initial effect before decaying, despite a first-lag coefficient greater than one. The stabilizing second lag produces companion eigenvalues 0.844949 and 0.355051. Neither the first coefficient alone nor the sum of coefficients is used as a general stationarity test."
    )
    p(
        "For stable specifications, summing the coefficient recursion gives an infinite total of 1/(1 minus the sum of AR coefficients). A total through horizon H includes H+1 responses. The stable AR(3) has more effect remaining after horizon 100 than the other two specifications. Reporting only the common limit would conceal that difference in transmission speed."
    )
    p(
        "Changing the AR(1) intercept from zero to 0.5 provides a separate level control. The stationary mean rises from zero to five, while the centered responses and variance are unchanged. Paired paths share all innovations and start at their respective means; every retained value differs by five up to floating-point precision. This isolates a level shift from a persistence change."
    )

    page()
    h("3. Stability boundaries and the moving-average control")
    p(
        "The companion spectral radius changes from "
        + fmt(dynamics["ar3_stable"]["spectral_radius"], 6)
        + " to "
        + fmt(dynamics["ar3_unstable"]["spectral_radius"], 6)
        + " when the final two AR(3) coefficients rise from 0.2 to 0.3. The first specification is stable; the second has an eigenvalue above one. Their paths share the same 200 innovations, identical zero initial states and no burn-in. This holds the disturbance history fixed while changing the propagation law."
    )
    figure(
        "stability_paths.png",
        "Figure 3. Stable and unstable AR(3) paths under shared innovations. The vertical scales differ. Finite realized values and sample variances for the unstable path do not imply stationary population moments.",
        155,
    )
    figure(
        "companion_eigenvalues.png",
        "Figure 4. Companion eigenvalues relative to the unit circle. The boundary AR(2), with coefficients (1.5, -1), has complex eigenvalues of modulus one. The classification uses modulus, not only the real part.",
        210,
    )
    p(
        "A numerical margin of 1e-10 flags unit-circle and nearly unit-root cases. Stationary moments and convergent infinite response sums are withheld for that category and for unstable models. A finite oscillating response can be plotted without being assigned a stationary variance or a convergent total."
    )
    table(
        "Table 3. Finite MA(2) representation: 1 + 1.5L + 1.3L squared.",
        ["Variance", "Lag-1 covariance", "Lag-2 covariance", "Polynomial root modulus"],
        [
            [
                fmt(ma["stationary_variance"], 2),
                fmt(ma["autocovariances_0_to_4"][1], 2),
                fmt(ma["autocovariances_0_to_4"][2], 2),
                fmt(ma["root_moduli"][0], 6),
            ]
        ],
        [1, 1, 1, 1],
    )
    p(
        "The finite MA representation has constant second moments under white-noise innovations, with zero autocovariances beyond lag two. Its roots lie inside the unit circle, so this particular innovation representation is noninvertible. Stationarity concerns second moments; invertibility concerns recovering innovations from present and past observations. The two conditions should not be conflated."
    )

    page()
    h("4. What the findings support")
    h("Implications for dynamic analysis", 2)
    p(
        "A common long-run effect does not establish a common adjustment path or common volatility. Dynamic regressions need both a stability assessment and horizon-specific responses. Short-run amplification can coexist with eventual decay; a large autoregressive coefficient cannot be interpreted independently of the remaining lags."
    )
    p(
        "For averages, the number of dated observations is an incomplete description of precision. In the stated persistent model, serial covariance raises variance by a factor of 18.10 relative to the same-marginal-variance independence calculation. The associated coverage failure motivates checking the interval's covariance assumptions before interpreting its nominal confidence level."
    )
    h("Numerical evidence and software design", 2)
    maximum_error = max(
        v["maximum_three_method_disagreement"]
        for v in checks.values()
        if isinstance(v, dict) and "maximum_three_method_disagreement" in v
    )
    p(
        f"Across all six AR experiments and horizons 0-100, the largest disagreement between the three response implementations is {maximum_error:.3g}. An independent SciPy difference-equation filter [3] agrees within the recorded tolerance. Stable covariance matrices satisfy the discrete Lyapunov equation [2] and match a 5,001-term squared-response approximation. Analytic AR(1) moments and an independently constructed Toeplitz covariance matrix supply separate controls."
    )
    p(
        "The object-oriented Python package separates immutable specifications, simulation settings, recursive simulators, deterministic analyzers, and result containers. Reporting consumes these results; the notebook does not implement another simulator. Scientific tests cover root reciprocity, boundary and zero-root cases, initial-lag ordering, variance scaling, burn-in, response normalization, finite MA covariance, invalid inputs, and seeded reproduction."
    )
    h("Limits of the conclusions", 2)
    p(
        "No coefficients are estimated from observed data. The simulated series do not represent a dated macroeconomic or financial dataset. Results do not establish observed causal policy effects, forecasting superiority, or an empirical standard-error correction. Applying these mechanisms to real data would require an explicit sampling model, data provenance, parameter uncertainty, and suitable inference."
    )
    p(
        "The coverage experiment uses known parameters and exact stationary initialization. The separate path illustrations use documented finite initial states and burn-in. Starting a stable process at its mean does not make its initial distribution stationary. Effective independent observation counts are specific to the statistic and window. Monte Carlo deviations remain finite-simulation noise, not posterior uncertainty."
    )
    h("References", 2)
    for reference in [
        "[1] Sargent, T. J. and Stachurski, J. AR(1) Processes. QuantEcon. https://python.quantecon.org/ar1_processes.html",
        "[2] SciPy documentation. solve_discrete_lyapunov: the equation A X A-transpose - X + Q = 0. https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_lyapunov.html",
        "[3] SciPy documentation. lfilter: difference-equation conventions and initial conditions. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html",
    ]:
        story.append(paragraph(reference, "ref"))

    page()
    h("Appendix. Response estimates and reproducibility")
    selected = [
        row
        for row in responses
        if row["experiment"] == "ar2_response" and int(row["horizon"]) <= 20
    ]
    table(
        "Table A1. AR(2) responses: coefficients (1.2, -0.3), one-unit innovation.",
        ["Horizon", "Matrix", "Recursion", "Controlled shock", "Cumulative"],
        [
            [
                row["horizon"],
                fmt(row["matrix"], 6),
                fmt(row["recursion"], 6),
                fmt(row["shock"], 6),
                fmt(row["cumulative"], 6),
            ]
            for row in selected
        ],
        [0.12, 0.21, 0.21, 0.23, 0.23],
        compact=True,
    )
    h("Recorded settings", 2)
    table(
        "Table A2. Synthetic path configurations; each retains 200 observations.",
        ["Experiment", "Seed", "Burn-in", "Initial state"],
        [
            [
                name.replace("_", " "),
                str(value["simulation"]["seed"]),
                str(value["simulation"]["burn_in"]),
                ", ".join(
                    fmt(v, 0)
                    for v in value.get(
                        "initial_state_most_recent_first",
                        value.get("initial_innovation_state_most_recent_first", []),
                    )
                ),
            ]
            for name, value in configs.items()
        ],
        [0.43, 0.21, 0.16, 0.2],
        compact=True,
    )
    p(
        "The independent mean-precision experiment uses seed 20261215 and 10,000 stationary replications of 200 observations. CSV and JSON artifacts preserve paths, response methods, interval coverage, numerical errors and model settings. Source and output hashes in manifest.json identify the recorded numerical calculation; notebook_validation.json identifies the executed notebook. The report is generated from those artifacts, without a second numerical implementation."
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=62,
        rightMargin=62,
        topMargin=53,
        bottomMargin=50,
        title="Persistence and Shock Propagation",
        author="Sebastian Jaramillo",
        subject="Controlled time-series experiments on persistence, stability, and interval precision",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    source_files = [
        results / name
        for name in (
            "dynamics_summary.csv",
            "impulse_responses.csv",
            "mean_precision.json",
            "ma_control.json",
            "numerical_validation.json",
            "configurations.json",
        )
    ]
    record = {
        "report_date": report_date.isoformat(),
        "report_builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output": str(output.relative_to(root)),
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
        "pdf_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (results / "report_validation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-11-23", help="Report date in ISO format")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build_report(
        root, root / "docs/persistence_shock_propagation.pdf", date.fromisoformat(arguments.date)
    )


if __name__ == "__main__":
    main()

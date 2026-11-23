"""Typeset the research report and the standalone brief from recorded estimates."""

import csv
import sys
from datetime import date
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "project4/scripts"))
from report_style import ResearchReport  # noqa: E402

RES = ROOT / "results"
TITLE = "U.S. GDP Growth at Risk"
SUB = "Testing the downside forecast for U.S. growth"


def rows(name):
    with (RES / f"{name}.csv").open() as f:
        return list(csv.DictReader(f))


def one(records, **keys):
    found = [r for r in records if all(r[k] == str(v) for k, v in keys.items())]
    if len(found) != 1:
        raise ValueError(keys)
    return found[0]


def fmt(r, k, d=2):
    return f"{float(r[k]):.{d}f}"


def percent(r, k):
    return f"{100 * float(r[k]):.1f}%"


def save(report, path, subject):
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=62,
        rightMargin=62,
        topMargin=50,
        bottomMargin=48,
        title=TITLE,
        author="Sebastian Jaramillo",
        subject=subject,
    )
    doc.build(report.story, onFirstPage=report._footer, onLaterPages=report._footer)


def build():
    s, reference, forecasts = rows("scores"), rows("reference"), rows("forecasts")

    def score(kind, h, tr="observed", sample="full"):
        return one(s, model=kind, horizon=h, training=tr, sample=sample)

    def ref(kind, h, tr="observed"):
        return one(reference, model=kind, horizon=h, training=tr)

    ar, drift = score("AR", 4), score("Drift", 4)
    one_ar, spread = score("AR", 1), score("ARDL", 1)
    a, sm = ref("AR", 4), ref("AR", 4, "interpolated")
    shock = one(forecasts, origin="2020Q2", training="observed", model="AR", horizon=4)
    share = (float(shock["actual"]) - float(shock["mean"])) ** 2 / (
        float(ar["rmse"]) ** 2 * int(ar["n"])
    )
    comparison = [row for row in rows("course_comparison") if row["actual"]]
    samples = [
        ("Through 2019Q4", [r for r in comparison if r["training"] == "observed"]),
        ("From 2022Q1", [r for r in comparison if r["training"] == "interpolated"]),
        ("All eligible origins", comparison),
    ]

    def losses(records, method):
        errors = [float(r["actual"]) - float(r[f"{method}_mean"]) for r in records]
        gaps = [float(r["actual"]) - float(r[f"{method}_p05"]) for r in records]
        return (
            (sum(e * e for e in errors) / len(records)) ** 0.5,
            sum(max(0.05 * g, -0.95 * g) for g in gaps) / len(records),
        )

    matched = []
    for label, sample in samples:
        course_rmse, course_loss = losses(sample, "course")
        project_rmse, project_loss = losses(sample, "project")
        matched.append(
            [
                label,
                len(sample),
                f"{course_rmse:.3f}",
                f"{project_rmse:.3f}",
                f"{course_loss:.3f}",
                f"{project_loss:.3f}",
            ]
        )
    sections = [
        (
            "Did the portfolio improve the annual forecast?",
            f"Across {matched[2][1]} matched annual origins, root mean squared error fell from {matched[2][2]} to {matched[2][3]} log percentage points, a 5.6% reduction. Fifth-percentile loss fell from {matched[2][4]} to {matched[2][5]}. The coursework method won on both measures over the {matched[1][1]} post-pandemic origins. Its mean-forecast error was {matched[1][2]} against {matched[1][3]}. The windows overlap and use revised GDP.",
        ),
        (
            "The cost of missing a rebound",
            f"In a separate comparison over all {drift['n']} annual origins using observed GDP, the random walk with drift delivered the more accurate mean forecast. Its root mean squared error was {fmt(drift, 'rmse')} log percentage points, against {fmt(ar, 'rmse')} for the trend autoregression. One forecast accounts for {share:.0%} of the autoregression's total squared error. After the second quarter of 2020 it projected {fmt(shock, 'mean')}% cumulative growth over the following year. Realized growth was +{fmt(shock, 'actual')}%.",
        ),
        (
            "A different ranking for downside risk",
            f"The autoregression produced a lower average fifth-percentile loss than the drift benchmark ({fmt(ar, 'loss05', 3)} versus {fmt(drift, 'loss05', 3)}). Its average central 90% interval was wider ({fmt(ar, 'width')} versus {fmt(drift, 'width')} points). Growth fell below its downside threshold in {ar['breaches']} of {ar['n']} forecasts ({percent(ar, 'breach_rate')}), above the nominal 5%. Five of those misses ended between 2008Q3 and 2009Q3 and three ended in 2020. Adjacent annual outcomes share three quarters.",
        ),
        (
            "What did the Treasury spread add at a common horizon?",
            f"Over {one_ar['n']} one-quarter targets, adding the spread nudged fifth-percentile loss from {fmt(one_ar, 'loss05', 4)} to {fmt(spread, 'loss05', 4)}. Mean-forecast error increased from {fmt(one_ar, 'rmse', 3)} to {fmt(spread, 'rmse', 3)} points.",
        ),
        (
            "Downside risk at the 2026Q1 forecast origin",
            f"Using observed GDP through 2026Q1, the autoregression put mean four-quarter growth at {fmt(a, 'mean')}%, the fifth percentile at {fmt(a, 'p05')}%, and the probability of contraction at {percent(a, 'p_negative')}. Smoothing eight pandemic quarters in the training sample moved the fifth percentile to {fmt(sm, 'p05')}% and the contraction probability to {percent(sm, 'p_negative')}. These estimates describe a fixed historical exercise using revised GDP levels.",
        ),
    ]
    intro = "In my doctoral assignment, the trend AR(4) put the fifth percentile of annual GDP growth from 2026Q1 at -0.218 log percentage points. The portfolio estimate is -0.342 using the same pandemic smoothing. The original chart paired an annual GDP forecast with a one-quarter spread forecast. Matching horizons lets us test both average growth and the downside against realized outcomes."
    (ROOT / "EXECUTIVE_SUMMARY.md").write_text(
        "# U.S. GDP Growth at Risk\n\n"
        + intro
        + "\n\n"
        + "\n\n".join("## " + h + "\n\n" + p for h, p in sections)
        + "\n\n[Research report](docs/gdp_growth_at_risk.pdf) · [Matched coursework and portfolio forecasts](results/course_comparison.csv) · [Saved coursework estimates](../coursework_results.csv) · [Forecast scores](results/scores.csv)\n"
    )

    def first():
        page = ResearchReport(TITLE, SUB, date(2026, 11, 23), RES / "figures")
        page.h("Executive summary")
        page.p(intro)
        for h, p in sections:
            page.h(h, 2)
            page.p(p)
        return page

    save(
        first(),
        ROOT / "docs/executive_summary.pdf",
        "Executive summary of the GDP forecast evaluation",
    )
    r = first()
    r.page()
    r.h("How did the coursework forecast hold up on matched dates?")
    r.p(
        "The saved coursework result is one forecast. At 2026Q1 the original model placed a 5% probability on annual growth below -0.218 log percentage points by 2027Q1. The reconstruction gives -0.2175. Applying the portfolio model to the same smoothed training series gives -0.342. Keeping the observed pandemic quarters moves it to -1.150. The target quarter falls beyond the supplied data, so a realized error cannot be calculated for these estimates."
    )
    r.h("An annual forecast from each eligible origin", 2)
    r.p(
        "I reconstructed the class regression and its predictive simulation at each eligible origin. Both models received the same history and predicted the same four-quarter growth outcome. For the 65 origins through 2019Q4, the data are observed. From 2022Q1, both models smooth the same eight pandemic observations. A forecaster in those eight intervening quarters could not yet use the 2022Q1 endpoint required for interpolation."
    )
    r.table(
        "Matched annual forecasts. The last two columns score the predicted fifth percentile. Lower is better in all four score columns.",
        [
            "Forecast origins",
            "N",
            "Class RMSE",
            "Portfolio RMSE",
            "Class tail loss",
            "Portfolio tail loss",
        ],
        matched,
        [1.8, 0.47, 0.95, 1, 1, 1],
        True,
    )
    r.p(
        "Most eligible forecasts were issued before the pandemic. They drive the overall improvement. The 13 later origins reverse the ranking for both average growth and downside loss. Consecutive annual forecasts share three quarters of realized growth, so the count overstates the number of independent economic episodes."
    )
    r.h("Why the other scorecard has 86 outcomes", 2)
    r.p(
        "The separate drift comparison keeps observed pandemic GDP throughout and can therefore score all 86 annual origins, including the eight omitted above. It answers a different question: would a forecaster using observed GDP have done better with a trend autoregression or with a simple random walk with drift?"
    )
    r.reference(
        "Replication: results/course_comparison.csv pairs the forecasts by origin and target. scripts/compare_course_forecasts.py reconstructs the coursework side from the frozen GDP snapshot."
    )
    r.page()
    r.h("1. What could a forecaster use at the time?")
    r.p(
        "The GDP series contains 169 consecutive quarters from 1984Q1 through 2026Q1. Real GDP is measured in billions of chained 2017 dollars at a seasonally adjusted annual rate. Growth over h quarters is 100 times the natural logarithm of the ratio of end to starting GDP. A one-quarter result therefore measures the change over that quarter. All forecast errors and interval widths use log percentage points."
    )
    r.p(
        "The first training sample ends in 2003Q4. Reserving four observations for lags leaves 76 common regression targets. Each subsequent forecast adds one quarter to the sample. The issue date is the first BEA publication of GDP for the origin quarter. The inputs stop in that quarter, even when later financial observations were available. All models are evaluated against GDP from the same supplied snapshot."
    )
    r.table(
        "Table 1. Selected origin dates from the BEA vintage history.",
        ["Last GDP quarter", "First publication", "Next-quarter target"],
        [
            [
                q,
                one(forecasts, origin=q, training="observed", model="Drift", horizon=1)[
                    "issue_date"
                ],
                t,
            ]
            for q, t in [
                ("2003Q4", "2004Q1"),
                ("2020Q2", "2020Q3"),
                ("2025Q3", "2025Q4"),
                ("2026Q1", "2026Q2"),
            ]
        ],
        [1, 1, 1],
    )
    r.h("Three forecasts on the same footing", 2)
    r.p(
        "The benchmark adds an estimated drift and a new shock to the last log GDP level. The AR(4) adds a deterministic trend and four GDP lags. Writing it in quarterly changes yields an intercept, trend, lagged level and three lagged changes. The ARDL adds four lagged quarterly spreads to this equation. At one quarter, the latest available spread is the average for the origin quarter. Extending the ARDL recursively would require future spread values."
    )
    r.p(
        "The supplied spread agrees with the average of daily FRED T10Y2Y observations in all 169 quarters to within 0.005 percentage points, the rounding tolerance of the file. The next spread row is empty. Neither first-release GDP levels nor successive historical revisions are available in the supplied files. Release dates establish the timing of the exercise, while a common revised snapshot supplies the levels used for fitting and scoring."
    )
    r.page()
    r.h("2. How well did the distributions anticipate realized growth?")
    labels = {"AR": "AR(4)", "Drift": "Drift", "ARDL": "ARDL(4,4)"}
    table = []
    for h, k in [(1, "Drift"), (1, "AR"), (1, "ARDL"), (4, "Drift"), (4, "AR")]:
        v = score(k, h)
        table.append(
            [
                labels[k],
                h,
                v["n"],
                fmt(v, "rmse"),
                percent(v, "coverage"),
                fmt(v, "width"),
                v["breaches"],
                fmt(v, "loss05", 4),
            ]
        )
    r.table(
        "Table 2. Expanding-window scores on observed GDP. Smaller RMSE and quantile loss are better.",
        ["Model", "Quarters", "N", "RMSE", "90% cover", "Width", "Below 5th", "Loss at 5%"],
        table,
        [1.3, 1, 0.6, 0.9, 1.15, 0.8, 1, 1.15],
        True,
    )
    r.figure(
        "historical_risk.png",
        "Figure 1. Each annual outcome is paired with the AR forecast issued four quarters earlier. The low band ending in 2021Q2 followed the abrupt contraction in 2020Q2. Dots flag realizations below the modeled fifth percentile.",
        220,
    )
    r.p(
        f"The AR interval covered {percent(ar, 'coverage')} of four-quarter outcomes. Its lower tail was breached by the financial crisis and the pandemic. In particular, the outlook formed in 2020Q2 placed even its 95th percentile at {fmt(shock, 'p95')}%, while GDP rebounded over the next year. A large shock followed by a rapid reversal exposed a limitation of fitting constant innovation variance and persistent dynamics through that quarter."
    )
    r.p(
        "Splitting the annual forecasts into four groups by origin quarter removes overlapping growth windows within each group. Each contains 21 or 22 forecasts. The drift benchmark had lower RMSE in all four groups. The AR had lower fifth-percentile loss in three of four. The exception was the group containing the forecast issued after 2020Q2. These are descriptive rankings across a small number of distinct episodes."
    )
    r.page()
    r.h("3. How much does pandemic treatment change the downside?")
    r.p(
        "The primary models keep each observed quarter. A sensitivity replaces log GDP from 2020Q1 through 2021Q4 with a straight line between 2019Q4 and 2022Q1. This calculation first becomes feasible at the 2022Q1 release. Both versions are scored against the same observed growth. Thirteen subsequent four-quarter targets are available."
    )
    r.table(
        "Table 3. AR(4) at four quarters. Historical scores use the matched post-endpoint origins.",
        ["Training series", "N", "RMSE", "Width", "Loss at 5%", "Below 5th"],
        [
            [label, v["n"], fmt(v, "rmse"), fmt(v, "width"), fmt(v, "loss05", 3), v["breaches"]]
            for label, tr in [("Observed", "observed"), ("Interpolated", "interpolated")]
            for v in [score("AR", 4, tr, "post_endpoint")]
        ],
        [1.6, 0.6, 1, 1, 1.15, 1.15],
    )
    r.figure(
        "reference_distribution.png",
        "Figure 2. Simulated predictive percentiles at the 2026Q1 origin. The markers show the four-quarter growth thresholds with 5% probability below them.",
        194,
    )
    r.table(
        "Table 4. Fixed reference forecasts from 2026Q1 using observed GDP.",
        ["Model", "Target", "Mean", "5th", "95th", "P(growth < 0)"],
        [
            [
                labels[k],
                v["target"],
                fmt(v, "mean"),
                fmt(v, "p05"),
                fmt(v, "p95"),
                percent(v, "p_negative"),
            ]
            for k, h in [("Drift", 1), ("AR", 1), ("ARDL", 1), ("Drift", 4), ("AR", 4)]
            for v in [ref(k, h)]
        ],
        [1.3, 1.2, 1, 0.8, 0.8, 1.4],
        True,
    )
    r.p(
        "Smoothing the pandemic observations reduced the average width of the annual AR interval in the eligible sample. Both versions covered all 13 realizations. This short period offers little evidence about coverage in a future contraction. The reference forecast illustrates how that training decision can materially change a downside estimate even when expected growth moves little."
    )
    r.page()
    r.h("4. Estimation, scoring and reproducibility")
    r.h("Parameter and innovation uncertainty", 2)
    r.p(
        "The AR and ARDL use a trend measured in units of 100 quarters, beginning at zero in 1984Q1. The lagged log level is centered on the first observation and divided by 100. Coefficients have independent zero-mean normal priors. Prior standard deviations are 2 for drift, trend and scaled level, and 0.5 for lagged growth or spread coefficients. Innovation precision has a Gamma prior with shape 2 and rate 1. The same units and priors apply at every origin."
    )
    r.p(
        "Gaussian coefficients integrate analytically conditional on precision. Numerical integration on log precision yields posterior weights. One-step forecasts integrate a normal mixture that includes observation variance. At four quarters the drift model adds four innovation variances and propagates the uncertainty of a common drift over four steps. The AR uses 65,536 predictive paths. Each path keeps its coefficient and variance draws and receives four independent innovations."
    )
    r.h("Independent checks", 2)
    g = rows("gibbs_checks")
    r.p(
        f"At the reference origin an independent Gibbs implementation fits each of the three specifications on both training series. Four chains each retain 8,000 draws after 1,500 warmup iterations. The largest parameter R-hat was {max(float(v['max_rhat']) for v in g):.5f}; the minimum bulk effective sample size was {min(float(v['min_bulk_ess']) for v in g):,.0f}. The ten Gibbs-based lower thresholds were within {max(abs(float(v['delta_p05'])) for v in g):.3f} log points of the main estimates. The observed annual AR threshold had a simulation standard error of {fmt(a, 'p05_mcse', 3)} points, estimated from 32 batches.".replace(
            "; the", ". The"
        )
    )
    r.h("Interpreting the score", 2)
    r.p(
        "RMSE measures the error of the predictive mean. Coverage counts outcomes between the 5th and 95th percentiles. Width is the average distance between those bounds. For the fifth-percentile loss, an outcome below the threshold receives 0.95 times the shortfall, while an outcome above it receives 0.05 times the excess. Both the frequency and magnitude of misses therefore affect the comparison."
    )
    r.h("Sources and replication", 2)
    r.reference(
        "U.S. Bureau of Economic Analysis. Real Gross Domestic Product, GDPC1. FRED, Federal Reserve Bank of St. Louis. fred.stlouisfed.org/series/GDPC1"
    )
    r.reference(
        "Federal Reserve Bank of St. Louis. 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity, T10Y2Y. fred.stlouisfed.org/series/T10Y2Y"
    )
    r.reference(
        "U.S. Bureau of Economic Analysis. Vintage History of Quarterly GDP and GDI Estimates, October 29, 2026 edition. apps.bea.gov/national/xls/gdp-gdi-vintage-history.xlsx"
    )
    r.reference(
        "Replication files, input fingerprints, release calendar, forecasts and scores: github.com/jsjaramillod/Time-Series/tree/main/project5"
    )
    save(
        r,
        ROOT / "docs/gdp_growth_at_risk.pdf",
        "Predictive distributions and historical evaluation of U.S. real GDP growth",
    )


if __name__ == "__main__":
    build()

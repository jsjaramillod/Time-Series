"""Typeset the payroll research report and its one-page executive brief."""

import csv
import sys
from datetime import date
from pathlib import Path
from statistics import mean

from reportlab import rl_config
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "project4/scripts"))
from report_style import ResearchReport  # noqa: E402

TITLE = "Forecasting U.S. Payrolls"
SUBTITLE = "What production, prices and financial markets add to employment history"
NAMES = {"AR": "Employment only", "Macro": "Three variables", "Full": "Five variables"}
RES = ROOT / "results"


def rows(name):
    with (RES / f"{name}.csv").open() as stream:
        return list(csv.DictReader(stream))


def select(records, **criteria):
    return [r for r in records if all(r[k] == str(v) for k, v in criteria.items())]


def number(r, key, digits=0):
    return f"{float(r[key]):,.{digits}f}"


def pct(r, key):
    return f"{float(r[key]):.1%}"


def signed(r, key):
    return f"{float(r[key]):+,.0f}"


def save(report, filename, subject):
    doc = SimpleDocTemplate(
        str(ROOT / "docs" / filename),
        pagesize=A4,
        leftMargin=62,
        rightMargin=62,
        topMargin=48,
        bottomMargin=47,
        title=TITLE + " | " + subject,
        author="Sebastian Jaramillo",
        subject=subject,
    )
    doc.build(report.story, onFirstPage=report._footer, onLaterPages=report._footer)


def new_report(subtitle):
    report = ResearchReport(TITLE, subtitle, date(2026, 11, 22), RES / "figures")
    report.story[2] = report.paragraph("Sebastian Jaramillo  |  22 November 2026", "date")
    return report


def build():
    rl_config.invariant = 1
    scores, raw, reference = rows("scores"), rows("forecasts"), rows("reference")
    primary = select(scores, intensity="0.5", vintage="first_release")
    overall = select(primary, episode="All")
    ar, macro, full = [select(overall, model=m)[0] for m in NAMES]
    observed = select(reference, training="observed", intensity="0.5")
    smoothed = select(reference, training="interpolated", intensity="0.5")
    obs = select(observed, model="Full")[0]
    sm = select(smoothed, model="Full")[0]
    hist = select(raw, intensity="0.5", model="Full")
    sq = lambda row: (float(row["mean"]) - float(row["first_release"])) ** 2  # noqa: E731
    crisis = [r for r in hist if "2020-03" <= r["target"] <= "2020-06"]
    stress_share = sum(map(sq, crisis)) / sum(map(sq, hist))
    after = [r for r in hist if "2022-01" <= r["target"] <= "2024-12"]
    before = [r for r in hist if "2015-01" <= r["target"] <= "2019-12"]

    def width(group):
        return mean(float(r["p95"]) - float(r["p05"]) for r in group)

    gain = 1 - float(macro["rmse"]) / float(ar["rmse"])
    financial = 1 - float(full["rmse"]) / float(macro["rmse"])
    intro = (
        "Would a labor-market analyst improve the next payroll forecast by adding industrial "
        "production, consumer prices, equity prices and the Treasury yield curve? Three Bayesian models "
        "face the same 216 monthly releases from January 2007 through December 2024. Each predicts "
        "both the job count and the probability of a decline."
    )
    sections = [
        (
            "Most of the gain came from production and prices",
            f"Adding industrial production and CPI reduced root mean squared error by {gain:.1%}, from "
            f"{float(ar['rmse']) / 1000:.2f} million to {float(macro['rmse']) / 1000:.2f} million jobs. Adding "
            f"equity prices and the Treasury spread reduced it a further {financial:.2%}. Financial information "
            f"made a clearer difference to loss probabilities. The average Brier score improved from "
            f"{number(macro, 'brier', 3)} to {number(full, 'brier', 3)}. Lower scores reward probabilities closer to the outcome.",
        ),
        (
            "Four pandemic releases dominated the errors",
            f"All three models missed every March–June 2020 first announcement with their nominal 90% intervals. "
            f"Those four months accounted for {stress_share:.1%} of the five-variable model's total squared error. "
            f"After the sharp movements entered its expanding training sample, the model's average 90% interval "
            f"spanned {width(after) / 1000:.2f} million jobs in 2022–24, compared with {width(before) / 1000:.2f} million in 2015–19.",
        ),
        (
            "How much history changes a risk judgment",
            f"With data through March 2026, the five-variable model put the chance of an April payroll decline "
            f"at {pct(obs, 'p_loss')}. Smoothing 26 pandemic-era months in the same history lowered it to "
            f"{pct(sm, 'p_loss')}. The first announcement was a gain of 115,000 jobs. The risk estimate depended "
            "materially on the treatment of an exceptional episode.",
        ),
    ]
    limit = (
        "The models use historical release dates and a common cutoff, but their inputs contain subsequent "
        "revisions. The backtest measures performance under that reconstruction. The fixed April 2026 "
        "illustration is separate from the 216 scored months."
    )
    brief = new_report("Executive brief  |  Monthly nonfarm employment")
    brief.p(intro)
    for h, p in sections:
        brief.h(h)
        brief.p(p)
    brief.table(
        "First-release scorecard. Prediction errors in thousands of jobs.",
        ["Information set", "Mean absolute error", "RMSE", "Brier score"],
        [
            [NAMES[x["model"]], number(x, "mae"), number(x, "rmse"), number(x, "brier", 3)]
            for x in overall
        ],
        [1.53, 1.48, 0.91, 1],
        True,
    )
    brief.h("Reading the evidence", 2)
    brief.p(limit)
    brief.story.append(
        Paragraph(
            '<a href="https://github.com/jsjaramillod/Time-Series/tree/main/project6" color="#304d58">Research report, forecast records and replication materials</a>',
            brief.styles["ref"],
        )
    )
    save(brief, "executive_summary.pdf", "Executive brief")
    (ROOT / "EXECUTIVE_SUMMARY.md").write_text(
        "# Forecasting U.S. Payrolls\n\n"
        + intro
        + "\n\n"
        + "\n\n".join("## " + h + "\n\n" + p for h, p in sections)
        + "\n\n"
        + limit
        + "\n\n"
        + "[Research report](docs/payroll_forecasts.pdf) · [Forecast records](results/forecasts.csv) · "
        + "[BLS April 2026 release](https://www.bls.gov/news.release/archives/empsit_05082026.htm)\n"
    )

    r = new_report(SUBTITLE)
    r.h("How much information earns its place?")
    r.p(intro)
    r.p(sections[0][1])
    r.table(
        "Table 1. Monthly first announcements, January 2007–December 2024. Job counts in thousands.",
        [
            "Information set",
            "RMSE",
            "Mean absolute error",
            "90% coverage",
            "Mean width",
            "Brier score",
        ],
        [
            [
                NAMES[a["model"]],
                number(a, "rmse"),
                number(a, "mae"),
                pct(a, "coverage"),
                number(a, "width"),
                number(a, "brier", 3),
            ]
            for a in overall
        ],
        [1.65, 0.78, 1.18, 1.03, 0.94, 0.83],
    )
    r.p(
        "The employment-only model uses the past year of payrolls. The three-variable model adds production "
        "and CPI. The five-variable model adds the S&P 500 price index and the 10-year Treasury yield minus "
        "the three-month bill rate. Each model uses twelve monthly lags and learns from January 1962 onward."
    )
    r.figure(
        "loss_probabilities.png",
        "Figure 1. Probability assigned to a negative monthly change. "
        "Ticks along the lower edge mark the 35 first announcements that reported job losses.",
        165,
    )
    r.p(
        "Over 2022–24 the five-variable model assigned an average loss probability of "
        f"{mean(float(a['p_loss']) for a in after):.1%}. All 36 first announcements in those years showed "
        "gains. The wide intervals behind that caution help explain why overall coverage exceeded its nominal rate."
    )

    r.page()
    r.h("When the models missed the labor market")
    r.p(
        "The error ranking depends on the episode. During 2008–09 the employment-only model came closest "
        "to the announced change on average by RMSE, while the five-variable model produced the lowest "
        "Brier score. All 24 announcements showed declines. In spring 2020, the difficult call was the "
        "speed and direction of the reversal after the collapse."
    )
    r.figure(
        "pandemic.png",
        "Figure 2. Posterior predictive means and the first BLS announcement in four "
        "consecutive target months of 2020. Negative values represent net job losses.",
        185,
    )
    r.table(
        "Table 2. Performance by target period. RMSE in thousands of jobs. Model order is employment only, three variables, five variables.",
        ["Target period", "Months", "Losses", "RMSE by model", "Brier score by model"],
        [
            [
                ep,
                g[0]["n"],
                g[0]["declines"],
                " / ".join(number(x, "rmse") for x in g),
                " / ".join(number(x, "brier", 3) for x in g),
            ]
            for ep in ["2008-09", "Mar-Jun 2020", "Other months"]
            for g in [select(primary, episode=ep)]
        ],
        [1.24, 0.65, 0.65, 1.78, 1.9],
        True,
    )
    may = select(crisis, target="2020-05")[0]
    r.p(
        f"For May 2020 the five-variable model projected a further loss of {abs(float(may['mean'])) / 1000:.2f} million "
        "jobs. BLS reported a gain of 2.51 million. The employment-only model projected a loss of 17.43 million. "
        "Additional indicators tempered the extrapolation but still left a large error."
    )
    r.p(sections[1][1])
    r.p(
        "Across the other 188 months, only nine first releases reported losses. Even the model with the lowest "
        "aggregate Brier score assigned too much risk to many later gains. A desk monitoring recession risk "
        "would want to examine this calibration alongside the average score."
    )

    r.page()
    r.h("April 2026 through two versions of history")
    r.p(
        "The fixed exercise uses 771 observations through March 2026. March payrolls were released on April 3, "
        "CPI on April 10 and industrial production on April 16. May 8 brought the first April payroll figure, "
        "22 days after the common model cutoff. The published gain was 115,000 jobs."
    )
    r.table(
        "Table 3. April forecasts under primary shrinkage. Changes and central 90% limits in thousands of jobs.",
        [
            "Training history",
            "Information set",
            "Mean",
            "Median",
            "5th pct.",
            "95th pct.",
            "P(loss)",
        ],
        [
            [
                a["training"].capitalize(),
                NAMES[a["model"]],
                signed(a, "mean"),
                signed(a, "median"),
                signed(a, "p05"),
                signed(a, "p95"),
                pct(a, "p_loss"),
            ]
            for a in observed + smoothed
        ],
        [1.24, 1.48, 0.67, 0.72, 0.78, 0.78, 0.81],
        True,
    )
    r.figure(
        "april_reference.png",
        "Figure 3. Five-variable predictive distributions at the same cutoff. "
        "Smoothing changes both the sign of the median and the width of the interval.",
        126,
    )
    r.p(
        f"The announced gain lay at percentile {100 * float(obs['cdf_115k']):.1f} under observed history "
        f"and {100 * float(sm['cdf_115k']):.1f} after smoothing. Econoday's survey forecast of +63,000 lay "
        f"at percentiles {100 * float(obs['cdf_63k']):.1f} and {100 * float(sm['cdf_63k']):.1f}, respectively. "
        "The survey represents a separate information set and serves as a reference point."
    )
    r.h("Employment lags in the price equation", 2)
    lags = select(rows("cpi_coefficients"), training="observed", intensity="0.5")
    r.p(
        "The five-variable model also estimates how past employment levels enter the CPI equation. "
        "For lags one through three, the posterior medians were "
        + ", ".join(number(a, "median", 3) for a in lags)
        + ". Their central 90% intervals were "
        + ", ".join("[" + number(a, "p05", 3) + ", " + number(a, "p95", 3) + "]" for a in lags)
        + ". All span zero. These are reduced-form conditional associations in log levels."
    )
    r.p(
        "The alternative history replaces the raw monthly levels and rates from December 2019 through "
        "January 2022 with a time-linear path between November 2019 and February 2022. It is a retrospective "
        "sensitivity experiment. Historical scoring retains every observed pandemic movement."
    )

    r.page()
    r.h("Release timing, revisions and shrinkage")
    r.p(
        "The first forecast trains through December 2006. Each new fit expands the sample by one month. "
        "The issue date is the latest of the origin month's employment, CPI and industrial production releases. "
        "Monthly market observations are treated as available by then. All variables stop at the same month, "
        "even when some data for the target month are already available. Lead times to the first announcement "
        "range from 8 to 35 calendar days, with a median of 18."
    )
    r.p(
        "The primary target is the first seasonally adjusted monthly change reported by BLS. A second target "
        "subtracts adjacent payroll levels from the frozen workbook. The first-release series has 35 declines "
        "and the revised series has 34. Historical input vintages for all predictors are unavailable."
    )
    r.table(
        "Table 4. Forecast sensitivity. Each row scores 216 outcomes. RMSE in thousands of jobs.",
        [
            "Target / prior variance",
            "RMSE, employment",
            "RMSE, three",
            "RMSE, five",
            "Brier, five",
            "90% cover, five",
        ],
        [
            [
                label,
                *[number(a, "rmse") for a in g],
                number(g[-1], "brier", 3),
                pct(g[-1], "coverage"),
            ]
            for target, shrink, label in [
                ("first_release", "0.5", "First / primary"),
                ("revised", "0.5", "Revised / primary"),
                ("first_release", "0.1", "First / tighter"),
                ("revised", "0.1", "Revised / tighter"),
            ]
            for g in [select(scores, episode="All", vintage=target, intensity=shrink)]
        ],
        [1.51, 1.08, 0.84, 0.84, 0.79, 0.95],
        True,
    )
    r.h("How small is the financial increment?", 2)
    repeat = select(rows("replay_scores"), episode="All", vintage="first_release")
    diff = float(repeat[1]["rmse"]) - float(repeat[2]["rmse"])
    r.p(
        f"On first announcements, the five-variable model reduced RMSE relative to the three-variable model "
        f"by {float(macro['rmse']) - float(full['rmse']):.1f} thousand jobs. A second simulation seed and longer "
        f"chains put the gap at {diff:.1f} thousand. Both runs place the incremental gain below 1%. "
        "The small gap and concentration of squared errors in 2020 limit a claim of general forecast superiority."
    )
    r.h("Where the inputs come from", 2)
    r.p(
        "The supplied monthly panel uses U.S. nonfarm employment, industrial production and all-items CPI. "
        "Every monthly equity observation agrees with the companion price-index file. Both Treasury rates agree "
        "with published GS10 and TB3MS values throughout the sample. October 2025 CPI was missing following "
        "the government shutdown. The reference fit uses the geometric midpoint of September and November CPI. "
        "This adjustment enters only after November CPI is available."
    )
    r.reference(
        "Sources: U.S. Bureau of Labor Statistics, CES monthly revisions table, May 8, 2026 Employment "
        "Situation and 2025 CPI shutdown FAQ. Federal Reserve Board, G.17 release dates. Federal Reserve "
        "Bank of St. Louis, ALFRED release calendars and FRED series. Econoday via CME Group, May 8, 2026. "
        "Linked sources and the dated input records accompany the replication package."
    )

    r.page()
    r.h("Estimation and numerical controls")
    r.h("Prior and estimation sample", 2)
    r.p(
        "All equations have an intercept and twelve lags. Employment, industrial production, CPI and equity "
        "prices enter in natural logs. The term spread enters in percentage points. The full system orders "
        "the spread, equities, employment, production and CPI. The forecasting exercise imposes no structural "
        "ordering on the joint innovations."
    )
    r.p(
        "Each own first-lag coefficient has prior mean one. Other coefficient means are zero. At lag h the "
        "own-lag prior variance is lambda divided by h squared. Cross-lag variances multiply this by 0.01 "
        "times the ratio of the dependent and predictor equation residual standard deviations. The intercept "
        "variance is 1,000. Residual scales come from the unrestricted OLS system fit within each training "
        "sample. The main lambda is 0.5 and the sensitivity uses 0.1."
    )
    r.p(
        "Before truncation the covariance prior is inverse Wishart with d + 2 degrees of freedom and a "
        "diagonal scale equal to the squared residual scales, where d is the number of equations. It is "
        "independent of the Gaussian coefficient prior. The payroll innovation standard deviation is bounded "
        "at 0.25 log units so the job-level predictive mean exists after exponentiation. No proposed covariance "
        "draw reached the bound in the recorded primary and sensitivity fits."
    )
    r.h("Forecast construction", 2)
    r.p(
        "Given each covariance draw, the sampler updates equation coefficients in turn using the full "
        "precision matrix. QR reduction and a prior-scaled singular value decomposition implement the Gaussian "
        "conditional update. A Bartlett factor supplies correlated joint innovations. For each draw of log "
        "growth g, the job change is the last payroll level in thousands times exp(g) minus that same level."
    )
    r.p(
        "The headline estimates integrate the conditional payroll shock analytically for each parameter "
        "draw. Conditional means include the lognormal variance correction. Inverting the averaged Gaussian "
        "CDF gives the median and interval endpoints. Simulated joint-draw means and loss frequencies are "
        "also stored. The Brier score averages squared probability errors. Coverage counts observations between "
        "the 5th and 95th percentiles, and width measures the distance between them."
    )
    r.h("Verification and scope", 2)
    r.p(
        "Four chains start at different fitted residuals. The historical primary run starts with 256 warm-up "
        "and 512 retained iterations per chain. The reference starts with 1,024 warm-up and 2,048 retained. "
        "Runs lengthen if a monitored rank-normalized R-hat exceeds 1.01 or bulk or tail effective sample size "
        "falls below 400. Monitoring covers payroll forecast location and scale, plus the three reported CPI "
        "coefficients in the full reference system. A small-system control checks the sampler against a dense "
        "Gaussian update and an independent inverse-Wishart implementation."
    )
    r.p(
        "Replication materials include the executed notebook, forecasts by release date, both outcome "
        "definitions, shrinkage and seed checks, source fingerprints and scientific tests. The frozen "
        "workbook is required for full re-estimation. All published scores can be reconstructed from the forecast file."
    )
    for label, url in [
        ("Monthly first estimates and revisions", "https://www.bls.gov/web/empsit/cesnaicsrev.htm"),
        (
            "April 2026 first announcement",
            "https://www.bls.gov/news.release/archives/empsit_05082026.htm",
        ),
        ("April survey comparison", "https://www.cmegroup.com/education/events/econoday/667021"),
        (
            "Research package and release records",
            "https://github.com/jsjaramillod/Time-Series/tree/main/project6",
        ),
    ]:
        r.story.append(Paragraph(f'<a href="{url}" color="#304d58">{label}</a>', r.styles["ref"]))
    save(r, "payroll_forecasts.pdf", "Research report")


if __name__ == "__main__":
    build()

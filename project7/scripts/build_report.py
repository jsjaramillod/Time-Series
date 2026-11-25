"""Typeset the payroll study and the one-page executive brief from its results."""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab import rl_config
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "project4/scripts"))
from report_style import ResearchReport  # noqa: E402

TITLE = "Payroll Forecasts After a Shock"
SUB = "How should a six-month forecast remember the pandemic?"
RES = ROOT / "results"
NAMES = ["Observed", "Interpolated", "Temporary volatility"]
LABELS = ["Constant variance", "Interpolation", "Temporary volatility"]


def sheet():
    page = ResearchReport(TITLE, SUB, date(2026, 11, 25), RES / "figures")
    page.story[2] = page.paragraph("Sebastian Jaramillo  |  25 November 2026", "date")
    return page


def save(page, name, subject):
    doc = SimpleDocTemplate(
        str(ROOT / "docs" / name),
        pagesize=A4,
        leftMargin=62,
        rightMargin=62,
        topMargin=50,
        bottomMargin=48,
        title=TITLE,
        author="Sebastian Jaramillo",
        subject=subject,
    )
    doc.build(page.story, onFirstPage=page._footer, onLaterPages=page._footer)


def rms(x):
    return (x.pow(2).mean()) ** 0.5


def build():
    rl_config.invariant = 1
    scores = pd.read_csv(RES / "scores.csv")
    ref = pd.read_csv(RES / "reference.csv").set_index("treatment")
    original = pd.read_csv(ROOT / "data/coursework.csv").set_index("quantity")
    course = original.loc["six_month_payroll_growth"]
    gamma = pd.read_csv(RES / "volatility_percentiles.csv").set_index("percentile").multiplier
    naive = pd.read_csv(RES / "persistence.csv").set_index("origin")
    a = pd.read_csv(RES / "forecasts.csv")
    sig = json.loads((RES / "signal.json").read_text())
    primary = scores.query('sample=="All" and outcome=="revised"').set_index("treatment")
    initial = scores.query('sample=="All" and outcome=="first_change_sum"').set_index("treatment")
    raw, sm, vol = [primary.loc[n] for n in NAMES]
    refvol = ref.loc[NAMES[2]]
    levels = pd.read_csv(ROOT / "data/september_release.csv").seasonally_adjusted_payroll_thousands
    class_jobs = refvol.origin_payroll * np.expm1(course["median"] / 100)
    gain = levels.iloc[1] - levels.iloc[0]
    b_revised = rms(naive.prediction - naive.revised)
    b_first = rms(naive.prediction - naive.first_change_sum)
    reduction = 1 - vol.rmse / raw.rmse
    lead = (
        "In my doctoral assignment, payrolls were projected to rise by about "
        f"{class_jobs:.0f},000 over the six months from March to September 2026. "
        f"The September release vintage showed a gain of {gain:.0f},000. The class model allowed pandemic shocks to have unusually large variance. "
        "Would keeping those observations improve later forecasts, compared with smoothing them away?"
    )
    findings = [
        (
            "What the assignment got right",
            f"The original model put the March 2020 innovation variance at {original.loc['march_variance_multiplier', 'median']:.1f} times its normal level. The independent fit put it at {gamma.loc[50]:.1f}. For the 2026 illustration, the original median missed by about {abs(gain - class_jobs):.0f},000 jobs, the refitted volatility model by {abs(gain - refvol['median']):.0f},000, and interpolation by {abs(gain - ref.loc[NAMES[1], 'median']):.0f},000.",
        ),
        (
            "The 28 matched forecasts",
            f"On revised employment levels, temporary volatility lowered root mean squared error from {raw.rmse:.0f},000 to {vol.rmse:.0f},000 jobs against constant variance, a {reduction:.1%} reduction. Smoothing the training series and starting from observed payroll conditions gave the smallest error, {sm.rmse:.0f},000. Repeating the previous six-month gain scored {b_revised:.0f},000.",
        ),
        (
            "What changes with the publication being forecast",
            f"Against the sum of six first announcements, the volatility model had the lowest error of the three statistical models ({initial.loc[NAMES[2], 'rmse']:.0f},000 jobs). Repeating the preceding six-month gain was more accurate at {b_first:.0f},000. These are different published measures of hiring. The rankings apply to overlapping windows that all ended in gains.",
        ),
        (
            "How much uncertainty was warranted?",
            f"All three 90% intervals included all 28 revised outcomes. The average width was {raw.width / 1000:.2f} million jobs with constant variance, {vol.width / 1000:.2f} million with temporary volatility and {sm.width / 1000:.2f} million with interpolation. These post-crisis outcomes give little evidence about the ability to forecast a new contraction.",
        ),
    ]

    def first():
        r = sheet()
        r.h("Executive summary")
        r.p(lead)
        for h, body in findings:
            r.h(h, 2)
            r.p(body)
        return r

    links = (
        "[Research report](docs/payroll_after_shocks.pdf) · [One-page brief](docs/executive_summary.pdf) · "
        "[Forecast scores](results/scores.csv) · [September outcome](data/september_release.csv)"
    )
    (ROOT / "EXECUTIVE_SUMMARY.md").write_text(
        "# "
        + TITLE
        + "\n\n"
        + lead
        + "\n\n"
        + "\n\n".join("## " + h + "\n\n" + p for h, p in findings)
        + "\n\n"
        + links
        + "\n"
    )
    save(first(), "executive_summary.pdf", "Executive brief on six-month employment forecasts")
    r = first()
    r.page()
    r.h("1. The forecast that started the study")
    r.p(
        f"The assignment put median six-month growth at {course['median']:.3f} in 100 log points, with a central 90% interval of [{course.p05:.3f}, {course.p95:.3f}]. Against observed growth of {refvol.bls_log_growth:.3f}, the original median understated the gain by {refvol.bls_log_growth - course['median']:.3f} log points. Its archived simulation used 1,000 retained draws and one future path per draw. Its actual computation date is unknown."
    )
    values = [
        [
            lab,
            f"{v.median_log:.3f}",
            f"{v['median']:,.0f}",
            f"{v['median'] - gain:+,.0f}",
            f"{v.p05:,.0f} to {v.p95:,.0f}",
        ]
        for n, lab in zip(NAMES, LABELS)
        for v in [ref.loc[n]]
    ]
    values.insert(
        0,
        [
            "Coursework",
            f"{course['median']:.3f}",
            f"{class_jobs:.0f}",
            f"{class_jobs - gain:+.0f}",
            f"{refvol.origin_payroll * np.expm1(course.p05 / 100):,.0f} to {refvol.origin_payroll * np.expm1(course.p95 / 100):,.0f}",
        ],
    )
    r.table(
        "March to September 2026. Job counts and errors in thousands. Signed error is forecast minus observed gain.",
        ["Model", "Median, log pts", "Median, jobs (000)", "Error (000)", "90% interval (000)"],
        values,
        [1.4, 1, 1.1, 0.9, 1.5],
        True,
    )
    r.p(
        f"The archived percentiles are converted using the forecasting workbook's March level of {refvol.origin_payroll:,.0f} thousand. The observed gain uses a single BLS vintage, from {levels.iloc[0]:,} thousand in March to {levels.iloc[1]:,} thousand in September. The separate monthly gain reported for September was 29 thousand. A simple carry-forward of the preceding six-month gain projected {naive.loc['2026-03', 'prediction']:.0f} thousand."
    )
    r.figure(
        "volatility_posterior.png",
        "Figure 1. The multiplier applies to innovation variance in all five series. The March median is about ten times normal in standard-deviation units. The path after March follows the fixed decay until its April 2022 reset.",
        195,
    )
    r.page()
    r.h("2. What improved on the same dates?")
    r.p(
        "At each origin from March 2022 to June 2024 the models use the same release cutoff, five series, twelve lags and six-month horizon. Revised employment is the difference between the two observed payroll levels in the frozen workbook. Monthly first announcements are also summed across the six months. Those announcements come from different vintages."
    )
    rows = []
    for n, lab in zip(NAMES, LABELS):
        v = primary.loc[n]
        rows.append(
            [
                lab,
                f"{v.rmse:.0f}",
                f"{v.mae:.0f}",
                f"{initial.loc[n, 'rmse']:.0f}",
                f"{v.width / 1000:.2f}",
                f"{int(v.covered)}/28",
            ]
        )
    rows.append(
        [
            "Repeat previous gain",
            f"{b_revised:.0f}",
            f"{(naive.prediction - naive.revised).abs().mean():.0f}",
            f"{b_first:.0f}",
            "—",
            "—",
        ]
    )
    r.table(
        "Historical scores. Errors are in thousands of jobs. The width and coverage columns refer to 90% intervals against revised levels.",
        [
            "Method",
            "Revised RMSE",
            "Revised MAE",
            "Initial sum RMSE",
            "Width (millions)",
            "Covered",
        ],
        rows,
        [1.6, 1, 1, 1.05, 1.02, 0.85],
        True,
    )
    r.figure(
        "payroll_paths.png",
        "Figure 2. Each point shows the cumulative change over six months ending on the horizontal date. The training histories differ. The primary forecasts start from identical observed conditions.",
        205,
    )
    r.p(
        "The two pandemic treatments gave more accurate forecasts on revised levels than either the constant-variance fit or repeating the preceding gain. The latter rule won against the sum of initial announcements. The constant-variance fit produced the widest intervals in each of the six nonoverlapping groups."
    )
    r.page()
    r.h("3. Which conclusions survive a change in the comparison?")
    phases = scores[(scores.outcome == "revised") & (scores["sample"] != "All")].pivot(
        index="sample", columns="treatment", values="rmse"
    )
    phase_counts = (
        scores.query('outcome=="revised" and treatment=="Observed"').set_index("sample").n
    )
    r.table(
        "Nonoverlapping checks. Each row starts one month later and retains every sixth origin. Entries are RMSE in thousands of jobs.",
        ["Origin offset", "Outcomes", *LABELS],
        [[k, int(phase_counts[k]), *[f"{v[n]:.0f}" for n in NAMES]] for k, v in phases.iterrows()],
        [1, 0.8, 1.3, 1.2, 1.55],
        True,
    )
    r.p(
        f"Temporary volatility improved on constant variance in every offset. Interpolation had lower error than temporary volatility in {(phases[NAMES[1]] < phases[NAMES[2]]).sum()} of six offsets. Each offset contains only four or five distinct six-month outcomes. Adjacent forecasts in the full comparison share five months of hiring."
    )
    sens = (
        pd.read_csv(RES / "smoothed_state_scores.csv")
        .query('sample=="All" and outcome=="revised"')
        .iloc[0]
    )
    r.h("The state handed to the model matters", 2)
    r.p(
        f"The main interpolation fit smooths its training history but starts the forecast with the last twelve observed months. When its starting lags are also smoothed, its revised-target error rises from {sm.rmse:.0f},000 to {sens.rmse:.0f},000 jobs. The test holds coefficients fixed at each origin. Early post-pandemic forecasts are particularly sensitive to the conditions used to start the forecast."
    )
    scenario = pd.read_csv(RES / "retrospective_scenario.csv").set_index("future_covariance")
    r.h("What if the shock window had been known in February 2020?", 2)
    r.p(
        f"Using the February 2020 state, the recorded six-month decline through August was {abs(scenario.observed_gain.iloc[0]) / 1000:.2f} million jobs. The lower end of the 90% interval was {scenario.loc['Constant', 'p05'] / 1000:.2f} million under ordinary shocks and {scenario.loc['Scaled', 'p05'] / 1000:.2f} million under elevated shocks. The scaled interval contained the decline. Both calculations use parameters learned through March 2022 and assume the shock dates are known. They illustrate the impact of volatility on the interval and are excluded from the scorecard."
    )
    r.page()
    r.h("4. Replication and limits")
    r.p(
        "The five series are the yield spread, equity prices, employment, industrial production and consumer prices. The spread enters in percentage points. All four levels enter as 100 times their natural logarithms. The expanding sample begins in January 1962. All methods receive the same observed-history prior calibration at each origin. The cutoff waits for payroll, CPI and industrial production releases. The underlying observations are revised."
    )
    r.h("Estimation and numerical checks", 2)
    r.p(
        "The models use a Gaussian coefficient prior centered on each variable's first own lag. The temporary model jointly estimates coefficients, the baseline innovation covariance and a 25-month variance increase using a Gibbs sampler. Each coefficient draw is held fixed for six forecast steps. Six joint innovations generate a simulated path. A second calculation integrates out the future Gaussian shocks exactly conditional on those parameters."
    )
    allfits = pd.concat([a, ref.reset_index()])
    repeat = (
        pd.read_csv(RES / "replay_scores.csv")
        .query('sample=="All" and outcome=="revised"')
        .set_index("treatment")
    )
    r.p(
        f"Across {len(allfits)} fits with four chains each, the largest monitored R-hat was {allfits.max_rhat.max():.4f}. The minimum bulk and tail effective sample sizes were {allfits.min_bulk_ess.min():,.0f} and {allfits.min_tail_ess.min():,.0f}. Changing seeds gave revised-target RMSEs of {repeat.loc[NAMES[0], 'rmse']:.0f}, {repeat.loc[NAMES[1], 'rmse']:.0f} and {repeat.loc[NAMES[2], 'rmse']:.0f} thousand for constant variance, interpolation and temporary volatility. The error rankings held for both target definitions."
    )
    r.h("Independent signal-extraction exercise", 2)
    r.p(
        f"The last of 200 observed messages was {sig['last_message']:.4f}. The filter put the underlying signal at {sig['median']:.4f}, with middle-half bounds of {sig['q25']:.4f} and {sig['q75']:.4f}. These reproduce the coursework values. Conditioning directly on all 200 observations gave the same answer to numerical precision. This control tests a calculation with known parameters."
    )
    r.h("Sources and reproduction", 2)
    r.reference(
        "U.S. Bureau of Labor Statistics. The Employment Situation, September 2026. Table B-1 and The Economics Daily historical levels. October 2026."
    )
    r.reference(
        "Lenza, M. and G. E. Primiceri (2022). How to estimate a vector autoregression after March 2020. Journal of Applied Econometrics, 37, 688–699. DOI 10.1002/jae.2895."
    )
    r.reference(
        "Saved coursework outputs, dated forecasts, replication commands and source links: github.com/jsjaramillod/Time-Series/tree/main/project7"
    )
    save(
        r,
        "payroll_after_shocks.pdf",
        "Six-month forecasts, archived coursework and a matched historical evaluation",
    )


if __name__ == "__main__":
    build()

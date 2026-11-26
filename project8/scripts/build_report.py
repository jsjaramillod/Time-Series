"""Typeset the research report and one-page brief using the released results."""

import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from reportlab import rl_config
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "project4/scripts"))
from report_style import ResearchReport  # noqa: E402

TITLE = "What Can 100 Predictors Tell Us?"
RESULTS = ROOT / "results"


def sheet(title, subtitle):
    page = ResearchReport(title, subtitle, date(2026, 11, 25), RESULTS / "figures")
    page.story[2] = page.paragraph("Sebastian Jaramillo  |  25 November 2026", "date")
    return page


def save(page, name, subject):
    repo = "https://github.com/jsjaramillod/Time-Series/tree/main/project8"
    page.story.append(
        Paragraph(
            f'<a href="{repo}" color="#24574f"><u>Project data notes, forecasts and code</u></a>',
            page.styles["ref"],
        )
    )
    doc = SimpleDocTemplate(
        str(ROOT / "docs" / name),
        pagesize=A4,
        leftMargin=62,
        rightMargin=62,
        topMargin=50,
        bottomMargin=48,
        title=page.title,
        author="Sebastian Jaramillo",
        subject=subject,
    )
    doc.build(page.story, onFirstPage=page._footer, onLaterPages=page._footer)


def build():
    rl_config.invariant = 1
    os.environ.setdefault(
        "SOURCE_DATE_EPOCH", str(int(datetime(2026, 11, 25, 12, tzinfo=timezone.utc).timestamp()))
    )
    scores = pd.read_csv(RESULTS / "scores.csv").set_index(["window", "method"])
    score = scores.loc["All"]
    archive = pd.read_csv(ROOT / "data/coursework.csv").set_index("quantity").value
    method = "Selected"
    base = score.at["PCA 2", "rmse"]
    best = score.at[method, "rmse"]
    gain = 100 * (1 - best / base)
    factor = pd.read_csv(RESULTS / "factor_parameters.csv").set_index(["model", "parameter"])
    bridge = pd.read_csv(RESULTS / "column_audit.csv").set_index("treatment")
    hours = pd.read_csv(RESULTS / "hours_responses.csv").set_index(["end", "tightness", "quarter"])
    orig = hours.loc["1994Q3", 0.5, 1]
    ext = hours.loc["2019Q4", 0.5, 1]
    last = hours.loc["2019Q4", 0.5, 12]
    uncertainty = pd.read_csv(RESULTS / "paired_uncertainty.csv")
    equity = factor.loc["Equities"]
    lead = (
        f"In my doctoral time-series assignment, compressing 100 predictors into two components gave the smallest one-step forecast error, {base:.2f}. "
        f"Re-estimating the three original methods recovered all three scores. Keeping all 100 predictors and shrinking their coefficients lowered the error to {best:.2f} "
        f"on the same 100 outcomes, a {gain:.2f}% reduction. The strength of the penalty was selected on earlier observations."
    )
    why = (
        "Two components summarize what moves together. Some information useful for predicting the target can lie outside those two directions. "
        "Ridge lets every predictor contribute while limiting how strongly the model can respond to any one coefficient. Both approaches improved on ordinary least squares here."
    )
    activity = (
        f"The activity factor recovered the original monthly persistence of {bridge.at['Coursework', 'phi_mean']:.3f}. "
        f"Correcting which series received eight income adjustments raised the estimated income loading from {bridge.at['Coursework', 'W875RX1']:.3f} to {bridge.at['Named series', 'W875RX1']:.3f} with the same sampler. "
        f"After the correction, the full fit estimated persistence of {factor.at[('Named series', 'phi'), 'mean']:.3f} with exceptional months replaced and {factor.at[('Observed', 'phi'), 'mean']:.3f} with all observed changes retained."
    )
    stocks = (
        f"In a separately documented sample of 4,126 trading days, market-factor loadings ranged from {equity.at['loading_IWM', 'mean']:.2f} for IWM to {equity.at['loading_SPY', 'mean']:.2f} for SPY. "
        "The funds tended to rise and fall together on the same day. The coursework preserved the figures but left the original price vintage and date range unverified."
    )
    impact = (
        f"Under the assumption that only technology permanently changes productivity, the assignment estimated a {100 * archive['1994 negative impact probability']:.2f}% chance that hours fell immediately after a favorable technology shock. "
        f"The refit estimated {100 * orig.p_negative:.2f}% through 1994Q3 and {100 * ext.p_negative:.2f}% through 2019Q4. "
        "In the extended sample, the central 80% interval for the twelfth quarter includes zero."
    )
    sections = [
        ("Forecasting with many predictors", lead),
        ("Interpreting the activity index", activity),
        ("A common market movement", stocks),
        ("Hours after a productivity gain", impact),
    ]
    brief = sheet("Common Signals and Forecast Accuracy", "Executive summary")
    for header, body in sections:
        brief.h(header, 2)
        brief.p(body)
    brief.h("Scope of the improvement", 2)
    scope = (
        "The forecast gain is measured in the anonymous target’s units on a single panel. "
        "The activity and market estimates describe historical co-movement. The hours estimates give an economic interpretation conditional on the long-run restriction."
    )
    brief.p(scope)
    save(
        brief,
        "executive_summary.pdf",
        "Executive summary of the forecasting, factor and hours results",
    )
    (ROOT / "EXECUTIVE_SUMMARY.md").write_text(
        "# Common Signals and Forecast Accuracy\n\nSebastian Jaramillo · 25 November 2026\n\n"
        + "\n\n".join(f"## {h}\n\n{b}" for h, b in [*sections, ("Scope of the improvement", scope)])
        + "\n"
    )
    p = sheet(
        TITLE,
        "Evidence from a doctoral forecast comparison, U.S. activity, equity returns and hours worked",
    )
    p.h("The forecast that improved")
    p.p(lead)
    p.p(why)
    rows = []
    for key, label in [
        ("OLS", "All predictors, OLS"),
        ("Ridge 15 raw", "All predictors, ridge (15), raw units"),
        ("PCA 2", "Two principal components"),
        ("Ridge 15", "All predictors, ridge (15), standardized"),
        (method, "Validated ridge (500), standardized"),
        ("AR(4)", "Four target lags alone"),
    ]:
        old = archive.get(key + " RMSE")
        rows.append(
            [
                label,
                "—" if pd.isna(old) else f"{old:.3f}",
                f"{score.at[key, 'rmse']:.3f}",
                f"{score.at[key, 'mae']:.2f}",
            ]
        )
    p.table(
        "Table 1. One-step forecasts for the same 100 target observations.",
        ["Model", "HW RMSE", "Refit RMSE", "Refit MAE"],
        rows,
        [0.52, 0.16, 0.18, 0.14],
    )
    p.p(
        "RMSE takes the square root of the average squared error, giving large misses extra weight. MAE is the average absolute miss. Both are in target units. The source labels the target Y and supplies an observation index. Its economic identity and calendar frequency are unknown."
    )
    p.h("How the choice was made", 2)
    p.p(
        "At each origin, the model saw only earlier outcomes and available predictors. Targets 201–300 determined the number of components or size of the ridge penalty. Ridge with a penalty of 500 won that comparison. The specification was then fixed for targets 301–400, with coefficients updated as new observations arrived."
    )
    p.p(
        "The portfolio’s other exercises ask whether broad activity can be measured and how hours respond to a technology shock. They use distinct data. Only the forecast exercise has observed outcomes against which to measure a reduction in error."
    )
    p.page()
    p.h("Where the forecast gain came from")
    p.p(
        "The original ridge penalty of 15 applied to slopes expressed in their raw units reproduces the submitted score. Giving every regressor unit training variance, including the four target lags, makes the same penalty comparable across variables. This step alone lowered RMSE from 98.45 to 96.46. Selecting the strength of shrinkage in the earlier window lowered it further."
    )
    p.figure(
        "forecast_errors.png",
        "Figure 1. Both curves accumulate errors from target 301 through the observation on the horizontal axis. Early values depend on very few observations.",
        190,
    )
    rows = []
    for win in ["301-350", "351-400", "All"]:
        s = scores.loc[win]
        rows.append(
            [
                win.replace("-", "–"),
                f"{s.at['PCA 2', 'rmse']:.2f}",
                f"{s.at[method, 'rmse']:.2f}",
                f"{s.at['PCA 2', 'rmse'] - s.at[method, 'rmse']:.2f}",
            ]
        )
    p.table(
        "Table 2. Difference in RMSE relative to the best coursework model.",
        ["Target observations", "Two components", "Validated ridge", "Gain"],
        rows,
        [0.34, 0.25, 0.25, 0.16],
    )
    p.p(
        f"The advantage appeared in both halves. Over all 100 cases, MAE also fell from {score.at['PCA 2', 'mae']:.2f} to {score.at[method, 'mae']:.2f}. In a paired bootstrap with five-observation blocks, the central 95% interval for the RMSE reduction was {uncertainty.iloc[0].p025:.2f} to {uncertainty.iloc[0].p975:.2f} target units. With ten-observation blocks it was {uncertainty.iloc[1].p025:.2f} to {uncertainty.iloc[1].p975:.2f}."
    )
    p.p(
        "The historical coursework scores were already known. The extension therefore tests a transparent chronological selection rule on the existing benchmark. An untouched second panel would be needed to measure how well the gain transfers."
    )
    p.page()
    p.h("What a common factor measures")
    p.p(
        "Sales, industrial production, payrolls and income give four monthly views of U.S. activity. The factor moves up when the indicators rise together. Persistence measures how much of a movement carries into the next month. A loading measures how strongly one standardized indicator moves with the factor."
    )
    p.p(activity)
    p.table(
        "Table 3. Monthly persistence and the link between income and activity.",
        ["Estimation and treatment", "Persistence", "Income loading"],
        [
            [
                "Original procedure, original columns",
                f"{bridge.at['Coursework', 'phi_mean']:.3f}",
                f"{bridge.at['Coursework', 'W875RX1']:.3f}",
            ],
            [
                "Original procedure, corrected columns",
                f"{bridge.at['Named series', 'phi_mean']:.3f}",
                f"{bridge.at['Named series', 'W875RX1']:.3f}",
            ],
            *[
                [
                    label,
                    f"{factor.at[(model, 'phi'), 'mean']:.3f}",
                    f"{factor.at[(model, 'loading_W875RX1'), 'mean']:.3f}",
                ]
                for model, label in [
                    ("Named series", "Four chains, named adjustments"),
                    ("Observed", "Four chains, observed changes"),
                ]
            ],
        ],
        [0.58, 0.2, 0.22],
        compact=True,
    )
    p.p(
        "In the two adjusted fits, March 2020 through March 2022 is also replaced by series means. Keeping all observations changes both that period and the eight exceptional income observations. The last row measures their combined sensitivity."
    )
    p.figure(
        "factor_history.png",
        "Figure 2. Posterior median and central 80% interval of each retrospective factor. A unit innovation fixes the scale within each panel. The monthly series uses the named adjustments.",
        225,
    )
    p.p(
        stocks
        + f" The daily persistence estimate was {equity.at['phi', 'mean']:.2f}. Strong same-day co-movement can coexist with little carryover from one trading day to the next."
    )
    p.page()
    p.h("Hours after a favorable technology shock")
    p.p(
        "A permanent gain in output per hour lets firms produce a given quantity with fewer hours. Whether hours actually fall depends on how output and demand respond. The structural exercise asks what U.S. quarterly data imply when only the technology shock can change productivity permanently."
    )
    p.p(impact)
    p.table(
        "Table 4. Impact quarter after a positive one-standard-deviation technology shock.",
        ["Sample and source", "Mean hours response¹", "Probability hours fall"],
        [
            [
                "Coursework through 1994Q3",
                f"{100 * archive['1994 impact mean']:.3f}",
                f"{100 * archive['1994 negative impact probability']:.2f}%",
            ],
            *[
                [f"Refit through {end}", f"{100 * row['mean']:.3f}", f"{100 * row.p_negative:.2f}%"]
                for end, row in [("1994Q3", orig), ("2019Q4", ext)]
            ],
        ],
        [0.45, 0.27, 0.28],
        compact=True,
    )
    p.figure(
        "hours_response.png",
        "Figure 3. Responses cumulate growth effects starting in the impact quarter. Shading gives each refit’s central 80% posterior interval. The dashed line reproduces the coursework median.",
        175,
    )
    p.p(
        f"¹ Responses are 100 times the change in log hours, approximately percent. At quarter 12 the extended median is {100 * last['median']:.3f}, with an 80% interval from {100 * last.p10:.3f} to {100 * last.p90:.3f}. The immediate decline is much better determined than its duration."
    )
    p.p(
        "Both fits use four growth lags. The available levels begin in 1947Q1, so the first usable regression observation is 1948Q2. The earlier fit uses 186 observations and the extension uses 287. Halving the coefficient-prior standard deviations leaves the probability of a negative impact above 96% in both samples."
    )
    p.h("Evidence and replication", 2)
    p.reference(
        "Archived EC 607 Problem Sets 9–10 provide the benchmark scores and hours responses. The project repository includes individual predictions, input fingerprints, posterior intervals and the executed notebook. Sources include FRED series OPHNFB and HOANBS and adjusted closes for SPY, QQQ, IWM and DIA."
    )
    p.reference(
        "Carter and Kohn (1994), On Gibbs sampling for state space models, Biometrika 81, 541–553. Galí (1999), Technology, Employment, and the Business Cycle, American Economic Review 89, 249–271."
    )
    save(
        p,
        "common_signals.pdf",
        "Forecast evaluation, retrospective factor estimates and conditional responses of U.S. hours",
    )


if __name__ == "__main__":
    build()

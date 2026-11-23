# Forecast construction

The common dependent variable is quarterly growth, measured as the first difference of 100 times log real GDP. A four-quarter forecast accumulates four such changes. The level data use a seasonally adjusted annual rate. The constant annualization factor cancels in the ratio of GDP levels.

## Dynamic equation

Let z be 100 times log GDP and d its first difference. The trend AR is

    d[t] = a + b t/100 + c (z[t-1] - z[0])/100
             + p1 d[t-1] + p2 d[t-2] + p3 d[t-3] + e[t].

Here t starts at zero in 1984Q1 and advances by one per quarter. This is a trend AR(4) in levels written in a different basis. Expanding the equation gives lag coefficients (1 + c/100 + p1, p2 - p1, p3 - p2, -p3). The prior is defined in the displayed growth-equation coordinates. The spread model adds T10Y2Y at lags one through four. The drift model uses only a and e. All likelihoods condition on the same first four observations. The [forecast protocol](protocol.md) specifies the prior scales and evaluation windows.

## Predictive calculation

The independent normal coefficient prior and Gamma precision prior yield a Gaussian coefficient conditional at fixed precision. Project 04 supplies a singular-value decomposition of the prior-scaled design and normalized quadrature weights on log precision. At one step, integrate the conditional normal forecast with mean x times the conditional coefficient mean and variance 1/h + x V(h) x', where h is precision. Evaluate the mixture CDF and invert it for each quantile. For drift over k steps the conditional variance becomes k/h + k² Var(a|h,data). A shared drift draw affects every step.

For the four-quarter AR, sample a precision node by its posterior mass and coefficients from the conditional multivariate normal. Carry those values along one four-quarter trajectory, update GDP and its lags, and draw a fresh Gaussian innovation at each step. All finite paths enter the estimate. The 5th percentile of 65,536 terminal growth draws estimates growth at risk. The standard error in the forecast file is the standard deviation of 32 batch estimates divided by the square root of 32 for simulated annual AR quantiles. A zero flags a deterministic quadrature calculation.

The fixed-origin cross-check uses the Gaussian Gibbs sampler in Project 02, with independent random streams for simulation and fresh innovations on every path. We check rank-normalized R-hat (at most 1.01), bulk and tail ESS (at least 400) and agreement of the predictive 5th percentiles within six combined batch standard errors. Identical seeds across the observed and interpolated versions of a model reduce numerical noise in comparisons. The separate result files identify simulation uncertainty and the economic dispersion of outcomes.

## Information and source data

The [release calendar](../data/release_calendar.csv) records the earliest published GDP estimate in each origin-quarter block in the BEA vintage table. The first publication for 2025Q3 was labeled “Initial” and occurred on December 23, 2025. The calendar determines when forecasts can be issued. Revisions in the supplied historical GDP levels still enter the fits and target scores. A retrospective evaluation using contemporaneous data vintages could produce different forecasts and rankings.

FRED publishes the daily 10-year minus 2-year constant-maturity Treasury yield spread in percentage points. The local quarterly series matches the available-day averages to two-decimal rounding. One empty 2026Q2 spread row falls beyond GDP coverage. The local inputs stay external. The fingerprints in the [run record](../results/run.json) identify the exact bytes. The calendar was extracted from the October 29, 2026 BEA workbook with SHA-256 127ef3a7e7c9923e7962dd43693139d44865acbb4e3a6db9b8c52b6de87b4a58.

Sources: [BEA real GDP via FRED](https://fred.stlouisfed.org/series/GDPC1), [Treasury yield spread via FRED](https://fred.stlouisfed.org/series/T10Y2Y), [BEA vintage history workbook](https://apps.bea.gov/national/xls/gdp-gdi-vintage-history.xlsx).

## Scoring

Each origin contributes one forecast per eligible model, training treatment and horizon. Coverage uses inclusive endpoints. A breach is an outcome strictly below the 5th percentile. With u equal to the outcome minus that percentile, quantile loss is 0.05u when u is positive and -0.95u otherwise. Scores use equal weights within a model, horizon and evaluation period.

The four annual origin groups start in 2003Q4, 2004Q1, 2004Q2 and 2004Q3 respectively. Adjacent outcomes within a group span disjoint four-quarter windows, though macroeconomic dependence may persist. Historical scores are descriptive. The endpoint-restricted pandemic comparison contains 16 quarterly and 13 annual outcomes. The 2026Q1 reference forecasts have no scored targets in the frozen snapshot.

## Matched reconstruction of the coursework forecast

The original saved output gives the fifth percentile of growth from 2026Q1 to 2027Q1. The rolling coursework forecasts are reconstructions. The final reconstruction gives −0.2175 versus the archived −0.2180 log percentage points. The script fits the diffuse-prior trend AR(4) on 100 times log real GDP, draws 20,000 coefficient and variance samples with seed 123, and propagates four innovations per path with seed 456. It checks the input fingerprint against the primary study. Both methods use observed GDP through 2019Q4 and the same interpolated log levels at origins from 2022Q1. The interpolation endpoint rules out the intervening eight origins. The final unscored reference is kept in the forecast file and excluded from both loss measures. [Comparison records](../results/course_comparison.csv) · [Reconstruction](../scripts/compare_course_forecasts.py)

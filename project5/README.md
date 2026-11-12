# Project 05: Probabilistic GDP Forecasts and Growth at Risk

How much could the U.S. economy grow over the next year, and how unfavorable could the outcomes be? This project will estimate predictive distributions for real GDP growth, with the lower tail as its central measure of downside risk. It will test whether those forecasts describe historical outcomes and whether financial information improves them.

**Status: research-design draft.** No models, prepared datasets, or reproduced forecasts are implemented here. The historical evaluation below is a proposed extension of the fixed reference experiment.

## Modeling growth and uncertainty

The analysis will start with an AR(4) using GDP history, an intercept, and a deterministic trend:

$$
y_t=c+\delta t+\sum_{i=1}^{4}\phi_i y_{t-i}+\varepsilon_t,
\qquad \varepsilon_t\sim\mathrm{i.i.d.}\ N(0,\sigma^2).
$$

Here $y_t=100\log(GDP_t)$, using quarterly real GDP (`GDPC1`). An ARDL(4,4) will then add four lagged values of a yield-spread predictor, with no contemporaneous predictor, to assess its incremental forecasting value.

Gibbs sampling will generate posterior parameter draws. Predictive simulation will retain each parameter draw along a recursive path while adding a new innovation at every step. The AR produces one- and four-quarter distributions for $g_h=y_{T+h}-y_T$. These are cumulative log-percentage changes, without annualizing quarterly growth. Growth at risk is the fifth percentile of the four-quarter distribution.

The ARDL supports a one-quarter forecast using released predictor values through $x_T$. A four-quarter forecast additionally needs $x_{T+1},x_{T+2},x_{T+3}$. A predictor-path assumption or joint model would be a separate extension.

## Historical forecast evaluation

The central evaluation will use an expanding window across successive quarterly forecast origins. The initial training span, evaluation dates, issue-date rule, and target-vintage policy will be fixed after auditing data coverage and releases, before examining performance.

At each origin, refit using available information and produce forecasts before observing their targets. Compare the AR(4) with a random walk with drift at one and four quarters; compare the ARDL with the AR at one quarter. Models must share origins, targets, and effective training windows.

Score predictive means with RMSE, central 90% intervals with coverage and width, and the fifth-percentile forecast with quantile loss. Report how often outcomes fall below the predicted lower-tail threshold. Together, these assess central accuracy, uncertainty, and downside-risk forecasts.

Four-quarter outcomes overlap across adjacent origins. Report that dependence and use nonoverlapping origins as a sensitivity check; do not treat quarterly annual-growth errors as independent evidence.

A forecast issue date must respect release calendars. Transformations and any tuning must use only the training information. Historical-vintage inputs are required for real-time claims; evaluation with revised data will be labeled pseudo-real-time, with its revision limitations stated.

## Pandemic observations and reference experiment

Observed GDP, including the pandemic contraction and recovery, will be the main analysis. Retaining extreme episodes matters when studying downside risk. The Gaussian, constant-variance model may still misrepresent tails; historical coverage and quantile scores will test that limitation.

A retrospective sensitivity will interpolate transformed GDP for 2020Q1–2021Q4 between 2019Q4 and 2022Q1. Preserve observed and adjusted series separately. No backtest may use the 2022Q1 endpoint before its release; the full adjustment belongs only to information sets where both endpoints are available.

Compare adjusted and observed-data models on the same eligible origins, scoring both against observed GDP under the same target-vintage policy.

The separate fixed reference experiment uses 1984Q1–2026Q1, targeting 2026Q2 and cumulative growth through 2027Q1. These labels identify reference outputs, not a current forecast. Its single origin does not replace the proposed historical evaluation.

## Inputs and decisions pending

Obtain and freeze GDP data and vintages; define the yield spread's maturities, identifiers, units, aggregation, and availability. Confirm proper priors for the intercept, trend, lag coefficients, and disturbance precision, including their parameterization. Evaluation dates and the treatment of first-release versus revised target observations remain to be specified.

## Implementation and deliverables

Develop the baseline and validate its sampler and recursive predictions first, then run the historical evaluation, and finally add the financial predictor. Check lag alignment, trend extrapolation, growth units, sampling diagnostics, and simulation stability of tail quantiles.

The proposed architecture separates data/vintage preparation, AR/ARDL specifications, posterior path simulation, forecast evaluation, and risk summaries. Shared Gaussian sampling from [Project 02](../project2/README.md) remains planned reuse, not an existing dependency.

Deliverables will include a reproducible notebook, data/prior/configuration records, predictive plots, forecast-score and coverage tables by horizon, the pandemic sensitivity comparison, and an interpretation of growth at risk supported by the historical evidence.

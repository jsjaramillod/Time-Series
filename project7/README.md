# Project 07: Employment Forecasts Through Extreme Episodes

How can we forecast employment during extreme episodes without deleting those observations? This project will test when explicitly modeling exceptional volatility helps a macroeconomic forecasting system. The economic focus is the distribution of cumulative employment changes over six months.

**Status: research-design draft.** No models, prepared datasets, or reproduced results are implemented here. The historical evaluation is a proposed extension of the fixed reference experiment.

## Three treatments, one forecasting question

Use the full BVAR from [Project 06](../project6/README.md): five variables, twelve monthly lags, and an intercept. Its inputs combine employment, industrial production, CPI, equity prices, and a Treasury yield spread.

| Treatment | Data | Disturbance covariance |
| --- | --- | --- |
| Observed-data baseline | Retain observed pandemic movements. | Constant. |
| Interpolation benchmark | Retrospective December 2019–January 2022 adjustment. | Constant. |
| Temporary volatility | Retain observed pandemic movements. | Time-varying scalar times a common covariance matrix. |

Match forecast origins, observed targets, vintage rules, effective training windows, shared parameter priors, units, and other transformations. Audit interpolation details and document the different pandemic windows; the treatments differ in more than variance alone.

All three forecasts must use the six-month horizon. The monthly forecast in Project 06 is not an equivalent benchmark. The fixed illustration retains the January 1962–March 2026 sample and a six-month cumulative forecast from March 2026, separately from the multi-origin evaluation.

## Temporary volatility and MCMC

The scaled BVAR is

$$
Y_t=Z_t\beta+\sqrt{s_t}\varepsilon_t,
\qquad \varepsilon_t\sim N(0,\Sigma).
$$

Outside March 2020–March 2022, $s_t=1$. Within those 25 months,

$$
s_t=\gamma\rho^{t-t_0},\qquad \rho=0.9,\qquad \gamma>0,
$$

where $t_0$ is March 2020. The scale returns to one in April 2022. This reference specification is motivated by Lenza and Primiceri (2022).

Gibbs updates will sample coefficients, disturbance precision, and inverse scale. For $d=\gamma^{-1}$, the Gamma prior has kernel $d^{a-1}\exp(-d/b)$, with shape $a$ and scale $b$ pending specification. Conditional on the scale, divide each complete regression row, including its intercept and lagged regressors, by the current-period $\sqrt{s_t}$.

MCMC captures parameter uncertainty. Predictive simulation will propagate joint innovations over six steps, retaining one parameter draw per path. Convert each cumulative employment log-growth draw $g_6$ into $E_T[\exp(g_6)-1]$ before reporting changes in thousands of jobs and predictive intervals.

## Historical evaluation without future information

Freeze the specification, priors, preprocessing, initial training span, origin selection, stress-period definitions, and target-vintage policy before computing scores. Use expanding estimation windows and retain only origins with realized six-month payroll targets.

At each issue date, fit models and preprocessing using only released training observations. Record a common monthly cutoff and the corresponding six-month target.

The primary three-way comparison must use common origins after the interpolation endpoints and the complete reference volatility window were available. It evaluates how treating an already observed crisis affects subsequent forecasts. Dates will be chosen after auditing release calendars and coverage; suitable later stress episodes can be evaluated only within that feasible sample.

The pandemic window and fixed decay rate were chosen retrospectively. Even with historical data vintages, this is a post-crisis evaluation of a frozen retrospective specification. It does not establish that the model could have identified or forecast the crisis prospectively. Revised-data runs must additionally be labeled pseudo-real-time.

During-crisis comparisons that require future interpolation endpoints or the completed volatility window belong to a separate retrospective scenario analysis, excluded from historical forecast-score tables. A prospective shock-detection and scaling rule remains an extension to be specified.

Score predictive means with RMSE and central 90% intervals with coverage and width, using the same observed payroll outcomes for every treatment. Never score against interpolated targets. Six-month outcomes overlap across adjacent monthly origins; report that dependence and repeat comparisons using nonoverlapping origins before drawing performance conclusions.

## Controls, implementation, and deliverables

A subordinate technical control will use the known-parameter Gaussian signal model and 200 observations in `message.csv`. Analytical Kalman filtering will recover the terminal signal distribution and its quartiles. This checks filtering machinery with fixed parameters, independently of the empirical BVAR and its MCMC estimation.

Validate scale dates, complete-row transformations, Gamma updates, covariance conventions, sampling diagnostics, and Monte Carlo uncertainty. Forecast checks must preserve horizon, units, information cutoffs, and matched evaluation samples.

The proposed implementation separates variance scaling, BVAR sampling, predictive paths, and origin-by-origin comparison records. Shared components remain planned. Missing inputs include the macroeconomic data and reference BVAR programs from Project 06, complete priors and scale hyperparameters, vintages and realized targets, and the signal control's data and initial-state prior.

Deliverables will include a reproducible notebook, data/prior/configuration records, a volatility-posterior plot, matched six-month forecast and score tables, interval diagnostics, and a separately labeled retrospective crisis sensitivity. Conclusions will distinguish post-crisis forecasting evidence from technical validation and retrospective scenarios.

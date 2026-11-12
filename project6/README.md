# Project 06: Macroeconomic Information and Payroll Forecasts

Does incorporating macroeconomic and financial information improve employment forecasts, and when does that information justify a more complex model? This project will compare progressively richer information sets, expressing forecasts as job gains or losses, predictive intervals, and the probability of an employment decline.

**Status: research-design draft.** Models, prepared datasets, and reproduced results are not implemented here. The three-model historical comparison is a proposed extension of the fixed reference analysis.

## Comparing information sets

The initial comparison will forecast one month ahead using:

| Model | Information |
| --- | --- |
| Univariate autoregression | Employment history. |
| Small BVAR | Employment, industrial production, and prices. |
| Full BVAR | The small system plus equity prices and a Treasury yield spread. |

For a controlled starting design, propose twelve monthly lags and an intercept in every model. The full BVAR preserves the reference five-variable, twelve-lag specification. Lag choices, variable scaling, and comparable prior rules across model dimensions will be documented before evaluation.

Inputs are `PAYEMS`, `INDPRO`, `CPIAUCSL`, the S&P 500 index, and a spread constructed from `DGS10` minus `TB3MS`. Employment, production, CPI, and equity levels enter in natural logarithms without multiplying by 100. The spread remains in documented yield units. The full reference ordering is spread, log equity prices, log employment, log production, and log CPI; the two yields become one modeled variable.

The BVARs will use independent Normal/Wishart priors for coefficients and disturbance precision, with Minnesota-style shrinkage. Their exact construction and dimension-specific scaling require audit. Sensitivity to shrinkage intensity will assess whether conclusions depend on that choice. Reduced-form coefficients describe conditional relationships, without establishing causal effects.

## From MCMC to interpretable forecasts

Gibbs sampling will represent uncertainty about coefficients and disturbance covariance. Predictive simulation will then draw joint future innovations in each BVAR. For every employment log-growth draw $g^{(s)}$, compute

$$
\Delta E^{(s)}=E_T[\exp(g^{(s)})-1].
$$

With `PAYEMS` retained in thousands of persons, the change is in thousands of jobs. Transform each draw before reporting means, medians, central 90% predictive intervals, and the predictive probability that $\Delta E<0$. This keeps parameter uncertainty, future shocks, and the nonlinear conversion in the forecast distribution.

## Historical evaluation

Use an expanding window over multiple monthly origins, including documented stress periods. Fix the initial training span, evaluation dates, episode definitions, and target-vintage policy after reviewing coverage and releases, before comparing performance.

Use common issue dates and a common last observation month for all models, determined by the latest fully released input set. Forecast the following month's employment change only when that payroll target remains unreleased; record the actual issue date and lead time. Match targets, units, effective training windows, and information cutoffs across models.

Compare predictive means using RMSE, central 90% intervals using coverage and width, and loss probabilities using Brier scores against the observed employment-decline indicator. Report overall and stress-period results, including the number of decline events, without presuming the larger model is superior.

Release calendars and unequal series availability require an explicit alignment policy. Fit scaling, priors or tuning rules, and missing-value treatments using training information only. Historical-vintage inputs are required for real-time claims; revised-data evaluation will be labeled pseudo-real-time. First-release and revised payroll targets must be distinguished consistently.

## Extreme episodes and the reference illustration

Observed data, including pandemic movements, will be the main analysis. The constant-covariance Gaussian model may misrepresent extreme outcomes; interval coverage and decline-probability scores will assess that limitation. Explicit pandemic-volatility modeling belongs to [Project 07](../project7/README.md).

The December 2019–January 2022 interpolation remains a retrospective reference benchmark, with its transformation order and endpoint handling to be audited. Project 07 will compare it with volatility scaling at a common horizon. A backtest cannot use the January 2022 endpoint before its release.

The January 1962–March 2026 sample and April 2026 target remain a separate fixed illustration. They do not define a current forecasting service or replace the multi-origin evaluation.

## Inputs, implementation, and deliverables

Pending inputs are frozen monthly data and vintages, the reference S&P 500 workbook, `BVAR.zip`, and complete prior definitions. Confirm monthly yield aggregation, price conventions, missing observations, and the documented treatment of any October 2025 CPI gap. Evaluation dates and release/target-vintage policies remain to be specified.

Build and validate the employment baseline first, then the small and full systems. Separate data preparation, model/prior specifications, samplers, predictive simulation, and forecast evaluation. Shared components remain proposed. Checks will cover lag alignment, covariance conventions, MCMC diagnostics, joint shocks, employment units, and reproducibility.

Deliverables will include a notebook, data/prior/configuration records, forecast and score tables for all three models, interval and loss-probability diagnostics, shrinkage sensitivity results, and an economic interpretation of when additional information earns its complexity.

# Project 02: Bayesian Monetary Policy and Recession Risk

How does the Federal Reserve respond to inflation and economic slack, and what can financial conditions tell us about future recessions? This project connects those questions through Bayesian inference: estimate policy responses, compare competing economic specifications, and produce recession probabilities that account for parameter uncertainty.

The proposed contribution is a transparent, reusable computational workflow. It starts with a Gaussian interest-rate regression estimated by Gibbs sampling, extends that model to asymmetric responses to economic activity, and then develops a latent-variable probit model for recession risk twelve months ahead. Markov chain Monte Carlo (MCMC) provides the draws needed to quantify uncertainty about parameters, long-run responses, and forecast probabilities.

**Status: research design; implementation and empirical validation are pending.** This directory currently contains this README. No estimation package, executable analysis, or empirically reproduced results have been published here. The required data archives and reference programs remain to be obtained.

The project extends [Bayesian learning for discrete data](../project_01_bayesian_discrete_models) toward dynamic economic relationships and model uncertainty. Reusable Python components will separate data preparation, model specifications, samplers, posterior results, and forecast evaluation.

## Economic questions

The policy analysis asks whether a single response to the output gap adequately describes the data, or whether positive and negative gaps call for different slope coefficients. It also distinguishes a positive long-run response to inflation from a response large enough to increase the real interest rate.

The forecasting analysis asks what the policy rate, equity returns, and the yield curve reveal about the probability of a U.S. recession twelve months later. It will assess forecasts against a simple benchmark while respecting the information available at each forecast origin.

These applications share an emphasis on uncertainty, but answer different questions. The policy regression describes conditional associations; causal conclusions would require additional identification assumptions. The probit model evaluates predictive relationships rather than estimating the causal effect of monetary policy on recessions.

## Modeling strategy

### Baseline interest-rate rule

The quarterly specification is

$$
FF_t = \beta_1 + \beta_2 g_t^\pi + \beta_3 g_t^y + \beta_4 FF_{t-1} + \varepsilon_t,
\qquad \varepsilon_t \sim \mathrm{i.i.d.}\ N(0,\sigma^2),
$$

where $g_t^\pi=\pi_t-\pi_t^*$ is the inflation gap and $g_t^y=y_t-y_t^*$ is the output gap. The reference sample is **1954Q3–2014Q3**, using quarterly average Federal Funds rates, GDP-deflator inflation, trend inflation computed as in Chan, Koop and Potter (2012), real GDP, and CBO potential GDP. The original data and reference code must establish the exact gap units, transformations, and treatment of the initial lag.

The baseline uses independent priors:

$$
\beta\sim N(0_4,10I_4),\qquad
h=1/\sigma^2\sim\operatorname{Gamma}(m=1,v=3).
$$

The Gamma parameters retain the reference notation. Its density and mapping to Python's shape/scale or shape/rate convention must be confirmed from `gen_h.m` before implementation. The independent coefficient prior must retain covariance $10I_4$; scaling it by $1/h$ would change the model.

The Gibbs sampler will alternate between the conditional posteriors of the coefficients and error precision. Planned summaries include OLS estimates, posterior medians, standard deviations, and central 90% credible intervals for the coefficients and error variance.

### Asymmetric responses and model comparison

The extended rule replaces the single output gap with two signed variables:

$$
g_{t,+}^y=\max(g_t^y,0),\qquad
g_{t,-}^y=\min(g_t^y,0).
$$

It retains the intercept, inflation gap, and lagged policy rate while estimating separate slopes on the positive and negative gaps. Negative gaps retain their sign. The five-coefficient prior must be documented consistently with the baseline, including the prior on the additional slope.

Posterior draws of the difference between the two gap coefficients will describe uncertainty about asymmetry. Overlapping marginal intervals or different posterior medians alone will not resolve that question.

Model comparison will use [Chib's (1995) marginal-likelihood method](https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/NormConstants/ChibJASA1995.pdf), evaluating the likelihood, prior, and posterior ordinate at the same parameter point. The two specifications must explain identical observations. The initial comparison will use equal prior model probabilities, log-density calculations, and stable normalization. Sensitivity to parameter and model priors will be reported separately.

### Long-run inflation response

For the baseline rule, define

$$
\theta=\frac{\beta_2}{1-\beta_4}.
$$

The analysis will report both $\Pr(\theta>0\mid D)$ and $\Pr(\theta>1\mid D)$ with distinct labels. The first measures evidence for a positive response; the second corresponds to the Taylor-principle threshold in this specification. A probability calculated at the zero threshold cannot establish the probability of exceeding one.

Draws with $\beta_4$ close to one can produce extreme ratios. Draws outside the stable region can lack a stable long-run interpretation. The analysis will inspect and report these cases; any stability-conditioned summary must state its conditioning explicitly.

### Recession probabilities twelve months ahead

The monthly model is

$$
\Pr(NBER_{t+12}=1\mid x_t,\beta)=\Phi(x_t'\beta),
$$

where $x_t$ contains an intercept, the Federal Funds rate, the three-month S&P 500 growth rate, and the ten-year Treasury yield minus the three-month Treasury bill yield. The reference data span **January 1960–February 2026**. The input programs must establish the equity-return convention, coefficient prior, and predictor/target alignment.

Estimation will use latent Gaussian responses with unit disturbance variance and truncated-normal updates, following [Albert and Chib (1993)](https://www.stat.cmu.edu/~brian/905-2009/all-papers/albert-chib-1993.pdf). The probit sampler will alternate between latent responses and coefficients. It requires a separate sampling routine from the Gaussian policy regression.

Averaging $\Phi(x_T'\beta^{(s)})$ across coefficient draws gives the posterior predictive recession probability. Variation in those probability draws describes parameter uncertainty; simulated binary recession outcomes represent a different object.

The initial evaluation design includes January–December 2007 forecast origins with January–December 2008 targets. A separate reference forecast uses a February 2026 origin and February 2027 target. At every origin, training pairs must include only targets available by that date, allowing for both the twelve-month horizon and recession-dating publication delays.

Evaluation using revised data and retrospective recession dates will be labeled **pseudo-real-time**. A **real-time** claim requires historical data vintages and announcement dates. The effective training window and available information will be recorded explicitly.

## Data and reference inputs

The following inputs are missing from this repository. They are needed to establish the data construction and validate the intended specifications.

| Required input | Purpose | Details to confirm |
| --- | --- | --- |
| `InterestRateRule.zip`, including its data file, `interest_rate_rule_estimate_gibbs.m`, `gen_beta.m`, and `gen_h.m` | Quarterly policy-rule estimation and reference checks | Columns, units, gap construction, initial lag, Gamma convention, and sampler settings. |
| `MarginalLikelihoodLinearRegression.m` | Reference calculation for Chib model comparison | Prior normalization and evaluation of the posterior ordinate. |
| `RecessionForecast.zip`, including `Recession_Forecast_Data.xlsx`, `Probit_Recession_Forecast.m`, `gen_ystar`, and `gen_beta` | Monthly recession-model estimation and reference checks | Coefficient prior, return definition, target shift, and estimation-window rules. |
| Earlier Python scripts or notebooks associated with the reference calculations, if available | Audit numerical discrepancies and implementation choices | Seeds, retained draws, initialization, transformations, and differences from the MATLAB reference programs. |

The first three inputs are needed for faithful replication of the reference specifications. The earlier Python work would help audit discrepancies. Rebuilding the datasets from public sources would be a separately documented extension because revised observations or reconstructed trend inflation may change the results.

A data manifest will record provenance, checksums, source or vintage, frequency, coverage, units, transformations, missing-value handling, and the effective estimation sample. Redistribution rights must be documented before any input data or reference materials are published.

## Reusable Python architecture

The following components are proposed; no classes or package modules have been implemented here yet. The architecture separates economic assumptions from computation so that related specifications can share tested routines.

| Proposed component | Responsibility |
| --- | --- |
| `InterestRateData` / `RecessionData` | Validate dates and units at each frequency, construct labeled design matrices, and align forecast targets. |
| `GaussianRegression` / `BayesianProbit` | Define each likelihood, parameterization, and prior specification. |
| `GaussianGibbsSampler` | Sample regression coefficients and precision under independent Normal/Gamma priors. |
| `ProbitGibbsSampler` | Sample truncated latent responses and probit coefficients. |
| `ChibModelComparison` | Compute log marginal likelihoods and posterior model weights with explicit priors and common-sample checks. |
| `PosteriorResult` | Store named draws, configuration, diagnostics, and data metadata; generate summaries without refitting. |
| `ExpandingWindowEvaluator` | Track forecast origins, target dates, available observations, predictive scores, and benchmark forecasts. |

The symmetric and asymmetric rules will construct different design matrices for the same Gaussian model and sampler. The probit model will use its own sampler while sharing result storage and evaluation conventions. Samplers will accept an explicit random generator and chain configuration.

A future notebook will orchestrate the analysis and explain the economic results. Estimation logic will remain in importable modules, allowing later AR/ARDL applications to reuse the regression machinery without building a general-purpose framework at the outset.

## Implementation sequence and deliverables

Each stage will produce an output that can be reviewed before the next extension.

1. **Validate the baseline regression.** Obtain and audit the policy inputs, resolve the Gamma convention, and implement the Gaussian Gibbs sampler. Check its behavior on known examples before producing the baseline posterior summaries and diagnostics.
2. **Evaluate asymmetry and model uncertainty.** Add signed gap specifications, verify common samples and priors, and produce slope-difference summaries, log marginal likelihoods, posterior model weights, and both long-run inflation-response probabilities.
3. **Develop recession forecasts.** Audit the monthly inputs and prior, implement the latent-variable probit sampler, and evaluate probability forecasts with explicit origin/target dates and information sets.
4. **Assemble a reproducible analysis.** Provide a notebook, dependency specification, data manifest, run configuration, tables, figures, and a validation report explaining numerical discrepancies and Monte Carlo uncertainty.

These are planned deliverables. The current deliverable is the project design in this README. The [portfolio overview](../README.md) will be updated as implementations and validated results become available.

## Validation and evidence standards

Validation will combine mathematical controls, simulated examples, sampling diagnostics, and checks on the information used for forecasting.

- **Internal mathematical control.** A finite comparison between a fair coin and a weighted coin provides an exact check on marginal likelihoods, prior model weights, and normalization. With equal model priors, equal conditional prior mass on head probabilities 1/3 and 2/3 for the weighted coin, and four heads in six flips, the posterior model probabilities are 729/1369 and 640/1369. These values were checked analytically during design preparation; the control will become a software test rather than a portfolio application.
- **Sampler behavior.** Small simulated Gaussian and probit examples will check posterior behavior, positive precision draws, latent-variable truncation, and stable probability calculations. Multiple chains with dispersed starts, trace plots, effective sample sizes, convergence diagnostics, and Monte Carlo standard errors will assess reliability. Similar summaries from different initial values alone do not establish convergence.
- **Data and timing.** Checks will cover unique chronological dates, missing values, signed gap construction, identical samples for model comparison, the twelve-month target shift, and publication delays. Future targets and transformations fitted on future data must not enter training.
- **Predictive performance.** Brier scores, log scores, and a simple benchmark will complement probability plots. Conclusions from the short 2008 evaluation window will remain descriptive.
- **Reference audit.** Existing numerical reference calculations are unverified and are not presented as reproduced findings. A negative value labeled as a term-spread standard deviation and nearly constant historical recession probabilities require investigation of the original code, transformations, and date alignment. The validation report will also check threshold labels, precision-to-variance conversion, and prior conventions. Discrepancies will be explained rather than eliminated by forcing numerical agreement.

The linked methodological papers support the estimation and comparison methods. They do not validate the empirical reference calculations. Independently reproduced findings will be reported only after the input data, implementation, and diagnostics have been checked.

## Connections to the broader portfolio

The [econometric themes](../README.md#econometric-themes) connect this project to persistence, shock propagation, growth risk, and joint macroeconomic forecasting. Gaussian regression sampling and marginal-likelihood comparison can support lag uncertainty and Bayesian model averaging. Posterior simulation can then extend to multi-step GDP forecasts and downside growth risk.

Further applications can investigate joint employment forecasts, changes in volatility, hidden economic states, and common factors across financial series. These extensions require additional data and validation; they are not implementations delivered by this project. The immediate scope remains the baseline policy regression, asymmetric model comparison, and recession-risk model.

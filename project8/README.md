# Project 08: Common Factors, Forecast Evaluation, and Structural Interpretation

How can many economic or financial series reveal a common signal, and what additional assumptions are needed to interpret a shock economically? This project will connect three stages: latent-factor estimation, forecast evaluation with many predictors, and structural interpretation of macroeconomic dynamics.

**Status: local research-design draft.** Data preparation, models, and numerical results have not been implemented or reproduced here. MCMC will support the dynamic-factor and Bayesian VAR analyses; the wide-predictor comparison uses OLS, ridge regression, and principal components as point-forecast methods.

## Common activity and equity-market signals

The first stage will estimate a single dynamic factor from four monthly measures: growth in industrial production, real personal income excluding transfers, real manufacturing and trade sales, and nonfarm payroll employment. Inspect `ci_data.xls` before applying transformations, then demean and standardize the inputs.

The model is

$$
y_{it}=\lambda_i f_t+e_{it},\qquad
f_t=\phi f_{t-1}+\eta_t,
$$

with independent Gaussian measurement disturbances and factor innovation variance normalized to one. A separate sign convention, consistent with the priors and applied jointly to factors and loadings, is needed before giving positive factor values an economic interpretation.

The Gibbs sampler will update loadings, factor persistence, measurement variances, and the complete latent factor path. The path update will use Carter and Kohn (1994) filtering and backward sampling. A filtered or smoothed conditional mean alone is not a draw of that path. The factor is assumed covariance stationary; its prior, sampling constraint, and initial-state distribution must be checked against the reference programs.

A second application will use daily returns $100\Delta\log(P_t)$ for SPY, QQQ, IWM, and DIA, demeaned and standardized. Price adjustment conventions, coverage, and dates remain unspecified. The intended output is a common equity-market signal with posterior uncertainty.

Report loading distributions and factor medians with central 80% intervals. The factor analyses are retrospective inference; any future forecasting use would require transformations and factor estimation restricted to each forecast origin.

## Forecasting with many predictors

The reference panel `Wide_Data.xlsx` contains 400 indexed observations, a target `Y`, and 100 predictors. Its economic identity is not specified. Compare three one-step methods, each using four target lags:

- OLS with all 100 predictors lagged once.
- Ridge with the same regressors, penalizing all slopes but not the intercept, with the fixed reference penalty $\lambda=15$.
- OLS with two lagged principal-component factors extracted from the predictor panel.

Use an expanding window: train through observation 300 to forecast 301, refit through 301 to forecast 302, and continue through forecast 400. The comparison produces 100 forecasts per method and reports RMSE in the target's units.

Ridge centering/scaling and penalty normalization must match the reference implementation. The PCA specification centers the predictors and extracts two components; standardizing them would be a separately documented change. Reestimate transformations and factor weights inside every training window, then apply the training transformation to the forecast inputs. Full-sample factor extraction would introduce future information.

## Technology shocks and hours worked

The structural stage will reassess the productivity–hours relationship using a Bayesian VAR and the long-run identification of Galí (1999), following Blanchard and Quah (1989). Inputs are quarterly `OPHNFB` and `HOANBS`, transformed to first differences of natural logarithms.

The VAR has four lags. Reference estimation windows are 1948Q1–1994Q3 and an extension through 2019Q4. Obtain sufficient presample observations for differencing and lags, and report the effective window. Minnesota-style coefficient priors will use zero own-lag means for these growth-rate variables; the complete prior specification remains to be verified.

The identifying restriction gives the nontechnology shock no long-run effect on the level of productivity. For each valid reduced-form posterior draw, form

$$
\Psi(1)=\left(I-\sum_{\ell=1}^{4}\Pi_\ell\right)^{-1},
\qquad L L'=\Psi(1)\Sigma\Psi(1)',
\qquad B=\Psi(1)^{-1}L,
$$

where $L$ is lower triangular with positive diagonal. Structural innovations have identity covariance, and $B$ maps them into reduced-form innovations. The positive long-run productivity response fixes the technology-shock sign.

Report the posterior probability of a negative impact response of hours, and cumulative log-hours responses over twelve quarters with central 80% intervals. State impact/horizon indexing and shock units explicitly. Check stability and near-singularity before interpreting long-run mappings; any posterior conditioning or discarded mass must be documented.

Compare sample windows with otherwise matched settings. Causal interpretation depends on the long-run restriction. Data revisions and Bayesian estimation also prevent treating this as an exact reproduction of the original published estimates.

## Inputs, reusable components, and validation

Required inputs are `ci_data.xls` and its programs (`ci.m`, `gen_lambda.m`, `gen_phi.m`, `gen_sig2.m`), frozen ETF prices, `Wide_Data.xlsx`, productivity/hours vintages, and the complete factor and BVAR priors. Missing factor sample dates, ETF coverage, variance parameterizations, and state initialization must be resolved before estimation.

The architecture will reuse the proposed state-space/filtering design in [Project 07](../project7/README.md) and BVAR machinery in [Project 06](../project6/README.md). Additional responsibilities are factor-path sampling, an expanding-window evaluator for the three point-forecast methods, and a long-run identification transform. These components remain proposed.

Validation will check normalization and sign handling, latent-path sampling, covariance and prior conventions, MCMC diagnostics, and Monte Carlo uncertainty. Forecast checks will verify lag/target alignment, training-only preprocessing, the fixed ridge penalty, and identical evaluation observations. Structural checks will verify the covariance factorization, zero long-run restriction, sign convention, and cumulative-response construction.

Deliver the stages separately: first the macroeconomic and equity factor estimates with uncertainty, then the wide-predictor forecast record and RMSE comparison, and finally structural impact probabilities and response plots for both samples. A reproducible notebook, data/prior records, and validation report will accompany each stage.

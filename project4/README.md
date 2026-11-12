# Project 04: Lag Uncertainty and Bayesian Model Averaging

How much history does a time-series model need, and how much does that choice affect what we learn about persistence? A short lag specification can miss important dynamics; a larger model introduces additional uncertain parameters. This project will compare autoregressive specifications while retaining uncertainty about the model order in the final inference.

The analysis will connect posterior sampling, marginal likelihoods, and Bayesian model averaging. Its initial application uses a reference series of 200 simulated observations to study persistence and shock responses. It will distinguish choosing one lag order from averaging a shared quantity across several models.

**Status: local research-design draft for review.** No estimation code, model probabilities, posterior figures, or replicated numerical results have been produced here. The reference dataset is missing. The regression, sampling, and dynamic-analysis components described in the related projects are also proposed components, not existing dependencies.

## Research question and scope

The central question is whether conclusions about persistence remain stable when the number of autoregressive lags is uncertain. The candidate set is AR(1) through AR(8). The core analysis will estimate these models, compare their posterior probabilities under alternative coefficient priors, and construct a model-averaged posterior for the sum of AR coefficients.

An AR(8) analysis will also describe uncertainty in finite-horizon impulse responses. The initial scope concerns inference from the supplied simulated series. Forecast comparisons and applications to observed economic data would require separate designs and inputs.

## Comparable models and likelihoods

For each model $M_p$, with $p=1,\ldots,8$, the specification is

$$
y_t=c+\sum_{i=1}^{p}\phi_i y_{t-i}+\varepsilon_t,
\qquad \varepsilon_t\sim\mathrm{i.i.d.}\ N(0,\sigma^2).
$$

Conditional Bayesian estimation will treat the first eight observations as common conditioning information. Every candidate will explain the same target vector, $y_9,\ldots,y_{200}$, giving 192 modeled observations. Smaller models will retain this start date even though they require fewer lags.

This common window makes the conditional marginal likelihoods comparable:

$$
p(y_{9:200}\mid y_{1:8},M_p).
$$

Changing the target observations between models would mix differences in lag specification with differences in the data being explained. The design will preserve target indices and conditioning information explicitly.

## Priors and posterior sampling

The reference coefficient prior has independent $\phi_i\sim N(0,0.5)$, where 0.5 is the variance. Two sensitivity cases will use variances 1 and 0.25. The precision prior is specified as $h=1/\sigma^2\sim\mathrm{Gamma}(1,3)$ in the reference notation.

Two details must be resolved before estimation and marginal-likelihood comparison:

- The exact Gamma density and its mapping to shape/scale or shape/rate parameters.
- The intercept prior, which is not specified in the available reference description.

The implementation must document proper, normalized parameter priors for every candidate model, keep priors on shared parameters consistent across orders, and retain the independent coefficient/precision specification. An arbitrary improper intercept prior would leave the marginal-likelihood calculation without a defined prior normalizing constant.

The proposed Gaussian Gibbs sampler will draw coefficients conditional on precision and precision conditional on coefficients. Its reuse from [Project 02](../project2/README.md) is an architectural plan; that sampler has not yet been implemented. Diagnostics and Monte Carlo uncertainty will accompany the posterior summaries.

## Evidence and posterior model probabilities

The initial model prior assigns equal probability, $P(M_p)=1/8$, to the eight candidate orders. Marginal likelihoods will be calculated with Chib's (1995) method, using the likelihood, prior, and posterior ordinate evaluated at the same parameter point. The methodological reference is linked in Project 02.

For each prior-variance setting, posterior model probabilities are

$$
w_p=P(M_p\mid D)
=\frac{p(D\mid M_p)P(M_p)}{\sum_{k=1}^{8}p(D\mid M_k)P(M_k)},
$$

where $D$ denotes the common target vector conditional on the first eight observations. Calculations will use log densities and stable normalization.

The three prior settings define separate sensitivity analyses. Their model weights will be recomputed independently rather than pooled into one model average. Changes in model probabilities will be interpreted in relation to the prior and its marginal-likelihood penalty; a more diffuse prior is not automatically a neutral comparison.

## Selecting a model and averaging inference

Selecting the order with the largest posterior probability yields one specification. Bayesian model averaging retains the full set of model probabilities when answering a question that is defined under every candidate.

The core shared quantity is

$$
S_p=\sum_{i=1}^{p}\phi_i.
$$

Its model-averaged posterior is the mixture

$$
p_S(s\mid D)=\sum_{p=1}^{8}p_{S_p}(s\mid D,M_p)w_p.
$$

To generate mixture draws, first draw a model index using the posterior weights. Then select a posterior parameter draw from that model and compute its coefficient sum. This preserves both within-model parameter uncertainty and uncertainty about the lag order.

The analysis will compare the AR(8) posterior for the coefficient sum with the model-averaged distribution. The coefficient sum is a useful common summary, but it is not a complete stability diagnostic for an AR model with several lags. Coefficient vectors of different dimensions will not be averaged without a defined common target.

A model-averaged predictive distribution would be a different object: a weighted mixture of model-specific predictive distributions. It is a possible forecasting extension, requiring forecast origins, horizons, benchmarks, and evaluation rules beyond the current inference study.

## Dynamic responses and stability

For the AR(8) specification, each posterior coefficient draw will define impulse responses through

$$
\psi_0=1,\qquad
\psi_j=\sum_{i=1}^{8}\phi_i\psi_{j-i},
$$

with negative-index responses set to zero. The planned output reports posterior medians and central 90% intervals at horizons one through ten. These are marginal intervals at individual horizons, not a simultaneous response band.

The dynamic checks will reuse the companion-matrix reasoning proposed in [Project 03](../project3/README.md). Normal coefficient priors allow unstable parameter draws. Stability will therefore be inspected draw by draw using the companion eigenvalues, rather than inferred only from the coefficient sum or imposed silently by discarding draws.

The core analysis reports finite-horizon responses and coefficient sums. Any later long-run-response calculation must be performed for each model and parameter draw with an explicit stability rule. Conditioning a model average on stability also changes its model weights; filtering draws while retaining the original weights would require a different interpretation.

## Required inputs and unresolved details

The required data file is `sim_arma_data.csv`, containing 200 observations on the reference series. The true generating order, generating coefficients, and random seed are not provided in the available specification. They will not be inferred from a filename or invented to recreate the data.

Before estimation, inspect the file's column names, ordering, missing values, and observation count. Record its provenance and checksum. The reference regression and marginal-likelihood programs, including `gen_beta.m`, `gen_h.m`, and `MarginalLikelihoodLinearRegression.m`, or equivalent parameterization documentation, are needed to resolve prior conventions and validate the evidence calculation.

Additional known-parameter simulations can support software validation. They would be explicitly designed test cases, not substitutes for the missing reference dataset. No observed economic dataset is required for the initial scope.

## Small, reusable Python design

The implementation will reuse the planned Gaussian sampler, marginal-likelihood calculation, posterior-result storage, and dynamic-analysis routines across the related projects. The additional responsibilities can remain small:

| Proposed component | Responsibility |
| --- | --- |
| `ARDesignBuilder` | Build lag matrices for each order while preserving the common target indices and conditioning window. |
| `LagOrderStudy` | Coordinate model/prior configurations and collect posterior draws, diagnostics, and log marginal likelihoods with their metadata. |
| `ModelAveragedInference` | Normalize model weights and generate mixture draws for explicitly defined scalar quantities. |
| `DynamicsAnalyzer` | Transform coefficient draws into stability checks and finite-horizon responses using the proposed Project 03 interface. |

Model fitting, evidence calculations, mixture construction, and plotting will remain separate. The design does not require a new general-purpose modeling library.

## Validation and deliverables

Implementation will proceed in three stages:

1. **Establish the data and priors.** Obtain the reference series, resolve the intercept and Gamma priors, freeze the 192-observation target window, and validate a single AR(8) estimation.
2. **Compare the lag orders.** Estimate AR(1)–AR(8) under each of the three coefficient-prior variances. Check sampler diagnostics and the numerical stability of the Chib calculation before interpreting posterior model probabilities.
3. **Average and explain inference.** Construct the coefficient-sum mixtures, compare them with the fixed AR(8) analysis, and report finite-horizon response uncertainty and sensitivity to priors.

Checks will verify identical target observations across models, lag alignment, normalized model probabilities, positive precision draws, and reproducible configurations. Known finite-mixture examples will validate the averaging procedure, including the limiting case in which all weight is assigned to one model. Companion-matrix and recursive calculations will cross-check response calculations.

Multiple chains, effective sample sizes, convergence diagnostics, and Monte Carlo standard errors will assess sampling reliability. The marginal-likelihood calculation will be checked for sensitivity to the evaluation point and simulation variability. Reference numbers will not be treated as reproduced findings without a validated dataset and executable calculation.

Planned deliverables are a reproducible notebook and configuration, a prior specification and data record, log-evidence/model-probability tables for each sensitivity case, coefficient-sum posterior plots, an AR(8) response plot, and a validation report. Any future forecasting extension must fit models and transformations using only information available at each forecast origin.

The connection across the portfolio is explicit: Project 02 proposes the sampling and evidence machinery, Project 03 explains fixed-parameter dynamics, and Project 04 will use those ideas to study inference when the dynamic specification itself is uncertain.

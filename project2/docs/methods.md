# Model definitions and numerical checks

This document explains the implemented models. All empirical estimates are conditional associations or predictions. Neither application identifies a causal effect of monetary policy.

## Data contracts

`InterestRateData` accepts one strictly chronological observation per quarter. It rejects missing periods, missing values, duplicate dates, and nonpositive GDP levels. It constructs the same transformations as the supplied MATLAB program:

- Response: `FF[1:]` in the workbook's percentage-point units.
- Inflation gap: `PI[1:] - PI_Target[1:]`.
- Output gap: `100 * (log(GDP[1:]) - log(GDP_POT[1:]))`.
- Lagged policy rate: `FF[:-1]`.
- Asymmetric terms: `max(gap, 0)` and `min(gap, 0)`, retaining the negative sign.

The supplied workbook has 240 rows, 1954Q4–2014Q3. Both regressions therefore use the same **239 responses, 1955Q1–2014Q3**. The design's earlier 1954Q3 start cannot be obtained from this file. No missing initial quarters are fabricated. `PI_Target` is accepted as supplied; its trend-inflation estimation and data vintage have not been independently reconstructed.

`RecessionData` parses six-digit `YYYYMM` dates explicitly, checks monthly continuity, and keeps predictors separate from future outcomes. It constructs the MATLAB **simple** three-month equity return, `100*(SP500_t/SP500_(t-3)-1)`, and the ten-year minus three-month yield spread. The supplied workbook contains 794 monthly rows, January 1960–February 2026. The first usable predictor month is April 1960. The first matched target is April 1961. The complete matched sample has 779 observations; the final twelve predictor rows remain available for forecasting even though their future outcomes are absent.

Input SHA-256 checksums, effective dates, units, and unavailable provenance information are recorded in the generated manifests. Original data, reference MATLAB programs, and the earlier notebook are not redistributed.

## Gaussian regression with independent priors

For response vector `y`, design `X`, coefficients `b`, and precision `h`, the likelihood is `y | b,h ~ N(Xb, I/h)`. The priors are independent: `b ~ N(mu,V)` and `h ~ Gamma(a0, rate=b0)`.

Completing the quadratic in `b` gives posterior precision `P = V^-1 + h X'X` and conditional mean `P^-1(V^-1 mu + h X'y)`. Conditional on `b`, precision has shape `a0+n/2` and rate `b0 + (y-Xb)'(y-Xb)/2`.

The coefficient prior is **not** multiplied by the error variance. This distinction changes both the Gibbs updates and model evidence. Linear systems use Cholesky factorization. If `P=LL'`, the random coefficient increment is obtained by solving `L' d=z` for a standard-normal `z`; solving with `L` would give the wrong covariance.

The reference `gen_h.m` defines Gamma parameters through mean `m` and degrees-of-freedom parameter `v`. Its conversion is:

```text
shape = v/2
rate  = v/(2m)
scale = 1/rate = 2m/v
```

Thus `m=1,v=3` gives shape **1.5**, rate **1.5**, mean **1**, and variance **2/3**. The Gibbs sampler stores `sigma2=1/h` and summarizes those transformed draws directly.

The portfolio design specifies `V=10I`, while the supplied MATLAB program uses `V=5I`. The primary run uses `10I`; a separate sensitivity run uses `5I`. Every coefficient, including the intercept and the additional asymmetric slope, receives the stated independent prior. Both regressions use identical observations and the same precision prior.

## Model evidence

[Chib (1995)](https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/NormConstants/ChibJASA1995.pdf) supplies the marginal-likelihood identity. At one common evaluation point `(b*,h*)`, the implementation calculates:

```text
log p(y) = log p(y|b*,h*) + log p(b*) + log p(h*)
           - log p(b*|h*,y) - log p(h*|y)
```

The first posterior ordinate is a normalized multivariate-normal density. The second averages normalized conditional Gamma densities `p(h*|b_draw,y)` over posterior coefficient draws. `logsumexp` prevents underflow. The reported Monte Carlo standard error uses the chain-preserving error of the averaged ordinate and a delta-method transformation to the log scale; it does not quantify prior or model uncertainty.

An independent control integrates coefficients analytically under `y|h ~ N(X mu, I/h + X V X')`, then integrates precision numerically using `z=log(h)` with the Jacobian `exp(z)`. Integration boundaries expand until both kernel tails lie 60 log units below its numerical maximum. The returned quadrature error is a numerical error estimate, not a proof of an exact infinite-domain integral. Tests additionally compare this control against an exactly integrated Gamma-noise case.

Model probabilities normalize log evidence plus log model priors. The primary comparison assigns each model probability 0.5. Sensitivities assign the asymmetric model prior probability 0.25 or 0.75. A finite fair-versus-weighted-coin calculation tests the normalization against exact rational probabilities `729/1369` and `640/1369`.

## Derived policy quantities and model adequacy

The long-run inflation response is `theta=b_inflation/(1-b_lagged_rate)`. `P(theta>0)` and `P(theta>1)` are reported separately. Only the second is the Taylor-principle threshold within this specification. All draws enter the unconditional summaries. The report also gives the share with `abs(b_lagged_rate)>=1`, the share with `abs(1-b_lagged_rate)<0.01`, and the explicitly conditioned probability `P(theta>1 | abs(b_lagged_rate)<1)`. Unstable draws do not have a stable long-run interpretation.

Asymmetry is summarized through joint draws of `b_positive-b_negative`, with a central credible interval and posterior sign probability. Comparing marginal interval overlap is not used as a test of equal slopes. Model probabilities answer a separate question and depend on prior widths.

Fixed-design conditional posterior predictive checks compare residual lag-one autocorrelation against simulated independent Gaussian errors. This checks an assumption of the fitted likelihood while conditioning on the observed lagged policy rate. It is neither a dynamic policy simulation nor an out-of-sample policy forecast. Residual serial dependence, historical regime changes, omitted variables, and the effective lower bound limit the economic interpretation even when MCMC diagnostics pass.

## Probit augmentation

The monthly model is `P(NBER_(t+12)=1 | x_t,b)=Phi(x_t'b)`. The coefficient prior is `N(0,10I)` and the latent disturbance variance is fixed at one for identification.

The sampler follows the latent-normal construction of [Albert and Chib (1993)](https://doi.org/10.1080/01621459.1993.10476321). Latent responses have mean `x_t'b`, unit variance, and are truncated above zero for recession outcomes and below zero otherwise. Conditional on these responses, coefficients follow the corresponding normal regression update with precision fixed at one. [SciPy's truncated-normal convention](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html) uses standardized truncation limits; this is tested even for extreme opposite-sign means.

At a forecast origin, `mean(Phi(x_origin'b_draw))` is the predictive recession probability. The central 90% interval across these probability draws describes **parameter uncertainty about the probability**, not a 90% interval for a binary future outcome. No extra Bernoulli simulation is needed to estimate the mean probability.

## Forecast timing and evaluation

The default exercise makes forecasts at the end of each month in 2007 for the corresponding month in 2008. At origin `o`, a training pair must satisfy both `target_date<=o` and `label_available_date<=o`; its predictor is twelve months earlier than its target. The forecast itself uses the distinct predictor row at `o`. This prevents both future-label leakage and the accidental use of the final training predictor as the forecast predictor.

Without an explicit label-availability calendar, the code assumes availability at the target month end plus a configurable delay. Zero delay is a **retrospective-label assumption**, not a real-time claim. A separate 18-month-delay sensitivity demonstrates the effect of withholding more recent labels; it is not a verified historical release schedule. An optional CSV can supply actual availability dates, but historical predictor vintages and publication times are still needed before claiming real-time performance.

[NBER's dating procedure](https://www.nber.org/research/business-cycle-dating/business-cycle-dating-procedure-frequently-asked-questions) explains that turning points are dated retrospectively. For example, [the December 2007 peak was announced on December 1, 2008](https://www.nber.org/research/business-cycle-dating/business-cycle-dating-committee-announcements). Current recession indicators therefore do not establish when every training label was known.

The benchmark is the Beta(1,1) posterior predictive recession frequency on exactly the same available training targets, `(recessions+1)/(n+2)`. Evaluation reports mean Brier loss and mean negative log score; smaller is better. Probabilities are clipped to `[1e-12,1-1e-12]` only for log-score calculation. No test-set preprocessing or fitted transformations are used.

The 2008 target window contains only recession months in the supplied indicator, so these scores cannot establish discrimination across expansions and recessions or broad calibration. Adjacent twelve-month forecasts also share much of their information. They are descriptive scores, with no independence-based significance claims. The February 2026-origin forecast targets February 2027 and has no observed outcome in the input workbook; it is excluded from scores and is labeled a historical-origin forecast rather than a current forecast.

## Diagnostics and reproducibility

Each fit uses four independent random streams spawned from one recorded NumPy seed. Chains begin at dispersed coefficient values. Warmup is discarded and no thinning is applied to stored draws. Summaries retain the chain dimension and use ArviZ's rank-normalized split R-hat, bulk/tail effective sample sizes, and Monte Carlo standard errors. The computational gate requires `R-hat<=1.01` and both ESS measures `>=400` for every reported parameter and policy-derived quantity. These practical thresholds cannot prove convergence or model correctness; see [Vehtari et al. (2021)](https://doi.org/10.1214/20-BA1221).

Tests independently check analytic conditionals, correlated-normal sampling orientation, exact coin probabilities, evidence by quadrature, Gaussian recovery, a probit posterior against one-dimensional numerical integration, rare-event truncation, date alignment, release-calendar filtering, and future-target exclusion. The notebook imports the package and reads generated results; it contains no duplicate estimation implementation.

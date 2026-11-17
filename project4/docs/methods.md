# Models, priors, and numerical methods

## Observations and the common conditional likelihood

The supplied `sim_arma_data.csv` contains 200 ordered numerical observations in one **headerless** column. The first value is data, not a column name. The loader requires the expected observation count, one numeric column, and finite values; it rejects blank observations rather than silently dropping them. Values are neither standardized nor demeaned. Input SHA-256, count, and summary statistics are recorded. The true data-generating process, generation parameters, seed, economic identity, and calendar frequency remain unverified. The raw file is external to the public repository.

Every candidate $M_p$, $p=1,\ldots,8$, uses

$$y_t=c+\sum_{i=1}^p\phi_i y_{t-i}+\varepsilon_t,\qquad \varepsilon_t\sim N(0,h^{-1}).$$

All models condition on the same initial eight observations and explain $y_9,\ldots,y_{200}$: 192 identical targets. Thus the compared likelihoods are $p(y_{9:200}\mid y_{1:8},M_p)$. Lag $j$ in the first row is $y_{9-j}$. Smaller models do not start earlier. Regression arrays and target indices are immutable, with a fingerprint binding their values and column order. An additional response/target-position fingerprint verifies comparability across all 24 fits.

These are conditional AR likelihoods, products of sequential normal transition densities. Regressors are observed lagged outcomes. Analytical coefficient integration evaluates that likelihood at the observed lag matrices; it does not assert that the entire unconditional time series is a fixed-design Gaussian vector. Initial observations have no extra model-specific density in this comparison. Ordinal positions are used throughout: no dates are fabricated.

## Proper independent priors

The primary specification is

$$c\sim N(0,100),\quad \phi_i\sim N(0,v),\quad h\sim\mathrm{Gamma}(a=1,b=3),$$

with $v=0.5$. The two coefficient-prior sensitivity cases use $v=1$ and $v=0.25$. These numbers are **variances**, not standard deviations. All coefficient priors, including the intercept, are independent of precision. Their covariance is $V=\mathrm{diag}(100,v,\ldots,v)$ and their mean is zero. Models receive equal prior probability $1/8$ within each sensitivity case. Different variance settings are separate studies, not components of one model mixture.

The Gamma density uses **shape and rate**:

$$p(h)=\frac{b^a}{\Gamma(a)}h^{a-1}e^{-bh},\quad h>0.$$

Consequently $E[h]=a/b=1/3$. This is not a shape-scale Gamma and not a mean-degree shorthand. In particular $1/E[h]$ is not $E[1/h]$: with shape one, the prior mean of the error variance diverges. The prior remains proper. The data produce a proper conditional posterior with finite variance summaries.

A proper intercept prior is needed to define absolute marginal likelihoods. The variance 100 is an explicit primary choice, held constant across lag orders. Deterministic integration additionally checks intercept variances 10 and 1000 at $v=0.5$. Another separate sensitivity uses Gamma shape 1.5 and rate 1.5, obtained from the alternative mean-degree convention $(m,\nu)=(1,3)$ via $a=\nu/2$, $b=\nu/(2m)$. Its precision mean is one, so it is a genuinely different prior. These supplementary checks are labeled integration-only; no extra Gibbs fits are claimed.

## Shared Gibbs sampler

Let $\beta=(c,\phi_1,\ldots,\phi_p)'$, with prior mean $m_0=0$. The full conditionals are

$$P_h=V^{-1}+hX'X,\quad m_h=P_h^{-1}(V^{-1}m_0+hX'y),$$

$$\beta\mid h,D\sim N(m_h,P_h^{-1}),$$

$$h\mid\beta,D\sim\mathrm{Gamma}\left(a+n/2,\ b+\tfrac12(y-X\beta)'(y-X\beta)\right).$$

The precision shape does not contain an added $k/2$ term because the coefficient prior is independent of precision. Cholesky solves generate normal draws with covariance $P_h^{-1}$. For $P_h=LL'$, the random increment solves $L'd=z$; solving $Ld=z$ would use the wrong covariance. NumPy's Gamma sampler takes scale, so the implemented draw uses the reciprocal conditional rate.

Project 04 composes the existing Project 02 Gaussian model, normalized priors, Gibbs sampler, posterior container, and Chib evidence machinery. A small shared-sampler metadata adapter supports ordinal labels while preserving existing dated labels. The Project 02 scientific tests are rerun to check that reuse.

Each of 24 main fits uses four independently spawned chains, 3,000 warmup draws and 10,000 retained draws per chain. Its seed is `20261117 + scenario_index*10000 + order*100`, with scenario order `[0.5,1.0,0.25]`. All chain dimensions are preserved. The gate requires rank-normalized split R-hat at most 1.01 and bulk and tail effective sample sizes at least 400, for coefficients, error variance, the coefficient sum, spectral radius, and responses at horizons 1-10. Constant horizon-zero responses are excluded from convergence diagnostics.

## Chib evidence and point checks

Chib's identity [1] is

$$\log p(D\mid M)=\log p(D\mid\beta^*,h^*,M)+\log p(\beta^*,h^*\mid M)-\log p(\beta^*,h^*\mid D,M).$$

The posterior ordinate factors as $p(\beta^*\mid h^*,D)p(h^*\mid D)$. The first factor is a known normal density; the second is estimated by averaging the normalized Gamma conditional density $p(h^*\mid\beta,D)$ over retained coefficient draws. Log-sum-exp avoids underflow. The evaluation point uses the pooled coefficient mean and pooled mean precision. An audit also evaluates precision at 0.8 and 1.2 times that point, with the same coefficient point. The identity should be invariant apart from numerical and Monte Carlo error.

The evidence gate compares each point with independent integration using tolerance `max(0.003, 6*MCSE)` in log units. Actual discrepancies are retained rather than rounded into the tolerance. A separate scalar-mean check uses `max(0.001, 6*MCSE)`; these are computational gates, not economic acceptance criteria.

## Independent SVD integration

The coefficient integral is calculated without reusing the Gibbs coefficient update. Let $V=LL'$, $A=XL=U\operatorname{diag}(s)R'$ and $r=y-Xm_0$. Writing $z_i=(U'r)_i$ and $r_\perp'r_\perp=r'r-\sum_i z_i^2$ gives

$$\log|h^{-1}I+XVX'|=-n\log h+\sum_i\log(1+hs_i^2),$$

$$r'(h^{-1}I+XVX')^{-1}r=h\left(r_\perp'r_\perp+\sum_i\frac{z_i^2}{1+hs_i^2}\right).$$

These identities give the integrated normal likelihood. Precision is integrated in $u=\log h$, including the Jacobian $h$. Adaptive quadrature bounds expand until each log-kernel tail is at least 80 below its located maximum. Relative integration errors are stored. A second coefficient-conditional matrix integral from Project 02 supplies a distinct numerical formulation; their log evidences must agree within $10^{-8}$.

For a scalar contrast $S=a'\beta$, conditional means and variances are also computed through the SVD. Gauss-Legendre integration averages those conditional normal distributions over precision. Node masses must reproduce adaptive normalization within $10^{-8}$; the default 160-node rule doubles if needed. The coefficient-sum contrast is $a=(0,1,\ldots,1)'$. Root finding in the integrated CDF gives central scalar intervals. A survival-function calculation directly averages $\Phi((m_S(h)-x)/s_S(h))$ to avoid cancellation in rare upper-tail probabilities.

Tests additionally integrate precision analytically first and integrate a one-coefficient posterior numerically. This reverses the integration order and checks evidence, mean, variance, and tail probability against the SVD calculation. An unused-regressor model has a closed Gamma evidence integral and unchanged normal coefficient prior, supplying another exact control.

## Posterior weights and shared-quantity mixtures

For each coefficient-prior case,

$$w_p=\frac{p(D\mid M_p)/8}{\sum_{j=1}^8p(D\mid M_j)/8}.$$

Primary weights use Chib evidence. Integration-based weights are retained as controls. A shared scalar has posterior mixture

$$p(S\mid D)=\sum_{p=1}^8w_p p(S_p\mid D,M_p),\qquad S_p=\sum_{i=1}^p\phi_i.$$

The numerical report distinguishes two representations of that same target. Weighted empirical Gibbs distributions give simulation summaries. Integrated conditional normal scalar distributions, using the same Chib weights, give higher-precision reference summaries and rare tails. A reported `integration_*` column identifies the latter. The component distributions, not unequal-length coefficient vectors, are averaged.

For component means $\mu_p$ and variances $V_p$,

$$\operatorname{Var}(S\mid D)=\underbrace{\sum_p w_pV_p}_{\text{within models}}+\underbrace{\sum_p w_p(\mu_p-\bar\mu)^2}_{\text{between models}}.$$

Empirical component variances use the population variance of each finite empirical distribution. Weighted empirical quantiles use the inverse cumulative mass convention. Integrated-reference quantiles solve the mixture CDF directly. Model averaging does not necessarily produce a wider interval than an arbitrary fixed high-order model.

The code also generates 100,000 scalar resamples per sensitivity case: select a model with probability $w_p$, then select a stored coefficient-sum draw from that model. Their means are compared with weighted empirical means. Public reported summaries avoid the extra resampling error by evaluating the weighted empirical or integrated mixture directly. These resamples are not additional independent Gibbs information and cannot recover tail events absent from the original component pools.

## Joint Monte Carlo uncertainty

Per-model log-evidence MCSE uses a delta method applied to the mean conditional density, preserving chain order. Model-weight and Gibbs-based BMA-mean MCSE additionally use 40 joint batches: ten contiguous batches of 1,000 retained draws within each of four chains. Each batch recomputes Chib model weights and coefficient-sum statistics together, retaining their within-batch dependence. The evaluation point is held at the pooled value. Standard deviations across batch estimates divided by the square root of 40 estimate numerical Monte Carlo error.

Batch MCSE is not posterior model uncertainty. In particular, no sampled rare events can yield a deceptively small estimated MCSE. Scalar integration is used for reported rare probabilities; passing mean and convergence checks does not certify tail-event accuracy. The MCSE attached to a Gibbs mixture mean is not attached to its deterministic integration reference.

## Finite responses and stability

For each retained coefficient vector,

$$\psi_0=1,\qquad\psi_j=\sum_{i=1}^{\min(j,p)}\phi_i\psi_{j-i}.$$

Every draw contributes; unstable draws are not removed. Posterior response medians and fifth/ninety-fifth percentiles at horizons 1-10 are pointwise marginal summaries, not a simultaneous band. Model-averaged responses mix the model-specific response distributions at each horizon. Responses are not calculated from an averaged padded coefficient vector.

Batch companion eigenvalues classify stability by $\rho(F)<1-10^{-10}$. A fixed set of posterior draws cross-checks vectorized responses and radii against Project 03's independent companion-matrix interface. A sum below one is necessary but not sufficient for strict stability in this real-coefficient causal AR setting. An AR(2) counterexample with sum below one and an explosive root is included in the tests.

If conditioning on the stability event $A$, the model weights become $w_pP(A\mid D,M_p)/\sum_jw_jP(A\mid D,M_j)$. Separate diagnostic weights implement that rule; they do not replace the primary unconditional mixture. Retained stable-draw proportions are simulation frequencies with limited rare-event resolution, not exact probabilities of one. Infinite-response inference would need an explicitly stated stability-conditioned target and is outside this finite-horizon analysis.

## Reproducibility and scope

Input identity, configuration, seed streams, package versions, source hashes for the three local packages, and public artifact hashes are recorded. Private coefficient archives are separated from public evidence. Cache reuse requires identical data/design fingerprints, priors, sampler settings, and sampling-source hashes. Replaying in a fresh output directory supplies an independent RNG run; verification compares the retained arrays, not only rounded tables. Exact equality is asserted only in the recorded environment.

All inferences remain conditional on the supplied series, common conditioning observations, declared priors, and the eight-model set. No generating AR order, macroeconomic identity, or forecast performance is inferred from the file name or from posterior model preference.

## References

1. Chib, S. (1995). [Marginal Likelihood from the Gibbs Output](https://doi.org/10.1080/01621459.1995.10476635). *Journal of the American Statistical Association*, 90, 1313-1321.
2. Hoeting, J. A., Madigan, D., Raftery, A. E., and Volinsky, C. T. (1999). [Bayesian model averaging: A tutorial](https://doi.org/10.1214/ss/1009212519). *Statistical Science*, 14, 382-401, with discussion. Correction: 15, 193-195; corrected-version notice on the [author's research page](https://sites.stat.washington.edu/raftery/Research/bma_2005.html).
3. Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Buerkner, P.-C. (2021). [Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC](https://doi.org/10.1214/20-BA1221). *Bayesian Analysis*, 16, 667-718.

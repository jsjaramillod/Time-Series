# Model definitions and numerical methods

## Scope and units

All parameters are fixed experimental inputs. Innovations are independent Gaussian draws with variance one in the primary experiments. The package accepts other positive innovation variances and uses their square roots when generating shocks. Observations are synthetic model units; a period is not assigned a calendar frequency. There is no empirical estimation or posterior sampling stage.

## Autoregressive model and stability

For coefficients ordered by lag, the model is

$$y_t=c+\phi_1 y_{t-1}+\cdots+\phi_p y_{t-p}+\varepsilon_t.$$

The simulator requires the explicit initial state $(y_{-1},y_{-2},\ldots,y_{-p})$. Each output starts at $t=0$. This ordering is independent of how a Python array happens to wrap negative indices. The companion matrix has first row $(\phi_1,\ldots,\phi_p)$ and ones on its first subdiagonal. With $e_1=(1,0,\ldots)'$, its state equation is $s_t=c e_1+F s_{t-1}+e_1\varepsilon_t$.

The causal model has stationary second moments when the companion spectral radius is strictly below one. Finite roots of $1-\phi_1z-\cdots-\phi_pz^p$ are reciprocals of nonzero companion eigenvalues. Trailing zero AR coefficients can imply zero eigenvalues without additional finite polynomial roots. The implementation handles that case explicitly.

Numerically, a spectral radius within $10^{-10}$ of one is labeled **unit-circle boundary or numerically near boundary**. Stationary moment calculations reject both that category and unstable inputs. This conservative numerical rule does not assert that every nearly unit-root coefficient is algebraically equal to one. The specified boundary AR(2), $(1.5,-1)$, has conjugate eigenvalues of modulus exactly one.

Independent algebraic controls supplement the root calculations. For the stable AR(3), the maximum absolute row-sum norm of $F^3$ is below one, implying $\rho(F)<1$. For the unstable AR(3), $f(\lambda)=\lambda^3-0.5\lambda^2-0.3\lambda-0.3$ is negative at one and positive at two, so continuity establishes a real root above one. The stable AR(2) eigenvalues are $0.6\pm\sqrt{0.06}$, both inside the unit circle. The boundary eigenvalues are $3/4\pm i\sqrt{7}/4$, whose squared modulus is $9/16+7/16=1$.

## Stationary mean and covariance

Taking expectations gives $\mu=c/(1-\sum_i\phi_i)$ under stability. The denominator is not a stand-alone stationarity test. The centered state covariance solves

$$P=FPF'+Q,\qquad Q=\sigma^2 e_1e_1'.$$

The implementation calls SciPy's discrete Lyapunov solver [2], then checks the residual $FPF'-P+Q$ directly. As an independent numerical control, $P_{11}$ is compared with $\sigma^2\sum_{j=0}^{5000}\psi_j^2$. This is a truncated representation check for the stated stable specifications, not a universal error bound for any nearly unstable input. AR(1) closed-form moments provide a separate exact test [1].

The lag-$k$ autocovariance is $(F^k P)_{11}$. Sample variances from a single path are recorded separately from population stationary variances. No stationary moments are assigned to unstable or boundary models, even though their finite simulated paths have computable sample statistics.

## Controlled impulse responses

A response is the change caused by a **one-unit innovation**, with subsequent innovations set to zero, holding initial values fixed. The horizon-zero response is one. Three implementations compute horizons 0 through 100:

1. Matrix powers: $\psi_j=e_1'F^je_1$.
2. Coefficient recursion: $\psi_0=1$ and $\psi_j=\sum_{i=1}^{\min(j,p)}\phi_i\psi_{j-i}$.
3. A shocked path minus an unshocked path, using identical zero initial states. Subtraction removes deterministic intercept effects, including for models without a stationary mean.

SciPy's `signal.lfilter` [3], with numerator one and denominator $(1,-\phi_1,\ldots,-\phi_p)$, supplies an additional difference-equation implementation. Numerical gates use relative and absolute tolerances of $10^{-11}$. The test suite also checks nonzero intercepts, alternating responses, unstable dynamics, and horizon zero.

For stable models, summing the recursion gives $\sum_{j\ge0}\psi_j=1/(1-\sum_i\phi_i)$. A response total through horizon $H$ includes $H+1$ terms. The reported remainder is the analytical infinite sum minus that finite sum. An unstable or boundary denominator is not reported as a convergent response total. These calculations concern one-time innovations; they should not be described as observed policy effects.

## Simulation design

The primary seed is 20261115, using NumPy's `default_rng` with its recorded PCG64 generator. Each experiment retains 200 observations:

| Experiment | Coefficients | Intercept | Initial lagged observations | Burn-in | Seed |
| --- | --- | --- | --- | --- | --- |
| Persistent AR(1) | 0.9 | 0 | 0 | 500 | 20261115 |
| Mean-adjusted AR(1) | 0.9 | 0.5 | 5 | 500 | 20261115 |
| Stable AR(3) | 0.5, 0.2, 0.2 | 0 | 0, 0, 0 | 0 | 20261116 |
| Unstable AR(3) | 0.5, 0.3, 0.3 | 0 | 0, 0, 0 | 0 | 20261116 |
| Response AR(2) | 1.2, -0.3 | 0 | 0, 0 | 500 | 20261117 |
| Boundary AR(2) | 1.5, -1 | 0 | 0, 0 | 0 | 20261116 |
| MA(2) | 1, 1.5, 1.3, including contemporaneous coefficient | 0 | Two zero past innovations | 500 | 20261118 |

AR(1) simulations share all 700 innovations. Their initial states differ by the stationary mean shift, so every retained observation differs by five. The two AR(3) simulations share all 200 innovations and initial conditions; their coefficient change is the only changed input. No burn-in is imposed on the unstable example. The stable AR(3) zero-state path remains a finite transient illustration. Starting stable examples at their mean and using finite burn-in elsewhere limits, but does not mathematically eliminate, initialization effects.

Every configuration records parameters, settings, initial-state ordering, seed, and full innovation-array checksum. CSV files preserve retained values and innovations. Runtime versions and package-source/output SHA-256 hashes identify the recorded calculation. Reproduction across different numerical libraries is assessed with numerical tolerances; exact equality is checked only within the recorded environment.

## Mean precision experiment

For $n$ stationary observations, the exact variance of their mean is

$$V_n=\frac{n\gamma_0+2\sum_{k=1}^{n-1}(n-k)\gamma_k}{n^2}.$$

This follows by summing all $n^2$ entries of the observation covariance matrix. The tests independently construct that Toeplitz matrix for AR(1). The comparison $\gamma_0/n$ sets all cross-time covariances to zero while holding marginal variance fixed. The ratio $V_n/(\gamma_0/n)$ is the finite-window variance inflation factor. The effective independent observation count for this particular mean is $\gamma_0/V_n$; it is not an effective sample size for every statistic or for MCMC output generally.

There are 10,000 independently generated AR(1) replications of length 200, with coefficient 0.9, intercept zero and innovation variance one, using seed 20261215. Each replication starts from an independent draw from the **exact stationary Gaussian distribution**, so the finite-window mean is exactly Gaussian. The two nominal 95% intervals use the known covariance-aware standard error and the known marginal-variance independence approximation. These are repeated-sample confidence intervals under specified parameters, not posterior intervals or feasible intervals with estimated coefficients.

The exact independence-interval coverage is $2\Phi(1.959964\sqrt{(\gamma_0/n)/V_n})-1$. The empirical covariance-aware coverage has binomial Monte Carlo standard error approximately $\sqrt{0.95(0.05)/10000}$. The standard error for the simulated mean-variance estimate is $V_n\sqrt{2/(R-1)}$ under independent Gaussian replications. Stochastic Monte Carlo deviations are reported, while deterministic identities form the pass/fail pipeline gates.

## Finite moving average

For $y_t=\sum_{j=0}^q\theta_j\varepsilon_{t-j}$ with finite coefficients and white-noise innovations, the autocovariance is $\gamma_k=\sigma^2\sum_{j=0}^{q-k}\theta_j\theta_{j+k}$ for $0\le k\le q$, and zero for higher lags. The experiment uses $\theta=(1,1.5,1.3)$, yielding variance 4.94 and autocovariances 3.45 and 1.30. Its polynomial roots have modulus 0.877058, inside the unit circle. The specified innovation representation is therefore noninvertible even though the process is covariance stationary. Invertibility concerns recovering innovations from past and present observations; it is separate from existence of stationary second moments.

## Primary references

1. Sargent, T. J. and Stachurski, J. [AR(1) Processes](https://python.quantecon.org/ar1_processes.html), QuantEcon. Stationary Gaussian distributions and AR(1) moment recursions.
2. SciPy documentation. [Discrete Lyapunov equation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_lyapunov.html). Solver convention: $AXA^H-X+Q=0$.
3. SciPy documentation. [Linear filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html). Difference-equation coefficient and initial-condition conventions.

All numeric findings are generated by this project's experiments, rather than copied from these references.

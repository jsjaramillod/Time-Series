# Executive summary

In my third doctoral assignment, a one-unit shock in a two-lag model generated a response of **1.20** one period later. Summing the computed responses gave **9.99999946**. The portfolio reproduces that path and verifies its theoretical cumulative effect of **10**. A second experiment asks how much confidence to place in an average of persistent observations. Across 10,000 simulations with known parameters, an intended 95% interval covered the true mean **35.27%** of the time under independence and **95.00%** when it used the serial covariance.

## Persistence and shock propagation

The analysis quantifies how dynamic specifications change the duration, accumulated effect, and variability of shocks. It uses fixed parameters and synthetic Gaussian innovations. The conclusions describe controlled mechanisms rather than estimated macroeconomic relationships. The [research report](docs/persistence_shock_propagation.pdf) develops the findings, figures, and supporting calculations.

## Main findings

**The same long-run effect can conceal different transmission paths.** Three stable models have a cumulative response of ten to a one-unit innovation. By horizon 20, the AR(1), AR(2), and AR(3) have accumulated 8.906, 9.677, and 7.394 units respectively. Their stationary variances are 5.263, 7.429, and 3.492, despite identical innovation variance. A shared infinite response total determines neither the speed of adjustment nor unconditional volatility.

**Short-run amplification can coexist with stability.** The AR(2) coefficients (1.2, -0.3) produce a response of 1.20 one period after a unit innovation, followed by decay. Its largest companion eigenvalue is 0.844949, below one. A first-lag coefficient above one is therefore insufficient to classify a multiple-lag process as unstable.

**Persistence sharply reduces the information in a time average.** In the stationary AR(1) with coefficient 0.9, the variance of a 200-observation average is 0.476316. Treating those observations as independent gives 0.026316, underestimating the variance by a factor of 18.10. The exact finite-window precision corresponds to 11.05 independent observations for estimating this mean. This statistic is specific to the model, window, and quantity being estimated.

**Ignoring serial dependence can invalidate nominal interval coverage.** Across 10,000 independent stationary replications, the nominal 95% interval constructed under independence covers the known mean only 35.27% of the time. Its exact model-implied coverage is 35.50%. The interval using the known serial covariance attains 95.00% empirical coverage. The Monte Carlo standard error at 95% coverage is 0.22 percentage points. These are repeated-sample intervals under known parameters, not posterior credible intervals or feasible intervals using estimated coefficients.

**A level shift is different from a persistence change.** Raising the AR(1) intercept from zero to 0.5 shifts the stationary mean from zero to five while leaving the response coefficients and variance unchanged. Paired simulations share all innovations and differ by five at every retained observation, up to floating-point precision. The shock half-life remains 6.58 generic periods.

**Stability and invertibility are distinct properties.** The AR(3) coefficient change from (0.5, 0.2, 0.2) to (0.5, 0.3, 0.3) raises the spectral radius from 0.939481 to 1.054380. Only the first has stationary moments in the causal representation. A separate boundary AR(2) has unit-modulus complex eigenvalues. The finite MA(2) control remains covariance stationary with variance 4.94, while its specified innovation representation is noninvertible.

## Evidence and interpretation

Responses through horizon 100 agree across recursion, companion-matrix powers, a controlled shock experiment, and an independent SciPy filter. The largest three-method discrepancy across all six AR experiments is below 3.0e-13 model units. Covariance calculations satisfy the Lyapunov equation and agree with an independent squared-response sum. The package records seeds, initial states, burn-in, versions, and artifact checksums; the notebook reads the same generated results.

These findings motivate checking stability and serial covariance before interpreting dynamic coefficients or interval precision. They do not identify observed economic causal effects, estimate real-world persistence, or establish forecast performance. Periods have no imposed monthly or quarterly meaning. An empirical extension requires a separately specified dataset and inference model.

See the [README](README.md), [methods and derivations](docs/methods.md), [computed results](results/report.md), and [executed notebook](notebooks/project03_analysis.ipynb).

# Persistence and shock propagation: computed results

All parameters are fixed design inputs. The observations are synthetic, measured in generic model units and periods. No economic coefficients are estimated.

## Dynamic classifications and response totals

| Experiment | classification | spectral_radius | stationary_mean | stationary_variance | long_run_sum | cumulative_at_20 | cumulative_at_100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ar1_persistent | stable | 0.90000000 | 0.00000000 | 5.26315789 | 10.00000000 | 8.90581011 | 9.99976095 |
| ar1_mean_shift | stable | 0.90000000 | 5.00000000 | 5.26315789 | 10.00000000 | 8.90581011 | 9.99976095 |
| ar3_stable | stable | 0.93948066 | 0.00000000 | 3.49206349 | 10.00000000 | 7.39377317 | 9.98233801 |
| ar3_unstable | unstable | 1.05438011 | Undefined | Undefined | Undefined | 21.38058989 | 2159.88963100 |
| ar2_response | stable | 0.84494897 | 0.00000000 | 7.42857143 | 10.00000000 | 9.67663697 | 9.99999955 |
| ar2_boundary | unit-circle boundary or numerically near boundary | 1.00000000 | Undefined | Undefined | Undefined | 4.10756588 | 2.96812328 |

Stationary moments and convergent long-run sums are undefined for the unstable and unit-circle specifications. Finite paths and finite response sums remain computable; they do not establish stationarity.

## Persistence and precision

For the AR(1) coefficient 0.9 and innovation variance one, stationary variance is 5.26315789 and the exact shock half-life is 6.578813 periods. An intercept change from zero to 0.5 shifts the stationary mean from zero to five without changing centered dynamics.

For a stationary window of 200 observations, the exact variance of its mean is 0.47631579, versus 0.02631579 under an independence assumption using the same marginal variance. Their ratio is 18.100000, equivalent to 11.049724 independent observations for estimating the mean.

Across 10,000 independent stationary replications, empirical 95% interval coverage is 95.00% when accounting for correlation and 35.27% under independence. Exact model-implied coverage of the independence interval is 35.4979%. Parameters and covariance are known; these are repeated-sample coverage statements, not posterior credible intervals.

## Moving-average control

The MA(2) polynomial is 1 + 1.5L + 1.3L^2. Its variance is 4.94; autocovariances at lags 1 and 2 are 3.45 and 1.30. Higher-lag autocovariances vanish. Its root moduli are 0.87705802; the supplied innovation representation is noninvertible despite covariance stationarity.

## Numerical verification and interpretation

All deterministic controls pass: **True**. Responses through horizon 100 agree across matrix powers, recursion, a controlled shock experiment, and an independent SciPy linear filter. Stationary covariance satisfies the Lyapunov equation and agrees with a 5,001-term squared-response sum. Detailed errors are recorded in `numerical_validation.json`.

The AR(3) path comparison uses exactly the same 200 innovations, zero lagged states and no burn-in. The stable coefficient vector is (0.5, 0.2, 0.2); the unstable vector is (0.5, 0.3, 0.3). AR(1) paired paths share 700 innovations and discard 500 warmup observations. All settings and source/output checksums are recorded. One simulated path cannot test an economic hypothesis or prove stationarity.

Read [methods](../docs/methods.md), the [executive summary](../EXECUTIVE_SUMMARY.md), and the [research report](../docs/persistence_shock_propagation.pdf). The executed notebook inspects these artifacts without implementing another simulator.

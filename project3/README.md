# Project 03: Persistence and Shock Propagation in Time Series

An economic disturbance can fade quickly, persist for many periods, or reveal that a model has no stable long-run behavior. Understanding that distinction is essential before interpreting a dynamic regression or using it to forecast. This project will use controlled time-series simulations to connect autoregressive parameters with persistence, covariance stationarity, and impulse responses.

The analysis will explain how shocks travel through a model and verify those responses through equivalent calculations. It will compare persistent, stable, and unstable specifications, distinguish an intercept from an unconditional mean, and show when a cumulative long-run response is well defined.

**Status: local design draft for review.** This directory currently contains this README. No simulator, notebook, figures, or numerical experiment results have been implemented or reproduced here. The core analysis will use known parameters and synthetic data. Posterior estimation and empirical applications are optional future extensions.

## Questions and scope

The project addresses four connected questions:

- How do autoregressive coefficients determine whether a disturbance decays, oscillates, or grows?
- How can two series with the same innovation variance have very different unconditional variability?
- How does an intercept determine the level around which a stable process fluctuates?
- Can companion-matrix calculations and a controlled shock experiment recover the same response at every horizon?

The core scope comprises fixed-parameter AR/MA examples, covariance-stationarity analysis, AR(1) and AR(3) simulations, and AR(2) impulse responses. Simulation here means generating innovations and evolving a known process. MCMC estimation and posterior response bands would require a separate inference stage.

## Model and dynamic representation

The simulation model is

$$
y_t=c+\sum_{i=1}^{p}\phi_i y_{t-i}+\varepsilon_t,
\qquad \varepsilon_t\sim\mathrm{i.i.d.}\ N(0,\sigma^2).
$$

With the state vector $s_t=(y_t,y_{t-1},\ldots,y_{t-p+1})'$, it becomes

$$
s_t=c e_1+F s_{t-1}+e_1\varepsilon_t,
$$

where the first row of the companion matrix $F$ contains the AR coefficients, its first subdiagonal contains ones, and $e_1$ selects the first state component.

For the causal representation used here, covariance stationarity requires all companion eigenvalues to have modulus strictly below one. Equivalently, all roots of the AR characteristic polynomial

$$
\phi(z)=1-\phi_1z-\cdots-\phi_p z^p
$$

must lie outside the unit circle. Finite characteristic roots are reciprocals of nonzero companion eigenvalues. Complex roots must be assessed by their modulus; roots on the boundary require separate treatment from stable roots.

For stable specifications, the unconditional mean is

$$
\mu=\frac{c}{1-\sum_{i=1}^{p}\phi_i}.
$$

For an AR(1), the unconditional variance is

$$
\operatorname{Var}(y_t)=\frac{\sigma^2}{1-\phi^2}.
$$

This provides a direct explanation for why variability increases as $|\phi|$ approaches one while innovation variance stays fixed. A finite simulated path will illustrate these properties; it will not establish stationarity by itself.

## Controlled experiments

The following specifications define the initial analysis. Their parameters are fixed inputs, not estimated economic relationships. Gaussian simulation experiments will use innovation variance one unless a configuration explicitly states otherwise.

| Experiment | Specification | Question it isolates |
| --- | --- | --- |
| Persistent AR(1) | $c=0$, $\phi=0.9$ | How does a stable but persistent process differ from its innovations? |
| Mean adjustment | $c=0.5$, $\phi=0.9$ | How does changing the intercept produce an unconditional mean of five without changing persistence? |
| Stable AR(3) | $c=0$, $\phi=(0.5,0.2,0.2)$ | How do several lags combine to propagate shocks in a stable process? |
| Unstable AR(3) | $c=0$, $\phi=(0.5,0.3,0.3)$ | How can a change in lag coefficients move the process outside the stability region? |
| AR(2) response | $c=0$, $\phi=(1.2,-0.3)$ | Do different impulse-response calculations agree? |
| Unit-circle boundary | $c=0$, $\phi=(1.5,-1)$ | Why do complex roots with modulus one fail the strict stability condition? |

A finite moving-average example,

$$
y_t=\varepsilon_t+1.5\varepsilon_{t-1}+1.3\varepsilon_{t-2},
$$

will provide a stationarity control. Its finite innovation representation has constant second moments under white-noise innovations. It helps distinguish the stationarity question from the invertibility question.

Each simulation will record its seed, initial state, sample size, and burn-in. Stable examples can start at their unconditional mean with a documented burn-in to limit initialization effects. Unstable examples will use explicit initial values and finite horizons without assigning them stationary moments. Shared innovation sequences across selected comparisons are a proposed design choice for isolating parameter effects.

## Impulse responses and long-run effects

The response to a one-unit innovation at horizon $j$ is

$$
\psi_j=\frac{\partial y_{t+j}}{\partial\varepsilon_t}
=e_1'F^j e_1,\qquad j=0,1,\ldots.
$$

The AR(2) example will be evaluated through three equivalent calculations:

1. **Companion-matrix powers:** extract the $(1,1)$ element of $F^j$.
2. **A controlled unit-shock experiment:** initialize the stable process at its mean, apply one unit innovation, set subsequent innovations to zero, and measure deviations from that mean.
3. **The coefficient recursion:** use $\psi_0=1$, $\psi_j=\sum_{i=1}^{p}\phi_i\psi_{j-i}$, and $\psi_j=0$ for negative indices as a numerical cross-check.

The initial comparison will cover horizons zero through twenty. Extending to horizon one hundred will examine convergence of the cumulative response. For a stable process,

$$
\sum_{j=0}^{\infty}\psi_j=\frac{1}{1-\sum_{i=1}^{p}\phi_i}.
$$

For the specified AR(2), this analytical target is ten. The finite-horizon sum must be labeled as an approximation to that limit. It will be reported as the sum of deviations following a one-time innovation, with the shock normalization and horizon stated explicitly.

## Inputs and reproducibility

No external empirical dataset is required for the core project. Inputs will be explicit configurations containing the intercept, coefficient vector, innovation variance, initial values, random seed, simulation length, burn-in, and response horizon.

The implementation will distinguish variance from standard deviation when generating innovations and retain generated series together with their configurations. The analytical targets in this README are validation references, not results from an implemented pipeline.

## Proposed object-oriented design

These components are proposed and have not been implemented.

| Component | Responsibility |
| --- | --- |
| `ARSpecification` | Store and validate coefficients, intercept, and innovation variance; expose the model order without conflating parameters with simulation settings. |
| `ARSimulator` | Generate innovations and recursive paths from a specification, explicit initial state, and random generator; support controlled shock experiments. |
| `DynamicsAnalyzer` | Construct the companion matrix, inspect eigenvalues and characteristic roots, compute supported stationary moments, and evaluate impulse responses. |
| `SimulationResult` | Store the generated series, innovation sequence, initial state, and configuration for reproducible comparisons. |
| `ImpulseResponseResult` | Store horizons, responses, cumulative sums, method labels, and numerical agreement measures. |

Plotting functions and a notebook will consume these results without embedding simulation logic in figure code. The simulator and dynamic analysis will remain usable independently: a root calculation should not require a random simulation, and a shock response should not depend on a noisy realized path.

## Validation and planned deliverables

Validation will link analytical identities with agreement between computational methods:

- Check companion-matrix dimensions and eigenvalue/root relationships, including complex and unit-circle cases. Document numerical tolerance and flag values near the boundary.
- Verify the AR(1) mean and variance formulas and the specified mean adjustment. Sample moments will be interpreted with finite-sample variation and persistence in mind.
- Check the stable and unstable AR(3) classifications algebraically as well as numerically.
- Require matrix-based, controlled-shock, and recursive responses to agree within a stated floating-point tolerance over the comparison horizon.
- Compare finite cumulative responses with their analytical limit only when stability conditions hold. Preserve the one-unit shock normalization and horizon-zero response.
- Check that fixed seeds and configurations reproduce paths, and that burn-in and initial conditions are recorded rather than hidden in defaults.

Planned deliverables are a small Python implementation, a reproducible analysis notebook, configuration records, simulation comparisons, impulse-response and cumulative-response plots, and a short interpretation of the connection between stability, persistence, and long-run effects. No empirical findings or posterior uncertainty bands are claimed at the design stage.

## Connections and optional extensions

This project complements [monetary policy and recession risk](../project2/README.md) by examining the dynamics underlying lagged relationships. It supplies a foundation for later lag selection, multi-step forecasting, and interpreting dynamic responses.

Reusable interfaces, shared-innovation comparisons, and explicit numerical agreement checks are proposed engineering additions to the core analytical and simulation scope. Further extensions could investigate a real economic series or propagate posterior coefficient draws into response distributions. Those would require a separately specified dataset or inference model; neither is part of the current implementation plan.

# Executive summary

Read the [full research report (PDF)](docs/monetary_policy_recession_risk.pdf) for the economic analysis, figures, and supporting estimates.

## Purpose and approach

The project examines two questions: how the Federal Reserve's policy rate is associated with inflation and economic slack, and whether financial indicators help forecast recession risk twelve months ahead. It implements Bayesian Gaussian policy rules and an expanding-window Bayesian probit model in a Python package with explicit data, model, sampling, and evaluation responsibilities.

The policy application compares symmetric and asymmetric output-gap responses over **239 quarters, 1955Q1–2014Q3**. The primary independent coefficient prior is `N(0,10I)`; `N(0,5I)` provides a concentration sensitivity. The monthly application uses the supplied January 1960–February 2026 data, a simple three-month equity return, the policy rate, and the term spread. Each specification uses four chains with 10,000 retained draws per chain after 3,000 warmup draws.

## Principal findings

The symmetric policy rule receives **96.4% posterior model probability** under equal model prior probabilities and the primary coefficient prior; the sensitivity result is **95.1%**. These weights compare the two specified models conditional on their priors and sample. They do not establish that either model is true.

The symmetric rule's long-run inflation response has a posterior median of **1.60**, a central 90% interval of **[0.95, 2.48]**, and a **93.4% probability of exceeding one**. Residual lag-one autocorrelation is approximately **0.176**, and a conditional posterior predictive check challenges the independent-error assumption. These estimates describe conditional associations, not causal policy effects.

For the twelve observed 2008 targets, the probit Brier loss is **0.756**, versus **0.777** for a matched training-frequency benchmark. Its negative log score is **2.154**, versus **2.134** for the benchmark; lower is better for both measures. The mixed ranking and a window containing only recession outcomes do not support a broad claim of forecasting superiority.

The **February 2026-origin** probability of recession in **February 2027** is **8.6%**, with a central 90% interval for the probability of **[6.3%, 11.3%]**. Its outcome is absent and excluded from evaluation. This is a historical-origin exercise, not a forecast updated to the report date. An assumed 18-month label delay raises that probability to **9.9%**, illustrating sensitivity to information availability.

## Evidence and limits

The local verification repeated all **30 model fits**, reproduced retained draw arrays exactly in the recorded environment, and passed **22 scientific tests**. Chib log-evidence estimates agree with independent quadrature within **0.0001 log units**. Chain diagnostics satisfy the configured gates, but convergence and numerical agreement do not resolve residual dependence, structural changes, or data-vintage limitations.

Forecast evaluation is **pseudo-real-time**: historical predictor vintages and verified label-announcement calendars are unavailable. The 18-month delay is an assumption, not a reconstructed publication schedule. Source checksums identify the supplied files without certifying their original economic provenance.

The project's contribution is a reproducible analytical workflow that makes assumptions, timing, numerical controls, and limitations inspectable. See the [README](README.md) for execution, the [empirical report](results/report.md) for tables and diagnostics, and [methods](docs/methods.md) for derivations.

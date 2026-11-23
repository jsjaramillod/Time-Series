# Executive summary

In my fourth doctoral assignment, I compared and averaged eight possible lag orders. The model using the previous two observations received **71.54%** posterior probability. The portfolio confirms that weight to two decimal places. For the eight-lag model, the mean sum of coefficients was **0.8006** in class and **0.8008** in the portfolio's integration check. The new question is how much these conclusions depend on prior beliefs: the two-lag model's weight ranges from **60.01% to 80.96%**, while model-averaged persistence stays between **0.8430 and 0.8504**.

## Lag uncertainty and model-averaged persistence

The study asks whether conclusions about persistence depend on selecting an autoregressive lag order. It compares AR(1)-AR(8), carries their posterior weights into a shared coefficient-sum distribution, and evaluates finite-horizon shock responses. The supplied series has 200 ordered observations and an unverified generating process. All models condition on the initial eight observations and explain the same 192 targets. The [research report](docs/lag_uncertainty_model_averaging.pdf) develops the analysis and supporting estimates.

## Model preference is more prior-sensitive than the shared coefficient-sum estimate

Under the primary independent coefficient prior, AR(2) receives **71.54% posterior model probability**. AR(4), AR(3), and AR(5) receive approximately **13.03%, 9.11%, and 5.38%**. AR(8) receives only **0.0114%**. These weights compare the declared alternatives; they do not identify the unknown generating process.

AR(2) remains the leading order under coefficient-prior variances 0.25 and 1, but its probability changes to **60.01%** and **80.96%**. More diffuse coefficient priors change the integrated evidence penalty for larger models. Across these settings, the model-averaged coefficient-sum mean remains between **0.8430 and 0.8504**. Stable conclusions about a shared quantity can coexist with visibly prior-sensitive model weights.

## Accounting for model uncertainty changes both the center and dispersion

The primary model-averaged coefficient-sum mean is **0.8478**, with central 90% interval **[0.7901, 0.9029]**. The selected AR(2) has mean **0.8518** and interval **[0.7998, 0.9037]**. Fixed AR(8) instead has mean **0.8008** and interval **[0.7276, 0.8739]**.

Within-model variation and differences between model-specific means both enter the mixture variance. Between-model uncertainty contributes **10.38%** of the primary total. The model average is wider than the selected AR(2) distribution while remaining narrower than AR(8). This comparison shows why accounting for model uncertainty does not imply wider intervals relative to every fixed specification.

The reported scalar reference summaries integrate precision after analytically integrating Gaussian coefficients, using Chib model weights. The Gibbs-based mixture mean is **0.84771**, with a joint-batch Monte Carlo standard error of approximately **0.00010**; its integration reference is **0.84777**. The two calculations target the same posterior quantity and separate simulation noise from posterior uncertainty.

## Finite shock responses reveal specification-dependent timing

Under model averaging, the one-period response to a one-unit innovation has posterior median **1.235**, with pointwise 90% interval **[1.122, 1.351]**. At horizon ten, its median is **0.072**, compared with **-0.129** under AR(8). The respective intervals **[-0.149, 0.324]** and **[-0.395, 0.168]** both include zero. The difference in central response does not establish a robust sign reversal.

All draws, including unstable ones, contribute to finite responses. The intervals are marginal at each horizon, not a simultaneous band. Responses are averaged as shared quantities; unequal-length coefficient vectors are not padded and averaged into an artificial model.

## Numerical reliability includes recognizing rare-event limits

The main study contains 24 Gibbs fits, with four chains and 10,000 retained draws per chain after 3,000 warmup iterations. Every parameter and continuous-derived diagnostic gate passes. Chib baseline log evidences agree with independent integration within **0.000306 log units**. A second integral based on conditional matrices agrees with the independent SVD calculation to approximately **6e-13 log units**.

Good convergence and accurate posterior means do not guarantee reliable rare-event counts. The primary integrated probability of a coefficient sum above one is approximately **3.44e-6**. For AR(2) alone, its probability is about **2.26e-6**, implying only **0.09 expected events in 40,000 posterior draws**. Zero sampled events therefore provide weak information about this tail. A direct integrated survival function resolves it without interpreting zero counts as zero probability.

A coefficient sum below one is not a sufficient stability test. Companion eigenvalues are inspected draw by draw, and model weights conditional on stability are kept separate from the primary weights. Extremely high observed stable-draw shares are reported as simulation frequencies rather than exact probabilities of one.

## Scope

The priors are proper and explicit, including a separate intercept prior and shape-rate Gamma convention. Additional intercept and precision-prior sensitivities use deterministic integration, separately from the 24 main fits. Source-data identity, seeds, diagnostics, code hashes, and artifact checksums make the calculation inspectable. The generating process and economic identity remain unverified; this is conditional inference for the supplied series, not an out-of-sample forecasting evaluation or an estimate of a causal economic effect.

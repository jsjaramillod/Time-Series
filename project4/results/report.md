# Lag uncertainty and model-averaged persistence: computed results

The supplied headerless file contains 200 ordered observations. All candidates condition on the same first eight and explain observations 9-200 (192 targets). The generating process, its parameters, seed, and any economic identity remain unverified.

## Model comparison

| Coefficient prior variance | AR order | Chib log evidence | Model probability | Log-evidence MCSE |
| --- | --- | --- | --- | --- |
| 0.5 | 1 | -302.239264 | 0.00000352 | 0.000071 |
| 0.5 | 2 | -290.017193 | 0.71535722 | 0.000094 |
| 0.5 | 3 | -292.078092 | 0.09109321 | 0.000114 |
| 0.5 | 4 | -291.720202 | 0.13029133 | 0.000139 |
| 0.5 | 5 | -292.605293 | 0.05376822 | 0.000147 |
| 0.5 | 6 | -294.459585 | 0.00841815 | 0.000161 |
| 0.5 | 7 | -296.636516 | 0.00095453 | 0.000189 |
| 0.5 | 8 | -298.763071 | 0.00011382 | 0.000206 |
| 1 | 1 | -302.186153 | 0.00000363 | 0.000071 |
| 1 | 2 | -289.872207 | 0.80964340 | 0.000097 |
| 1 | 3 | -292.221908 | 0.07723810 | 0.000111 |
| 1 | 4 | -292.110492 | 0.08634131 | 0.000131 |
| 1 | 5 | -293.394980 | 0.02389862 | 0.000151 |
| 1 | 6 | -295.597329 | 0.00264183 | 0.000178 |
| 1 | 7 | -298.107322 | 0.00021470 | 0.000190 |
| 1 | 8 | -300.563880 | 0.00001841 | 0.000196 |
| 0.25 | 1 | -302.689838 | 0.00000486 | 0.000069 |
| 0.25 | 2 | -290.966376 | 0.60012347 | 0.000098 |
| 0.25 | 3 | -292.775711 | 0.09827809 | 0.000121 |
| 0.25 | 4 | -292.248815 | 0.16645025 | 0.000144 |
| 0.25 | 5 | -292.691099 | 0.10695553 | 0.000156 |
| 0.25 | 6 | -294.191631 | 0.02385231 | 0.000173 |
| 0.25 | 7 | -296.048300 | 0.00372554 | 0.000188 |
| 0.25 | 8 | -297.857914 | 0.00060994 | 0.000210 |

The primary maximum-probability order is AR(2), with posterior probability 71.535722%. Equal model prior probabilities and the declared independent parameter priors define these conditional comparisons.

## Coefficient-sum inference

The model-averaged integrated-reference mean is 0.847769; its central 90% interval is [0.790141, 0.902894]. The fixed AR(8) mean is 0.800792, with interval [0.727621, 0.873913]. These scalar reference calculations integrate precision after analytically integrating coefficients; Gibbs mixtures provide an independent simulation comparison.

Within-model posterior variance contributes 0.00105923 and between-model variance contributes 0.00012268. The latter is 10.380% of the mixture variance. The selected-model interval is [0.799848, 0.903685]. Model averaging need not widen inference relative to an arbitrary fixed high-order model.

The integrated-reference probability that the coefficient sum exceeds one is 3.441213883e-06. Tail event counts in finite Gibbs samples can be zero; zero observed events are not reported as proof of zero posterior probability. A coefficient sum below one is not a sufficient stability condition.

## Dynamics and stability

Finite responses use a one-unit innovation, including a contemporaneous response of one. Horizons 1-10 have marginal 90% posterior intervals, not simultaneous bands. Both fixed AR(8) and model-averaged responses preserve all draws, including unstable draws. Model averaging is performed on the responses, not by padding and averaging coefficient vectors.

The model-weighted proportion of retained draws classified as stable is 0.99999999. This is a simulation proportion, not an exact probability of one. Separate weights conditional on the stability event are recorded and are not substituted for the primary weights.

## Reproduction and limitations

The main study runs 24 model/prior fits with the recorded independent chain streams. Additional intercept and Gamma-prior checks use deterministic integration and do not claim extra Gibbs fits. Diagnostic gates, evidence at three evaluation points, SVD and conditional-matrix integration agreement, scalar posterior controls, and companion-response checks are recorded in validation.json and the control tables.

Data remain external, and private draw archives are ignored. Parameter inference is conditional on the supplied series and finite model set. No generating order is inferred from the file name, and no forecast superiority or empirical causal effect is claimed. See [methods](../docs/methods.md), the [executive summary](../EXECUTIVE_SUMMARY.md), and the [research report](../docs/lag_uncertainty_model_averaging.pdf).

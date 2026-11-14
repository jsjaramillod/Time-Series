# Reference audit

The implementation was checked against the supplied local MATLAB programs and the earlier `homeworks.ipynb`. Stored notebook outputs were treated as historical references, never as proof that the new implementation reproduced a result. The original files remain unchanged and are not included in this repository.

| Item | Evidence in supplied material | Implementation decision |
| --- | --- | --- |
| Policy coefficient prior | Design says `10I`; `interest_rate_rule_estimate_gibbs.m` and several notebook cells use `5I`. | Primary `10I` and separate `5I` sensitivity; same scheme for both competing models. |
| Gamma convention | `gen_h.m` explicitly converts mean/degree parameters to MATLAB shape/scale. | Shape `v/2`, rate `v/(2m)`; for `(m,v)=(1,3)`, both equal 1.5. |
| Sample coverage | Workbook begins 1954Q4 and MATLAB drops the first response to form a lag. | Report the actual effective 1955Q1–2014Q3 sample, 239 observations. |
| Inflation gap | `PI-PI_Target` in MATLAB. | Retain supplied series; do not claim independent reproduction of trend inflation. |
| Output gap | MATLAB uses `100*(log(GDP)-log(GDP_POT))`. | Same transformation; negative asymmetric gap retains its sign. |
| Long-run threshold | Earlier notebook computes the sign probability at zero. | Report positive response and response above one separately, with stability diagnostics. |
| Recession equity return | MATLAB uses simple three-month returns; earlier notebook uses log returns. | Primary results follow MATLAB simple percent returns; do not force agreement with log-return outputs. |
| Latest forecast predictor | Earlier notebook sets `x_T=X[-1]` after dropping future targets. | Preserve all feature rows and select the actual requested origin independently from training pairs. |
| Training window | Earlier notebook filters predictor dates while its response is shifted twelve months into the future. | Filter on target maturity **and** assumed/provided label availability. |
| Date parsing | Source uses numeric `YYYYMM` dates. | Explicit calendar parsing; never default nanosecond interpretation. |
| Forecast naming | Earlier plot labels the 2007 exercise real time. | Label revised-data, retrospective-label evaluation pseudo-real-time. |
| Probability calculation | MATLAB averages simulated binary predictions; Python also calculates CDF means. | Average conditional probit probabilities to remove unnecessary Bernoulli simulation noise. |
| Standalone marginal-likelihood program | `MarginalLikelihoodLinearRegression.m` was not found in the supplied directory. | Implement Chib's published identity and validate against independent quadrature and exact controls. Do not claim comparison with the missing file. |
| Negative reported standard deviation | Mentioned as a concern in the project design; no verified reference summary establishes its origin. | Compute SD from draws; do not invent an explanation or copy an impossible value. |

The policy-run metadata includes SHA-256 hashes of the available Gaussian MATLAB programs. The recession manifest hashes its source workbook. The notebook and MATLAB files do not establish the original economic data download dates, vintages, or redistribution rights; these remain explicitly unverified.

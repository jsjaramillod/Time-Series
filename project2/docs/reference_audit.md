# Specification audit

This document records the statistical and data-timing choices behind the reported results. Each choice is explicit so that readers can reproduce the calculation and distinguish numerical validation from economic interpretation.

| Item | Specification and justification | Verification or limitation |
| --- | --- | --- |
| Policy coefficient prior | The primary independent prior is `N(0,10I)`. A separate `N(0,5I)` run measures sensitivity to a more concentrated coefficient prior. | Both policy models use the same prior scheme and the same effective sample. Model probabilities remain conditional on these choices. |
| Precision prior | With mean-degree parameters `(m,v)`, use Gamma shape `v/2` and rate `v/(2m)`. For `(1,3)`, both equal 1.5. | Conditional-distribution tests check rate/scale conversion. This prior is independent of the coefficient prior. |
| Quarterly sample | The workbook begins in 1954Q4. Constructing a one-quarter rate lag leaves 239 observations, 1955Q1–2014Q3. | Calendar validation rejects missing or duplicate quarters. Earlier periods are not reconstructed. |
| Inflation gap | Use the supplied `PI-PI_Target`. | The construction and historical vintage of the supplied inflation target have not been independently verified. |
| Output gap | Use `100*(log(GDP)-log(GDP_POT))`. In the asymmetric rule, `min(gap,0)` retains its negative sign. | Positive input levels and the design-matrix transformations are checked. |
| Long-run response | Calculate the inflation coefficient divided by `1-lagged_rate` for every posterior draw. Report probabilities above zero and above one separately. | Report unstable and near-unit-root draws; a sign probability is not evidence for a response greater than one. |
| Equity return | Use the simple three-month percentage return, `100*(SP500_t/SP500_(t-3)-1)`. | The return convention and its three-month lag are tested explicitly. It is not interchangeable with a log return. |
| Forecast-origin features | Preserve feature rows independently of whether a future target exists. | The February 2026 predictor can generate a February 2027 probability while its unobserved outcome remains excluded from scoring. |
| Training window | Require both target maturity and assumed or supplied label availability at each origin. | Tests exclude future labels. The default zero delay is retrospective; 18 months is an assumed sensitivity, not a verified release calendar. |
| Monthly dates | Parse numeric `YYYYMM` explicitly as calendar months. | Validate monthly continuity and reject malformed or duplicate dates. |
| Forecast interpretation | Describe the exercise as pseudo-real-time. | Revised predictors and unavailable historical vintages prevent a verified real-time claim. |
| Probability calculation | Average conditional probit CDF values over posterior draws. | This integrates parameter uncertainty without additional Bernoulli simulation noise. Probability intervals concern the probability parameter, not the binary realization. |
| Marginal likelihood | Apply Chib's posterior-ordinate identity with explicit prior normalization. | Independent quadrature and exact controls validate the calculation. Agreement does not establish economic adequacy. |
| Posterior dispersion | Compute standard deviations directly from retained draws. | Negative standard deviations are invalid; no unexplained historical value is carried forward. |

Input manifests record workbook SHA-256 checksums, observed coverage, and effective samples. These checksums establish byte identity for reproduction; they do not establish original download dates, historical data vintages, or redistribution rights. Those aspects remain unverified. See [methods](methods.md) for equations and primary methodological references.

## Reconstructing the original forecast dates

The coursework workbook stored dates as numeric YYYYMM. The original historical fit interpreted those values as timestamps near 1970, admitting all 779 labeled pairs even at the January 2007 origin. It also selected the last row with a known forward outcome for its endpoint prediction. That row held February 2025 predictors. The portfolio separates feature availability from label availability and uses a simple three-month equity return in place of the coursework log return.

On the twelve realized 2008 targets, the saved probabilities had Brier loss 0.767 and negative log loss 2.086. The dated portfolio forecasts scored 0.756 and 2.154. A separate historical-frequency benchmark scored 0.777 and 2.134. Lower is better on each metric. All twelve outcomes were recessions and the coursework fit had access to future information. These scores describe a single episode. [Saved probabilities](../../coursework_results.csv) · [Dated forecasts](../results/recession/recession_forecasts.csv)

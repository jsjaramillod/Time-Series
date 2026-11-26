# Time Series and Bayesian Econometrics

An applied econometrics portfolio focused on Bayesian inference, temporal dynamics, and probabilistic forecasting. The projects examine how assumptions about economic relationships, model structure, and uncertainty affect the conclusions we draw from data.

The central questions connect learning and prediction: how beliefs change with new observations, how economic shocks propagate, and how models translate incomplete information into forecasts. Bayesian updating anchors the existing work. Project 02 uses Markov chain Monte Carlo (MCMC), including Gibbs sampling, to study monetary policy and recession risk. The GDP and payroll studies evaluate dated forecasts, including a six-month employment comparison through an extreme episode. Project 08 reproduces a 100-predictor forecast comparison and estimates common activity and market factors and the response of hours to technology.

## Projects and current status

| Project | Economic or statistical question | Material and implementation status |
| --- | --- | --- |
| [01 — Bayesian Learning and Posterior Prediction for Discrete Data](project1/README.md) | How do prior beliefs and observations shape posterior predictions? | Existing [Python notebook](project1/project_01_bayesian_discrete_models.ipynb), PGA scoring data, figures, and project description. |
| [02 — Bayesian Monetary Policy and Recession Risk](project2/README.md) | Do policy responses differ across economic conditions, and which financial signals anticipate recessions? | Implemented Python package, scientific tests, empirical policy analysis, and pseudo-real-time recession forecasts; documented model and data limitations. |
| [03 — Persistence and Shock Propagation in Time Series](project3/README.md) | How do autoregressive dynamics determine stability, persistence, and the response to a controlled shock? | Implemented Python package, controlled simulations, scientific controls, executed notebook, and research report. |
| [04 — Lag Uncertainty and Bayesian Model Averaging](project4/README.md) | How much history does a model need, and how does lag uncertainty affect inference about persistence? | Implemented conditional AR comparison, Gibbs sampling, independent evidence controls, model-averaged posteriors, and research report. |
| [05 — Probabilistic GDP Forecasts and Growth at Risk](project5/README.md) | How much could U.S. GDP grow over the next year, and how well do predictive distributions describe downside outcomes? | Executed quarterly GDP backtest, release calendar, downside-risk forecasts and pandemic sensitivity. |
| [06 — Macroeconomic Information and Payroll Forecasts](project6/README.md) | When does additional macroeconomic and financial information justify greater complexity in one-month employment forecasts? | Three Bayesian models scored on 216 payroll releases, with first-release and revised targets, shrinkage checks and a fixed April 2026 illustration. |
| [07 — Employment Forecasts Through Extreme Episodes](project7/README.md) | When does modeling exceptional volatility help six-month employment forecasts while retaining extreme observations? | Implemented six-month BVAR comparison, 28 matched origins, scored September 2026 coursework forecast, and a verified signal-extraction exercise. |
| [08 — Common Factors, Forecast Evaluation, and Structural Interpretation](project8/README.md) | Can common signals aid prediction, and what long-run assumptions support the interpretation of technology shocks? | Reproduced three coursework scores, lowered holdout RMSE by 1.67%, audited activity-factor inputs and re-estimated U.S. hours responses. |

The first five studies start with results from my doctoral time-series coursework. The PGA study reproduces the predictive mean and checks its upper tail. The policy study reproduces the original model odds and repairs the recession forecast dates. The shock experiment tests interval coverage, and the lag-order study tests sensitivity to prior beliefs. For GDP, a comparison of 78 matched annual forecasts finds a smaller overall error for the portfolio model and a smaller error for the coursework method over 13 post-pandemic origins. Project 06 starts from the original payroll estimate and tests its pandemic smoothing rule on matched dates. Project 07 evaluates the original six-month payroll forecast against the September 2026 release and compares three pandemic treatments across 28 matched origins. Project 08 recovers all three original forecast scores and reduces RMSE from 97.22 to 95.60 using a ridge penalty chosen on earlier observations.

[Saved coursework estimates](coursework_results.csv) are transcribed from the author's archived notebook. Each row identifies its output cell. The source notebook's SHA-256 is `8cb38041e00a5569e3167f3bb8cf90567daf5892520f434a1124f3c104a1f242`. The [GDP comparison](project5/results/course_comparison.csv) contains newly reconstructed forecasts and their outcomes.

## Repository contents

The following paths exist in the repository. `project_01_bayesian_discrete_models` is the legacy entry point for the Project 01 overview.

```text
Time-Series/
├── README.md
├── coursework_results.csv
├── project_01_bayesian_discrete_models
├── project1/
│   ├── README.md
│   ├── project_01_bayesian_discrete_models.ipynb
│   ├── data/
│   │   └── pga_scores.xlsx
│   └── figures/
├── project2/
│   ├── README.md
│   ├── pyproject.toml
│   ├── requirements-repro.txt
│   ├── src/bayes_policy/
│   ├── tests/
│   ├── docs/
│   ├── notebooks/
│   └── results/
├── project3/
│   ├── README.md
│   ├── pyproject.toml
│   ├── src/shock_dynamics/
│   ├── tests/
│   ├── docs/
│   ├── notebooks/
│   └── results/
├── project4/
│   ├── README.md
│   ├── pyproject.toml
│   ├── src/lag_uncertainty/
│   ├── tests/
│   ├── docs/
│   ├── notebooks/
│   └── results/
├── project5/
│   ├── README.md
│   ├── src/gdp_risk/
│   ├── data/
│   ├── tests/
│   ├── notebooks/
│   ├── docs/
│   └── results/
├── project6/
│   ├── README.md
│   ├── src/payroll_risk/
│   ├── data/
│   ├── tests/
│   ├── notebooks/
│   ├── docs/
│   └── results/
├── project7/
│   ├── README.md
│   ├── src/payroll_volatility/
│   ├── data/
│   ├── tests/
│   ├── notebooks/
│   ├── docs/
│   └── results/
└── project8/
    ├── src/common_signals/
    ├── notebooks/
    ├── docs/
    └── results/
```

Read the existing Project 01 notebook on GitHub or open it locally in Jupyter after preparing a Python environment with NumPy, pandas, SciPy, Matplotlib, and an Excel reader such as openpyxl. Project 02 has its own installable package, dependency specification, command-line pipelines, and scientific test suite; see its README for reproduction commands. Project 03 also has an installable numerical package and reproducible pipeline; its synthetic experiments require no external dataset. Project 04 reuses the shared packages for conditional AR comparison and Bayesian model averaging; its source series remains external. Project 05 includes an executable forecast evaluation and verified source fingerprints. Project 06 includes a payroll forecast package and dated forecast records. Project 07 implements temporary-volatility estimation, a six-month payroll comparison and an analytical signal filter. Project 08 provides forecast-by-forecast comparisons, factor posterior summaries and quarterly hours responses.

<a id="research-progression"></a>

## Econometric themes

These themes connect the concrete projects in the catalog. Bayesian learning has existing materials in Project 01. Policy-response inference and recession probabilities are implemented in Project 02. Persistence and controlled shock transmission are implemented in Project 03, and lag-order comparison and model-averaged inference in Project 04. Project 05 evaluates predictive distributions for GDP. Project 06 evaluates how real and financial indicators change employment forecasts. Project 07 tests the treatment of past shocks at a common six-month horizon. Project 08 quantifies the predictive gain from shrinkage and the sensitivity of activity and hours estimates.

| Theme | Economic or statistical question | Methods and application |
| --- | --- | --- |
| Bayesian learning | How do prior beliefs and sample information shape predictions? | Conjugate models, prior sensitivity, and posterior prediction for binary and count outcomes. |
| Policy responses and model uncertainty | Does allowing asymmetric responses change what we learn about monetary policy? | Gibbs sampling, marginal likelihoods, and posterior model probabilities for competing interest-rate rules. |
| Recession risk | What information do interest rates, equity returns, and the yield curve contain about future downturns? | Bayesian probit models, probability forecasts, and evaluation with explicit forecast origins and information sets. |
| Persistence and specification uncertainty | How do shocks propagate, and how much do conclusions depend on lag order? | Controlled AR simulations and companion matrices; posterior lag comparison and Bayesian model averaging. |
| Growth and employment forecasts | Does added information improve predictions and their uncertainty estimates? | AR/ARDL GDP distributions and growth at risk; comparisons of univariate, small-BVAR, and full-BVAR employment forecasts, evaluated across historical origins. |
| Exceptional volatility and filtering | How does the treatment of a past crisis affect subsequent employment forecasts? | Three treatments compared at six months; post-crisis evaluation separated from retrospective crisis scenarios; a subordinate known-parameter Kalman control. |
| Many predictors and common factors | When do compression and shrinkage improve predictions? | Dynamic-factor inference with latent-path sampling; OLS/ridge/PCA point forecasts with training-only transformations and expanding-window evaluation. |
| Structural interpretation | Which identifying assumptions connect observed macroeconomic dynamics to technology shocks? | A long-run exclusion restriction in a Bayesian VAR, with a convention that orients the positive technology shock. |

Project 08's wide-predictor panel has no verified economic identity for its target variable. Its forecast comparison reports performance in the units of the supplied target Y.

These connections guide the software design as well as the research questions. Regression sampling and model comparison can support both policy specifications and lag uncertainty; posterior simulation can support predictive distributions and uncertainty about dynamic responses. Forecast evaluation provides a common way to assess whether added complexity improves predictions.

## Modeling approach

The existing Python notebook combines analytical derivations, simulation, visualization, and posterior prediction. Project 02 implements reusable object-oriented components for data validation, posterior sampling, model comparison, and forecast evaluation. Project 03 adds immutable dynamic specifications, explicit simulators, deterministic analyzers, and reusable result containers. Project 04 composes those shared packages with aligned AR designs, independent scalar integration, and model-mixture inference. Economic specifications will share computational routines while keeping their assumptions, data transformations, and interpretations explicit.

Implemented MCMC components support independent-prior Bayesian regression and latent-variable probit. Project 06 implements independent-prior BVAR inference and joint predictive simulation. Project 08 implements joint latent-path sampling for monthly activity and daily market factors. Known-parameter dynamics use direct simulation and matrix calculations; Kalman controls use analytical filtering. The wide-predictor OLS/ridge/PCA comparison evaluates point forecasts.

Forecast designs emphasize observed-data baselines, matched horizons, release calendars, and explicit vintage policies. Pandemic interpolation remains a retrospective sensitivity. Project 07 separates feasible post-crisis evaluation from scenarios that use retrospective crisis information.

Validation will connect exact calculations and simulated controls with sampling diagnostics, data provenance, and historical forecast scores. Reference results, independently reproduced findings, and proposed extensions will remain clearly distinguished.

The portfolio emphasizes modeling judgment alongside computation: which assumptions drive a conclusion, what information was available at a forecast date, and how much uncertainty remains.

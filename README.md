# Time Series and Bayesian Econometrics

An applied econometrics portfolio focused on Bayesian inference, temporal dynamics, and probabilistic forecasting. The projects examine how assumptions about economic relationships, model structure, and uncertainty affect the conclusions we draw from data.

The central questions connect learning and prediction: how beliefs change with new observations, how economic shocks propagate, and how models translate incomplete information into forecasts. Bayesian updating anchors the existing work. Planned extensions use Markov chain Monte Carlo (MCMC), including Gibbs sampling, to study monetary policy, recession risk, growth, employment, and common economic and financial signals.

## Projects and current status

| Project | Economic or statistical question | Material and implementation status |
| --- | --- | --- |
| [01 — Bayesian Learning and Posterior Prediction for Discrete Data](project_01_bayesian_discrete_models) | How do prior beliefs and observations shape posterior predictions? | Existing [Python notebook](project1/project_01_bayesian_discrete_models.ipynb), PGA scoring data, figures, and project description. |
| [02 — Bayesian Monetary Policy and Recession Risk](project2/README.md) | Do policy responses differ across economic conditions, and which financial signals anticipate recessions? | Research design; implementation pending. |
| [03 — Persistence and Shock Propagation in Time Series](project3/README.md) | How do autoregressive dynamics determine stability, persistence, and the response to a controlled shock? | Research design; implementation pending. |
| [04 — Lag Uncertainty and Bayesian Model Averaging](project4/README.md) | How much history does a model need, and how does lag uncertainty affect inference about persistence? | Research design; implementation pending. |
| [05 — Probabilistic GDP Forecasts and Growth at Risk](project5/README.md) | How much could U.S. GDP grow over the next year, and how well do predictive distributions describe downside outcomes? | Research design; implementation pending. |
| [06 — Macroeconomic Information and Payroll Forecasts](project6/README.md) | When does additional macroeconomic and financial information justify greater complexity in one-month employment forecasts? | Research design; implementation pending. |
| [07 — Employment Forecasts Through Extreme Episodes](project7/README.md) | When does modeling exceptional volatility help six-month employment forecasts while retaining extreme observations? | Research design; implementation pending. |
| [08 — Common Factors, Forecast Evaluation, and Structural Interpretation](project8/README.md) | Can common signals aid prediction, and what long-run assumptions support the interpretation of technology shocks? | Research design; implementation pending. |

Projects 02–08 currently contain research designs in their READMEs. Their estimation code, shared components, prepared datasets, and empirically reproduced results remain to be developed. Each design identifies its inputs, methods, validation criteria, and planned deliverables, keeping reference calculations separate from reproduced findings.

## Repository contents

The following paths exist in the repository. `project_01_bayesian_discrete_models` is a text file containing the Project 01 description.

```text
Time-Series/
├── README.md
├── project_01_bayesian_discrete_models
├── project1/
│   ├── project_01_bayesian_discrete_models.ipynb
│   ├── data/
│   │   └── pga_scores.xlsx
│   └── figures/
├── project2/
│   └── README.md
├── project3/
│   └── README.md
├── project4/
│   └── README.md
├── project5/
│   └── README.md
├── project6/
│   └── README.md
├── project7/
│   └── README.md
└── project8/
    └── README.md
```

Read the existing Project 01 notebook on GitHub or open it locally in Jupyter after preparing a Python environment with NumPy, pandas, SciPy, Matplotlib, and an Excel reader such as openpyxl. There is currently no shared dependency manifest or automated environment setup. Projects 02–08 have no executable pipelines yet.

<a id="research-progression"></a>

## Econometric themes

These themes connect the concrete projects in the catalog. Bayesian learning has existing materials in Project 01; the other themes are specified in the designs for Projects 02–08.

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

Project 08's wide-predictor panel has no verified economic identity for its target variable. Its forecast comparison will report performance in that target's units without labeling it as GDP or industrial production.

These connections guide the software design as well as the research questions. Regression sampling and model comparison can support both policy specifications and lag uncertainty; posterior simulation can support predictive distributions and uncertainty about dynamic responses. Forecast evaluation provides a common way to assess whether added complexity improves predictions.

## Modeling approach

The existing Python notebook combines analytical derivations, simulation, visualization, and posterior prediction. The planned implementation uses reusable object-oriented components for data validation, posterior sampling, model comparison, and forecast evaluation. Economic specifications will share computational routines while keeping their assumptions, data transformations, and interpretations explicit.

Planned MCMC components support Bayesian regression, probit, BVAR, and dynamic-factor inference. Known-parameter dynamics use direct simulation and matrix calculations; Kalman controls use analytical filtering. The wide-predictor OLS/ridge/PCA comparison evaluates point forecasts.

Forecast designs emphasize observed-data baselines, matched horizons, release calendars, and explicit vintage policies. Pandemic interpolation remains a retrospective sensitivity. Project 07 separates feasible post-crisis evaluation from scenarios that use retrospective crisis information.

Validation will connect exact calculations and simulated controls with sampling diagnostics, data provenance, and historical forecast scores. Reference results, independently reproduced findings, and proposed extensions will remain clearly distinguished.

The portfolio emphasizes modeling judgment alongside computation: which assumptions drive a conclusion, what information was available at a forecast date, and how much uncertainty remains.

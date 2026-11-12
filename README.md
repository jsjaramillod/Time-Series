# Time Series and Bayesian Econometrics

An applied econometrics portfolio focused on Bayesian inference, temporal dynamics, and probabilistic forecasting. The projects examine how assumptions about economic relationships, model structure, and uncertainty affect the conclusions we draw from data.

The central questions connect learning and prediction: how beliefs change with new observations, how economic shocks propagate, and how models translate incomplete information into forecasts. Bayesian updating anchors the existing work. Planned extensions use Markov chain Monte Carlo (MCMC), including Gibbs sampling, to study monetary policy, recession risk, growth, employment, and common economic and financial signals.

## Projects and current status

| Project | Economic or statistical question | Material and implementation status |
| --- | --- | --- |
| [01 — Bayesian learning for discrete data](project_01_bayesian_discrete_models) | How do prior beliefs and observed data shape posterior and predictive distributions? | [Python notebook](project1/project_01_bayesian_discrete_models.ipynb), PGA scoring data, figures, and project description. |
| [02 — Bayesian monetary policy and recession risk](project2/README.md) | Do policy responses differ across economic conditions, and what can financial indicators tell us about recession risk? | Research design, methods, proposed Python architecture, replication targets, and data requirements. Implementation and empirical replication are pending. |

The Project 02 design distinguishes reference results from independently checked calculations and future empirical replication. Its README defines the proposed methods, required inputs, and validation criteria.

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
└── project2/
    └── README.md
```

Read the existing notebook on GitHub or open it locally in Jupyter after preparing a Python environment with NumPy, pandas, SciPy, Matplotlib, and an Excel reader such as openpyxl. There is currently no shared dependency manifest or automated environment setup. Project 02 has no executable pipeline yet; its architecture and deliverables are explicitly planned.

<a id="research-progression"></a>

## Econometric themes

These themes organize the portfolio around related economic and statistical questions. Bayesian learning is implemented in Project 01; monetary policy and recession risk are specified in the Project 02 design. The other applications below are planned extensions, with no separate implementations currently published in this repository.

| Theme | Economic or statistical question | Methods and application |
| --- | --- | --- |
| Bayesian learning | How do prior beliefs and sample information shape predictions? | Conjugate models, prior sensitivity, and posterior prediction for binary and count outcomes. |
| Policy responses and model uncertainty | Does allowing asymmetric responses change what we learn about monetary policy? | Gibbs sampling, marginal likelihoods, and posterior model probabilities for competing interest-rate rules. |
| Recession risk | What information do interest rates, equity returns, and the yield curve contain about future downturns? | Bayesian probit models, probability forecasts, and evaluation with explicit forecast origins and information sets. |
| Persistence and shock propagation | How long do shocks matter, and how uncertain is that persistence? | Autoregressive models, stationarity, impulse responses, lag selection, and Bayesian model averaging. |
| Growth and employment forecasts | What does the predictive distribution imply about downside GDP risk and future job gains or losses? | AR/ARDL growth forecasts, growth at risk, and joint macroeconomic forecasting with Bayesian VARs and shrinkage priors. |
| Changing volatility and hidden states | How can we account for unusual volatility while extracting signals from noisy observations? | Pandemic variance scaling, state-space models, Kalman filtering, and comparisons at matched forecast horizons. |
| Many predictors and common factors | When do compression and shrinkage improve forecasts relative to using every predictor? | Dynamic factor models, latent-state sampling, principal components, ridge regression, and expanding-window forecast comparisons. |
| Structural interpretation | Which identifying assumptions connect observed macroeconomic dynamics to technology shocks? | Long-run restrictions in Bayesian VARs and posterior uncertainty about productivity and hours responses. |

These connections guide the software design as well as the research questions. Regression sampling and model comparison can support both policy specifications and lag uncertainty; posterior simulation can support predictive distributions and uncertainty about dynamic responses. Forecast evaluation provides a common way to assess whether added complexity improves predictions.

## Modeling approach

The existing Python notebook combines analytical derivations, simulation, visualization, and posterior prediction. The planned implementation uses reusable object-oriented components for data validation, posterior sampling, model comparison, and forecast evaluation. Economic specifications will share computational routines while keeping their assumptions, data transformations, and interpretations explicit.

Validation will connect exact calculations and simulated examples with MCMC diagnostics, data provenance, and forecast evaluation that respects information available at each origin. Reference results, independently reproduced findings, and proposed extensions will remain clearly distinguished.

The portfolio emphasizes modeling judgment alongside computation: which assumptions drive a conclusion, what information was available at a forecast date, and how much uncertainty remains.

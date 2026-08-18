# Time Series and Bayesian Econometrics

This repository contains a collection of applied mini-projects in time-series econometrics, Bayesian inference, forecasting, and uncertainty quantification. The goal of the project is to present econometric and statistical methods in a clear, reproducible, and employer-facing format.

Each mini-project combines mathematical derivations, computational implementation, data analysis, visualization, and interpretation. The projects are designed to demonstrate both technical depth and applied modeling judgment, with an emphasis on economic, financial, and probabilistic data.

# Skills Demonstrated: 
 - Bayesian inference and posterior updating
 - Conjugate prior models
 - Posterior predictive distributions
 - Simulation-based analysis
 - Time-series modeling and forecasting
 - Model comparison and interpretation
 - Statistical programming in Python
 - Data visualization and reproducible research
 - Econometric reasoning for applied economic and financial problems

# Repository Structure
```text
Time-Series/
├── README.md
├── requirements.txt
├── data/
│   └── raw/
├── figures/
├── reports/
├── notebooks/
├── project_01_bayesian_discrete_models/
│   ├── README.md
│   ├── notebook.ipynb
│   ├── report.pdf
│   └── figures/
└── project_02_bayesian_macro_policy/
    ├── README.md
    ├── pyproject.toml
    ├── .gitignore
    ├── Makefile
    │
    ├── data/
    │   ├── README.md
    │   ├── raw/
    │   └── processed/
    │
    ├── notebooks/
    │   └── bayesian_macro_policy_analysis.ipynb
    │
    ├── src/
    │   └── bayesian_macro/
    │       ├── __init__.py
    │       ├── config.py
    │       ├── results.py
    │       │
    │       ├── data/
    │       │   ├── interest_rate.py
    │       │   └── recession.py
    │       │
    │       ├── models/
    │       │   ├── coin_comparison.py
    │       │   ├── gaussian_gibbs.py
    │       │   └── probit_gibbs.py
    │       │
    │       ├── analysis/
    │       │   ├── interest_rate_rule.py
    │       │   ├── recession_forecast.py
    │       │   └── chib.py
    │       │
    │       └── visualization/
    │           └── plots.py
    │
    ├── benchmarks/
    │   └── expected_results.json
    │
    ├── tests/
    │   ├── test_coin_comparison.py
    │   ├── test_interest_rate_data.py
    │   ├── test_gaussian_gibbs.py
    │   ├── test_probit_data.py
    │   └── test_replication_targets.py
    │
    ├── figures/
    └── results/
        ├── posterior_summaries.csv
        ├── model_probabilities.csv
        ├── recession_forecasts.csv
        └── replication_report.csv
```


# How to Run

Clone the repository and install the required packages:

git clone <repository-url>
cd Time-Series
pip install -r requirements.txt

Then open the project notebooks:

jupyter notebook

Each mini-project folder contains its own README with a description of the research question, methodology, results, and interpretation.

# Professional Relevance

This portfolio is intended to demonstrate applied econometric modeling skills relevant for quantitative research, data science, economic analysis, and financial modeling roles. The projects emphasize not only implementation, but also interpretation: how assumptions affect results, how uncertainty is quantified, and how statistical models can be used to answer practical questions.

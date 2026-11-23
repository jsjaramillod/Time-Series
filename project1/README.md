# Project 01: Bayesian Learning and Posterior Prediction for Discrete Data

## The coursework forecast

In my first doctoral time-series assignment, a Bayesian count model predicted a mean of **71.2245 strokes** for a PGA round and a **30.11%** chance of a score above 75. The portfolio notebook recovers both estimates from the same 7,577 rounds. How well does that distribution describe high scores? Only **684 rounds (9.03%)** actually exceeded 75. The mean fits the sample closely, but the model assigns too much probability to high scores. This is a check on the fitted sample.

[Executed notebook](project_01_bayesian_discrete_models.ipynb) · [PGA scores](data/pga_scores.xlsx) · [Saved coursework estimates](../coursework_results.csv)

## Overview

This project studies Bayesian inference for discrete-data models, focusing on how prior beliefs are updated after observing data. The project develops two conjugate Bayesian models: the Beta-Bernoulli model for binary outcomes and the Gamma-Poisson model for count data.

The project combines analytical derivations, simulation, visualization, and an empirical application. The main goal is to show how Bayesian models can be used not only to estimate unknown parameters, but also to quantify uncertainty and make probabilistic predictions about future observations.

## Motivation

Many real-world problems involve discrete outcomes. Examples include customer conversions, default events, arrivals, transaction counts, insurance claims, and scoring outcomes. Bayesian methods provide a natural framework for these problems because they combine prior information with observed data and produce full posterior distributions rather than only point estimates.

This project illustrates that workflow in simple but powerful settings. The Bernoulli model shows how learning works for binary outcomes, while the Poisson model shows how Bayesian inference can be applied to count data and posterior prediction.

## Methods

The project covers the following Bayesian models and concepts:

* Bayes’ rule for posterior updating
* Beta-Bernoulli conjugacy
* Gamma-Poisson conjugacy
* Posterior mean and posterior variance
* Maximum likelihood estimation versus Bayesian estimation
* Prior sensitivity analysis
* Posterior predictive distributions
* Negative Binomial posterior predictive distribution
* Simulation-based visualization

## Project Components

### 1. Posterior Updating with a Discrete Parameter Space

The project begins with a simple finite-parameter example. Given a likelihood and prior distribution over a scalar parameter, Bayes’ rule is used to compute the posterior distribution, posterior mean, maximum likelihood estimator, and marginal likelihood.

This section introduces the basic mechanics of Bayesian updating.

### 2. Beta-Bernoulli Model

The second section studies binary outcomes modeled as Bernoulli trials. A Beta prior is placed on the success probability, which leads to a Beta posterior distribution.

The analysis compares the posterior mean with the maximum likelihood estimator and studies how the prior affects inference. Simulations show that informative priors can strongly influence the posterior when the sample size is small, while the likelihood dominates as the sample size grows.

### 3. Gamma-Poisson Model

The third section studies count outcomes modeled with a Poisson likelihood. A Gamma prior is placed on the Poisson intensity parameter, producing a Gamma posterior distribution.

The project derives the posterior distribution, posterior mean, posterior variance, and maximum likelihood estimator. It also derives the posterior predictive distribution and shows that it has a Negative Binomial form.

### 4. Empirical Application: PGA Tour Scoring Data

The final section applies the Gamma-Poisson model to PGA Tour scoring data. The data consist of 18-hole scores from PGA Tour events during the 2018 season.

The model treats each score as a count outcome generated from a Poisson distribution with unknown mean parameter. A Gamma prior is chosen to reflect reasonable prior beliefs about professional golf scores. The posterior distribution is then used to compute predictive probabilities for future rounds.

The posterior predictive analysis answers questions such as:

* What is the expected score for a future round?
* What is the most likely future score?
* What is the probability of shooting above 75?
* What is the probability of shooting between 65 and 70?

## Key Results

For the PGA scoring application, the data contain 7,577 observations. The average observed score is approximately 71.22 strokes, with a standard deviation of approximately 3.20.

Using a Gamma-Poisson Bayesian model, the posterior predictive mean is approximately 71.22, and the posterior predictive mode is 71. The model estimates that the probability of a future score above 75 is approximately 30.1%, while the probability of a score between 65 and 70 inclusive is approximately 25.9%.

The predictive mean is within 0.001 stroke of the observed mean (71.2237). For a score above 75, the fitted probability exceeds the observed share by 21.08 percentage points.

## Skills Demonstrated

* Bayesian inference
* Conjugate prior modeling
* Probability modeling for discrete data
* Posterior predictive inference
* Simulation and visualization
* Statistical programming in Python
* Mathematical derivation and applied interpretation
* Communication of uncertainty

## Reproduce the example

Open the [analysis notebook](project_01_bayesian_discrete_models.ipynb) with this directory as the working directory. The workbook is included at [data/pga_scores.xlsx](data/pga_scores.xlsx). The notebook uses NumPy, pandas, SciPy, Matplotlib and openpyxl. The final PGA comparison counts the observed rounds above 75 and compares their share with the predictive probability.

```sh
python -m pip install numpy pandas scipy matplotlib openpyxl jupyterlab
jupyter lab project_01_bayesian_discrete_models.ipynb
```

"""Build a results-reading notebook with explicit opt-in estimation."""

from pathlib import Path

import nbformat as nbf


def main():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    notebook = nbf.v4.new_notebook()
    notebook.metadata.kernelspec = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    notebook.cells = [
        md(
            "# Lag Uncertainty and Bayesian Model Averaging\n\nIn my fourth doctoral assignment, I compared and averaged eight possible lag orders. The model using the previous two observations received **71.54%** posterior probability. The portfolio confirms that weight to two decimal places. For the eight-lag model, the mean sum of coefficients was **0.8006** in class and **0.8008** in the portfolio's integration check. The new question is how much these conclusions depend on prior beliefs: the two-lag model's weight ranges from **60.01% to 80.96%**, while model-averaged persistence stays between **0.8430 and 0.8504**. [Saved coursework estimates](../../coursework_results.csv)\n\nHow does the lag-order decision affect inference about persistence and shock transmission? [Executive summary](../EXECUTIVE_SUMMARY.md) | [Methods](../docs/methods.md) | [Research report](../docs/lag_uncertainty_model_averaging.pdf)\n\nThe supplied series has an unverified generating process. All results are conditional on the declared models, priors, and common conditioning observations."
        ),
        code(
            "from pathlib import Path\nimport json\nimport os\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import Image, display\nfrom lag_uncertainty import ModelAveragedInference\nfrom lag_uncertainty.study import LagOrderStudy\n\ncandidates = [Path.cwd(), Path.cwd().parent, Path.cwd() / 'project4', Path.cwd().parent / 'project4']\nPROJECT = next(p for p in candidates if (p / 'results/study_summary.json').is_file())\nRESULTS = PROJECT / 'results'\nif os.environ.get('RUN_ESTIMATION') == '1':\n    if not os.environ.get('AR_DATA'):\n        raise ValueError('Set AR_DATA to the explicit source CSV path for an intentional rerun')\n    LagOrderStudy(os.environ['AR_DATA'], RESULTS).run()\nload = lambda name: json.loads((RESULTS / name).read_text())\nload('data_manifest.json')"
        ),
        md(
            "## A common likelihood window\n\nThe input is a headerless one-column file with 200 observations. The first eight condition every AR(1)-AR(8) model. Every candidate explains the same 192 targets at ordinal positions 9-200. No calendar, generating order, or generation seed is assigned."
        ),
        code(
            "metadata = load('run_metadata.json')\ndisplay({k: metadata[k] for k in ['coefficient_variance_order', 'intercept_variance', 'precision_prior', 'model_prior', 'sampler']})\nvalidation = load('validation.json')\nassert validation['all_passed']\ndisplay({k: validation[k] for k in ['fits_count', 'common_target_positions_inclusive', 'common_modeled_observations']})"
        ),
        md(
            "## Posterior model preference and prior concentration\n\nEach coefficient-prior variance defines a separate eight-model comparison with equal model prior probabilities. Chib evidence supplies the primary weights. Independent SVD and conditional-matrix integrals are retained as numerical controls."
        ),
        code(
            "comparison = pd.read_csv(RESULTS / 'model_comparison.csv')\nweights = comparison.pivot(index='order', columns='coefficient_variance', values='posterior_model_probability')\ndisplay(weights)\ndisplay(Image(filename=str(RESULTS / 'figures/lag_order_probabilities.png')))"
        ),
        md(
            "## Selecting and averaging are different targets\n\nThe coefficient sum excludes the intercept. A model average mixes the scalar posterior distributions; it does not pad and average coefficient vectors. The table distinguishes weighted empirical Gibbs summaries from integrated scalar reference calculations using the same Chib weights."
        ),
        code(
            "persistence = pd.read_csv(RESULTS / 'persistence_summary.csv')\ndisplay(persistence[['coefficient_variance', 'target', 'mean', 'mean_mcse_joint_batches', 'integration_mean', 'integration_sd', 'integration_p05', 'integration_p95', 'integration_between_model_variance_share']])\ndisplay(Image(filename=str(RESULTS / 'figures/persistence_posteriors.png')))"
        ),
        md(
            "## A small transparent mixture control\n\nThis example is an illustrative finite mixture, not an estimate for the supplied series. The shared API computes its within- and between-component variance and samples a model index before selecting a conditional scalar draw."
        ),
        code(
            "toy = ModelAveragedInference([[0, 2], [4, 6]], [0.25, 0.75], labels=[1, 2])\ndisplay(toy.summary())\nvalues, labels = toy.draw(50000, seed=91)\ndisplay(pd.Series({'resample_mean': values.mean(), 'second_model_fraction': np.mean(labels == 2)}))\nresampling_checks = pd.read_csv(RESULTS / 'mixture_resampling_checks.csv')\ndisplay(resampling_checks[['coefficient_variance', 'resamples', 'weighted_empirical_mean', 'resample_mean', 'resampling_standard_error_of_mean']])"
        ),
        md(
            "## Finite-horizon response uncertainty\n\nThe innovation is one unit at horizon zero. All draws contribute to horizons 1-10, including unstable ones. Intervals are marginal at individual horizons, not simultaneous bands. The model average acts on response distributions, not on averaged coefficients."
        ),
        code(
            "responses = pd.read_csv(RESULTS / 'impulse_responses.csv')\ndisplay(responses.query('coefficient_variance == 0.5 and horizon in [1, 5, 10]'))\ndisplay(Image(filename=str(RESULTS / 'figures/response_uncertainty.png')))"
        ),
        md(
            "## Stability and rare-event resolution\n\nThe coefficient sum is not a sufficient root-stability test. Stable-draw proportions are finite-simulation frequencies. Under the primary prior, AR(2) may have zero observed unstable or sum-above-one draws while scalar integration still assigns a positive rare upper-tail probability. Resampling from the same Gibbs pools cannot recover events absent from them."
        ),
        code(
            "stability = pd.read_csv(RESULTS / 'stability.csv')\ncontrol = pd.read_csv(RESULTS / 'scalar_integration_controls.csv')\ndisplay(stability.query('coefficient_variance == 0.5')[['order', 'stable_draw_proportion', 'sum_below_one_but_not_stable_proportion', 'draws_retained', 'draws_removed_for_stability']])\ndisplay(control.query('coefficient_variance == 0.5')[['order', 'mcmc_sum_probability_above_one', 'integration_probability_sum_above_one']])"
        ),
        md(
            "## Numerical controls and convergence\n\nThe model fits retain four chains. Diagnostics cover original parameters and continuous derived quantities, not the constant horizon-zero response. Joint contiguous batches recompute model weights and sum summaries together. Monte Carlo error and between-model posterior uncertainty are different quantities."
        ),
        code(
            "parameters = pd.read_csv(RESULTS / 'posterior_parameters.csv')\nderived = pd.read_csv(RESULTS / 'posterior_derived.csv')\nchecks = pd.read_csv(RESULTS / 'evidence_point_checks.csv')\ndisplay(pd.Series({'maximum_rhat': max(parameters.r_hat.max(), derived.r_hat.max()), 'minimum_bulk_ess': min(parameters.ess_bulk.min(), derived.ess_bulk.min()), 'maximum_chib_baseline_error': comparison.absolute_chib_integration_error.max(), 'maximum_svd_vs_conditional_error': comparison.svd_vs_conditional_integration_error.max(), 'maximum_point_check_error': checks.absolute_error_vs_quadrature.max()}))\ndisplay(Image(filename=str(RESULTS / 'figures/ar8_chain_traces.png')))"
        ),
        md(
            "## Additional normalized-prior checks\n\nProper-intercept and alternative Gamma-parameterization scenarios use deterministic integration at coefficient variance 0.5. They are separate checks, not extra claimed Gibbs fits or pooled prior alternatives."
        ),
        code(
            "sensitivity = pd.read_csv(RESULTS / 'additional_prior_sensitivity.csv')\ndisplay(sensitivity.query('order == 2')[['scenario', 'posterior_model_probability', 'bma_sum_mean', 'bma_sum_sd', 'bma_sum_p05', 'bma_sum_p95']])\ndisplay(load('batch_mcse.json')['0.5'])"
        ),
        md(
            "## Reproduction and scope\n\nAll source/data settings are recorded. Public tables and this notebook can be inspected without the external CSV; a full rerun requires its matching bytes. Private archives are excluded from Git. Source and output checksums identify the recorded calculation, while scientific controls support numerical verification.\n\nThe study does not identify the unknown generating process, establish a macroeconomic causal effect, or evaluate forecast performance. Read the report for the interpretation of model sensitivity, pointwise responses, and rare-event limits."
        ),
        code(
            "display(load('run_metadata.json'))\nprint('All recorded fit controls passed:', validation['all_passed'])"
        ),
    ]
    path = Path(__file__).resolve().parents[1] / "notebooks/project04_analysis.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)
    print(path)


if __name__ == "__main__":
    main()

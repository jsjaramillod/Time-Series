"""Build the results-oriented notebook. Execute it with a project Python kernel."""

from pathlib import Path

import nbformat as nbf

root = Path(__file__).resolve().parents[1]
notebook = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
notebook.cells = [
    md(
        "# Bayesian Monetary Policy and Recession Risk\n\nIn my second doctoral assignment, the simpler Federal Reserve policy rule received **95.06%** posterior probability. Re-estimation gives **95.06%** with the same coefficient prior. The **96.44%** result uses a different prior. The practical change is in the recession forecast: the original **12.36%** estimate labeled February 2027 used a predictor row from February 2025. Selecting the February 2026 row and using the project's equity-return convention gives **8.64%**. The corrected forecast has the intended twelve-month horizon. Its outcome is unobserved in the study. [Saved coursework estimates](../../coursework_results.csv)\n\n"
        "An applied econometrics investigation with reusable Python models, explicit timing contracts, "
        "and independent numerical controls. This notebook inspects the generated empirical artifacts; "
        "it does not contain a second estimation implementation.\n\n"
        "**Questions:** Does an asymmetric output-gap response justify another policy coefficient? "
        "What do financial signals imply about recession probabilities twelve months ahead? "
        "[Full methods](../docs/methods.md) · [Specification audit](../docs/reference_audit.md) · "
        "[Empirical report](../results/report.md)"
    ),
    code(
        "from pathlib import Path\nimport json\nimport os\nimport pandas as pd\n"
        "from IPython.display import Image, display\n"
        "ROOT = Path.cwd()\n"
        "if not (ROOT / 'results').is_dir():\n"
        "    ROOT = ROOT / 'project2' if (ROOT / 'project2/results').is_dir() else ROOT.parent\n"
        "assert (ROOT / 'results/policy_posterior.csv').is_file(), 'Run the policy pipeline first'\n"
        "RESULTS = ROOT / 'results'\n"
        "pd.options.display.float_format = '{:.5f}'.format"
    ),
    md(
        "## Optional reproduction from the private inputs\n\n"
        "The default notebook run reads committed results and requires no workbooks. To regenerate "
        "estimates, set `RUN_ESTIMATION=1`, `POLICY_DATA`, and `RECESSION_DATA` to the external workbook "
        "paths before starting the kernel. The commands in the README provide the equivalent CLI. "
        "No originals are copied into the repository."
    ),
    code(
        "if os.environ.get('RUN_ESTIMATION') == '1':\n"
        "    from bayes_policy.pipeline import run_policy, run_recession\n"
        "    from bayes_policy.reporting import build_report\n"
        "    from bayes_policy.samplers import SamplerConfig\n"
        "    config = SamplerConfig(draws=10000, warmup=3000, chains=4, seed=20261114)\n"
        "    policy_validation = run_policy(os.environ['POLICY_DATA'], RESULTS, config)\n"
        "    recession_validation = run_recession(os.environ['RECESSION_DATA'], RESULTS / 'recession', config)\n"
        "    delayed_validation = run_recession(os.environ['RECESSION_DATA'], RESULTS / 'recession_delay18',\n"
        "                                       config, label_delay_months=18)\n"
        "    assert policy_validation['sampler_diagnostics_passed']\n"
        "    assert policy_validation['independent_evidence_controls_passed']\n"
        "    assert recession_validation['all_diagnostics_passed']\n"
        "    assert delayed_validation['all_diagnostics_passed']\n"
        "    build_report(RESULTS)\n"
        "else:\n"
        "    print('Inspecting generated empirical artifacts; no estimation rerun requested.')"
    ),
    md(
        "## Audit the actual sample\n\n"
        "The quarterly workbook begins in 1954Q4. One lag leaves 239 effective responses, "
        "1955Q1–2014Q3. Trend inflation and data vintage are accepted as supplied, not independently "
        "reconstructed. The missing earlier quarters are not invented."
    ),
    code(
        "manifest = json.loads((RESULTS / 'data_manifest.json').read_text())\n"
        "display(pd.Series(manifest, name='Policy input provenance').to_frame())"
    ),
    md(
        "## Policy responses and prior sensitivity\n\n"
        "Both policy specifications use independent Normal/Gamma priors. The primary coefficient "
        "covariance is 10I; the 5I sensitivity evaluates a more concentrated coefficient prior. Negative output "
        "gaps keep their sign. Intervals below are central 90% posterior intervals."
    ),
    code(
        "posterior = pd.read_csv(RESULTS / 'policy_posterior.csv')\n"
        "display(posterior[posterior['model'].isin(['symmetric_v10', 'asymmetric_v10'])])\n"
        "display(Image(filename=str(RESULTS / 'figures/policy_coefficients.png')))"
    ),
    md(
        "## Model complexity must earn its place\n\n"
        "Chib evidence and the independent quadrature control agree closely. The comparison uses "
        "identical response observations and explicit prior model probabilities. A posterior model "
        "weight measures preference within these candidates; it is not a probability that the chosen "
        "economic model is universally correct."
    ),
    code(
        "evidence = pd.read_csv(RESULTS / 'model_comparison.csv')\n"
        "display(evidence)\n"
        "assert evidence['evidence_control_passed'].all()"
    ),
    md(
        "## Long-run response and asymmetry\n\n"
        "The positive-response and Taylor-principle thresholds are distinct: theta>0 versus theta>1, "
        "where theta is the inflation coefficient divided by one minus the lag coefficient. "
        "Unstable and near-unit-root draws remain visible. The joint slope difference measures "
        "asymmetry more directly than comparing two marginal intervals."
    ),
    code(
        "derived = json.loads((RESULTS / 'policy_derived.json').read_text())\n"
        "display(pd.json_normalize(derived).set_index('model'))\n"
        "display(Image(filename=str(RESULTS / 'figures/policy_uncertainty.png')))"
    ),
    md(
        "## A reliable sampler can fit an inadequate likelihood\n\n"
        "Four-chain rank-normalized split R-hat, bulk/tail ESS, and MCSE assess simulation reliability. "
        "The conditional residual check examines economic-model adequacy. Residual serial dependence "
        "challenges the independent Gaussian error assumption; regime changes and the lower bound "
        "also limit interpretation. These regressions do not identify causal policy effects."
    ),
    code(
        "validation = json.loads((RESULTS / 'validation.json').read_text())\n"
        "display(pd.json_normalize(validation['models'], sep='.').T)\n"
        "assert validation['sampler_diagnostics_passed']\n"
        "display(Image(filename=str(RESULTS / 'figures/policy_traces.png')))"
    ),
    md(
        "## Recession forecasts: origin, target, availability\n\n"
        "A 2007 forecast uses that origin's predictors and trains only on already matured targets, "
        "with the declared availability rule. The primary run assumes zero additional label delay "
        "and retrospective recession indicators. Predictor vintages and actual announcement times "
        "are unverified, so the exercise is **pseudo-real-time**. The February 2026-origin forecast "
        "targets February 2027 and has no outcome in the input; it is a historical-origin forecast."
    ),
    code(
        "recession_manifest = json.loads((RESULTS / 'recession/recession_manifest.json').read_text())\n"
        "display(pd.Series({k: v for k, v in recession_manifest.items() if k != 'metadata'},\n"
        "                  name='Monthly input provenance').to_frame())\n"
        "forecasts = pd.read_csv(RESULTS / 'recession/recession_forecasts.csv')\n"
        "display(forecasts)\n"
        "assert (pd.to_datetime(forecasts.last_training_target) <= pd.to_datetime(forecasts.origin)).all()\n"
        "assert (pd.to_datetime(forecasts.latest_label_availability) <= pd.to_datetime(forecasts.origin)).all()\n"
        "assert forecasts.diagnostics_passed.all()\n"
        "display(Image(filename=str(RESULTS / 'recession/recession_probabilities.png')))"
    ),
    md(
        "## Scores and an honest benchmark\n\n"
        "The Beta(1,1)-smoothed historical-frequency benchmark uses the same training labels. "
        "Brier loss and negative log score reward accurate probability forecasts; smaller is better. "
        "All twelve 2008 targets are recessions in the supplied indicator, so these scores cannot "
        "establish broad calibration or discrimination. Intervals around model probabilities "
        "describe parameter uncertainty, not intervals for binary outcomes."
    ),
    code(
        "scores = json.loads((RESULTS / 'recession/recession_scores.json').read_text())\n"
        "display(pd.DataFrame({k: scores[k] for k in ['probit', 'benchmark']}).T)\n"
        "assert scores['all_diagnostics_passed']\n"
        "endpoint = forecasts[forecasts.origin.str.startswith('2026-02')]\n"
        "assert endpoint.actual.isna().all()\n"
        "display(endpoint[['origin', 'target_date', 'probability', 'probability_p05', 'probability_p95']])"
    ),
    md(
        "## Withholding recently dated outcomes\n\n"
        "An 18-month label-delay sensitivity changes the available training sample. It demonstrates "
        "dependence on label availability; it does not reconstruct a historical NBER announcement "
        "calendar or turn revised predictor data into a real-time experiment."
    ),
    code(
        "delayed = pd.read_csv(RESULTS / 'recession_delay18/recession_forecasts.csv')\n"
        "delayed_scores = json.loads((RESULTS / 'recession_delay18/recession_scores.json').read_text())\n"
        "display(pd.DataFrame({'primary': scores['probit'], 'delay18': delayed_scores['probit']}).T)\n"
        "display(delayed[['origin', 'last_training_target', 'latest_label_availability', 'training_n',\n"
        "                 'probability', 'diagnostics_passed']])\n"
        "assert delayed_scores['all_diagnostics_passed']"
    ),
    md(
        "## What this project demonstrates\n\n"
        "Economic specification, Bayesian computation, prior sensitivity, software design by "
        "composition, input provenance, temporal validation, and communication of model limitations. "
        "The package and scientific controls are reusable. The results do not establish causality, "
        "real-time forecasting performance, or performance across all business-cycle regimes. "
        "Original input vintages and the policy model's residual structure remain material limitations."
    ),
]
notebook.cells[16:16] = [
    md(
        "## Financial signals in the endpoint model\n\n"
        "The February 2026-origin coefficient posterior summarizes 779 matched observations. "
        "Coefficients describe changes in the latent probit index, not direct percentage-point "
        "probability changes or causal effects. Conditional predictor relationships need to be "
        "interpreted jointly with the model and its timing assumptions."
    ),
    code(
        "endpoint_coefficients = pd.read_csv(RESULTS / 'recession/probit_2026-02-28_diagnostics.csv')\n"
        "display(endpoint_coefficients)\n"
        "display(Image(filename=str(RESULTS / 'recession/probit_endpoint_densities.png')))\n"
        "display(Image(filename=str(RESULTS / 'recession/probit_endpoint_traces.png')))"
    ),
]
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
nbf.write(notebook, root / "notebooks/project02_analysis.ipynb")
print("Created", len(notebook.cells), "cells")

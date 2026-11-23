"""Build an artifact-reading notebook with an optional explicit simulation rerun."""

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
            "# Persistence and Shock Propagation\n\nIn my third doctoral assignment, a one-unit shock in a two-lag model generated a response of **1.20** one period later. Summing the computed responses gave **9.99999946**. The portfolio reproduces that path and verifies its theoretical cumulative effect of **10**. A second experiment asks how much confidence to place in an average of persistent observations. Across 10,000 simulations with known parameters, an intended 95% interval covered the true mean **35.27%** of the time under independence and **95.00%** when it used the serial covariance. [Saved coursework estimates](../../coursework_results.csv)\n\nControlled time-series experiments on stability, persistence, and the precision of averages. Parameters are fixed; all data are synthetic. [Executive summary](../EXECUTIVE_SUMMARY.md) | [Methods](../docs/methods.md) | [Research report](../docs/persistence_shock_propagation.pdf)"
        ),
        code(
            "from pathlib import Path\nimport json\nimport os\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import Image, display\nfrom shock_dynamics import ARSpecification, DynamicsAnalyzer\nfrom shock_dynamics.pipeline import run_experiments\n\ncandidates = [Path.cwd(), Path.cwd().parent, Path.cwd() / 'project3', Path.cwd().parent / 'project3']\nPROJECT = next(p for p in candidates if (p / 'results/dynamics_summary.csv').is_file())\nRESULTS = PROJECT / 'results'\nif os.environ.get('RUN_SIMULATIONS') == '1':\n    run_experiments(RESULTS, seed=20261115, replications=10000)\nload = lambda name: json.loads((RESULTS / name).read_text())\nsummary = pd.read_csv(RESULTS / 'dynamics_summary.csv').set_index('experiment')\nsummary[['coefficients', 'intercept', 'innovation_variance', 'burn_in']]"
        ),
        md(
            "## Stability is a property of the model\n\nThe companion eigenvalues determine stability in the causal representation. A finite path is an illustration, not a stationarity test. Stationary moments and infinite response sums are deliberately absent for unstable and unit-circle models."
        ),
        code(
            "summary[['classification', 'spectral_radius', 'minimum_finite_root_modulus', 'stationary_mean', 'stationary_variance', 'long_run_sum']]"
        ),
        code("display(Image(filename=str(RESULTS / 'figures/companion_eigenvalues.png')))"),
        md(
            "## A controlled shock isolates transmission\n\nThe response at horizon zero is one. Matrix powers, coefficient recursion, and the difference between shocked and unshocked trajectories must agree. SciPy's linear filter supplies a fourth numerical control. All three stable zero-intercept models have an infinite cumulative response of ten, while their finite-horizon totals differ."
        ),
        code(
            "response = DynamicsAnalyzer(ARSpecification((1.2, -0.3))).impulse_response(100)\ncomparison = pd.DataFrame({'horizon': response.horizons, **response.methods, 'cumulative': response.cumulative})\ndisplay(comparison.iloc[:21])\nprint('Maximum three-method disagreement:', response.maximum_disagreement)\nprint('Analytical infinite sum:', DynamicsAnalyzer(ARSpecification((1.2, -0.3))).long_run_sum)"
        ),
        code(
            "display(Image(filename=str(RESULTS / 'figures/shock_propagation.png')))\ndisplay(summary.loc[['ar1_persistent', 'ar2_response', 'ar3_stable'], ['cumulative_at_20', 'cumulative_at_100', 'remaining_sum_after_100']])"
        ),
        md(
            "## The intercept changes the mean, not centered dynamics\n\nBoth AR(1) paths use exactly the same innovations and 500 burn-in periods. Their initial states differ by five, so changing the intercept from zero to 0.5 shifts every retained observation by five. The theoretical variance remains 1/(1-0.9²)."
        ),
        code(
            "paths = pd.read_csv(RESULTS / 'simulation_paths.csv').pivot(index='period', columns='experiment', values='value')\nprint('Maximum error relative to a five-unit shift:', np.abs(paths.ar1_mean_shift - paths.ar1_persistent - 5).max())\ndisplay(Image(filename=str(RESULTS / 'figures/mean_adjustment.png')))"
        ),
        md(
            "## Persistence reduces the information in an average\n\nEach replication starts from the exact stationary Gaussian distribution. The finite-window mean variance uses the full autocovariance sum. The independence comparison keeps the same marginal variance but sets cross-time covariances to zero. Coverage refers to known-parameter repeated-sample intervals, not posterior probabilities."
        ),
        code(
            "precision = load('mean_precision.json')\nkeys = ['replications', 'length', 'exact_mean_variance', 'independence_mean_variance', 'variance_inflation_factor', 'effective_independent_observations', 'correlation_aware_coverage', 'independence_coverage', 'exact_independence_coverage']\npd.Series({key: precision[key] for key in keys}, name='Computed result')"
        ),
        code("display(Image(filename=str(RESULTS / 'figures/mean_precision.png')))"),
        md(
            "## Shared innovations expose a stability change\n\nThe two AR(3) paths start from zero, use the same 200 innovations, and discard no observations. Their vertical scales differ. The unstable path has no stationary variance; its finite sample variance is not an estimate of a stationary population quantity."
        ),
        code("display(Image(filename=str(RESULTS / 'figures/stability_paths.png')))"),
        md(
            "## Stationarity and invertibility answer different questions\n\nThe finite MA(2) representation has constant second moments under white-noise innovations. Its polynomial roots are inside the unit circle, so that innovation representation is noninvertible. Autocovariances vanish beyond lag two."
        ),
        code(
            "ma = load('ma_control.json')\ndisplay(pd.Series({k: v for k, v in ma.items() if k != 'specification'}))\ndisplay(Image(filename=str(RESULTS / 'figures/ma_autocovariance.png')))"
        ),
        md(
            "## Numerical controls and reproducibility\n\nThe covariance solution must satisfy the Lyapunov equation and agree with a long squared-response sum. Roots, response conventions, variance scaling, initial-lag order, burn-in, and seeded reproduction have separate scientific tests. Checksums preserve the exact source and artifact identities for the recorded run."
        ),
        code(
            "checks = load('numerical_validation.json')\nassert checks['all_deterministic_controls_pass']\ndisplay(pd.DataFrame({k: v for k, v in checks.items() if isinstance(v, dict)}).T)\ndisplay(load('run_metadata.json'))"
        ),
        md(
            "## Scope of the conclusions\n\nThe experiment quantifies model-implied mechanisms. It does not estimate macroeconomic coefficients, identify observed causal effects, or evaluate a live forecast. Periods have no imposed monthly or quarterly frequency. Generalizing requires a specified empirical dataset and an inference model.\n\nTo regenerate from the project directory: `shock-dynamics --output results --seed 20261115 --replications 10000`. The notebook normally reads committed artifacts; set `RUN_SIMULATIONS=1` only for an intentional rerun."
        ),
    ]
    output = Path(__file__).resolve().parents[1] / "notebooks/project03_analysis.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    print(output)


if __name__ == "__main__":
    main()

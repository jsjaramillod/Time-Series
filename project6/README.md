# Macroeconomic Information and U.S. Payroll Forecasts

My original April 2026 payroll forecast from doctoral coursework was **+61,140 jobs**. The first BLS estimate was **+115,000**, a miss of 53,860. That forecast used five monthly indicators and smoothed the pandemic years. Which indicators improve the job count across many releases? When does retaining an extreme episode help the next forecast? The study compares three information sets on 216 first announcements and separately tests the original smoothing choice on matched dates.

[Executive brief (PDF)](docs/executive_summary.pdf) · [Research report (PDF)](docs/payroll_forecasts.pdf) · [Executive summary](EXECUTIVE_SUMMARY.md) · [Executed notebook](notebooks/project06_analysis.ipynb)

## Reading the estimates

| Information set | RMSE, thousands of jobs | Mean absolute error, thousands | 90% coverage | Loss-probability Brier score |
| --- | ---: | ---: | ---: | ---: |
| Employment only | 2,187 | 467 | 95.4% | 0.147 |
| Employment, production, CPI | 1,647 | 380 | 96.3% | 0.150 |
| Macro plus equity and term spread | 1,645 | 372 | 96.3% | 0.141 |

The evaluation covers complete calendar years from just before the 2008–09 downturn through the last full year before the 2025 CPI interruption. Rows use the same 216 first announcements and the primary prior. Thirty-five announcements reported losses. Lead time from issue to release ranged from 8 to 35 calendar days. Regressors use a frozen, revised workbook, so the calendar reconstruction is pseudo-real-time. The report also scores revised targets, tests tighter shrinkage, and separates the financial crisis from the first pandemic releases.

In April the two five-variable refits predicted **+46,436 jobs** with smoothing and **−141,062** with observed history. Their absolute errors were **68,564** and **256,062**. The employment-only model with observed history was closer at **+83,826** (error **31,174**). The original calculation and the refits use different March payroll snapshots. [Archived estimate and provenance](data/README.md#original-april-comparison)

A matched comparison applies the same five-variable implementation to all 34 eligible targets from March 2022 through December 2024. Smoothing reduced mean absolute error from **226,414 to 145,702 jobs**. The corresponding 90% intervals covered 31 of 34 first releases with smoothing and 34 of 34 with observed history. Every first release in this window was a gain. [Matched forecast records](results/matched_histories.csv)

## Reproduction

From this directory on Python 3.11 or later, install the shared diagnostics and payroll package. Verification and notebook exploration work from the published forecast records.

```sh
python -m pip install -e ../project2 -e '.[dev]'
pytest -q
python scripts/verify_results.py
```

To refit all 1,308 primary and sensitivity specifications and repeat the 648 primary historical fits with a second seed, supply the exact external workbook described in the [data contract](data/README.md). The model uses all available monthly observations back to January 1962, twelve lags, a common issue date, and fixed prior settings at every origin.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 payroll-risk /path/to/data_aggregated.xlsx --workers 4
python scripts/compare_histories.py /path/to/data_aggregated.xlsx
python scripts/figures.py
python scripts/build_report.py
```

The publication bundle has checksums for the recorded outputs. Different numerical libraries can change simulation results slightly. The optional source audit uses an Excel reader for the legacy equity file (`pip install xlrd`) and a date-indexed GS10/TB3MS CSV from FRED. Run `python scripts/audit_inputs.py PANEL.xlsx EQUITY.xls YIELDS.csv OUTPUT.json`.

## Implementation and evidence

`PayrollPanel` enforces the data contract and origin-specific transformations. `MinnesotaPrior` supplies coefficient moments. `BayesianVAR` samples coefficients and innovation covariances. `ForecastDraws` converts the predictive law to job counts. The study driver runs matched origins, records chain diagnostics, and evaluates first-release and revised outcomes.

- [Historical forecasts](results/forecasts.csv) and [scores](results/scores.csv) include point, interval and event forecasts under both shrinkage settings.
- [Independent-seed scores](results/replay_scores.csv) repeat the primary comparison with longer chains.
- [April 2026 forecasts](results/reference.csv) and [CPI equation coefficients](results/cpi_coefficients.csv) document the fixed reference exercise.
- [Release records](data/release_calendar.csv), [BLS first estimates](data/first_releases.csv) and the [input audit](results/input_audit.json) identify the data behind the comparison.

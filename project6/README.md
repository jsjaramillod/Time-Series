# Macroeconomic Information and U.S. Payroll Forecasts

Does an economist gain more from watching production and prices, or from also watching financial markets? Across 216 monthly payroll releases from January 2007 to December 2024, adding industrial production and CPI to employment history reduced root mean squared error by about 25%. Adding equity prices and the Treasury spread changed that error by less than 1%. The four releases from March through June 2020 fell outside every model's nominal 90% interval.

[Executive brief (PDF)](docs/executive_summary.pdf) · [Research report (PDF)](docs/payroll_forecasts.pdf) · [Executive summary](EXECUTIVE_SUMMARY.md) · [Executed notebook](notebooks/project06_analysis.ipynb)

## Reading the estimates

| Information set | RMSE, thousands of jobs | Mean absolute error, thousands | 90% coverage | Loss-probability Brier score |
| --- | ---: | ---: | ---: | ---: |
| Employment only | 2,187 | 467 | 95.4% | 0.147 |
| Employment, production, CPI | 1,647 | 380 | 96.3% | 0.150 |
| Macro plus equity and term spread | 1,645 | 372 | 96.3% | 0.141 |

The evaluation covers complete calendar years from just before the 2008–09 downturn through the last full year before the 2025 CPI interruption. Rows use the same 216 first announcements and the primary prior. Thirty-five announcements reported losses. Lead time from issue to release ranged from 8 to 35 calendar days. Regressors use a frozen, revised workbook, so the calendar reconstruction is pseudo-real-time. The report also scores revised targets, tests tighter shrinkage, and separates the financial crisis from the first pandemic releases.

In the April 2026 retrospective illustration, the five-variable model assigned a 56.5% probability to a payroll decline. Replacing 26 pandemic-era months with a smooth path lowered it to 41.4%. BLS initially reported a gain of 115,000 jobs. This example measures sensitivity to the training history.

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

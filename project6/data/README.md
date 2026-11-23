# Inputs and release timing

## Monthly observations

The frozen workbook contains 771 consecutive month-start rows, January 1962 through March 2026. The study reads the following columns from its first sheet. The external workbook and the accompanying equity file are identified by SHA-256 in the [input audit](../results/input_audit.json).

| Workbook column | Units and transformation |
| --- | --- |
| Date | Reference month, recorded on its first day. |
| EMP | Seasonally adjusted U.S. total nonfarm payrolls in thousands. Natural log for estimation. [PAYEMS](https://fred.stlouisfed.org/series/PAYEMS) |
| IP | Seasonally adjusted industrial production index. Natural log. [INDPRO](https://fred.stlouisfed.org/series/INDPRO) |
| CPI | Seasonally adjusted all-items consumer price index. Natural log. [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL) |
| SP500 | Supplied monthly S&P 500 price index. Natural log. All 771 values match the separate equity workbook. Its daily sampling rule is undocumented. |
| TenYearBond, ThreeMonthBill | Monthly U.S. Treasury rates in percent. Long rate minus bill rate enters in percentage points. Both inputs match the published monthly series at every observation. [GS10](https://fred.stlouisfed.org/series/GS10) · [TB3MS](https://fred.stlouisfed.org/series/TB3MS) |

The source construction averages available daily ten-year observations by month. The archived regression uses the workbook's monthly values, which agree with GS10 at published precision. The monthly equity and Treasury information is treated as available after the source month closes. Each issue date also waits for payroll, CPI and industrial production.

October 2025 CPI is the only missing input. The April 2026 illustration uses the geometric mean of the September and November index levels (324.653742). Every historical forecast ends before the gap. The loader requires the November observation to use this interpolation. BLS documents the interruption to CPI collection. [BLS collection FAQ](https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi-faq.htm)

## Release records and outcomes

[Release calendar](release_calendar.csv) records the publication of the origin month's indicators, the forecast issue date and the first publication of the target payroll number. The issue date is the latest of the origin month's three macro releases. Equity and Treasury data end in the same observation month. Partial data for the target month are excluded. Dates are calendar days. The October 2013 delay is retained.

To rebuild the calendar, save the ALFRED release-date text exports for releases 50 and 10 as `release-50.txt` and `release-10.txt`. Save the Federal Reserve's HTML release-date page as `fed.txt`. Run `python scripts/extract_calendar.py DIRECTORY data/release_calendar.csv`. The Fed's dated monthly list identifies regular G.17 releases. The selected ALFRED calendars use the first employment release in each publication month and the final CPI release in February when seasonal-adjustment records add a second date. [Employment dates](https://alfred.stlouisfed.org/release/downloaddates?rid=50&ff=txt) · [CPI dates](https://alfred.stlouisfed.org/release/downloaddates?rid=10&ff=txt) · [Federal Reserve G.17 dates](https://www.federalreserve.gov/releases/g17/release_dates.htm)

[First releases](first_releases.csv) are the first seasonally adjusted **over-the-month changes** in the BLS CES revisions table. The second score uses the difference of adjacent employment levels from the frozen workbook. These are distinct outcomes. The historical regressors also contain revisions, so this is a pseudo-real-time exercise. Neither the workbook's date of extraction nor historical vintages for all five variables are established. The April 2026 first release was +115,000. The separate Econoday survey estimate was +63,000. Its timing and information set differ from the model's April 16 cutoff. [BLS vintage table](https://www.bls.gov/web/empsit/cesnaicsrev.htm) · [April release](https://www.bls.gov/news.release/archives/empsit_05082026.htm) · [Econoday survey](https://www.cmegroup.com/education/events/econoday/667021)

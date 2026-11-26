# Input records

The submitted source is titled *EC 607 Time-Series Econometrics, Problem Sets 9–10*. The activity notebook labels its section HW9. In the portfolio this work is Project 08. The [archived statistics](coursework.csv) and [archived impulse responses](coursework_hours.csv) distinguish printed notebook outputs from tables in the submitted paper. The notebook SHA-256 is `8cb38041e00a5569e3167f3bb8cf90567daf5892520f434a1124f3c104a1f242`. The submitted manuscript source SHA-256 is `2a25f55c28e334b7fdc2b32cb355f1a13517b3c9de00dede1cb128746e3d4912`.

| Frozen input | Observations and units |
| --- | --- |
| ci_data.xlsx | 711 monthly levels, January 1967–March 2026. By header: real manufacturing and trade sales (CMRMTSPL), industrial production (INDPRO), nonfarm payrolls (PAYEMS) and real personal income excluding transfers (W875RX1). Use 100 times the log difference. |
| Wide_Data.xlsx | 400 indexed rows, one target Y and 100 predictors. The supplied file establishes neither a calendar frequency nor an economic identity for Y. Scores are in Y's units. |
| OPHNFB.csv and HOANBS.csv | Quarterly productivity and hours indexes for the nonfarm business sector, 1947Q1–2026Q1. Both structural fits stop at their declared end quarter. Local historical data can differ from later revisions. |
| Daily adjusted closes | SPY, QQQ, IWM and DIA. A November 25, 2026 Yahoo Finance extraction contains 4,128 common USD price dates from December 30, 2009 to May 29, 2026. Factor estimation uses 4,126 daily returns starting in January 2010. Adjusted close accounts for splits and distributions. |

The first four indicator names in the coursework preprocessing function differ from the actual order of the workbook columns. Eight corrections for unusual income growth were applied to the second data column, industrial production. The named-series fit applies them to W875RX1. The month pairs are December–January in 1992–93, 1993–94, 2004–05 and 2012–13. Both adjusted versions replace March 2020 through March 2022 inclusive with each series' mean. An additional fit keeps every observed change. Full-sample transformations are used for retrospective inference only.

The equity submission retained plots. The specific historical price file, date selection and adjustment convention used in class could not be verified. The new daily sample tests the same market interpretation with a documented source. Adjusted price histories can change when retrieved again. A strict replay requires the [frozen input hashes](../results/inputs.json). A fresh price pull can be used in a separate output directory with its own recorded hash. The external input files remain outside the repository.

[FRED productivity](https://fred.stlouisfed.org/series/OPHNFB) · [FRED hours](https://fred.stlouisfed.org/series/HOANBS) · [Yahoo Finance SPY history](https://finance.yahoo.com/quote/SPY/history/) · [QQQ](https://finance.yahoo.com/quote/QQQ/history/) · [IWM](https://finance.yahoo.com/quote/IWM/history/) · [DIA](https://finance.yahoo.com/quote/DIA/history/)

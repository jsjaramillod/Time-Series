# Forecasting U.S. Payrolls

In my doctoral time-series course, my five-variable model put the April 2026 gain at 61,140 jobs. The first announcement was +115,000, an absolute error of 53,860. Which indicators earn their place across many releases, and how much does the treatment of the pandemic matter? The historical comparison tests three information sets on the same 216 first announcements from January 2007 to December 2024.

## Most of the gain came from production and prices

Adding industrial production and CPI reduced root mean squared error by 24.7%, from 2.19 million to 1.65 million jobs. Adding equity prices and the Treasury spread reduced it a further 0.14%. Financial information made a clearer difference to loss probabilities. The average Brier score improved from 0.150 to 0.141. Lower scores reward probabilities closer to the outcome.

## Four pandemic releases dominated the errors

All three models missed every March–June 2020 first announcement with their nominal 90% intervals. Those four months accounted for 91.2% of the five-variable model's total squared error. After the sharp movements entered its expanding training sample, the model's average 90% interval spanned 2.92 million jobs in 2022–24, compared with 0.67 million in 2015–19.

## The original pandemic adjustment held up after the shock

The five-variable refit with the same smoothing rule missed April by 68,564 jobs. Keeping the pandemic observations raised the miss to 256,062. Across 34 earlier first releases from March 2022 to December 2024, smoothing lowered the model’s mean absolute error from 226,000 to 146,000 jobs. All 34 announcements were gains.

The models use historical release dates and a common cutoff, but their inputs contain subsequent revisions. The backtest measures performance under that reconstruction. The fixed April 2026 comparison uses saved estimates from two different input snapshots. The 34 paired forecasts use the same revised inputs and model settings.

[Research report](docs/payroll_forecasts.pdf) · [Forecast records](results/forecasts.csv) · [BLS April 2026 release](https://www.bls.gov/news.release/archives/empsit_05082026.htm)

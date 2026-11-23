# Forecasting U.S. Payrolls

Would a labor-market analyst improve the next payroll forecast by adding industrial production, consumer prices, equity prices and the Treasury yield curve? Three Bayesian models face the same 216 monthly releases from January 2007 through December 2024. Each predicts both the job count and the probability of a decline.

## Most of the gain came from production and prices

Adding industrial production and CPI reduced root mean squared error by 24.7%, from 2.19 million to 1.65 million jobs. Adding equity prices and the Treasury spread reduced it a further 0.14%. Financial information made a clearer difference to loss probabilities. The average Brier score improved from 0.150 to 0.141. Lower scores reward probabilities closer to the outcome.

## Four pandemic releases dominated the errors

All three models missed every March–June 2020 first announcement with their nominal 90% intervals. Those four months accounted for 91.2% of the five-variable model's total squared error. After the sharp movements entered its expanding training sample, the model's average 90% interval spanned 2.92 million jobs in 2022–24, compared with 0.67 million in 2015–19.

## How much history changes a risk judgment

With data through March 2026, the five-variable model put the chance of an April payroll decline at 56.5%. Smoothing 26 pandemic-era months in the same history lowered it to 41.4%. The first announcement was a gain of 115,000 jobs. The risk estimate depended materially on the treatment of an exceptional episode.

The models use historical release dates and a common cutoff, but their inputs contain subsequent revisions. The backtest measures performance under that reconstruction. The fixed April 2026 illustration is separate from the 216 scored months.

[Research report](docs/payroll_forecasts.pdf) · [Forecast records](results/forecasts.csv) · [BLS April 2026 release](https://www.bls.gov/news.release/archives/empsit_05082026.htm)

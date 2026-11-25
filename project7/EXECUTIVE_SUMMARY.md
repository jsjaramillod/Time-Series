# Payroll Forecasts After a Shock

In my doctoral assignment, payrolls were projected to rise by about 289,000 over the six months from March to September 2026. The September release vintage showed a gain of 394,000. The class model allowed pandemic shocks to have unusually large variance. Would keeping those observations improve later forecasts, compared with smoothing them away?

## What the assignment got right

The original model put the March 2020 innovation variance at 103.6 times its normal level. The independent fit put it at 103.7. For the 2026 illustration, the original median missed by about 105,000 jobs, the refitted volatility model by 79,000, and interpolation by 40,000.

## The 28 matched forecasts

On revised employment levels, temporary volatility lowered root mean squared error from 611,000 to 473,000 jobs against constant variance, a 22.6% reduction. Smoothing the training series and starting from observed payroll conditions gave the smallest error, 459,000. Repeating the previous six-month gain scored 596,000.

## What changes with the publication being forecast

Against the sum of six first announcements, the volatility model had the lowest error of the three statistical models (665,000 jobs). Repeating the preceding six-month gain was more accurate at 515,000. These are different published measures of hiring. The rankings apply to overlapping windows that all ended in gains.

## How much uncertainty was warranted?

All three 90% intervals included all 28 revised outcomes. The average width was 7.89 million jobs with constant variance, 3.20 million with temporary volatility and 3.08 million with interpolation. These post-crisis outcomes give little evidence about the ability to forecast a new contraction.

[Research report](docs/payroll_after_shocks.pdf) · [One-page brief](docs/executive_summary.pdf) · [Forecast scores](results/scores.csv) · [September outcome](data/september_release.csv)

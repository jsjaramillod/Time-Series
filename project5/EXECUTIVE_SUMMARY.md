# U.S. GDP Growth at Risk

In my doctoral assignment, the trend AR(4) put the fifth percentile of annual GDP growth from 2026Q1 at -0.218 log percentage points. The portfolio estimate is -0.342 using the same pandemic smoothing. The original chart paired an annual GDP forecast with a one-quarter spread forecast. Matching horizons lets us test both average growth and the downside against realized outcomes.

## Did the portfolio improve the annual forecast?

Across 78 matched annual origins, root mean squared error fell from 2.191 to 2.069 log percentage points, a 5.6% reduction. Fifth-percentile loss fell from 0.442 to 0.405. The coursework method won on both measures over the 13 post-pandemic origins. Its mean-forecast error was 0.578 against 0.625. The windows overlap and use revised GDP.

## The cost of missing a rebound

In a separate comparison over all 86 annual origins using observed GDP, the random walk with drift delivered the more accurate mean forecast. Its root mean squared error was 2.31 log percentage points, against 3.30 for the trend autoregression. One forecast accounts for 57% of the autoregression's total squared error. After the second quarter of 2020 it projected -11.50% cumulative growth over the following year. Realized growth was +11.68%.

## A different ranking for downside risk

The autoregression produced a lower average fifth-percentile loss than the drift benchmark (0.419 versus 0.457). Its average central 90% interval was wider (5.96 versus 4.64 points). Growth fell below its downside threshold in 8 of 86 forecasts (9.3%), above the nominal 5%. Five of those misses ended between 2008Q3 and 2009Q3 and three ended in 2020. Adjacent annual outcomes share three quarters.

## What did the Treasury spread add at a common horizon?

Over 89 one-quarter targets, adding the spread nudged fifth-percentile loss from 0.1855 to 0.1852. Mean-forecast error increased from 1.800 to 1.818 points.

## Downside risk at the 2026Q1 forecast origin

Using observed GDP through 2026Q1, the autoregression put mean four-quarter growth at 2.04%, the fifth percentile at -1.15%, and the probability of contraction at 14.4%. Smoothing eight pandemic quarters in the training sample moved the fifth percentile to -0.34% and the contraction probability to 8.0%. These estimates describe a fixed historical exercise using revised GDP levels.

[Research report](docs/gdp_growth_at_risk.pdf) · [Matched coursework and portfolio forecasts](results/course_comparison.csv) · [Saved coursework estimates](../coursework_results.csv) · [Forecast scores](results/scores.csv)

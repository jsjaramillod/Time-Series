# U.S. GDP Growth at Risk

How much weight should an economist put on the downside of a GDP forecast? An expanding-window comparison of U.S. quarterly data from 1984Q1 through 2026Q1 evaluates forecasts of both average growth and the point below which a model places 5% of outcomes. The choice of model depends on which forecast is scored.

## The cost of missing a rebound

The random walk with drift delivered the more accurate annual growth forecast on average. Its root mean squared error was 2.31 log percentage points, against 3.30 for the trend autoregression. One forecast accounts for 57% of the autoregression's total squared error. After the second quarter of 2020 it projected -11.50% cumulative growth over the following year. Realized growth was +11.68%.

## A different ranking for downside risk

The autoregression produced a lower average fifth-percentile loss than the drift benchmark (0.419 versus 0.457). Its average central 90% interval was wider (5.96 versus 4.64 points). Growth fell below its downside threshold in 8 of 86 forecasts (9.3%), above the nominal 5%. Five of those misses ended between 2008Q3 and 2009Q3 and three ended in 2020. Adjacent annual outcomes share three quarters.

## Limited incremental information from the yield curve

Across 89 one-quarter outcomes, adding four lags of the 10-year minus 2-year Treasury spread changed fifth-percentile loss from 0.1855 to 0.1852. Mean-forecast error rose from 1.80 to 1.82 points. The small change in the tail score offers little basis for choosing the larger model in this sample.

## Downside risk at the 2026Q1 forecast origin

Using observed GDP through 2026Q1, the autoregression put mean four-quarter growth at 2.04%, the fifth percentile at -1.15%, and the probability of contraction at 14.4%. Smoothing eight pandemic quarters in the training sample moved the fifth percentile to -0.34% and the contraction probability to 8.0%. These estimates describe a fixed historical exercise using revised GDP levels.

[Research report](docs/gdp_growth_at_risk.pdf) · [Forecast scores](results/scores.csv) · [Forecast records](results/forecasts.csv)

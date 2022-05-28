# Expected returns estimators

One of the most, if not the most, important inputs when executing optimization methods, where the objective function, more or less aims to maximize expected return, is the vector of expected returns for each asset.
A small increase in the expected return of just one of a portfolio’s assets can potentially force half of the assets from the resulting optimal portfolio, see [^1].

In this document, we always refer to the expected returns with the bold greek letter $\boldsymbol \mu$, alternatively as the expectation operator of the asset returns $\mathbb{E}[{\mathbf{R}}]$.

Here we offer many estimators of expected returns, both using linear and logarithmic returns, with or without compounding.
Additionally, `scikit-portfolio` introduces other estimators for the expected returns based on `median` or exponentially weighted averages, which turn out to be slightly better in the backtesting results than the standard mean historical returns.

## Returns estimator base class
This base class is at the core of all returns estimators. THe `.fit` method takes either *price* data or *return* data, depending on the initialization parameter `returns_data`. Moreover you can always modifiy the parameter with the '.set_returns_data(True|False)' method, returning a new estimator.


<hr>

## Mean historical linear-returns `MeanHistoricalLinearReturns`  [📖](../returns_api#meanhistoricallinearreturns)
Mean historical returns are simply computed as the historical **average** of the geometric returns over all data.
In other words, given the returns time series $r_t$ for $t=1,\dots,\T$, the mean historical returns are obtained as

\begin{equation}
\hat{\boldsymbol \mu} = \frac{1}{T} \sum_{t=1}^T r_t
\end{equation}


<hr>

## Compounded historical linear-returns `CompoundedHistoricalLinearReturns` [📖](../returns_api#compoundedhistoricallinearreturns)
Compounded historical returns are simply computed as the geometric **average** of the linear historical returns.
In other words, given the returns time series $r_t$ for $t=1,\dots,\T$, the compounded historical returns are obtained as:

\begin{equation}
\hat{\boldsymbol \mu} = \left(\prod_{t=1}^T 1+ r_t\right)^{1/T}
\end{equation}


<hr>

## Mean historical log-returns `MeanHistoricalLogReturns` [📖](../returns_api#meanhistoricallogreturns)
Here, rather than using linear returns, we compute the average of the log returns $\log(p_t/p_{t-1})$:

\begin{equation}
\hat{\boldsymbol \mu} = \frac{1}{T} \sum_{t=1}^T \left( \log p_t - \log p_{t-1} \right)
\end{equation}


<hr>

## Compounded historical log-returns `CompoundedHistoricalLogReturns` [📖](../returns_api#compoundedhistoricallogreturns)
Here, rather than using linear returns, we compute the average of the log returns $\log(p_t/p_{t-1})$, and geometric average

\begin{equation}
\hat{\boldsymbol \mu} = \left(\prod_{t=1}^T 1 + (\log p_t - \log p_{t-1})\right)^{1/T} 
\end{equation}


<hr>

## Median historical linear returns `MedianHistoricalLinearReturns` [📖](../returns_api#medianhistoricallinearreturns)
Like for `MeanHistoricalLinearReturns`, but using **median** rather than average.
In other words, given the returns time series $r_t$ for $t=1,\dots,T$, the median historical returns are obtained as

\begin{equation}
\hat{\boldsymbol \mu} = \textrm{median} \left( \{ r_{1},\ldots, r_{T} \} \right)
\end{equation}

<hr>

## Median historical log returns `MedianHistoricalLogReturns` [📖](../returns_api#medianhistoricallogreturns)
Like for `MeanHistoricalLinearReturns`, but using **median** rather than average.
In other words, given the log-returns time series $\log p_t/p_{t-1}$ for $t=1,\dots,T$, the median historical log-returns are obtained as:

\begin{equation}
\hat{\boldsymbol \mu} = \textrm{median} \left( \{ \log (p_{t+1}/p_{t})_{t=1}, \ldots, \log( p_T/p_{T-1})_{t=T} \} \right)
\end{equation}

<hr>

## Exponential Moving Average Returns `EMAHistoricalLinearReturns` [📖](../returns_api#emahistoricalreturns)
Estimates the (annualized if frequency=252) expected returns as the exponential moving average of linear historical returns.
Compounding is set to false by default.

<hr>

## Rolling Median Returns `RollingMedianReturns` [📖](../returns_api#rollingmedianreturns)
Estimates the returns from the average of the rolling median over a `window` of 20 observations, by default.

## CAPM expected returns `CAPMReturns` [📖](../returns_api#capmreturns)
Compute a return estimate using the Capital Asset Pricing Model. Under the CAPM, asset returns are equal to market returns plus a $\beta$ term encoding the relative risk of the asset.
The formula for calculating the expected return of an asset given its risk is as follows:

\begin{equation}
\mathbb{E}[R_i] = r_f + \beta_i \left( \mathbb{E}\lbrackR_m\rbrack - r_f\right)
\end{equation}

where
- $\mathbb{E}[R_i]$ is the expected return of asset $i$
- $r_f$ is the risk-free rate (default value is `0`)
- $\beta_i$ is the beta of the investment
- $ \left( \mathbb{E}\lbrackR_m\rbrack - r_f\right)$ is the market risk premium

The beta of a potential investment is a measure of how much risk the investment will add to a portfolio that looks like the market. If a stock is riskier than the market, it will have a beta greater than one. If a stock has a beta of less than one, the formula assumes it will reduce the risk of a portfolio.

See this page on [Investopedia](https://www.investopedia.com/terms/c/capm.asp) for more information about the CAPM.

<hr>

# References
[^1]: Best, M.J., and Grauer, R.R., (1991). On the Sensitivity of Mean-Variance-Ecient Portfolios to Changes in Asset Means: Some Analytical and Computational Results. The Review of Financial Studies, pp. 315-342.
[^3]: Lo, Andrew. "The statistics of Sharpe ratio". Financial Analysts Journal (2003).
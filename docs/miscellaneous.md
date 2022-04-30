# Miscellanous portfolio

Not all portfolio methods require hard mathematical optimization. Some of them are much simpler and require very simple calculations.
Here we list the simple portfolio optimization methods implemented in `scikit-portfolio`.
Some of them which are only useful for coding convenience, and to maintain beautiful optimization pipelines.

## Equally weighted portfolio
The equally weighted portfolio. Absolutely the simplest portfolio strategy to adopt. Interestingly, many authors claim that this portfolio achieves the best risk-reward ratio most of the time.
It is documented in the literature that due to estimation errors, mean-variance efficient portfolios often deliver no higher out-of-sample Sharpe ratios than does the naïve equally-weighted portfolio (EWP).
DeMiguel, Garlappi, and Uppal, (2007) [^1] is commonly cited to dismiss optimization based methods, in favor of equally weighted portfolios.

The proportion of weights given to each asset in the portfolio is equal to $1/N$.

\begin{split}
\begin{equation}
\begin{aligned}
w_i &= \frac{1}{N} & & \forall i = 1,\ldots,N
\end{aligned}
\end{equation}
\end{split}

!!! note
	Other authors (see Kritzman [^2]) instead have contrasting views about the better efficiency of the 1/N portfolio and say that the minimum volatility portfolio yields better out-of-sample results .
	Please do your own research to convince yourself which of the two are right.


## Single Asset portfolio
Here for coding convenience. A portfolio where 100% of weight is given to a specific asset $i^\star$, chosen in the initialization phase.


\begin{split}
\begin{equation}
\begin{aligned}
\begin{cases}
w_i = N^{-1} & i = i^\star \\
w_i =0 & \textrm{otherwise}
\end{cases}
\end{aligned}
\end{equation}
\end{split}

```python
from skportfolio import SingleAsset
from skportfolio.datasets import load_tech_stock_prices

print(SingleAsset(asset="TSLA").fit(load_tech_stock_prices()).weights_)
```

## Inverse variance portfolio
In this portfolio, asset weights are specified inversely proportional to asset returns variance.

\begin{equation}
w_i = \frac{1/\sigma^2 \lbrack r_i  \rbrack}{\sum_{j=1}^N 1/\sigma^2 \lbrack r_j \rbrack}
\end{equation}

```python
from skportfolio import InverseVariance
from skportfolio.datasets import load_tech_stock_prices

print(InverseVariance().fit(load_tech_stock_prices()).weights_)
```

## Inverse volatility portfolio
In this portfolio, asset weights are specified inversely proportional to asset returns volatility $\sigma=\sqrt{\sigma^2}$.
Very similar to Inverse Variance Portfolio.

\begin{equation}
w_i = \frac{1/\sigma \lbrack r_i  \rbrack}{\sum_{j=1}^N 1/\sigma \lbrack r_j \rbrack}
\end{equation}


```python
from skportfolio import InverseVolatility
from skportfolio.datasets import load_tech_stock_prices

print(InverseVolatility().fit(load_tech_stock_prices()).weights_)
```

## Capitalization weighted
First the market capitalization is calculated as the prices of the asset times the number of outstanding shares, then weights are obtained as the sum-one normalized average of the market capitalization over time.

```python
import pandas as pd
from skportfolio import CapWeighted
from skportfolio.datasets import load_tech_stock_prices

prices = load_tech_stock_prices()
shares = pd.Dataframe(index=prices.index, columns=prices.columns, data=1_000_000)  # fake shares, 1M outstanding shares for each stock
print(CapWeighted().fit(X=prices, y=shares).weights_)
```

# References
[^1]: DeMiguel, Victor, Lorenzo Garlappi, and Raman Uppal. "Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy?." The review of Financial studies 22.5 (2009): 1915-1953 [http://faculty.london.edu/avmiguel/DeMiguel-Garlappi-Uppal-RFS.pdf](http://faculty.london.edu/avmiguel/DeMiguel-Garlappi-Uppal-RFS.pdf)
[^2]: Kritzman, Page & Turkington (2010) "In defense of optimization: The fallacy of 1/N". Financial Analysts Journal, 66(2), 31-39.
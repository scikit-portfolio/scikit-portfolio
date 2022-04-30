# Efficient Conditional Drawdown At Risk Frontier
Conditional drawdown at risk is a measure often used in portfolio optimization for **effective risk management** introduced by Cheklov, Uryasev and Zabarankin in 2000 [^1],[^2].
The CDar (conditional drawdown-at-risk) is the average of all drawdowns for all the instances that drawdown exceeded a certain threshold.
Drawdown is a measure of downside risk.

Conditional drawdown is the average of all drawdowns, or cumulative losses, in excess of a certain threshold.
That threshold is referred to as drawdown-at-risk. It is very similar to how expected shortfall is defined.

CDar in the context of portfolio optimization was first proposed by Alexeei Chekhlov, Stanislav Uryasev and Michael Zabarankin in 2003. 
The main goal of their paper was to show that the optimization of CDar can be solved using linear programming, a kind of problems that `cvxpy` is very good at solving.

The portfolio *conditional drawdown at risk* is based on using portoflio drawdowns rathar than returns.
For the definition [look here](https://en.wikipedia.org/wiki/Drawdown_(economics)) and the original [`PyPortfolioOpt` version](https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html#efficientcdar).

Here we offer three specific classes to handle the efficient CDar frontier, namely the portfolio with the Minimum CDar, and the portfolios of lowest CDar given target return or the highest return given CDar. 

## Minimum CDar portfolio
The minimum CDar portfolio identifies the point with the least Conditional Drawdown at risk along the CDar efficient frontier.

## Efficient return on mean-cdar frontier


## Efficient risk on mean-cdar frontier



# References
[^1]: Chekhlov, Alexei, Stanislav Uryasev, and Michael Zabarankin. "Portfolio optimization with drawdown constraints." Supply chain and finance. 2004. 209-228.
[^2]: Chekhlov, Alexei, Stanislav Uryasev, and Michael Zabarankin. "Drawdown measure in portfolio optimization." International Journal of Theoretical and Applied Finance 8.01 (2005): 13-58.
<hr>
#  Efficient Conditional Value At Risk Frontier
Conditional value at risk is a measure often used in portfolio optimization for **effective risk management**.
The CVar (conditional value-at-risk) also called expected shortfall is a popular measure of **tail risk**.
CVaR is derived by taking a weighted average of the "extreme" losses in the tail of the distribution of possible 
returns, beyond the value at risk (VaR) cutoff point $\beta$.

For example, if we calculate the CVaR to be 10% for $\beta=0.95$, we can be $95\%$ confident that the worst-case average daily loss will be 3 %.

In other words, one can see CVar in terms of percentiles, so that it boils down to the average of all losses so severe that they only occur $(1−\beta)\%$ of the times.

We will adopt the following notation: 

- *w* for the vector of portfolio weights
- *r* for a vector of asset returns (daily), with probability distribution $p(r)$.
- $L(w, r) = - w^T r$ for the loss of the portfolio
- $\alpha$ for the portfolio value-at-risk (VaR) with confidence $\beta$.

The CVaR can then be written as:

\begin{equation}
CVaR(w, \beta) = \frac{1}{1-\beta} \int_{L(w, r) \geq \alpha (w)} L(w, r) p(r)dr.
\end{equation}

This is a nasty expression to optimize because we are essentially integrating over VaR values. The key insight
of Rockafellar and Uryasev (2001) [^3] is that we can can equivalently optimize the following convex function:

\begin{equation}
F_\beta (w, \alpha) = \alpha + \frac{1}{1-\beta} \int [-w^T r - \alpha]^+ p(r) dr,
\end{equation}

where $[x]^+ = \max(x, 0)$. The authors prove that minimising $F_\beta(w, \alpha)$ over all $w, \alpha$ minimises the CVaR.
Suppose we have a sample of *T* daily returns (these can either be historical or simulated). 
The integral in the expression becomes a sum, so the CVaR optimization problem reduces to a linear program:

\begin{equation*}
\begin{aligned}
& \underset{w, \alpha}{\text{minimise}} & & \alpha + \frac{1}{1-\beta} \frac 1 T \sum_{i=1}^T u_i \\
& \text{subject to} & & u_i \geq 0  \\
&&&  u_i \geq -w^T r_i - \alpha. \\
\end{aligned}
\end{equation*}

This formulation introduces a new variable for each datapoint (similar to Efficient Semivariance), so
you may run into performance issues for long returns dataframes. At the same time, you should aim to
provide a sample of data that is large enough to include tail events. 

## Minimum CVar portfolio [📖](../efficient_cvar_api#minimum_cvar)

<hr>

## Efficient return on mean-cvar frontier [📖](../efficient_cvar_api#efficient-return-optimization-on-cvar-frontier)

<hr>


## Efficient risk on mean-cvar frontier [📖](../efficient_cvar_api#efficient-risk-optimization-on-cdar-frontier)


<hr>

## References
[^3]: Chekhlov, A.; Rockafellar, R.; Uryasev, D. (2005). [Drawdown measure in portfolio optimization](http://www.math.columbia.edu/~chekhlov/IJTheoreticalAppliedFinance.8.1.2005.pdf)
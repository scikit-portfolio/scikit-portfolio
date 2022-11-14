# Black-Litterman

Investment analysts can incorporate subjective opinions (based on estimations from investment analysts) into market equilibrium returns using the Black-Litterman model, an asset allocation strategy. 
The Black-Litterman model offers a systematic method to estimate the mean and covariance of asset returns by combining analyst perspectives and equilibrium returns as opposed to solely relying on historical asset returns.

|                      | Markowitz Mean variance                      | Black-Litterman approach                                   |
|----------------------|----------------------------------------------|------------------------------------------------------------|
| $\boldsymbol \mu$    | mean, median, CAPM etc of historical returns | blended historical returns and analyst foreseen returns    |
| $\boldsymbol \Sigma$ | (shrinked) covariance of historical returns  | blended (historical) covariance with posterior uncertainty |

In the Black-Litterman model, the blended expected return are obtained from the following expression, derived from Bayesian theory.

\begin{equation}
\boldsymbol \mu = \lbrack \mathbf{P}^T \boldsymbol \Omega^{-1} + (\tau \boldsymbol \Sigma)^{-1}  \rbrack^{-1} \lbrack \mathbf{P}^T \boldsymbol \Omega \mathbf{P} + (\tau \boldsymbol \Sigma){-1} \boldsymbol \Pi \rbrack
\end{equation}

and the posterior estimation covariance is updated as:

\begin{equation}
\boldsymbol \Sigma(\boldsymbol \mu) = \boldsymbol \Sigma + \lbrack \mathbf{P}^T \boldsymbol \Omega \mathbf{P} + (\tau \boldsymbol \Sigma)^{-1} \rbrack^{-1}
\end{equation}

In other words, the Black Litterman model tells the Bayesian optimal way of integrating the historical expected returns with the analyst view.
Importantly, under this framework, the analyst has the possibility to both specify *absolute views* or *relative views*.

Absolute views express the analyst expectations of average return for a given asset or groups of assets.
For example one can say the AAPL will return 5% on a yearly basis. 

!!! warning
    The analyst views must be expressed in the **same time frame** as the expected returns.
    For example if the expected returns estimator computes the expected daily returns, the analyst views must be expressed as daily expected returns.

The posterior returns and covariance $\boldsymbol \mu$ and $\hat{\boldsymbol \Sigma}$ can then be used as input to another mean variance optimization method.
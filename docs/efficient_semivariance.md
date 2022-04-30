
# 2. Mean-Semivariance frontier
Here instead of the classical Markowitz efficient frontier optimization where the covariance is taken into
account, we use the semivariance, i.e. the variance of only negative returns, as investors are more interested
 Here instead of the classical Markowitz efficient frontier optimization where the covariance is taken into
account, we use the semivariance, i.e. the variance of only negative returns, as investors are more interested
negative volatility, while positive volatility is good.
The optimization problem this portfolio method solves is the following:

\begin{split}
    \begin{equation*}
    \begin{aligned}
    & \underset{w}{\text{maximise}} & & \mathbf{w}^T \boldsymbol \mu \newline
    & \text{subject to} & & n^T n \leq s^*  \newline
    & & & B w - p + n = 0 \newline
    & & & w^T \mathbf{1} = 1 \newline
    & & & n \geq 0 \newline
    & & & p \geq 0. \newline
    \end{aligned}
    \end{equation*}
\end{split}

where $\boldsymbol \mu$ are the expected returns, $\mathbf{B}$ is the $T \times N$ scaled matrix of excess returns `B = (returns - benchmark) / sqrt(T)`.
The implementation is based on Markowitz et al. (2019)[^1].

!!! warning
    As stated on the pyportfolioopt page, finding portfolios on the mean-semivariance frontier is computationally harder than standard mean-variance optimization: the `pyportfolioopt` implementation uses `2T + N` optimization variables, meaning that for 50 assets and 3 years of data, there are about 1500 variables. 
    While EfficientSemivariance allows for additional constraints/objectives in principle, you are much more likely to run into solver errors. I suggest that you keep EfficientSemivariance problems small and minimally constrained.


Here we offer three specific portfolio estimators over the mean-semivariance frontier:

## 2.1 Minimum semi-volatility 

::: skportfolio.frontier._efficientfrontier.MinimumSemiVolatility

## 2.1 Efficient return on mean-semivariance frontier
::: skportfolio.frontier._efficientfrontier.MeanSemiVarianceEfficientReturn

## 2.3 Efficient risk on mean-semivariance frontier
::: skportfolio.frontier._efficientfrontier.MeanSemiVarianceEfficientReturn

<hr>

# References
[^1]: Markowitz et al., "Avoiding the Downside: a practical review of the critical line algorithm for mean-semivariance portfolio optimization" [https://www.hudsonbaycapital.com/documents/FG/hudsonbay/research/599440_paper.pdf](https://www.hudsonbaycapital.com/documents/FG/hudsonbay/research/599440_paper.pdf)
# Notation
- There are $N$ assets and $T$ temporal samples.
- The asset prices are encoded in a $T$ rows, $N$ columns array $\mathbf{P} \in \mathbb{R}^{T \times N} = \{ P_{ti} \}$
- The asset returns are encoded in a $T$ rows, $N$ columns array $\mathbf{R} \in \mathbb{R}^{T \times N} = \{ R_{ti} \}$
- We denote the $i$-th asset price at time $t$ as $P_{ti}$.
- We denote the $i$-th asset historical return at time $t$ as $R_{ti}$.
- The demeaned $i$-th asset return matrix is $\mathbf{Z} = \mathbf{R} - \frac{1}{T}(\mathbf{R}^T\mathbf{1}) = \{ Z_{ti} \} = \{ R_{ti} - \langle R_{ti} \rangle_t \}$.
- The temporal average operator is $\langle \cdot_{ti} \rangle_t$ takes a $T \times N$ array and calculates a $N$ dimensional array with the temporal averages. It is equivalent to $\mathbf{A}^T\mathbf{1}/T$ where $\mathbf{A} \in \mathbb{R}^{T \times N}$ and $\mathbf{1} \in \mathbb{T}$.
- The covariance matrices are denoted by uppercase bold symbol $\boldsymbol \Sigma$ and are $N \times N$ matrices.


# Problem background and historical introduction

Probability, statistics, and optimization are appropriate mathematical foundations for quantitative analysis of investment decisions.
As a result, a vast number of financial theories and models addressing this process have been developed.
Harry Markowitz established the field of portfolio theory in 1952[^1], when he proposed a model that became known as Modern Portfolio Theory (MPT).
The model implies that an investor wants to maximize the expected return on a portfolio while taking a certain degree of risk.
Portfolios that match these characteristics are referred to be efficient portfolios, while those with the same projected return but more risk are referred to as sub-optimal portfolios.
This approach encouraged investment experts to reconsider their asset allocations, and model adopters to shift their holdings in accordance with Markowitz (1952) and his successors' theories and models.
As time has passed, various flaws in MPT have been discovered (some of which had been identified by Markowitz since the beginning[^2]), prompting the development of new models that aim to address these flaws.

The risk measure in MPT, the standard deviation of asset returns, was deemed to be an unwise choice by Rom & Ferguson (1994)[^3].
They proposed the Post-Modern Portfolio Theory (PMPT) model, which uses the standard deviation of negative asset returns as the risk measure, which tends to better reflect reality, instead of the standard deviation of positive asset returns.

In contrast to Markowitz (1952)[^1] and Rom & Ferguson (1994)[^3], who employed quadratic models, Konno & Yamazaki (1991)[^4] proposed a linear model that used the mean-absolute deviation as a risk measure.
The model was found to operate similarly to the previous quadratic model, but because of its linearity, it significantly decreased the complexity of the mathematical procedures.

Black & Litterman (1991)[^7] solved MPT's problem of requiring asset expected returns input by devising a methodology that allows the portfolio manager to supply a relative perspective of specified sub-groups rather than projected returns.

Indeed, there have been numerous attempts to improve portfolio optimization algorithms.
Digitalization has had an impact on the practical work of investment professionals, in addition to mathematical development in the field of portfolio theory.
Optimization software appeared on the market as computing power grew, and it quickly became popular.
Portfolio management is available through programs like **Bloomberg**, **Axioma**, and **Barra**, as well as the possibility to optimize your portfolio in terms of asset weights.

These systems, on the other hand, are often sophisticated and come with a significant monthly charge.
Both of these characteristics could make it difficult for a retail user, or a person interested in portfolio optimization to perform simple experiments.

## Portfolio optimization
Portfolio optimization is the process of selecting asset weights in order to achieve an optimal portfolio, based on an objective function. 
Typically, the objective is to maximize expected return or to minimize financial risk. It can also be a combination of the two.

## Modern portfolio theory
MPT (Modern Portfolio Theory), or **mean-variance** analysis, is a theory pioneered by Harry Markowitz in 1952.
It assumes that investors make rational decisions and expect a higher return for increased risk.
According to the theory, it is possible to construct a portfolio which maximizes the expected return, given a certain level of risk.
Such portfolio is said to be on the *efficient frontier*. 
An investor would not take on extra risk if it does not mean larger returns. 
Conversely, the investor must take on more risk if the goal is to achieve higher returns.
A key insight in this theory is that the return and risk of an asset should not be viewed separately, since the two factors together affect a portfolios
overall risk and return (Markowitz, 1952). 

Despite its groundbreaking theories, MPT has faced criticism. To begin with, it requires the input of expected returns, which requires the investor to predict future outcomes. In practice, this is often done by extrapolating historical data. 
Such predictions often fail to take new circumstances into account, which results in predictions that are flawed. 
Also, as the risk measure of MPT is variance, the optimization model become quadratic, since variance is quadratic.
For large portfolios, this implies heavy computations, which can make the model inefficient in a computational sense.

Additionally, MPT assumes that the asset returns follow a Gaussian distribution, which has two serious implications. 
Firstly, it underestimates the probability of large and important movements in the price of an asset. 
Secondly, by relying on the correlation matrix, it fails to capture the relevant dependence structure among the assets.
This limits the practical usefulness of MPT (Rachev & Mittnik, 2006).

Nevertheless, MPT has contributed with strong theoretical value. The findings of Markowitz can be formulated in three different ways.
The three different views can be seen below, equation 1, 3 and 5.

- Maximization of portfolio return
- Minimization of portfolio risk
- Optimization of a combination of return and risk

## The Efficient Frontier
Markowitz introduced the efficient frontier, which is a cornerstone of MPT (1952).
It's the set of optimal portfolios that provides the highest expected return for a given level of risk, or the lowest risk for a given level of expected return.
Portfolios that do not correspond with the efficient frontier are sub-optimal by definition.
The efficient frontier is commonly depicted as a hyperbola, with the portfolio return on the vertical axis and risk or volatility on the horizontal axis, like in the following figure:

![efficient_frontier](imgs/efficient_frontier.svg)


##  Purpose and problematization
Decisions to invest, keep, decrease or dispose of holdings are based on information and conclusions derived from a collaborative fundamental process.

The implementation of ideas and the portfolio construction process is however, within relevant constraints, delegated to the manager responsible for the portfolio.
In this process a variety of tools are being used today to assist and support the portfolio managers, most prominently **Bloomberg’s** portfolio function.
This approach has historically worked well for the fund management company, based on their returns. 
The tools that are used today are however described as somewhat cumbersome and complicated to use. 
The company believes that quantitative input in the portfolio construction process is an important decision support for the portfolio managers and wants to find a more comprehensive tool to use in the day-to-day decisions.


The key characteristics of `scikit-portfolio` are: 

1. **Simplicity**
2. **Speed**
3. **Accuracy**
4. **Usability**

To improve **simplicity**, we wanted to create a tool with a straightforward API and a small number of input parameters. 
Despite this, users have a plethora of fine-tuning choices at their disposal, the majority of which may be determined automatically at runtime via hyperparameters optimization.
The `fit`-`transform`-`predict` paradigma from *scikit-learn* is used to interface with other libraries in the data science toolkit.
The API was chosen to be the same as the one available in *scikit-learn* because of its widespread use among non-financial sector practitioners.
This choice makes portfolio optimization accessible to a wide range of users, including data scientists, machine learning engineers, and researchers.
This is likely to increase the utility of the optimization tool.

The request for speed has influenced our choice of optimization algorithm and the implementation of it.
For this reason we largely rely on the excellent `cvxpy` which with its highly flexible interface but accurate routines is becoming the *de-facto* standard in convex optimization for Python.
Moreover, `cvxpy` not being a library of optimization routine, but rather a *lingua-franca* for specifying complex convex problems, makes it possible for the final user to specify the solver of choice. By default, the ECOS[^6] solver, covers a large domain of problems, but commercial and more performant solvers are available to `cvxpy` such as the excellent Gurobi solver[^5].

The third and final requested characteristic, *accuracy*, is proven through testing and comparison of `scikit-portfolio` solutions with other commercial packages, such as Matlab Financial Toolbox.

Quantitative investment managers and risk managers use portfolio optimization to determine the proportions of 
various assets in a portfolio. The goal of portfolio optimization is to maximize a measure of a portfolio's return 
as a function of a measure of portfolio's risk, or viceversa, to minimize risk given some minimum level of return. 

**`scikit-portfolio`**  provides a large set of portfolio optimization methods to perform capital allocation, asset 
allocation, and risk assessment, as well as a backtesting framework for portfolio allocation backtesting strategies

Portfolio optimization is a problem that requires to satisfy three **conflicting 💣️ criteria**:

- **Minimize a measure of risk ⚠️**
- **Match or exceed a measure of return 😎️**
- **Satisfy additional constraints, such as minimum allocations in certain assets.👨‍🎓️️**

A portfolio is completely specified by asset weights, specifying  the proportion of individual assets to be held  in 
terms of total capital invested.

A proxy for **risk** is a function that characterizes either the variability or losses associated with portfolio 
choices. Typically, higher risk mean higher chance of losing the capital.
Portfolio returns measures are instead a way to quantify the gross or net benefits associated with portfolio choices. 

It is now clear that risk and returns are conflicting objectives, in the sense that there must exist for each 
combination of them, a certain point such that increases in one optimization dimension (return) does not lead to 
further  decreases in the other dimension (risk), and viceversa.
The locus of those points is called the `efficient frontier` on a risk-satisfaction cartesian plot.



[^1]: Markowitz H., "Portfolio selection", J. Fin, Vol. 7, No. 1. (1952), pp. 77-91 [url](https://www.math.hkust.edu.hk/~maykwok/courses/ma362/07F/markowitz_JF.pdf)
[^2]: Markowitz itself noted that the average portfolio return and standard deviation were not good measures. Cited from its 1952 paper: *"One suggestion as to tentative $\mu_i$, $\sigma_{ij}$ to use the observed $\mu_i$, $\sigma_{ii}$ for some period of the past. I believe that better methods, which take
into account more information, can be found."*
[^3]: Rom, Brian M., and Kathleen W. Ferguson. "Post-modern portfolio theory comes of age." Journal of investing 3.3 (1994): 11-17.
[^4]: Konno, H, Yamazaki H. "Mean-absolute deviation portfolio optimization model and its applications to Tokyo stock market." Management science 37.5 (1991): 519-531.
[^5]: Various authors, "Gurobi Optimizer Reference Manual", (2022), [https://www.gurobi.com](https://www.gurobi.com)
[^6]: Domahidi, A., Chu E., and Boyd S. "ECOS: An SOCP solver for embedded systems." 2013 European Control Conference (ECC). IEEE, 2013.
[^7]: Black, Fischer, and Robert Litterman. "Asset allocation: combining investor views with market equilibrium." Goldman Sachs Fixed Income Research 115 (1990).
# Information theory based portfolio optimization

This section collects all the portfolio optimization methods based on information theory.
One example of this kind of methods is based on evaluating the volatility through Shannon entropy of returns, the higher
the larger the risk.

## REPO: Returns Entropy Portfolio Optimization
Here we provide the implementation of the research paper *An Entropy-Based Approach to Portfolio Optimization* [^1].
The basic idea of the method is to use Shannon entropy of the portfolio returns distribution as the risk proxy. 

### Portfolio Estimator object
The REPO optimization aims to solve the following constrained optimization problem: 

\begin{split}\begin{equation*}
\begin{aligned}
& \underset{\mathbf{w}}{\text{minimize}} & & H\left( r_P \right) \\
& \text{subject to} & &  \mathbf{w}^T \boldsymbol \mu \geq \rho_{\textrm{target}} \\
&&& \mathbf{w}^T \mathbf{1} = 1\\
&&& w_{min} \leq \mathbf{w}_i \leq w_{max} \\
\end{aligned}
\end{equation*}\end{split}

where $H(r_P)$ is the Shannon entropy of the portfolio returns distribution, $r_t = \sum\limits_{i=1}^N w_i r_{ti}$.
The entropy is estimated through a kernel density estimator, regulated parameterized by a number of variables.

### REPO Function


# References
[^1]: Mercurio et al. "An Entropy-Based Approach to Portfolio Optimization" [https://pubmed.ncbi.nlm.nih.gov/33286106/](https://pubmed.ncbi.nlm.nih.gov/33286106/)

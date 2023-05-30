"""
Robust portfolio estimators
-----------------------------

Here we collect robust portfolio estimators.

A robust portfolio estimator is one that takes into consideration not only pointwise estimation 
of returns and covariance, but larger influence sphere centered around the sample estimates.
"""
from typing import List, Union

import cvxpy as cp
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats.distributions import chi2

from skportfolio._base import PortfolioEstimator
from skportfolio.riskreturn import (
    BaseReturnsEstimator,
    BaseRiskEstimator,
    MeanHistoricalLinearReturns,
    SampleCovariance,
)


def chi2inv(a, dof):
    """
    Returns the inverse chi square statistics
    Parameters
    ----------
    a
    dof: degrees of freedom
    Returns
    -------
    """
    return chi2.ppf(a, dof)


def bayesian_robust_frontier_sample_estimate(
    num_ptfs: int,
    risk_matrix: Union[pd.DataFrame, np.ndarray],
    expected_returns: Union[pd.Series, np.ndarray],
    n_jobs: int = 1,
):
    """

    Parameters
    ----------
    num_ptfs: int
        Number of portfolios to generate
    risk_matrix: Union[pd.DataFrame, np.ndarray]
    expected_returns: Union[pd.Series, np.ndarray]
        A dataframe with n_assets columns and n_portfolio returns
    n_jobs: int
        Number of parallel jobs to compute frontier points
    Returns
    -------
    """
    n_assets = risk_matrix.shape[0]

    def solve_min_vol_at_target_return(target_return=None):
        _w = cp.Variable(shape=(n_assets,), nonneg=True)
        objective = 0.5 * cp.quad_form(_w, risk_matrix)
        constraints = [cp.sum(_w) == 1]
        if target_return is not None:
            constraints.append((target_return == (_w.T @ expected_returns)))
        try:
            cp.Problem(cp.Minimize(objective), constraints=constraints).solve()
        except cp.error.SolverError:
            return np.nan * np.zeros((n_assets,))
        if _w.value is not None:
            return _w.value
        return np.nan * np.zeros((n_assets,))

    # Perform a minimum volatility portfolio
    min_vol_w = solve_min_vol_at_target_return(target_return=None)
    # calculates the returns of the minimum volatility portfolio
    min_vol_return = min_vol_w.dot(expected_returns)
    # determine return of maximum-return portfolio, clearly the maximum over all returns
    max_ret_return = expected_returns.max()
    # slice efficient frontier in NumPortf equally thick horizontal sectors in the upper branch only
    target_returns = np.linspace(min_vol_return, max_ret_return, num_ptfs)
    # compute the NumPortf compositions and risk-return coordinates of the optimal
    # allocations relative to each slice
    # Starts with min_vol portfolio then increase til the maximum volatility maximum return portfolio

    # run in parallel over all other returns til the last portfolio with the maximum return weights_composition
    ptfs_weights_composition = np.array(
        Parallel(n_jobs=n_jobs,)(
            delayed(solve_min_vol_at_target_return)(target_returns[i])
            for i in range(0, num_ptfs)
        )
    )

    # computes all the portfolio expected returns by batch scalar product of portfolio weights with expected returns
    ptfs_expected_return = np.einsum(
        "ij,j", ptfs_weights_composition, expected_returns, optimize="greedy"
    )
    # computes all the portfolio volatilites by batch scalar product of portfolio weights with risk matrix
    ptfs_volatility = np.sqrt(
        np.einsum(
            "ij,jk,ik->i",
            ptfs_weights_composition,
            risk_matrix,
            ptfs_weights_composition,
            optimize="greedy",
        )
    )

    return (
        ptfs_expected_return,
        ptfs_volatility,
        ptfs_weights_composition,
    )


class RobustBayesian(PortfolioEstimator):
    """
    Implementation of the Bayesian Robust Allocation algorithm by Attilio Meucci.
    The implementation is based on the Matlab code provided by Meucci, with a number of computational tricks.
    It rolls over the prices dataframe based on the window parameter. It finally averages all the weights
    of the robust bayesian portfolios into a .weights_ attribute.
    """

    def __init__(
        self,
        window: int = 120,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        risk_estimator: BaseRiskEstimator = SampleCovariance(),
        n_portfolios: int = 10,
        robustness_param_loc: float = 0.1,
        robustness_param_scatter: float = 0.1,
        n_jobs: int = 1,
    ):
        """
        Initializes the RobustBayesian method by Attilio Meucci.

        Parameters
        ----------
        window: int
            The number of rows to roll over
        rets_estimator: BaseReturnsEstimator, default MeanHistoricalLinearReturns
            The expected returns estimator.
        risk_estimator: BaseRiskEstimator, default SampleCovariance
            The risk estimator.
        n_portfolios: int, default 10
            Number of portfolios to average over
        robustness_param_loc: float, default 0.1
            The level of confidence in the estimation risk for posterior averages.
            The higher the values, the better for less experienced investors, whereas lower values are more suitable for
            very self-confident investors with clear views in mind.

        robustness_param_scatter: float, default 0.1
            Denotes the confidence of investor to estimation risk in posterior covariance.
            The higher the values, the better for less experienced investors, whereas lower values are more suitable for
            very self-confident investors with clear views in mind.
        n_jobs: int, default 1
            Set to -1 for maximum parallelism with multicore processors

        See Also
        --------

        [Link to original article](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=681553)
        """
        super().__init__()
        self.window = window
        self.n_portfolios = n_portfolios
        self.rets_estimator = rets_estimator
        self.risk_estimator = risk_estimator
        self.robustness_param_loc = robustness_param_loc
        self.robustness_param_scatter = robustness_param_scatter
        self.n_jobs = n_jobs

    def _fit(self, rolling_prices: pd.DataFrame):
        self.all_weights_: List[Union[pd.Series]] = []
        S_hat = self.risk_estimator.fit(
            rolling_prices
        ).risk_matrix_  # sample covariance
        mu_hat = self.rets_estimator.fit(
            rolling_prices
        ).expected_returns_  # sample mean returns
        N = rolling_prices.shape[1]  # number of assets (columns)
        T = rolling_prices.shape[0]  # number of time points, or samples (rows)
        Q = self.n_portfolios

        _, ds_hat, w_hat = bayesian_robust_frontier_sample_estimate(
            num_ptfs=Q, risk_matrix=S_hat, expected_returns=mu_hat, n_jobs=self.n_jobs
        )
        # Bayesian prior
        S0 = np.diag(np.diag(S_hat))
        m0 = np.squeeze(0.5 * S0 @ np.ones([N, 1]) / N)  # prior mean?
        T0 = 2 * T
        nu0 = 2 * T
        # Bayesian posterior parameters
        T1 = T + T0
        m1 = 1 / T1 * (mu_hat * T + m0 * T0)
        nu1 = T + nu0
        # fmt: off
        S1 = (1/ nu1*( S_hat*T + S0*nu0+np.outer(mu_hat-m0, mu_hat-m0) / (1/T + 1/T0)))
        # fmt: on
        _, _, w1 = bayesian_robust_frontier_sample_estimate(
            num_ptfs=self.n_portfolios, risk_matrix=S1, expected_returns=m1
        )
        # robustness parameters
        q_m2 = chi2inv(self.robustness_param_loc, N)
        g_m = np.sqrt(q_m2 / T1 * nu1 / (nu1 - 2))
        q_s2 = chi2inv(self.robustness_param_scatter, N * (N + 1) / 2)
        pick_vol = int(np.round(0.8 * self.n_portfolios)) - 1
        v = ds_hat[pick_vol] ** 2
        g_s = v / (
            (nu1 / (nu1 + N + 1) + np.sqrt(2 * nu1**2 * q_s2 / ((nu1 + N + 1) ** 3)))
        )

        targets = []
        for wu in w1:
            # Bayesian allocation
            new_target = -np.inf
            if wu.T @ S1 @ wu <= g_s:
                new_target = m1 @ wu - g_m * np.sqrt(wu.T @ S1 @ wu)
            targets.append(new_target)
        # put the portfolio with the maximum target
        self.all_weights_.append(
            pd.Series(index=rolling_prices.columns, data=w1[np.argmax(targets)])
        )

    def fit(self, X: pd.DataFrame, y=None, **fit_params) -> PortfolioEstimator:
        def rolling_pipe(dataframe, window, fctn):
            return pd.Series(
                [
                    dataframe.iloc[i - window : i].pipe(fctn) if i >= window else None
                    for i in range(1, len(dataframe) + 1)
                ],
                index=dataframe.index,
            )

        X.pipe(rolling_pipe, self.window, self._fit)
        self.weights_ = (
            pd.concat(self.all_weights_, axis=1).mean(1).rename(self.__class__.__name__)
        )
        return self


class RobustMarkowitz(PortfolioEstimator):
    """
    Implementation of the robust markowitz efficient frontier.
    """

    def __init__(
        self, risk_aversion_kappa: float = 1.1, risk_aversion_lambda: float = 0.5
    ):
        self.risk_aversion_kappa = risk_aversion_kappa
        self.risk_aversion_lambda = risk_aversion_lambda
        pass

    def fit(self, X, y=None, **fit_params) -> PortfolioEstimator:
        n_assets = X.shape[1]
        if not self.returns_data:
            returns = X.pct_change()
        else:
            returns = X
        k = self.risk_aversion_kappa
        lamda = self.risk_aversion_lambda
        mu = self.rets_estimator.set_returns_data(True).fit(returns).expected_returns_
        q = self.risk_estimator.set_returns_data(True).fit(returns).risk_matrix_
        sigma = np.diag(np.diag(q))
        x_robust = cp.Variable(shape=(n_assets,), name="x", pos=True)
        z_robust = cp.Variable(shape=(n_assets,), name="z", pos=True)

        constraints = [
            cp.sum(x_robust) == 1,
            cp.quad_form(x_robust, sigma) <= cp.dot(x_robust, x_robust),
        ]
        objective = (
            -cp.scalar_product(mu, x_robust)
            + k * z_robust
            + lamda * cp.quad_form(x_robust, q),
        )
        problem = cp.Problem(cp.Minimize(objective), constraints=constraints)
        problem.solve()
        w_optim = pd.Series(data=x_robust.value, index=X.columns)

        """
        function new_weights = robustOptimFcn(current_weights, pricesTT) 
        % Robust portfolio allocation
        
        nAssets = size(pricesTT, 2);
        assetReturns = tick2ret(pricesTT);
        
        Q = cov(table2array(assetReturns));
        SIGMAx = diag(diag(Q));
        
        % Robust aversion coefficient
        k = 1.1;
        
        % Robust aversion coefficient
        lambda = 0.05;
        
        rPortfolio = mean(table2array(assetReturns))';
        
        % Create the optimization problem
        pRobust = optimproblem('Description','Robust Portfolio');
        
        % Define the variables
        % xRobust - x  allocation vector
        xRobust = optimvar('x',nAssets,1,'Type','continuous','LowerBound',0.0,'UpperBound',0.1);
        zRobust = optimvar('z','LowerBound',0);
        
        % Define the budget constraint
        pRobust.Constraints.budget = sum(xRobust) == 1;
        
        % Define the robust constraint
        pRobust.Constraints.robust = xRobust'*SIGMAx*xRobust - zRobust*zRobust <=0;
        pRobust.Objective = -rPortfolio'*xRobust + k*zRobust + lambda*xRobust'*Q*xRobust;
        x0.x = zeros(nAssets,1);
        x0.z = 0;
        opt = optimoptions('fmincon','Display','off');
        [solRobust,~,~] = solve(pRobust,x0,'Options',opt);
        new_weights = solRobust.x;
        
        end
        Parameters
        ----------
        X
        y
        fit_params

        Returns
        -------

        """
        pass

"""
Ensemble portfolio estimators
-----------------------------

Here we collect baggind-based and bootstrap based portfolio meta-estimators.

A "bagging" portfolio meta-estimator fits base portfolio estimator on random subsets of the original dataset, then
aggregating the individual weights (either by voting or by averaging) to form a final prediction.

A "bootstrap" based portfolio estimator instead is based on the idea to reduce the extreme sensitivity of
optimized weights from the returns estimator, by aggregating multiple results over randomly perturbed returns through
resampling from a multivariate normal given the sample covariance matrix and returns estimates.
"""
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import cvxpy as cp
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from pypfopt.exceptions import OptimizationError
from scipy.stats.distributions import chi2

from skportfolio._base import PortfolioEstimator
from skportfolio.riskreturn import BaseReturnsEstimator
from skportfolio.riskreturn import BaseRiskEstimator
from skportfolio.riskreturn import MeanHistoricalLinearReturns
from skportfolio.riskreturn import PerturbedReturns
from skportfolio.riskreturn import SampleCovariance


def chi2inv(a, dof):
    return chi2.ppf(a, dof)


class MichaudResampledFrontier(PortfolioEstimator):
    """
    Generic class to resample efficient frontier estimators.
    The idea, similar to bootstrapping, is based on the idea of Resampled Efficient Frontier by Michaud (1998)
    Here we run n_iter independent fit of a specified portfolio estimator with a stochastic perturbation of the
    expected returns based on multivariate normal sampling of returns given sample expected returns and covariance.
    """

    _required_parameters = ["ptf_estimator"]

    def __init__(
        self,
        ptf_estimator: PortfolioEstimator,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        risk_estimator: BaseRiskEstimator = SampleCovariance(),
        n_iter: Optional[int] = 100,
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        agg_func: Optional[
            Union[
                str,
                Callable[
                    [Union[pd.DataFrame, np.ndarray]], Union[pd.Series, np.ndarray]
                ],
            ]
        ] = "mean",
        returns_estimator_random_state: Optional[int] = None,
    ):
        """
        Initializes the ResampledEfficientFrontier object. Explicitly required to define the ptf_estimator.

        Parameters
        ----------
        ptf_estimator: PortfolioEstimator
            The user provided estimator for the base optimal portfolio allocation. Default value is `MinimumVolatility`.
        rets_estimator: BaseReturnsEstimator
            The expected returns estimator. Default value is the `MeanHistoricalValue` estimator.
        risk_estimator: BaseRiskEstimator
            The risk matrix estimator. Default value is the sample covariance we use `SampleCovariance` estimator.
        n_iter: int
            Number of Monte Carlo resampling steps
        n_jobs: Optional[int]
            Number of parallel jobs. If None then 1 single worker is launched. If -1 all cores are used.
        random_state: Optional[np.random.RandomState]
            The random number generator for replicability purpose.
        agg_func: Callable, str
            Aggregation method of weights. Default mean over all weights.
        returns_estimator_random_state: None
            The random state of the internal PerturbedReturns estimator. Set it to a specific number only if you need
            always the same expected returns. Not the case in most situations, though it may be useful for debugging.
        """
        super().__init__()
        self.ptf_estimator: PortfolioEstimator = ptf_estimator
        self.rets_estimator: BaseReturnsEstimator = rets_estimator
        self.risk_estimator: BaseRiskEstimator = risk_estimator
        self.n_iter: int = n_iter
        self.random_state: Optional[np.random.RandomState] = random_state
        self.n_jobs: Optional[int] = n_jobs
        self.perturbed_estimator: PerturbedReturns = PerturbedReturns(
            rets_estimator=self.rets_estimator,
            risk_estimator=self.risk_estimator,
        )
        self.all_weights_: Optional[List[np.ndarray]] = None
        self.agg_func: Union[
            str,
            Callable[[Union[pd.DataFrame, np.ndarray]], Union[pd.Series, np.ndarray]],
        ] = agg_func
        self.returns_estimator_random_state: Optional[
            np.random.RandomState
        ] = returns_estimator_random_state
        self.risk_rewards_: Optional[List[Tuple[float, float]]] = None

    def fit(self, X, y=None, **fit_kwargs) -> PortfolioEstimator:
        # modifies the returns estimator of the provided portfolio estimator
        self.ptf_estimator.rets_estimator = self.perturbed_estimator.reseed(
            self.random_state
        )

        def _compute_weights_risk_reward(_x):
            # this trick is to send to workers newly initialized random state objects
            if self.n_jobs == 1:
                self.perturbed_estimator.reseed(self.returns_estimator_random_state)
            else:
                self.perturbed_estimator.reseed(
                    np.random.default_rng(self.returns_estimator_random_state)
                )
            try:
                ptf = self.ptf_estimator.fit(_x)
                return ptf.weights_, *ptf.risk_reward()
            # in case it is not possible to fit we return a triple of Nan, to be filtered afterwards
            except (AttributeError, OptimizationError) as portfolio_error:
                return (np.nan,) * 3

        # run n_iter portfolio optimization, where each estimator has the expected returns perturbed by a random term
        # if n_jobs=1 do not spawn any parallel process, simply do a list comprehension
        if self.n_jobs == 1 or self.n_jobs is None:
            out = [_compute_weights_risk_reward(X) for _ in range(self.n_iter)]
        else:
            # otherwise use joblib Parallel, it takes some overhead for initialization but on large portfolios
            # it can run much faster
            out = Parallel(n_jobs=self.n_jobs)(
                delayed(_compute_weights_risk_reward)(X) for _ in range(self.n_iter)
            )
        # filter unfitted results
        out = [o for o in out if not np.isnan(o[2])]
        self.all_weights_ = [o[0] for o in out]
        self.risk_rewards_ = [(o[1], o[2]) for o in out]
        # finally collects all the obtained weights and average by mean
        self.weights_ = pd.concat(self.all_weights_, axis=1).aggregate(
            self.agg_func, axis=1
        )
        self.weights_ = (self.weights_ / self.weights_.sum()).rename(
            f"{self.__class__.__name__}[{self.ptf_estimator.__class__.__name__}]"
        )
        return self


class SubsetResampling(PortfolioEstimator):
    """
    A stochastic method that randomly chooses a subset of assets from the universe building
    a new portfolio by averaging over many scenarios.
    https://blog.thinknewfound.com/2018/07/machine-learning-subset-resampling-and-portfolio-optimization/
    """

    def __init__(
        self,
        ptf_estimator: PortfolioEstimator,
        subset_size: float = 0.8,
        n_iter: int = 100,
        n_jobs: int = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        agg_func: Optional[
            Union[
                str,
                Callable[
                    [Union[pd.DataFrame, np.ndarray]], Union[pd.Series, np.ndarray]
                ],
            ]
        ] = "mean",
    ):
        """
        Initializes the SubsetResampling object. We explicitly require to define the ptf_estimator
        Parameters
        ----------
        ptf_estimator: PortfolioEstimator
            Original noise-sensitive portfolio estimator object, for example MaxSharpe or MaxOmegaRatio
        n_iter:
            Number of random subset of portfolios to average over
        n_jobs: int
            Number of parallel jobs (uses joblib backend)
        random_state: int
            Integer representing the seed of the random state
        agg_func: str or Callable
            How to aggregate all weights. By default uses an average over the ensemble of all weights. Other aggregation
            functions can be specified such as `np.median`.
        """

        super().__init__()
        self.ptf_estimator = ptf_estimator
        self.subset_size = subset_size
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.agg_func = agg_func
        self.all_weights_: List[Union[pd.Series, np.ndarray]] = []

    def fit(self, X, y=None, **fit_kwargs) -> PortfolioEstimator:
        num_assets = X.shape[1]
        n_subsets = (
            int(np.ceil(X.shape[1] ** self.subset_size))
            if isinstance(self.subset_size, float)
            else self.subset_size
        )

        def _sample_fit(_x):
            """
            Here we select a subsample of asssets using the pd.DataFrame.sample with axis=1, to be fed to the portfolio
            estimator
            Parameters
            ----------
            _x: pd.DataFrame
                The prices or returns data to be fed to portfolio estimator
            Returns
            -------
                A pd.Series with the portfolio weights, reindexed with the entire universe of assets as from `X`.
            """
            try:
                return (
                    self.ptf_estimator.fit(
                        X.sample(n=n_subsets, axis=1, random_state=self.random_state)
                    )
                    .weights_.reindex_like(X.T)
                    .fillna(0)
                )
            except (AttributeError, OptimizationError) as portfolio_error:
                return pd.Series(data=[np.nan] * num_assets, index=X.columns)

        # run joblib parallel only if n_jobs is not 1, to avoid useless overhead or dependencies
        if self.n_jobs != 1:
            self.all_weights_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_sample_fit)(X) for _ in range(self.n_iter)
            )
        else:
            self.all_weights_ = [_sample_fit(X) for _ in range(self.n_iter)]
        self.all_weights_ = [w for w in self.all_weights_ if w.notna().all()]
        self.weights_ = pd.concat(self.all_weights_, axis=1).agg(self.agg_func, axis=1)
        self.weights_ = (
            (self.weights_ / self.weights_.sum())
            .rename(
                f"{self.__class__.__name__}[{self.ptf_estimator.__class__.__name__}]"
            )
            .sort_index()
        ).loc[
            X.columns
        ]  # this last stpe is needed to have the same order of columns as from original input data
        return self


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
    def __init__(
        self,
        window: int = 120,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        risk_estimator: BaseRiskEstimator = SampleCovariance(),
        n_portfolios: int = 10,
        robustness_param_loc: float = 0.1,
        robustness_param_scatter: float = 0.1,
        n_jobs: int = -1,
    ):
        """
        Implementation of the Bayesian Robust Allocation algorithm by Attilio Meucci.

        The implementation is based on the Matlab code provided by Meucci, with a number of computational tricks.
        It rolls over the prices dataframe based on the window parameter. It finally averages all the weights
        of the robust bayesian portfolios into a .weights_ attribute.
        All the portfolio weights are stored in the .all_weights_ attribute that can be accessed.

        Parameters
        ----------
        window: int
            The number of rows to roll over
        rets_estimator: BaseReturnsEstimator
            The expected returns estimator, default value MeanHistoricalLinearReturns
        risk_estimator: BaseRiskEstimator
            The risk estimator, default value SampleCovariance
        n_portfolios: int
            Number of portfolios to average over
        robustness_param_loc: float
            The level of confidence in the estimation risk for posterior averages.
            The higher the values, the better for less experienced investors, whereas lower values are more suitable for
            very self-confident investors with clear views in mind.

        robustness_param_scatter: float
            Denotes the confidence of investor to estimation risk in posterior covariance.
            The higher the values, the better for less experienced investors, whereas lower values are more suitable for
            very self-confident investors with clear views in mind.
        n_jobs: int
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
        self.all_weights_: List[Union[pd.Series]] = []
        self.n_jobs = n_jobs

    def _fit(self, rolling_prices: pd.DataFrame):
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
        S1 = (
            1
            / nu1
            * (
                S_hat * T
                + S0 * nu0
                + np.outer(mu_hat - m0, mu_hat - m0) / (1 / T + 1 / T0)
            )
        )
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

    def fit(self, X: pd.DataFrame, y=None, **fit_kwargs) -> PortfolioEstimator:
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

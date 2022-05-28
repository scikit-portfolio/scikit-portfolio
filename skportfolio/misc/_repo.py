"""
Information theory inspired portfolio optimization
"""
from abc import ABCMeta
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from pypfopt.expected_returns import returns_from_prices
from scipy.optimize import minimize
from scipy.stats import entropy as spentropy
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.bandwidths import select_bandwidth

from skportfolio._base import PortfolioEstimator
from skportfolio.riskreturn import BaseReturnsEstimator
from skportfolio.riskreturn import MeanHistoricalLinearReturns
from skportfolio.riskreturn import all_returns_estimators


def repo(
    prices: Union[np.ndarray, pd.DataFrame],
    rets_estimator: BaseReturnsEstimator,
    kernel: str = "gaussian",
    bandwidth: Union[float, str] = "silverman",
    space_size: int = 100,
    alpha: float = 0.0,
    n_iter: int = 1,
    weight_bounds: Tuple[float, float] = (0.0, 1.0),
    returns_data: bool = False,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    n_jobs: int = 1,
):
    """
    This method is based on the estimation of Shannon entropy of the distribution of portfolio returns as a measure
     of volatility.
    Entropy is estimated through KDE method with specified kernel, bandwidth and other parameters.
    Typically the best parameters are found through cross-validation and hyperparameters optimization.

    Parameters
    ----------
    prices: pd.DataFrame
        Asset prices with index representing time as timestamp or date
    rets_estimator: BaseReturnsEstimator
        Returns estimator, estimator of expected returns from data
    kernel: str
        The KDE kernel to be used. Now only the following kernels are supported:
        ‘gaussian’, ‘tophat’, ‘epanechnikov’, ‘exponential’, ‘linear’, ‘cosine’
    bandwidth: float
        The bandwidth of the kernel.
        Should be chosen via hyperparameters optimization
    space_size: int
        The number of bins used to discretize the portfolio returns. 100 bins are typically good enough for daily data.
        However, it is better chosen via hyper-parameters optimization.
    alpha: float
        Multiobjective scalar. The higher, the more impact is given to high portfolio return instead of minimum volatility.
        Default 0, typically chosen via hyper-parameters optimization.
    n_iter:
        Number of random inizializations. Default 1, otherwise keeps the best solution.
    weight_bounds:
        Minimum and maximum weight to set to constrain the solution
    returns_data:
        Specify whether input prices are really prices or returns
    random_state: Union[int, np.random.RandomState]
        Specify the random number generator seed as integer to be passed to RandomState or directly a RandomState object
    n_jobs: int
        Number of parallel jobs, if 1 default serial calculation is done.

    Returns
    -------
    The portfolio weights after returns entropy optimization
    """

    tickers = prices.columns
    if not returns_data:
        returns = returns_from_prices(prices)
    else:
        returns = prices
    n_assets = returns.shape[1]
    ret_range = np.linspace(returns.min().min(), returns.max().max(), int(space_size))

    opts = {
        "ftol": 1e-5,
        "maxiter": 500,
        "eps": 1e-8,  # as default of LBFGSB
        "disp": False,
        "iprint": False,
    }

    mean_ret = rets_estimator.set_returns_data(returns_data=returns_data).fit_transform(
        prices
    )

    def objective(_w: np.ndarray):
        ret = returns.dot(_w)
        bw = bandwidth
        if isinstance(bandwidth, str) and (
            bandwidth in ("silverman", "scott", "normal_reference")
        ):
            bw = select_bandwidth(ret, bw, None)
        kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(ret.values[:, None])
        Rp = np.exp(
            kde.score_samples(ret_range[:, None])
        )  # take exponential of log-probability density
        return spentropy(np.squeeze(Rp)) - alpha * mean_ret.dot(_w)

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}
    best_sol, best_fval = None, np.inf
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    for u in range(n_iter):
        sol = minimize(
            objective,
            x0=random_state.random(n_assets),
            method="SLSQP",
            constraints=constraints,
            bounds=[(weight_bounds[0], weight_bounds[1])] * n_assets,
            options=opts,
        )
        if sol["fun"] < best_fval:
            best_sol = sol["x"]

    return pd.Series(dict(zip(tickers, best_sol)), name="repo")


"""
Portfolio estimator implementing the **REPO** method.
"""


class REPO(PortfolioEstimator, metaclass=ABCMeta):
    """
    Portfolio estimator implementing the **REPO** method.
    """

    def __init__(
        self,
        returns_data: bool = False,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        kernel: str = "gaussian",
        bandwidth: Union[float, str] = "silverman",
        space_size: int = 100,
        alpha: float = 0.0,
        n_iter: int = 1,
        random_state=None,
        l2_gamma=0,
        min_weight=0,
        max_weight=1,
    ):
        super(REPO, self).__init__()
        self.returns_data = returns_data
        self.rets_estimator = rets_estimator
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.space_size = space_size
        self.alpha = alpha
        self.n_iter = n_iter
        self.random_state = random_state

        self.l2_gamma = l2_gamma
        self.min_weight = min_weight
        self.max_weight = max_weight

    def fit(self, X, y=None):
        self.weights_ = repo(
            X,
            rets_estimator=self.rets_estimator,
            kernel=self.kernel,
            bandwidth=self.bandwidth,
            space_size=self.space_size,
            alpha=self.alpha,
            n_iter=self.n_iter,
            weight_bounds=(self.min_weight, self.max_weight),
            returns_data=self.returns_data,
            random_state=self.random_state,
        )
        return self

    def grid_parameters(self):
        return dict(
            alpha=np.logspace(-3, 1, 10),
            bandwidth=np.logspace(1, 4, 20),
            kernel=[
                "gaussian",
                "tophat",
                "epanechnikov",
                "exponential",
                "linear",
                "cosine",
            ],
            space_size=np.round(np.logspace(1, 4, 50)),
            rets_estimator=all_returns_estimators,
        )
"""
Ensemble portfolio estimators
-----------------------------

Here we collect baggind-based and bootstrap based portfolio meta-estimators.

A "bagging" portfolio meta-estimator fits base portfolio estimator on random subsets of the
original dataset, then aggregating the individual weights (either by voting or by averaging) to
form a final prediction.

A "bootstrap" based portfolio estimator instead is based on the idea to reduce the extreme
sensitivity of optimized weights from the returns estimator, by aggregating multiple results
over randomly perturbed returns through resampling from a multivariate normal given the sample
covariance matrix and returns estimates.
"""
from itertools import takewhile
import sys
import warnings
from typing import Callable, List, Optional, Union

import cvxpy as cp
import numpy as np
import pandas as pd
from numpy.random import Generator

from skportfolio._base import PortfolioEstimator
from skportfolio._simple import EquallyWeighted
from skportfolio.riskreturn import (
    BaseReturnsEstimator,
    BaseRiskEstimator,
    MeanHistoricalLinearReturns,
    PerturbedReturns,
    SampleCovariance,
)


class MichaudResampledFrontier(PortfolioEstimator):
    """
    Generic class to resample efficient frontier estimators.
    The idea, similar to bootstrapping, is based on the idea of Resampled Efficient Frontier
    by Michaud (1998). Here we run n_iter independent fit of a specified portfolio estimator
    with a stochastic perturbation of the expected returns based on multivariate normal
    sampling of returns given sample expected returns and covariance.
    """

    def __init__(
        self,
        estimator: PortfolioEstimator = EquallyWeighted(),
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        risk_estimator: BaseRiskEstimator = SampleCovariance(),
        n_iter: Optional[int] = 100,
        random_state: Optional[Union[int, Generator]] = None,
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
        Initializes the ResampledEfficientFrontier object. Explicitly required to define the ptf_estimator.

        Parameters
        ----------
        estimator: PortfolioEstimator
            The user provided estimator for the base optimal portfolio allocation. Default value is `MinimumVolatility`.
        rets_estimator: BaseReturnsEstimator
            The expected returns' estimator. Default value is the `MeanHistoricalValue` estimator.
        risk_estimator: BaseRiskEstimator
            The risk matrix estimator. Default value is the sample covariance we use `SampleCovariance` estimator.
        n_iter: int
            Number of Monte Carlo resampling steps
        random_state: Optional[Union[int,Generator]]
            The random number generator for reproducibility purpose.
            Set it to a specific number only if you need always the same expected returns.
            Not the case in most situations, though it may be useful for debugging.
        agg_func: Callable, str
            Aggregation method of weights. Default mean over all weights.
        """
        super().__init__()
        self.estimator = estimator
        self.n_iter = n_iter
        self.random_state = random_state
        self.rets_estimator = rets_estimator
        self.risk_estimator = risk_estimator
        self.agg_func = agg_func

    def fit(self, X, y=None, **fit_params) -> PortfolioEstimator:
        # modifies the returns' estimator of the provided portfolio estimator
        # initialize the generator to avoid large memory allocation
        perturbed_estimators = (
            PerturbedReturns(
                rets_estimator=self.rets_estimator,
                risk_estimator=self.risk_estimator,
                random_state=r if self.random_state is None else self.random_state,
            )
            for r in np.random.random_integers(
                low=0, high=sys.maxsize, size=self.n_iter
            )
        )

        # also here only use generators for faster and memory friendly evaluation
        all_weights_ = (
            self.estimator.set_returns_estimator(ptb).fit(X, y).weights_
            for ptb in perturbed_estimators
        )

        # finally collects all the obtained weights and average by mean
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            name = f"{self.__class__.__name__}[{self.estimator.__class__.__name__}]"
            self.weights_ = (
                pd.concat((w for w in all_weights_ if w.notna().all()), axis=1)
                .aggregate(self.agg_func, axis=1)
                .rename(name)
            )
        return self


class SubsetResampling(PortfolioEstimator):
    """
    A stochastic method that randomly chooses a subset of assets from the universe building
    a new portfolio by averaging over many scenarios.
    https://blog.thinknewfound.com/2018/07/machine-learning-subset-resampling-and-portfolio-optimization/
    Also based on this paper:
    https://www.bengillen.com/uploads/1/2/3/8/123891022/subsets.pdf
    """

    def __init__(
        self,
        estimator: PortfolioEstimator = EquallyWeighted(),
        subset_size: float = 0.8,
        n_iter: int = 100,
        random_state: Optional[Union[int, np.random.Generator]] = None,
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
        estimator: PortfolioEstimator
            Original noise-sensitive portfolio estimator object, for example MaxSharpe or MaxOmegaRatio
        n_iter:
            Number of random subset of portfolios to average over
        random_state: int
            Integer representing the seed of the random state
        agg_func: str or Callable
            How to aggregate all weights. By default uses an average over the ensemble of all weights. Other aggregation
            functions can be specified such as `np.median`.
        """

        super().__init__()
        self.estimator = estimator
        self.subset_size = subset_size
        self.n_iter = n_iter
        self.random_state = random_state
        self.agg_func = agg_func
        self.all_weights_ = []

    def fit(self, X, y=None, **fit_params) -> PortfolioEstimator:
        if self.random_state is None:
            generator = np.random.default_rng()
        else:
            generator = np.random.default_rng(seed=self.random_state)

        num_assets = X.shape[1]
        n_subsets = (
            int(np.ceil(num_assets**self.subset_size))
            if isinstance(self.subset_size, float)
            else self.subset_size
        )

        # also here we use generators for more pythonic
        # and less memory hungry processing
        all_weights = (
            self.estimator.fit(
                X.sample(n=n_subsets, axis=1, random_state=generator)
            ).weights_
            for _ in range(self.n_iter)
        )

        name = f"{self.__class__.__name__}[{self.estimator.__class__.__name__}]"
        self.weights_ = (
            pd.concat(takewhile(lambda x: x.notna().all(), all_weights), axis=1)
            .aggregate(self.agg_func, axis=1)
            .reindex(X.columns, fill_value=0.0)
            .rename(name)
        )
        # enforce renormalization
        self.weights_ /= self.weights_.sum()

        return self

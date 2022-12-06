"""
Contains definitions of expected returns estimators
"""
from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Union

import cvxpy as cp
import numpy as np
import pandas as pd
from pypfopt import expected_returns as expret
from sklearn.base import BaseEstimator, TransformerMixin

from skportfolio.riskreturn.expected_returns import (
    capm_return,
    ema_historical_return,
    mean_historical_log_return,
    mean_historical_return,
    median_historical_log_return,
    median_historical_return,
    rolling_median_returns,
)


class BaseReturnsEstimator(TransformerMixin, BaseEstimator, metaclass=ABCMeta):
    """
    Base class for return estimator.
    It provides the basic infrastructure for all estimators of expected returns.
    It either can take price data or returns data, by specifying the `returns_data=False` or `True` respectively.
    The returns calculation is done with the standard `.pct_change()` of Pandas, unless otherwise specified by
    specification of the `returns_function` callable.
    """

    def __init__(
        self,
        returns_data: bool = False,
        returns_function: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ):
        self.returns_data = returns_data
        self.expected_returns_ = None
        self.returns_function = returns_function
        self.random_state = None

    def set_returns_data(self, returns_data):
        self.returns_data = returns_data
        return self

    @abstractmethod
    def _set_expected_returns(self, X, y=None, **fit_params):
        pass

    def fit(self, X, y=None, **fit_params):
        """
        Base method for fitting a returns estimator
        Parameters
        ----------
        X
        y
        fit_params

        Returns
        -------

        """
        self._set_expected_returns(X, y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """
        Base fit_transform method for returns estimators
        Parameters
        ----------
        X
        y
        fit_params

        Returns
        -------

        """
        self.fit(X, y)
        return self.expected_returns_

    # for use with stochastic based returns calculators
    def reseed(self, seed: Optional[Union[int, np.random.Generator]]):
        """
        Change the seed on the current instance, so that new calls to random
        numbers produce a different result
        Parameters
        ----------
        seed: int or numoy Generator

        Returns
        -------
        self
        """
        if self.random_state is None:
            self.random_state = np.random.default_rng()
        if isinstance(seed, np.random.Generator):
            self.random_state = seed
        elif isinstance(seed, int):
            self.random_state = np.random.default_rng(seed)
        return self


class MeanHistoricalLinearReturns(BaseReturnsEstimator):
    def _set_expected_returns(self, X, y=None, **fit_params):
        self.expected_returns_ = mean_historical_return(X, self.returns_data)


class MeanHistoricalLogReturns(BaseReturnsEstimator):
    def _set_expected_returns(self, X, y=None, **fit_params):
        self.expected_returns_ = mean_historical_log_return(
            X,
            self.returns_data,
        )


class CompoundedHistoricalLinearReturns(BaseReturnsEstimator):
    def _set_expected_returns(self, X, y=None, **fit_params):
        self.expected_returns_ = expret.mean_historical_return(
            X, self.returns_data, compounding=True
        )


class CompoundedHistoricalLogReturns(BaseReturnsEstimator):
    def _set_expected_returns(self, X, y=None, **fit_params):
        self.expected_returns_ = expret.mean_historical_return(
            X,
            self.returns_data,
            compounding=True,
            log_returns=True,
        )


class MedianHistoricalLinearReturns(BaseReturnsEstimator):
    def _set_expected_returns(self, X, y=None, **fit_params):
        self.expected_returns_ = median_historical_return(X, self.returns_data)


class MedianHistoricalLogReturns(BaseReturnsEstimator):
    def _set_expected_returns(self, X, y=None, **fit_params):
        self.expected_returns_ = median_historical_log_return(X, self.returns_data)


class EMAHistoricalReturns(BaseReturnsEstimator):
    def __init__(self, returns_data=False, span=60):
        super().__init__(returns_data)
        self.span = span

    def _set_expected_returns(self, X, y=None, **fit_params):
        self.expected_returns_ = ema_historical_return(X, self.returns_data, self.span)


class CAPMReturns(BaseReturnsEstimator):
    def __init__(
        self,
        returns_data=False,
        risk_free_rate=0.0,
        benchmark=None,
    ):
        super().__init__(returns_data)
        self.risk_free_rate = risk_free_rate
        self.benchmark = benchmark

    def _set_expected_returns(self, X, y=None, **fit_params):
        self.expected_returns_ = capm_return(
            prices_or_returns=X,
            benchmark=y,
            returns_data=self.returns_data,
            risk_free_rate=self.risk_free_rate,
        )


class RollingMedianReturns(BaseReturnsEstimator):
    def __init__(self, returns_data=False, window=20):
        super().__init__(returns_data=returns_data)
        self.window = window

    def _set_expected_returns(self, X, y=None, **fit_params):
        self.expected_returns_ = rolling_median_returns(
            X, self.returns_data, self.window
        )


class MarketImpliedReturns(BaseReturnsEstimator):
    def _set_expected_returns(self, X, y=None, **fit_params):
        """
        This implementation
        https://it.mathworks.com/help/finance/black-litterman-portfolio-optimization.html
        Parameters
        ----------
        **fit_params
        X: prices or returns
        y: must be benchmark returns

        Returns
        -------
        """
        if y is None:
            raise ValueError("Must provide y as the benchmark portfolio returns")
        if not self.returns_data:
            X = X.pct_change().dropna()
            y = y.pct_change().dropna()
        sigma = X.cov()

        n = X.shape[1]
        w = cp.Variable(shape=(n,), pos=True)
        expr = 0.5 * cp.sum_squares(X.values @ w - y.values)
        problem = cp.Problem(
            objective=cp.Minimize(expr=expr), constraints=[cp.sum(w) == 1]
        )
        problem.solve()
        sharpe_benchmark = y.mean() / y.std()
        delta = sharpe_benchmark / np.sqrt(w.value.T @ sigma @ w.value)
        self.pi = delta * sigma @ w.value
        self.expected_returns_ = self.pi


all_returns_estimators = [
    MeanHistoricalLinearReturns(),
    MeanHistoricalLogReturns(),
    CompoundedHistoricalLinearReturns(),
    CompoundedHistoricalLogReturns(),
    MedianHistoricalLinearReturns(),
    MedianHistoricalLogReturns(),
    EMAHistoricalReturns(),
    CAPMReturns(),
    RollingMedianReturns(),
    MarketImpliedReturns(),
]

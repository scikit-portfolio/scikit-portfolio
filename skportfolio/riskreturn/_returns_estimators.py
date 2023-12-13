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
    """
    Mean historical returns are simply computed as the historical **average** of the geometric
    returns over all prices.
    """

    def _set_expected_returns(self, X, y=None, **fit_params):
        self.expected_returns_ = mean_historical_return(
            prices_or_returns=X, returns_data=self.returns_data
        )


class MeanHistoricalLogReturns(BaseReturnsEstimator):
    """
    Rather than using linear returns, we compute the average of the log returns
    """

    def _set_expected_returns(self, X, y=None, **fit_params):
        self.expected_returns_ = mean_historical_log_return(
            prices_or_returns=X, returns_data=self.returns_data, frequency=1
        )


class CompoundedHistoricalLinearReturns(BaseReturnsEstimator):
    """
    Compounded historical returns are simply computed as the geometric
    average of the linear historical returns. In other words, given the returns time series
    """

    def _set_expected_returns(self, X, y=None, **fit_params):
        self.expected_returns_ = expret.mean_historical_return(
            prices=X, returns_data=self.returns_data, compounding=True, frequency=1
        )


class CompoundedHistoricalLogReturns(BaseReturnsEstimator):
    """
    Rather than using linear returns, we compute the geometric average of the log returns.
    """

    def _set_expected_returns(self, X, y=None, **fit_params):
        self.expected_returns_ = expret.mean_historical_return(
            prices=X,
            returns_data=self.returns_data,
            frequency=1,
            compounding=True,
            log_returns=True,
        )


class MedianHistoricalLinearReturns(BaseReturnsEstimator):
    """
    Like for `MeanHistoricalLinearReturns`, but using median rather than average.
    """

    def _set_expected_returns(self, X, y=None, **fit_params):
        self.expected_returns_ = median_historical_return(
            prices_or_returns=X, returns_data=self.returns_data, frequency=1
        )


class MedianHistoricalLogReturns(BaseReturnsEstimator):
    """
    Like for `MeanHistoricalLogReturns`, but using **median** rather than average.
    """

    def _set_expected_returns(self, X, y=None, **fit_params):
        self.expected_returns_ = median_historical_log_return(
            prices_or_returns=X, returns_data=self.returns_data, frequency=1
        )


class EMAHistoricalReturns(BaseReturnsEstimator):
    """
    Estimates the (annualized if frequency=252) expected returns as the exponential moving
    average of linear historical returns. Compounding is set to false by default.
    Span set default to 60 rows.
    """

    def __init__(self, returns_data=False, compounding=False, span=60):
        super().__init__(returns_data)
        self.span = span
        self.compounding = compounding

    def _set_expected_returns(self, X, y=None, **fit_params):
        self.expected_returns_ = ema_historical_return(
            prices_or_returns=X,
            returns_data=self.returns_data,
            frequency=1,
            span=self.span,
            compounding=self.compounding,
        )


class CAPMReturns(BaseReturnsEstimator):
    """
    Compute a return estimate using the Capital Asset Pricing Model.
    Under the CAPM, asset returns are equal to market returns plus a $\beta$ term encoding the
    relative risk of the asset.
    """

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
            returns_data=self.returns_data,
            frequency=1,
            risk_free_rate=self.risk_free_rate,
            benchmark=y,
        )


class RollingMedianReturns(BaseReturnsEstimator):
    """
    Estimates the returns from the average of the rolling median over a `window` of
    20 observations, by default.
    """

    def __init__(self, returns_data=False, window=20):
        super().__init__(returns_data=returns_data)
        self.window = window

    def _set_expected_returns(self, X, y=None, **fit_params):
        self.expected_returns_ = rolling_median_returns(
            prices_or_returns=X,
            returns_data=self.returns_data,
            frequency=1,
            window=self.window,
        )


class MarketImpliedReturns(BaseReturnsEstimator):
    """
    Market implied returns, estimate of the expected returns based on a benchmark
    """

    def _set_expected_returns(self, X, y=None, **fit_params):
        """
        Based on this implementation:

        https://it.mathworks.com/help/finance/black-litterman-portfolio-optimization.html

        Parameters
        ----------
        X: pd.DataFrame
            prices or returns
        y: pd.Series
            Must be benchmark returns

        Other Parameters
        **fit_params

        Returns
        -------
        """
        if y is None:
            raise ValueError("Must provide y as the benchmark portfolio returns")
        if not self.returns_data:
            X = X.pct_change().dropna()
            y = y.pct_change().dropna()
        sigma = X.cov()

        n_assets = X.shape[1]
        weights = cp.Variable(shape=(n_assets,), pos=True)
        expr = 0.5 * cp.sum_squares(X.values @ weights - y.values)
        problem = cp.Problem(
            objective=cp.Minimize(expr=expr), constraints=[cp.sum(weights) == 1]
        )
        problem.solve()
        sharpe_benchmark = y.mean() / y.std()
        delta = sharpe_benchmark / np.sqrt(weights.value.T @ sigma @ weights.value)
        self.pi_ = delta * sigma @ weights.value
        self.expected_returns_ = self.pi_


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

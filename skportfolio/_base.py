from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod

import numpy as np
import pandas as pd
from pypfopt.expected_returns import returns_from_prices
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from skportfolio.metrics import sharpe_ratio
from skportfolio.riskreturn import BaseReturnsEstimator
from skportfolio.riskreturn import BaseRiskEstimator


def assert_is_prices(X: pd.DataFrame):
    if (X < 0).any().any():
        raise ValueError(
            "You are likely feeding returns and not prices. Positive reals are expected in the predict method."
        )


class PortfolioEstimator(
    BaseEstimator,
    metaclass=ABCMeta,
):
    """
    The base class of all portfolio estimators.
    It defines the predict method, which is common for all
    """

    @abstractmethod
    def __init__(self, returns_data: bool = False):
        self.returns_data = returns_data

    @abstractmethod
    def fit(self, X, y=None, **fit_params) -> PortfolioEstimator:
        """
        Fit the portfolio model from asset prices.

        Parameters
        ----------
        X: pd.DataFrame
            Asset prices
        y: None
            There for compatibility
        Returns
        -------
        self
            The current estimator
        """

    def predict(self, X) -> pd.Series:
        """
        Applies the estimated weights to the prices to get the portfolio value.
        In other words, generates the equity curve of the portfolio.

        Parameters
        ----------
        X: pd.DataFrame
            The prices expressed as a pandas dataframe

        Returns
        -------
        pd.Series
            The estimated portfolio value time series
        """
        check_is_fitted(self)
        assert_is_prices(X)
        return X.dot(self.weights_).rename(str(self))

    def score(self, X, y=None, **kwargs):
        """
        Score the portfolio using one of the metrics expressed as PortfolioScorer
        in the `metrics` module.
        Parameters
        ----------
        X: pd.DataFrame
            Dataframe with asset prices. Do not feed returns here!
        y: ignored

        Other Parameters
        -----------------
        Additional parameters to specify to the portfolio returns scorer,
        like risk_free_rate for Sharpe ratio.
        Alternatively you can also specify the score function to be used,
        like sharpe_ratio, or sortino_ratio.
        Returns
        -------
        float
            The specific score value from the specified, Sharpe ratio is returned if not specified.
        """
        if "score_function" not in kwargs:
            score_function = sharpe_ratio
        else:
            score_function = kwargs.pop("score_function")
        return score_function(returns_from_prices(self.predict(X)), **kwargs)

    def set_dummy_weights(self, X):
        """
        Creates weights with NaN values, typically used when method cannot converge
        or solution is unfeasible.
        Parameters
        ----------
        X: pd.DataFrame
            Prices or returns data, only used to pick asset names
        Returns
        -------
            A series of assets with NaN weights.
        """
        self.weights_ = pd.Series(data=np.nan, index=X.columns, name=str(self))
        return self

    def set_returns_data(self, returns_data=False):
        """
        Inform subsequent .fit method that the input data X are returns,
        otherwise prices are expected.
        Some portfolio estimators work independently with both, but is very
        important to specify what is contained
        in the `.fit` argument `X`.
        Parameters
        ----------
        returns_data

        Returns
        -------

        """
        self.returns_data = returns_data
        return self

    def set_returns_estimator(self, rets_estimator: BaseReturnsEstimator):
        """
        Modify the base returns estimator with a specified rets_est

        Parameters
        ----------
        rets_estimator: BaseReturnsEstimator
            The new returns estimator
        Returns
        -------
        self
        """
        if not isinstance(rets_estimator, BaseReturnsEstimator):
            raise TypeError("Must set a base returns estimator")
        self.rets_estimator = rets_estimator
        return self

    def set_risk_estimator(self, risk_estimator: BaseRiskEstimator):
        """
        Modify the base risk estimator with a new risk estimator

        Parameters
        ----------
        risk_estimator: BaseRiskEstimator
            The new risk estimator
        Returns
        -------
        self
        """
        if not isinstance(risk_estimator, BaseRiskEstimator):
            raise TypeError("Must set a base risk estimator")
        self.risk_estimator = risk_estimator
        return self

    def info(self):
        out = f"Model name: {self.__class__.__name__}\n{pd.Series(self.get_params())}\n"
        out += "\nWeights"
        if self.weights_ is not None:
            out += f"\n{self.weights_[self.weights_>0].round(4).to_string()}"
        out += "\n"
        return out

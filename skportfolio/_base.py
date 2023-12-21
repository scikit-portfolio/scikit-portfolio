"""
Base modulo defining the portfolio estimator base class used throughout the entire library and
derived from the scikit-learn BaseEstimator
"""

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
        raise ValueError("You are likely feeding returns and not prices.")


class PortfolioEstimator(
    BaseEstimator,
    metaclass=ABCMeta,
):
    """
    The base class of all portfolio estimators.
    It defines the predict method, which is common for all as well as the .fit and the .score
    methods which are necessary to all derived classes.
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

    def predict_proba(self, X):
        """
        Predicts the asset positions and equity curve based on historical price data and predefined
        weights.
        Here the initial portfolio value is **always** hypothesized to be $1.
        The asset positions are expressed in $, as well as the equity curve.

        Parameters
        ----------
        X : pd.DataFrame
            Historical price data with columns representing different assets and rows as timestamps.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the calculated equity curve with the same structure as input 'X'.
        """
        check_is_fitted(self)
        assert_is_prices(X)

        # Convert to NumPy array for faster computation
        prices_np: np.ndarray = X.values
        # Calculate returns and fill NaN values with 0.0
        returns_np: np.ndarray = X.pct_change().dropna().values
        weights_np: np.ndarray = np.array(self.weights_)

        # Calculate position in currency units (dollars or euro for example) for the initial day
        # using dot product. This value will be normalized at the end in order to start from 1
        # unit of currency
        position_0 = prices_np[0, :].dot(weights_np)
        initial_value = prices_np[0, :].sum()

        # Initialize positions and equity curve arrays
        positions_np = np.zeros_like(prices_np)
        positions_np[0, :] = position_0

        # Vectorized computation of positions and equity curve
        np.cumprod(1.0 + returns_np, axis=0, out=positions_np[1:])
        positions_np[1:, :] *= position_0

        positions_np /= initial_value
        return pd.DataFrame(data=positions_np, index=X.index, columns=X.columns)

    def predict(self, X):
        return self.predict_proba(X).sum(1)

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
        returns_data:
            True if next calls to fit are providing returns rather than prices, False otherwise

        Returns
        -------
        The current PortfolioEstimator instance
        """
        self.returns_data = returns_data
        return self

    def set_returns_estimator(self, rets_estimator: BaseReturnsEstimator):
        """
        Replace the current returns estimator with another one

        Parameters
        ----------
        rets_estimator: BaseReturnsEstimator
            The new returns estimator
        Returns
        -------
        The current PortfolioEstimator instance
        """
        if not isinstance(rets_estimator, BaseReturnsEstimator):
            raise TypeError("Must set a base returns estimator")
        self.rets_estimator = rets_estimator
        return self

    def set_risk_estimator(self, risk_estimator: BaseRiskEstimator):
        """
        Replace the current risk estimator with another one

        Parameters
        ----------
        risk_estimator: BaseRiskEstimator
            The new risk estimator
        Returns
        -------
        The current PortfolioEstimator instance
        """
        if not isinstance(risk_estimator, BaseRiskEstimator):
            raise TypeError("Must set a base risk estimator")
        self.risk_estimator = risk_estimator
        return self

    def info(self) -> str:
        """
        Prints some information about the current PortfolioEstimator instance
        Returns
        -------
        str
        """
        out = f"Model name: {self.__class__.__name__}\n{pd.Series(self.get_params())}\n"
        out += "\nWeights"
        if self.weights_ is not None:
            out += f"\n{self.weights_[self.weights_>0].round(4).to_string()}"
        out += "\n"
        return out

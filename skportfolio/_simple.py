"""
Implementation of basic portfolio estimators, such as single asset full allocation, 
equally weighted allocation and other methods not based on convex optimization.
"""
from abc import ABCMeta
from typing import Sequence
import numpy as np
import pandas as pd
from pypfopt.expected_returns import returns_from_prices
from skportfolio._base import PortfolioEstimator


def single_asset_weights(tickers: Sequence[str], asset: str) -> pd.Series:
    """
    Creates a series of portfolio weights where asset is assigned 100% of the weight.
    Default value of zero is assigned to the rest
    Parameters
    ----------
    tickers: names of assets
    asset: asset to set the weight to 1

    Returns
    -------
    A pd.Series with the weights of a portfolio containing a single asset

    Examples:
    >> single_asset_weight(["BTC/USDT", "ETH/USDT", "LTC/USDT"], "BTC/USDT")
    """
    w = pd.Series(index=tickers, data=0, name=f"{asset}_only")
    w.loc[asset] = 1
    return w


class EquallyWeighted(PortfolioEstimator, metaclass=ABCMeta):
    """
    The equally weighted portfolio.
    All the asset weights are computed as 1/N where N is the total number of
    assets in the provided prices dataframe provided to the `.fit` method.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params) -> PortfolioEstimator:
        n = X.shape[1]
        self.weights_ = pd.Series(index=X.columns, data=1 / n, name=str(self))
        return self


class InverseVariance(PortfolioEstimator, metaclass=ABCMeta):
    """
    The inverse variance portfolio.
    """

    def __init__(self, returns_data: bool = False):
        self.returns_data = returns_data

    def fit(self, X, y=None, **fit_params) -> PortfolioEstimator:
        if self.returns_data:
            inverse_covariance = 1.0 / X.cov()
        else:
            inverse_covariance = 1.0 / returns_from_prices(X).cov()
        self.weights_ = pd.Series(
            index=X.columns,
            data=np.diag(inverse_covariance) / (np.diag(inverse_covariance).sum()),
            name=str(self),
        )
        return self


class CapWeighted(PortfolioEstimator, metaclass=ABCMeta):
    """
    A portfolio with weights proportional to total ticker capitalization.
    Requires additional data y in the fit method in the form of a Pandas dataframe
    with market capitalization.
    """

    def __init__(self, returns_data: bool = False):
        self.returns_data = returns_data

    def fit(self, X, y=None, **fit_params) -> PortfolioEstimator:
        if y is None:
            raise ValueError(
                "Must provide shares as a dataframe together with prices or returns"
            )
        if not isinstance(y, pd.DataFrame):
            raise ValueError("Must provide number of shares dataframe")
        if not X.index.equals(y.index):
            raise IndexError("Unequal indices between prices and number of shares")
        market_cap = (y.mul(X)).mean(axis=0)
        self.weights_: pd.Series = market_cap / market_cap.sum()
        return self


class InverseVolatility(PortfolioEstimator, metaclass=ABCMeta):
    """
    The inverse volatility portfolio.
    """

    def __init__(self, returns_data: bool = False):
        self.returns_data = returns_data

    def fit(self, X, y=None, **fit_params) -> PortfolioEstimator:
        if self.returns_data:
            inv_cov = 1.0 / np.sqrt(np.diag(X.cov()))
        else:
            inv_cov = 1.0 / np.sqrt(np.diag(returns_from_prices(X).cov()))
        self.weights_: pd.Series = pd.Series(
            index=X.columns,
            data=inv_cov / np.nansum(inv_cov),
            name=str(self),
        )
        return self


class SingleAsset(PortfolioEstimator, metaclass=ABCMeta):
    def __init__(self, asset: str):
        self.asset = asset

    def fit(self, X, y=None, **fit_params) -> PortfolioEstimator:
        if self.asset not in X.columns:
            raise ValueError("Asset must be one of the those provided.")
        self.weights_ = single_asset_weights(tickers=X.columns, asset=self.asset)
        return self

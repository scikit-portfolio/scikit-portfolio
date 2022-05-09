from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple
from typing import Union

import cvxpy as cp
import numpy as np
import pandas as pd
from pypfopt.expected_returns import returns_from_prices
from sklearn.base import BaseEstimator

from skportfolio.metrics import sharpe_ratio
from skportfolio.riskreturn import BaseReturnsEstimator
from skportfolio.riskreturn import BaseRiskEstimator


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


def redistribute_non_allocable(
    weights: pd.Series,
    total_portfolio_value: float,
    min_amount: float,
    rank_gamma: float = 0.5,
    min_weight_tolerance: float = 1e-6,
):
    """
    Redistributes the weights of non-allocable assets (those who once multiplied by the total available
    investment amount are under a `min_amount` threshold, hence tend to remain in the non-allocable
    liquidity)

    Parameters
    ----------
    :param weights: pd.Series
        Portfolio weights. Long only, sum to 1
    :param total_portfolio_value: float
        The total available amount to allocate
    :param min_amount: float
        The minimum amount that can be allocated
    :param rank_gamma: float
        A regularization term. The higher the more similar the weights to the original weights.
        If rank_gamma=0, the resulting reallocation tends to push all left allocable to the asset
        with largest weight. If rank_gamma>>1 then a large portion of the leftovers still remain
        non allocable, given the min_amount constraint.
        Default: 0.5 as from many experiments it looks like with this value, the number of violations
        is smallest with small cosine distance from the original portfolio.
    :param min_weight_tolerance: float
        The minimum weight to consider as zero
    Returns
    -------
    The weights, redistributed in a way such that the most assets which are are closest to the
    allocability amount tend to increment at the expenses of the smallest weights.
    """
    amounts = weights * total_portfolio_value
    mask = (amounts < min_amount) & (amounts > min_weight_tolerance)
    amount_min = amounts[mask].sort_values(ascending=False)  # descending order
    remaining_weights: pd.Series = weights[mask]

    if remaining_weights.empty:
        return weights

    # we should reallocate some weight to reach minimum amount
    amount_min_rank = amount_min.rank(ascending=False, method="min")
    V = amount_min_rank.values
    # number of asset left to allocate
    n_left = amount_min.shape[0]
    x = cp.Variable(shape=(n_left,), pos=True)
    # here we setup an optimization problem. Weights with low rank are preferred to be
    # incremented, being V based on the ranking
    obj = cp.sum(cp.multiply(V, x))
    if rank_gamma > 0:
        obj += n_left * rank_gamma * cp.sum_squares(remaining_weights.values - x)

    constr = [x >= 0, x <= 1, cp.sum(x) == remaining_weights.sum()]
    probl = cp.Problem(objective=cp.Minimize(obj), constraints=constr)
    probl.solve()
    # Now put back the redistributed weights
    relloacted_weights = weights.copy()
    # Put back the redistributed weights
    relloacted_weights.loc[amount_min_rank.index] = x.value
    return relloacted_weights


def discrete_allocation(
    weights: pd.Series,
    latest_prices,
    total_portfolio_value_cents: float,
    multiplier: int = 100,
):
    """
    Allocates the weights into specific amounts of stake currency,
    respecting the fact that final amounts must be integers.
    Parameters
    ----------
    :param weights: pd.Series
    :param latest_prices: pd.DataFrame
    :param total_portfolio_value_cents: float
    :param multiplier: int

    Returns
    -------
    Allocation of assets in terms of integers in stake currency
    """
    from pypfopt import DiscreteAllocation

    lda = DiscreteAllocation(
        weights.to_dict(),
        latest_prices,
        total_portfolio_value=total_portfolio_value_cents * multiplier,
    )
    lda_weights = lda.greedy_portfolio()[0]

    return (
        pd.Series(lda_weights).sort_values(ascending=False).fillna(0) / multiplier
    ) * latest_prices


def clean_weights(weights: pd.Series, cutoff=1e-4, rounding=5):
    """
    Helper method to clean the raw weights, setting any weights whose absolute
    values are below the cutoff to zero, and rounding the rest.

    :param weights: pd.Series
        Portfolio weights
    :param cutoff: the lower bound, defaults to 1e-4
    :type cutoff: float, optional
    :param rounding: number of decimal places to round the weights, defaults to 5.
                     Set to None if rounding is not desired.
    :type rounding: int, optional
    :return: asset weights
    :rtype: OrderedDict
    """
    if weights is None:
        raise AttributeError("Weights not yet computed")
    cleaned_weights = weights.copy()
    cleaned_weights[np.abs(cleaned_weights) < cutoff] = 0
    if rounding is not None:
        if not isinstance(rounding, int) or rounding < 1:
            raise ValueError("rounding must be a positive integer")
        cleaned_weights = cleaned_weights.round(rounding)

    return cleaned_weights


class PortfolioEstimator(
    BaseEstimator,
    metaclass=ABCMeta,
):
    @abstractmethod
    def fit(self, X, y=None) -> PortfolioEstimator:
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
        pass

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
        if self.weights_ is None:
            raise ValueError("Unfitted estimator. Please run .fit to compute weights.")
        if (X < 0).any().any():
            raise ValueError(
                "You are likely feeding returns and not prices. Positive reals are expected in the predict method."
            )
        return X.dot(self.weights_).rename(str(self))

    def score(self, X, y=None, **kwargs):
        """
        Score the portfolio using one of the metrics expressed as PortfolioScorer in the `metrics` module.

        Parameters
        ----------
        X: pd.DataFrame
            Dataframe with asset prices. Do not feed returns here!
        y: ignored
        **kwargs: Dict
            Additional parameters to specify to the portfolio returns scorer, like risk_free_rate for Sharpe ratio.
            Alternatively you can also specify the score function to be used, like sharpe_ratio, or sortino_ratio
        Returns
        -------
        float
            The specific score value from the specified, Sharpe ratio is returned if not specified.
        """
        if "score_function" not in kwargs:
            score_function = sharpe_ratio
        else:
            score_function = kwargs.pop("score_function")
        return score_function(returns_from_prices(X.dot(self.weights_)), **kwargs)

    def set_dummy_weights(self, X):
        """
        Creates weights with NaN values, typically used when method cannot converge or solution is unfeasible.
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
        Inform subsequent .fit method that the input data X are returns, otherwise prices are expected.
        Some portfolio estimators work independently with both, but is very important to specify what is contained
        in the `.fit` argument `X`.
        Parameters
        ----------
        returns_data

        Returns
        -------

        """
        self.returns_data = returns_data
        return self

    def set_returns_estimator(self, rets_est: BaseReturnsEstimator):
        """
        Modify the base returns estimator with a specified rets_est

        Parameters
        ----------
        rets_est: BaseReturnsEstimator
            The new returns estimator
        Returns
        -------
        self
        """
        if not isinstance(rets_est, BaseReturnsEstimator):
            raise TypeError("Must set a base returns estimator")
        self.rets_estimator = rets_est
        return self

    def set_risk_estimator(self, risk_est: BaseRiskEstimator):
        """
        Modify the base risk estimator with a new risk estimator

        Parameters
        ----------
        risk_est: BaseRiskEstimator
            The new risk estimator
        Returns
        -------
        self
        """
        if not isinstance(risk_est, BaseRiskEstimator):
            raise TypeError("Must set a base risk estimator")
        self.risk_estimator = risk_est
        return self

    def grid_parameters(self) -> Dict[str, Sequence[Any]]:
        raise NotImplementedError(
            "Must implement abstract method for each derived subclass"
        )


    def info(self, **kwargs):
        out = f"Model name: {self.__class__.__name__}\n{pd.Series(self.get_params())}\n"
        out += "\nWeights"
        if self.weights_ is not None:
            out += f"\n{self.weights_[self.weights_>0].round(4).to_string()}"
        out += "\n"
        return out


class EquallyWeighted(PortfolioEstimator, metaclass=ABCMeta):
    """
    The equally weighted portfolio.
    All the asset weights are computed as 1/N where N is the total number of
    assets in the provided prices dataframe provided to the `.fit` method.
    """

    def fit(self, X, y=None) -> PortfolioEstimator:
        n = X.shape[1]
        self.weights_ = pd.Series(index=X.columns, data=1 / n, name=str(self))
        return self


    def grid_parameters(self) -> Dict[str, Sequence[Any]]:
        return {}


class InverseVariance(PortfolioEstimator, metaclass=ABCMeta):
    """
    The inverse variance portfolio.
    """

    def fit(self, X, y=None) -> PortfolioEstimator:
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


    def grid_parameters(self) -> Dict[str, Sequence[Any]]:
        return {}


class CapWeighted(PortfolioEstimator, metaclass=ABCMeta):
    def fit(self, X, y=None) -> PortfolioEstimator:
        if y is None:
            raise ValueError(
                "Must provide shares as a dataframe together with prices or returns"
            )
        elif not isinstance(y, pd.DataFrame):
            raise ValueError("Must provide number of shares dataframe")
        elif not X.index.equals(y.index):
            raise IndexError("Unequal indices between prices and number of shares")
        mkcap = (y.mul(X)).mean(axis=0)
        self.weights_: pd.Series = mkcap / mkcap.sum()
        return self


    def grid_parameters(self) -> Dict[str, Sequence[Any]]:
        return {}


class InverseVolatility(PortfolioEstimator, metaclass=ABCMeta):
    """
    The inverse volatility portfolio.
    """

    def fit(self, X, y=None) -> PortfolioEstimator:
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


    def grid_parameters(self) -> Dict[str, Sequence[Any]]:
        return {}


class SingleAsset(PortfolioEstimator, metaclass=ABCMeta):
    def __init__(self, asset: str):
        super().__init__()
        self.asset = asset

    def fit(self, X, y=None) -> PortfolioEstimator:
        if self.asset not in X.columns:
            raise ValueError("Asset must be one of the those provided.")
        return single_asset_weights(tickers=X.columns, asset=self.asset)

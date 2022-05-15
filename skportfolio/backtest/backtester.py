# Here we should implement a backtester that takes one or more portfolio estimator objects,
# possibly a rebalancing policy, transaction costs
from typing import Union, Tuple, TypeVar, Sequence
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from skportfolio import PortfolioEstimator


def transaction_costs(
    old_portfolio: pd.Series,
    new_portfolio: pd.Series,
    buy_costs: float = 0.01,  # buying 1 share costs you 1% of that share
    sell_costs: float = 0.01,  # selling 1 share costs you 1% of that share
):
    """
    Calculates the incurred transaction costs for transfering between a portfolio
    with old_portfolios [$] and new_portfolio [$].

    Parameters
    ----------
    old_portfolio: pd.Series
    new_portfolio: pd.Series
    buy_costs: float
    sell_costs: float

    Returns
    -------
    Transaction costs
    """
    capital_allocation_difference = new_portfolio - old_portfolio
    capital_to_buy = (
        (capital_allocation_difference * (capital_allocation_difference > 0))
        .abs()
        .sum()
    )
    capital_to_sell = (
        (capital_allocation_difference * (capital_allocation_difference < 0))
        .abs()
        .sum()
    )
    return capital_to_buy * buy_costs + capital_to_sell * sell_costs


class Strategy(BaseEstimator, MetaEstimatorMixin):
    def __init__(
        self,
        initial_weights: pd.Series,
        initial_portfolio_value: float,
        estimator: PortfolioEstimator,
        rebalance_frequency: Union[int, str],
        lookback_periods: Union[int, pd.offsets.BaseOffset],
        turnover: pd.DatetimeIndex,
        transaction_costs: Union[float, Tuple[float, float]],
    ) -> None:
        self.initial_weights = initial_weights
        self.initial_portfolio_value = initial_portfolio_value
        self.estimator = estimator
        self.rebalance_frequency = rebalance_frequency
        self.lookback_periods = lookback_periods
        self.turnover = turnover
        self.transaction_costs = transaction_costs

    def fit(self, X, y, **kwargs) -> TypeVar["Strategy"]:
        return self


class Backtester(BaseEstimator):
    def __init__(self, strategy: Union[Strategy, Sequence[Strategy]]):
        self.strategy = strategy

    def fit(self, X, y=None, **kwargs):
        idx_freq = X.index.freq
        idx_freqstr = X.index.freqstr

        if idx_freqstr is None:
            raise IndexError("Please resample your data to given frequency")

        triggers = pd.date_range(start=X.index.min(), end=X.index.max(), freq="M")

        self.weights_ = pd.DataFrame(index=X.index, columns=X.columns, data=[])

        self.weights_.loc[
            (X.index.min() + pd.offsets.Day(self.strategy.lookback_periods)), :
        ] = self.strategy.initial_weights

        # base_offset = pd.offsets.BaseOffset(self.strategy.lookback_periods, freq=idx_freqstr)
        for t in X.index[self.strategy.lookback_periods :]:
            df_win = X.loc[(t - (self.strategy.lookback_periods * idx_freq)) : t, :]
            new_portfolio = self.strategy.estimator.fit(df_win).weights_
            self.weights_.loc[t, :] = new_portfolio
            trigger = np.any(df_win.index.max() == triggers)
            if trigger:
                old_portfolio = (
                    self.strategy.portfolio_values[-1]
                    * df_win.loc[df_win.index.min() - (1 * idx_freq), :]
                )
                trx_cost = transaction_costs(
                    old_portfolio=old_portfolio, new_portfolio=new_portfolio
                )
                self.strategy.portfolio_values.append(
                    self.strategy.portfolio_values[-1] - trx_cost
                )
        return self

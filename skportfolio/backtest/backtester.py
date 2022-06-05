# Here we should implement a backtester that takes one or more portfolio estimator objects,
# possibly a rebalancing policy, transaction costs
from typing import Union, Tuple, Sequence
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from skportfolio import PortfolioEstimator, equity_curve


def calculate_transaction_costs(
    old_portfolio: pd.Series,
    new_portfolio: pd.Series,
    buy_costs: float,  # buying 1 share costs you 1% of that share
    sell_costs: float,  # selling 1 share costs you 1% of that share
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
        lookback_periods: Union[
            int,
            Tuple[int, int],
            pd.offsets.BaseOffset,
            Tuple[pd.offsets.BaseOffset, pd.offsets.BaseOffset],
        ],
        transaction_costs: Union[float, Tuple[float, float]],
    ) -> None:
        """

        Parameters
        ----------
        initial_weights
        initial_portfolio_value
        estimator
        rebalance_frequency: str
        lookback_periods:
            Minimum and maximum lookback periods. If a scalar is specified the same value is set for both
        transaction_costs
        """
        self.initial_weights = initial_weights
        self.initial_portfolio_value = initial_portfolio_value
        self.estimator = estimator
        self.rebalance_frequency = rebalance_frequency
        self.lookback_periods = lookback_periods
        self.transaction_costs = transaction_costs
        self.portfolio_value = []

    def fit(self, X, y, **kwargs):
        return self


class Backtester(BaseEstimator):
    def __init__(
        self, strategy: Union[Strategy, Sequence[Strategy]], warmup_period: int
    ):
        """

        Parameters
        ----------
        strategy
        warmup_period
        """
        self.strategy = strategy
        self.warmup_period = warmup_period

    def fit(self, X: pd.DataFrame, y=None, **kwargs):
        """

        Parameters
        ----------
        X
        y
        kwargs

        Returns
        -------

        """
        # idx_freq = X.index.freq
        # idx_freqstr = X.index.freqstr
        #
        # if idx_freqstr is None:
        #     raise IndexError("Please resample your data to given frequency")

        # Initializes all equity price and all portfolio weights
        self.weights_ = pd.DataFrame(index=X.index, columns=X.columns, data=[])
        self.equity_ = pd.Series(index=X.index)

        # Compute weights on initial warmup period
        warmup_prices = X.iloc[: self.warmup_period, :]
        # calculate the estimated prices on the warmup period
        # this acts as a kind of pre-initialization of weights, which otherwise should be set equal
        self.strategy.estimator.fit(warmup_prices)
        self.weights_.iloc[self.warmup_period, :] = self.strategy.estimator.weights_
        # updates the equity curve: the initial equity value starts from 0 where it started at `initial_portfolio_value`
        self.equity_.iloc[: self.warmup_period] = equity_curve(
            self.strategy.estimator.predict(warmup_prices),
            initial_value=self.strategy.initial_portfolio_value,
        )

        # calculates the rebalancing dates. At these dates, the portfolio gets rebalanced
        # with the new updated historical data. Transaction costs are accounted for and subtracted from portfolio value
        from collections import deque

        rebalance_dates = deque(
            pd.date_range(
                start=X.index.min(),
                end=X.index.max(),
                freq=self.strategy.rebalance_frequency,
            ).tolist()
        )
        self.costs_ = pd.Series(index=rebalance_dates)

        # First get the indices for the trailing strategy
        # using the well tested pd.Series.rolling function on a fake dataset
        # for quicker collection of results
        # TODO this trick should be made quicker through a generator function
        idx = []

        def collect_indices(df):
            """
            An helper function only needed to globally calculated the indices bounds from the rolling window
            """
            idx.append((df.index.min(), df.index.max()))
            return 0  # there is no need of performing any computation other the exploiting the calculation of indices

        pd.Series(index=X.index[self.warmup_period :], data=0).rolling(
            min_periods=self.strategy.lookback_periods[0],
            window=self.strategy.lookback_periods[1],
        ).apply(collect_indices)

        # then from the correctly created indices, iterates over all these windows
        # manually to populate the weights
        portfolio_value = self.strategy.initial_portfolio_value
        self.turnover_ = []
        self.positions_ = []
        self.paper_ = []
        self.weights_ = []
        self.equity_ = [portfolio_value]
        self.costs_ = []
        self.portfolio_value_ = [portfolio_value]

        # Iterates over all windows of the rolling window
        for i, (t_start, t_end) in enumerate(idx):
            # takes a slice of X with the current rolling window
            df_win = X[t_start:t_end].dropna(
                how="all", axis=0
            )  # eliminates oversamplings            # fit the portfolio optimization method on df_win
            # the portfolio estimator is the one obtained from the given strategy
            self.strategy.estimator.fit(df_win)
            # the old strategy weights were the latest available, excluding the all NaN rows
            self.weights_.append(self.strategy.estimator.weights_)
            self.equity_.append(
                equity_curve(
                    self.strategy.estimator.predict(df_win),
                    initial_value=self.equity_[-1]
                )
            )
            # then look at rebalancing events. If a rebalancing event is found, then the portfolio value
            # should be updated
            if t_end in rebalance_dates:
                self.costs_[t_end] = calculate_transaction_costs(
                    old_portfolio=self.portfolio_value_[-1] * self.weights_[-1],
                    new_portfolio=self.portfolio_value_[0] * self.weights_[0],
                    buy_costs=self.strategy.transaction_costs[0],
                    sell_costs=self.strategy.transaction_costs[1],
                )
                self.equity_[-1] -= self.costs_[t_end]
                self.portfolio_value_.append(self.equity_[-1])
                rebalance_dates.popleft()
                print(f"Rebalance at {t_end}")
                print(f"Portfolio Value {portfolio_value}")

        return self

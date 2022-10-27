# Here we should implement a backtester that takes one or more portfolio estimator objects,
# possibly a rebalancing policy, transaction costs
from typing import Union, Tuple, Sequence
from collections import deque
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from skportfolio import PortfolioEstimator, equity_curve


def calculate_transaction_costs(
    old_weights: pd.Series,
    new_weights: pd.Series,
    old_portfolio_value: float,
    new_portfolio_value: float,
    buy_costs: float,  # buying 1 share costs you 1% of that share
    sell_costs: float,  # selling 1 share costs you 1% of that share
):
    """
    Calculates the incurred transaction costs for transfering between a portfolio
    with old_portfolios [$] and new_portfolio [$].

    Parameters
    ----------
    old_weights: pd.Series
    new_weights: pd.Series
    buy_costs: float
    sell_costs: float

    Returns
    -------
    Transaction costs
    """
    weights_difference = new_weights - old_weights
    capital_allocation_difference = (
        new_portfolio_value - old_portfolio_value
    ) * weights_difference
    capital_to_buy = (
        (
            capital_allocation_difference
            * buy_costs
            * (capital_allocation_difference > 0)
        ).abs()
    ).sum()
    capital_to_sell = (
        (
            capital_allocation_difference
            * sell_costs
            * (capital_allocation_difference < 0)
        ).abs()
    ).sum()
    return capital_to_buy + capital_to_sell


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
        buy_sell_fees_pct: Union[float, Tuple[float, float]],
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
            buy_sell_fees_pct
        buy_sell_fees_pct:
            Fees for buy and sell. Fixed schedule
        """
        self.initial_weights = initial_weights
        self.initial_portfolio_value = initial_portfolio_value
        self.estimator = estimator
        self.rebalance_frequency = rebalance_frequency
        self.lookback_periods = lookback_periods
        self.buy_sell_fees_pct = buy_sell_fees_pct

        # Populated by calls to fit method on **all** initial dataset.
        self.original_index = None
        self.rebalance_dates_ = []
        self._last_position = []
        # Populated by calls to partial_fit method
        self._list_all_weights = []
        self._list_weights_prebalance = []
        self.transaction_costs_ = []
        self._positions = []
        self._total_position = []
        self.equity_ = []
        self._rebalance_idx = []
        self.all_weights_ = None

    def fit(self, X, y=None, **kwargs):
        self.rebalance_dates_ = pd.date_range(
            start=X.index.min(),
            end=X.index.max(),
            freq=self.rebalance_frequency,
        ).tolist()
        self.original_index = X.index

        return self

    def partial_fit(self, X, y=None, **kwargs):
        self._list_all_weights.append(self.estimator.fit(X).weights_)
        self._positions.append(
            self.estimator.predict(X.tail(1)).iat[0] * self.estimator.weights_
        )
        self._total_position.append(self._positions[-1].dot(self.estimator.weights_))
        if X.index.max() in self.rebalance_dates_:
            self._rebalance_idx.append(len(self._positions))
            self._list_weights_prebalance.append(self._list_all_weights[-1])
            self._last_position.append(self._positions[-1])

        if kwargs.get("last_fit", False):
            self.equity_ = equity_curve(
                df=pd.concat(self._positions, axis=1, ignore_index=True)
                .T.set_index(self.original_index[: len(self._positions)])
                .sum(1),
                initial_value=self.initial_portfolio_value,
            ).rename("equity")
            num_rebalance_events = len(self.rebalance_dates_)
            # -1 because first buy does not suffer transaction costs
            for rebalance_idx_cur in range(num_rebalance_events):
                self.transaction_costs_.append(
                    calculate_transaction_costs(
                        old_weights=self._list_weights_prebalance[
                            rebalance_idx_cur - 1
                        ],
                        new_weights=self._list_weights_prebalance[rebalance_idx_cur],
                        old_portfolio_value=self.equity_[
                            self._rebalance_idx[rebalance_idx_cur] - 1
                        ],
                        new_portfolio_value=self.equity_[
                            self._rebalance_idx[rebalance_idx_cur]
                        ],
                        buy_costs=self.buy_sell_fees_pct[0],
                        sell_costs=self.buy_sell_fees_pct[1],
                    )
                )
            # subtract the transaction costs from the equity curve
            for i, t in enumerate(self.rebalance_dates_[:-1]):
                iat = self.equity_.index.get_loc(t, method="nearest")
                if iat != -1:
                    self.equity_.iloc[iat:] -= (
                        self.transaction_costs_[i]
                        * self.equity_.iat[iat]
                        / (sum(self._positions[0]))
                    )
            # rebuilds the all_weights array
            self.all_weights_ = (
                pd.concat(self._list_all_weights, axis=1)
                .T.set_index(self.original_index[-len(self._list_all_weights) :])
                .reindex(self.original_index)
                .fillna(self.initial_weights)
            )
            print("done")

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
        # calculates the rebalance dates
        self.strategy.fit(X)
        # Compute weights on initial warmup period
        warmup_prices = X.iloc[: self.warmup_period, :]
        # calculate the estimated prices on the warmup period
        # this acts as a kind of pre-initialization of weights,
        # which otherwise should be set equal
        self.strategy.partial_fit(warmup_prices)

        # First get the indices for the trailing strategy
        # using the well tested pd.Series.rolling function on a fake dataset
        # for quicker collection of results
        # TODO this trick should be made quicker through a generator function
        idx = []

        def collect_indices(df):
            idx.append((df.index.min(), df.index.max()))
            # there is no need of performing any computation other the exploiting the calculation of indices
            return 0

        pd.Series(index=X.index[self.warmup_period :], data=0).rolling(
            min_periods=self.strategy.lookback_periods[0],
            window=self.strategy.lookback_periods[1],
        ).apply(collect_indices)

        # then from the correctly created indices, iterates over all these windows
        # manually to populate the weights

        # Iterates over all windows of the rolling window
        for i, (t_start, t_end) in enumerate(idx):
            # takes a slice of X with the current rolling window
            prices_slice = X[t_start:t_end].dropna(
                how="all", axis=0
            )  # eliminates oversamplings            # fit the portfolio optimization method on df_win
            # partial_fit of the strategy, populates with the weights calculated
            self.strategy.partial_fit(prices_slice, last_fit=t_end == X.index.max())

        return self

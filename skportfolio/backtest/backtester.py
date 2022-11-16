# Here we should implement a backtester that takes one or more portfolio estimator objects,
# possibly a rebalancing policy, transaction costs

from typing import Union, Tuple, List, Optional, Callable
from itertools import chain
from collections import deque
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore
from skportfolio._base import PortfolioEstimator
from skportfolio.backtest.strategy import Strategy
from skportfolio.backtest.fees import fixed_transaction_costs, TransactionCostsFcn
from skportfolio._constants import APPROX_DAYS_PER_YEAR

#
# class Backtester(BaseEstimator):
#     def __init__(
#         self,
#         strategies: Union[Strategy, List[Strategy]],
#         warmup_period: int,
#         initial_portfolio_value: float,
#     ):
#         """
#
#         Parameters
#         ----------
#         strategies
#         warmup_period
#         """
#         self.strategies = strategies
#         self.warmup_period = warmup_period
#         self.initial_portfolio_value = initial_portfolio_value
#
#     def fit(self, X, y=None, **kwargs):
#         if isinstance(self.strategies, Strategy):
#             self.strategies = [self.strategies]
#         # Compute weights on initial warmup period
#         warmup_prices = X.iloc[: self.warmup_period, :]
#         # calculate the estimated prices on the warmup period
#         # this acts as a kind of pre-initialization of weights,
#         # which otherwise should be set equal
#         # inplace operation here!
#         # TODO this is hard to make it parallel
#         # move all the Strategy and make it immutable, the backtesting logic shall go inside here
#         for strategy in self.strategies:
#             strategy.step(warmup_prices, y, warmup_step=True, **kwargs)
#         T = X.shape[0]
#
#         # Iterates over all windows of the rolling window
#         for i in range(self.warmup_period + 1, T):
#             X_slice = X.iloc[:i, :]
#             for strategy in self.strategies:
#                 strategy.step(
#                     X_slice,
#                     y=y.iloc[self.warmup_period : i] if y is not None else None,
#                     trigger_rebalance=(
#                         (i - self.warmup_period) % strategy.rebalance_frequency
#                     )
#                     == 0,
#                     **kwargs,
#                 )
#
#         # the variable containing the backtester result
#         self.equity_curve_ = pd.concat(
#             (
#                 strategy.all_positions_.sum(1).rename(str(strategy.estimator))
#                 for strategy in self.strategies
#             ),
#             axis=1,
#         )
#         self.average_turnover_ = [
#             np.mean(strategy.turnover_) for strategy in self.strategies
#         ]
#         # now remove the transaction costs
#         return self
#
#     def fit_predict(self, X, y=None, **kwargs):
#         self.fit(X, y, **kwargs)
#         return self.equity_curve_
#
#     def predict(self, X, y=None, **kwargs):
#         check_is_fitted(self, attributes="equity_curve_")
#         return self.equity_curve_


def rolling_expanding_window_with_warmup(seq, warmup, n_min, n_max):
    it = iter(range(len(seq)))  # makes it iterable
    if warmup > 0:
        win = deque((next(it, None) for _ in range(warmup)), maxlen=warmup)
        yield win
    # roll it forward at least warmup steps
    win = deque((next(it, None) for _ in range(n_min)), maxlen=n_max)
    yield win
    for e in it:
        win.append(e)
        yield win


def sliding_dataframes(*arrays, warmup=0, n_min=0, n_max=2**32):
    """
    Like scikit-learn train_test_split but with rolling-expanding window
    Parameters
    ----------
    arrays
    warmup
    n_min
    n_max

    Returns
    -------
    """
    n_dataframes = len(arrays)
    if n_dataframes == 0:
        raise ValueError("At least one array required as input")

    array_is_none = [a is None for a in arrays]
    n_samples = arrays[0].shape[0]
    for df in arrays:
        if not isinstance(df, (pd.DataFrame, pd.Series)) and df is not None:
            raise TypeError("This method only supports pandas dataframes and series")
        if df is not None and df.shape[0] != n_samples:
            raise ValueError("Specify equal length dataframes or series")

    # the first dataframe is the one dictating
    indices = rolling_expanding_window_with_warmup(
        arrays[0], warmup=warmup, n_min=n_min, n_max=n_max
    )

    for index in indices:
        yield list(
            chain.from_iterable(
                (a.iloc[index] if a is not None else None,) for a in arrays
            )
        )


def _make_zero_weights_with_cash(tickers):
    weights = pd.Series(data=0, index=tickers)
    weights.loc["__CASH__"] = 1
    return weights


def rebalance_function(current_weights, prices, estimator, **kwargs):
    return estimator.fit(prices).weights_, current_weights


def update_positions(positions, weights, current_value, t):
    positions.loc[t] = weights * current_value
    return positions


def last_total_prices(prices):
    """
    Computes the sum of the last row of data
    Parameters
    ----------
    prices

    Returns
    -------

    """
    return prices.iloc[-1:].sum()


class Backtester(PortfolioEstimator):
    def __init__(
        self,
        estimator: PortfolioEstimator,
        name: Optional[str] = None,
        rebalance_frequency: int = 1,
        initial_weights: Optional[pd.Series] = None,
        initial_portfolio_value: float = 10_000,
        warmup_period: int = 0,
        window_size: Union[
            int,
            Tuple[int, int],
        ] = 0,
        transaction_costs: Union[float, Tuple[float, float], TransactionCostsFcn] = 0.0,
        risk_free_rate: float = 0.0,
        rates_frequency: int = APPROX_DAYS_PER_YEAR,
        warm_start: bool = False,
    ):
        """

        Parameters
        ----------
        estimator: PortfolioEstimator
            Portfolio estimator to run backtest on.
        name: str, default=None
            Name of the backtesting strategy
        rebalance_frequency: int, default=None
            Number of rows to call the rebalance function.
        initial_weights: pd.Series, default=None
            Initial portfolio weights.
        initial_portfolio_value: float or int, default=10,000 units of currency.
            Initial portfolio value in currency units.
        warmup_period: int
            Number of rows to consider to calculate initial portfolio weights from.
        window_size: Union[int, Tuple[int,int]]
            The minimum and maximum window size. Default 0 means expanding window
        transaction_costs:
            Either a fixed rate, a pair of buy-sell rates, or a more complex function with signature
            variable_transaction_costs(delta_pos, *args).
        risk_free_rate: float
            The risk-free-rate. The portfolio earns this rate on the cash allocation.
        rates_frequency: int, default 252 (skportfolio._constants.APPROX_DAYS_PER_YEAR)
            The rate frequency, for annualized rates (number of business days in a year use 252)
        warm_start: bool, default False
            Whether to warm start the estimator as in many partial_fit estimators
        """

        self.estimator = estimator
        self.name = name
        self.rebalance_frequency = rebalance_frequency
        self.initial_weights = initial_weights
        self.initial_portfolio_value = initial_portfolio_value
        self.warmup_period = warmup_period
        self.window_size = window_size
        self.transaction_costs = transaction_costs
        self.risk_free_rate = risk_free_rate
        self.rates_frequency = rates_frequency
        self.warm_start = warm_start

        # private internals variables
        self._current_value = None
        self._normalization_constant = None
        self._iteration = 0
        self._last_position = None
        self._transaction_cost_fcn = None
        self._min_window_size = None
        self._max_window_size = None

        # attributes to be read after fit
        self.all_weights_ = None
        self.equity_curve_ = None
        self.positions_ = None
        self.positions_at_rebalance_ = None
        self.rebalance_dates_ = None
        self.transaction_costs_ = None
        self.turnover_ = None

    def _cold_start(self, X):
        self.n_samples = X.shape[0]
        # private variables conversion logic
        if isinstance(self.window_size, (list, tuple)):
            self._min_window_size = self.window_size[0]
            self._max_window_size = self.window_size[1]
        elif isinstance(self.window_size, int):
            min_window_size, max_window_size = self.window_size, self.n_samples

        if isinstance(self.transaction_costs, (float, int)):
            self._transaction_cost_fcn = lambda delta_pos: 2 * (
                sum(abs(delta_pos) * self.transaction_costs),
            )
        elif isinstance(self.transaction_costs, (tuple, list)):
            self._transaction_cost_fcn = lambda delta_pos: (
                sum(abs(delta_pos * self.transaction_costs[0])),
                sum(abs(delta_pos * self.transaction_costs[1])),
            )
        elif callable(self.transaction_costs):
            self._transaction_cost_fcn = self.transaction_costs

        # Define internals initialization
        if self.initial_weights is None:
            self.initial_weights = _make_zero_weights_with_cash(tickers=X.columns)
        self.all_weights_ = [self.initial_weights]
        # this is the normalization constant to apply to all positions
        self._normalization_constant = (
            X.iloc[0, :].sum() / X.shape[1] * self.initial_portfolio_value
        )
        self.rebalance_dates_ = []  # no rebalance yet
        # all the costs have been spent in buying the initial portfolio
        self.transaction_costs_ = []
        self.positions_ = pd.DataFrame(columns=X.columns.tolist() + ["__CASH__"])
        self.positions_ = update_positions(
            self.positions_,
            weights=self.all_weights_[-1],
            current_value=self.initial_portfolio_value,
            t=X.index[-1],
        )
        # allocation is zero everywhere
        # first rebalance is considered at the first step
        self.positions_at_rebalance_ = self.positions_
        self.turnover_ = []

    def fit(self, X, y=None, **kwargs):
        """
        Handles the initialization logic of internals
        Parameters
        ----------
        X
        y
        kwargs

        Returns
        -------

        """
        # here cold restart is always done, as each call to fit must reset the previous state
        self._cold_start(X)

        # add the trigger rebalance column
        for counter, (prices_slice, benchmark_slice) in enumerate(
            sliding_dataframes(
                X.assign(rebalance=np.arange(len(X)) % self.rebalance_frequency),
                y,
                warmup=self.warmup_period,
                n_min=self._min_window_size,
                n_max=self._max_window_size,
            )
        ):
            # rebalance can only be triggered once the warmup period was surpassed
            trigger_rebalance = (
                counter % self.rebalance_frequency
                if counter >= self.warmup_period
                else False
            )
            self.partial_fit(
                X=prices_slice,
                y=benchmark_slice,
                trigger_rebalance=trigger_rebalance,
                **kwargs,
            )
        # self.average_turnover_ = np.mean(self.turnover_)
        # now remove the transaction costs
        return self

    def partial_fit(self, X, y=None, **kwargs):
        trigger_rebalance = kwargs.get("trigger_rebalance", False)
        if not self.warm_start and self._iteration == 0:
            self._cold_start(X)

        self.all_weights_.append(self.estimator.fit(X).weights_)
        self.equity_curve_ = pd.concat(
            (
                self.equity_curve_,
                self._normalization_constant / X.dot(self.all_weights_[-1]),
            )
        )
        self._iteration += 1
        return self

    def fit_predict(self, X, y=None, **kwargs):
        self.fit(X, y, **kwargs)
        return self.equity_curve_

    def predict(self, X, y=None, **kwargs):
        check_is_fitted(self, attributes="equity_curve_")
        return self.equity_curve_

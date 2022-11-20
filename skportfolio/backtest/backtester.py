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


def rolling_expanding_window_with_warmup(seq, warmup, n_min, n_max):
    it = iter(range(len(seq)))  # makes it iterable
    if warmup > 0:
        win = deque((next(it, None) for _ in range(warmup)), maxlen=warmup)
        yield win
    # roll it forward at least warmup steps
    win = deque((next(it, None) for _ in range(n_min)), maxlen=n_max)
    # yield win
    for e in it:
        win.append(e)
        yield win


def sliding_dataframes(*arrays, warmup=0, n_min=0, n_max=None):
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
    weights = pd.Series(data=0.0, index=tickers)
    weights.loc["__CASH__"] = 1
    return weights


def find_index(current_index: pd.DatetimeIndex, full_index: pd.DatetimeIndex):
    out = np.argwhere(current_index.max() == full_index)
    if not out:
        return 0
    else:
        return int(out)


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
        transactions_budget: float = 0.1,
        risk_free_rate: float = 0.0,
        rates_frequency: int = APPROX_DAYS_PER_YEAR,
        warm_start: bool = False,
        precompute_returns: bool = True,
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
            The minimum and maximum window size. Default 0 means expanding window.
            Each time the backtesting engine calls a strategy rebalance function, a window of asset price data
            (and possibly signal data) is passed to the rebalance function.
            The rebalance function can then make trading and allocation decisions based on a rolling window of
            market data. The window_size property sets the size of these rolling windows.
            Set the window in terms of time steps. The window determines the number of rows of data from
            the asset price timetable that are passed to the rebalance function.
            The window_size property can be set in two ways. For a fixed-sized rolling window of data
            (for example, "50 days of price history"), the window_size property is set to a single scalar value
            (N = 50). The software then calls the rebalance function with a price timetable containing exactly N rows
            of rolling price data.
            Alternatively, you can define the window_size property by using a 1-by-2 vector [min max]
            that specifies the minimum and maximum size for an expanding window of data.
            In this way, you can set flexible window sizes. For example:

            [10 None] — At least 10 rows of data
            [0 50] — No more than 50 rows of data
            [0 None] — All available data (that is, no minimum, no maximum); this is the default value
            [20 20] — Exactly 20 rows of data; this is equivalent to setting window_size to the scalar value 20

            The software does not call the rebalance function if the data is insufficient to create a valid rolling
            window, regardless of the value of the RebalanceFrequency property.
            If the strategy does not require any price or signal data history, then you can indicate that the
            rebalance function requires no data by setting the window_size property to 0.
        transaction_costs: float, Tuple[float,float], Callable[delta_pos, *args], default 0
            Either a fixed rate, a pair of buy-sell rates, or a more complex function with signature
            variable_transaction_costs(delta_pos, *args).
        transactions_budget: float, default 10%
            The maximum amount to spend in transaction costs, expressed as a percentage
            of total initial portfolio value.
        risk_free_rate: float, default 0.0
            The risk-free-rate. The portfolio earns this rate on the cash allocation.
        rates_frequency: int, default 252 (skportfolio._constants.APPROX_DAYS_PER_YEAR)
            The rate frequency, for annualized rates (number of business days in a year use 252)
        warm_start: bool, default False
            Whether to warm start the estimator as in many partial_fit estimators
        precompute_returns: bool, default True
            Whether to precompute the prices returns. When using `fit` method this is the
            suggested behaviour to speed up calculation. Otherwise.

        """

        self.estimator = estimator
        self.name = name
        self.rebalance_frequency = rebalance_frequency
        self.initial_weights = initial_weights
        self.initial_portfolio_value = initial_portfolio_value
        self.warmup_period = warmup_period
        self.window_size = window_size
        self.transaction_costs = transaction_costs
        self.transactions_budget = transactions_budget
        self.risk_free_rate = risk_free_rate
        self.rates_frequency = rates_frequency
        self.warm_start = warm_start
        self.precompute_returns = precompute_returns

        # private internals variables
        self._iteration = 0
        self._transaction_cost_fcn: Optional[Callable] = None
        self._min_window_size: Optional[int] = 0
        self._max_window_size: Optional[int] = None

        # attributes to be read after fit
        self.all_weights_: Optional[pd.DataFrame] = None
        self.equity_curve_: Optional[pd.Series] = None
        self.positions_: Optional[pd.DataFrame] = None
        self.positions_at_rebalance_: Optional[pd.DataFrame] = None
        self.rebalance_dates_: Optional[List] = []
        self.returns_: Optional[pd.Series] = None
        self.buy_sell_costs_: Optional[pd.DataFrame] = None
        self.turnover_ = None

    def _cold_start(self, X, **kwargs):
        self.n_samples = X.shape[0]
        # private variables conversion logic
        if isinstance(self.window_size, (list, tuple)):
            self._min_window_size = self.window_size[0]
            self._max_window_size = self.window_size[1]
        elif isinstance(self.window_size, int):
            # in case 0 is specified, add one for having at least one row of data, hence expanding window
            self._min_window_size, self._max_window_size = (
                self.window_size,
                self.n_samples,
            )
        else:
            raise ValueError("Not a supported window size specification")

        if isinstance(self.transaction_costs, (float, int)):
            self._transaction_cost_fcn = lambda delta_pos: (
                abs(sum(delta_pos * (delta_pos > 0))) * self.transaction_costs,
                abs(sum(delta_pos * (delta_pos < 0))) * self.transaction_costs,
            )
        elif isinstance(self.transaction_costs, (tuple, list)):
            self._transaction_cost_fcn = lambda delta_pos: (
                sum(-delta_pos * (delta_pos > 0)) * self.transaction_costs[0],
                sum(-delta_pos * (delta_pos < 0)) * self.transaction_costs[1],
            )
        elif callable(self.transaction_costs):
            self._transaction_cost_fcn = self.transaction_costs

        if self.rebalance_frequency < 0:
            raise ValueError("Rebalancing frequency must be greater or equal than 1")

        # Define internals initialization
        self.all_weights_ = pd.DataFrame(columns=X.columns.tolist(), dtype=float)
        self.equity_curve_ = pd.Series(name=self.name + "_equity", dtype=float)
        self.rebalance_dates_ = [
            X.index[i] for i in np.arange(len(X)) if (i % self.rebalance_frequency) == 0
        ]
        # no rebalance yet, but the schedule is prepared
        # all the costs have been spent in buying the initial portfolio
        self.buy_sell_costs_ = pd.DataFrame(
            columns=["buy", "sell"], index=self.rebalance_dates_
        )
        self.positions_ = pd.DataFrame(columns=X.columns.tolist(), dtype=float)
        self.returns_ = pd.Series(name=self.name + "_returns", dtype=float)
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
        # all next calls to partial_fit are done like if they come from another invoker
        self.warm_start = True
        # add the trigger rebalance column
        if self.precompute_returns:
            returns = X.pct_change()
        else:
            returns = None
        for counter, (prices_slice, benchmark_slice, returns_slice) in enumerate(
            sliding_dataframes(
                X,
                y,
                returns,
                warmup=self.warmup_period,
                n_min=self._min_window_size,
                n_max=self._max_window_size,
            )
        ):
            # rebalance can only be triggered once the warmup period was surpassed
            self.partial_fit(
                X=prices_slice,
                y=benchmark_slice,
                precomputed_returns=returns_slice,
                full_index=X.index,
                **kwargs,
            )
        # now remove the transaction costs
        return self

    def partial_fit(self, X, y=None, **kwargs):
        """
        The method to be repeatedly called on slices of the price data.
        Parameters
        ----------
        X: pd.DataFrame
            A slice of prices data
        y: pd.Series
            Any additional variable needed to the estimator `fit` method
        kwargs:
            Any additional argument to pass to the estimator `fit` method, together
            with the full_index variable that describes the full temporal interval
            the X data come from.

        Returns
        -------

        """
        full_index = kwargs["full_index"]

        X_rets = None
        if "precomputed_returns" in kwargs:
            X_rets = kwargs["precomputed_returns"]
        else:
            X_rets = X.pct_change()
        last_index = X.index[-1]
        self._iteration = find_index(X.index, full_index=full_index)

        needs_rebalance = (self._iteration > 0) and (
            last_index in self.rebalance_dates_
        )
        if not self.warm_start and self._iteration == 0:
            self._cold_start(X)

        if self.initial_weights is None:
            self.initial_weights = self.estimator.fit(X).weights_
            self.all_weights_.loc[last_index, :] = self.initial_weights
        elif self.initial_weights is not None and not needs_rebalance:
            self.all_weights_.loc[last_index, :] = self.initial_weights
        elif self.initial_weights is not None and needs_rebalance:
            self.all_weights_.loc[last_index, :] = self.estimator.fit(X).weights_

        self.returns_.loc[last_index] = X_rets.loc[last_index].dot(
            self.all_weights_.loc[last_index]
        )
        # TODO this should be made faster through caching of intermediate results
        self.positions_ = (
            (1 + X_rets)
            .cumprod()
            .fillna(1)
            .mul(self.initial_portfolio_value)
            .rmul(self.all_weights_.iloc[-1, :])
        )
        self.equity_curve_ = self.positions_.sum(1)

        if needs_rebalance:
            # we must ensure that our data slice contains the index of previous rebalance
            # if this is not true because our sliding window is too small we need to
            # 1. either do nothing
            # 2. inform the user with a warning, to increase the time window or to switch to
            # expanding window
            prev_index = self.rebalance_dates_[
                self.rebalance_dates_.index(last_index) - 1
            ]
            if prev_index in self.positions_.index:
                delta_pos = (
                    self.positions_.loc[last_index, :]
                    - self.positions_.loc[prev_index, :]
                )
                buy_cost, sell_cost = self._transaction_cost_fcn(delta_pos)
                self.buy_sell_costs_.loc[last_index] = (
                    buy_cost,
                    sell_cost,
                )
                self.positions_.loc[last_index] -= buy_cost + sell_cost
                self.equity_curve_.loc[last_index] -= buy_cost + sell_cost
                self.returns_.loc[last_index] = (
                    self.equity_curve_.tail(2).pct_change().iat[-1]
                )

        return self

    def fit_predict(self, X, y=None, **kwargs):
        self.fit(X, y, **kwargs)
        return self.equity_curve_

    def predict(self, X, y=None, **kwargs):
        check_is_fitted(self, attributes="equity_curve_")
        return self.equity_curve_

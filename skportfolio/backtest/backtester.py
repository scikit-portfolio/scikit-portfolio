"""
Implementation of a vectorial backtester taking a single portfolio estimator
and other parameters as the rebalancing frequency and transaction costs, to produce
an equity curve as result
"""

from collections import deque
from functools import partial
from typing import Union, Tuple, List, Optional, Callable

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.base import MetaEstimatorMixin, RegressorMixin
from tqdm.auto import tqdm

from skportfolio._base import PortfolioEstimator
from skportfolio._constants import APPROX_DAYS_PER_YEAR
from skportfolio._simple import EquallyWeighted
from skportfolio.backtest.fees import TransactionCostsFcn, basic_percentage_fee
from skportfolio.metrics import sharpe_ratio, summary


def rolling_expanding_window(seq, n_min, n_max):
    """
    Emits the elements over a rolling or expanding window of an iterable sequence
    Parameters
    ----------
    seq
    n_min
    n_max

    Returns
    -------

    """
    current_iterator = iter(range(len(seq)))  # makes it iterable
    # roll it forward at least warmup steps
    win = deque((next(current_iterator, None) for _ in range(n_min)), maxlen=n_max)
    # yield win
    for element in current_iterator:
        win.append(element)
        yield win


def make_position(
    cash_weight: float,
    asset_weights: Union[np.ndarray, List[float]],
    asset_names: List[str],
    cash_name: str,
    portfolio_value: float,
) -> pd.Series:
    """
    Utility function to define a position as a pd.Series
    Parameters
    ----------
    cash_weight: float
    asset_weights: float
    asset_names: List[str]
    cash_name: str
    portfolio_value: float

    Returns
    -------
    """
    all_names = (cash_name, *asset_names)
    all_weights = (cash_weight, *asset_weights)
    return pd.Series(dict(zip(all_names, all_weights)), dtype=float).mul(
        portfolio_value
    )


class Backtester(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    """
    The main backtester estimator class.
    This object is particular. While it supports many different
    calls to its fit method, every time the results are reset.
    From a sklearn perspective this looks more like a Regressor
    rather than a model selection estimator like LogisticRegressionCV.
    This class can to be used in conjunction to hyperparameters search
    estimators like RandomizedSearchCV or GridSearchCV provided that
    the `cv` argument is set to [slice(None), slice(None)]
    """

    def __init__(
        self,
        estimator: Optional[PortfolioEstimator] = None,
        name: Optional[str] = None,
        rebalance_frequency: int = 1,
        initial_weights: Optional[pd.Series] = None,
        initial_portfolio_value: float = 10_000,
        warmup_period: int = 0,
        window_size: Union[
            int,
            Tuple[int, Optional[int]],
        ] = None,
        transaction_costs: Union[float, Tuple[float, float], TransactionCostsFcn] = 0.0,
        transactions_budget: float = 0.1,
        risk_free_rate: float = 0.0,
        cash_borrow_rate: float = 0.0,
        rates_frequency: int = APPROX_DAYS_PER_YEAR,
        score_fcn: Optional[
            Callable[
                [
                    pd.Series,
                ],
                float,
            ]
        ] = None,
        show_progress: bool = False,
    ):
        """
        Principal vectorized backtester for use with any PortfolioEstimator object.
        It runs the strategy intended by the PortfolioEstimator and periodically calculates a
        rebalance collecting all the positions over the complete interval.

        Parameters
        ----------
        estimator: PortfolioEstimator, default None
            Portfolio estimator to run backtest on. If set to None an
            EquallyWeighted strategy is adopted.
        name: str, default=None
            Name of the backtesting strategy. If set to None the name of the
            estimator is used.
        rebalance_frequency: int, default=1
            Number of rows to call the rebalance function.
        initial_weights: pd.Series, default=None
            Initial portfolio weights. If set to None, initial portfolio position
            is all allocated in cash.
        initial_portfolio_value: float or int, default=10,000 units of currency.
            Initial portfolio value in currency units.
        warmup_period: int
            Number of rows to consider to calculate initial portfolio weights from.
        window_size: Optional[Union[int, Tuple[int,int]]]
            The minimum and maximum window size. Default None means expanding window.
            Each time the backtesting engine calls a strategy rebalance function,
            a window of asset price data (and possibly signal data) is passed to the
            rebalance function. The rebalance function can then make trading and
            allocation decisions based on a rolling window of market data. The window_size
            property sets the size of these rolling windows.
            Set the window in terms of time steps. The window determines the number of
            rows of data from the asset price timetable that are passed to the
            rebalance function.
            The window_size property can be set in two ways. For a fixed-sized
            rolling window of data (for example, "50 days of price history"), the
            window_size property is set to a single scalar value
            (N = 50). The software then calls the rebalance function with a price
            timetable containing exactly N rows of rolling price data.
            Alternatively, you can define the window_size property by using a
            1-by-2 vector [min max]
            that specifies the minimum and maximum size for an expanding window of data.
            In this way, you can set flexible window sizes. For example:

            [10 None] — At least 10 rows of data
            [0 50] — No more than 50 rows of data
            [0 None] — All available data (that is, no minimum, no maximum): default
            [20 20] — Exactly 20 rows of data; equivalent to setting window_size to 20

            The software does not call the rebalance function if the data is insufficient to
            create a valid rolling window, regardless of the value of the RebalanceFrequency
            property.
            If the strategy does not require any price or signal data history, then you can
            indicate that the rebalance function requires no data by setting the window_size
             property to 0.
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
        score_fcn: Callable
            The score function to evaluate the strategy performance.
            When None, default is set to Sharpe Ratio scorer
        show_progress: bool, default False
            Whether to show the actual progress
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
        self.cash_borrow_rate = cash_borrow_rate
        self.rates_frequency = rates_frequency
        self.score_fcn = score_fcn
        self.show_progress = show_progress

        # private internals variables
        self._asset_names: Optional[List[str]] = None
        self._transaction_cost_fcn: Optional[TransactionCostsFcn] = None
        self._min_window_size: Optional[int] = 0
        self._max_window_size: Optional[int] = None

        self._reinit()

    def _reinit(self):
        # attributes to be read after fit
        self.positions_: Optional[pd.DataFrame] = None
        self.equity_curve_: Optional[pd.Series] = None
        self.returns_: Optional[pd.Series] = None
        self.rebalance_dates_: Optional[List] = []
        self.buy_sell_costs_: Optional[pd.DataFrame] = None
        self.turnover_: Optional[pd.Series] = None
        self.weights_: Optional[pd.Series] = None

    def fit(self, X, y=None, **fit_params) -> "Backtester":
        """
        Fit the backtester on a rolling-expanding window base to the data, rebalancing when
        needed and keeping into account transaction costs.

        Parameters
        ----------
        X: pd.DataFrame,
            Prices dataframe.
        y: pd.Series, default=None
            Other benchmarks to be passed to the portfolio estimator method.
            Useful when backtesting within the Black-Litterman framework where the market
            implied returns are necessary.

        Other Parameters:
        ------------------
        rebalance_signal: List[bool], Tuple[bool]
            A list of the same lenght as the prices, of booleans, indicating
            a rebalance signal. When true a rebalancing event is forced

        Returns
        -------
        self
        """
        self._reinit()
        if self.estimator is None:
            self.estimator = EquallyWeighted()
        asset_returns = X.pct_change().dropna()
        self._asset_names = X.columns.tolist()
        if "CASH" in self._asset_names:
            raise ValueError(
                "CASH is reserved for liquidity hence cannot be an asset name. Please "
                "rename your dataframe column."
            )
        n_samples, n_assets = X.shape
        rebalance_signal = fit_params.get("rebalance_signal", (False,) * n_samples)
        # private variables conversion logic
        if isinstance(self.window_size, (list, tuple)):
            self._min_window_size = self.window_size[0]
            self._max_window_size = (
                self.window_size[1] if self.window_size[1] is not None else n_samples
            )
        elif isinstance(self.window_size, int):
            # in case 0 is specified, add one for having at least one row of data,
            # hence expanding window
            self._min_window_size, self._max_window_size = (
                self.window_size,
                n_samples,
            )
        elif self.window_size is None:
            self._min_window_size, self._max_window_size = (
                0,
                n_samples,
            )
        else:
            raise ValueError("Not a supported window size specification")

        if isinstance(self.transaction_costs, (float, int, tuple, list)):
            self._transaction_cost_fcn = partial(
                basic_percentage_fee, transaction_costs=self.transaction_costs
            )
        elif callable(self.transaction_costs):
            self._transaction_cost_fcn = self.transaction_costs

        if self.rebalance_frequency < 0:
            raise ValueError("Rebalancing frequency must be greater or equal than 1")

        if self.initial_weights is None:
            initial_positions = make_position(
                1.0,
                [0.0] * n_assets,
                self._asset_names,
                "CASH",
                self.initial_portfolio_value,
            )
        else:
            if self.initial_weights.shape[0] != X.shape[1]:
                raise ValueError(
                    "Invalid number of initial weights, provide all weights of each asset"
                )
            initial_positions = make_position(
                1.0 - self.initial_weights.sum(),
                self.initial_weights.values,
                self._asset_names,
                "CASH",
                self.initial_portfolio_value,
            )
        if self.score_fcn is None:
            self.score_fcn: Callable = sharpe_ratio
        self.buy_sell_costs_: pd.DataFrame = pd.DataFrame(
            columns=["buy", "sell"], index=self.rebalance_dates_, data=0
        )
        self.turnover_: pd.Series = pd.Series(
            dtype=float, index=self.rebalance_dates_, data=0
        )
        self.returns_: pd.Series = pd.Series(dtype=float, index=X.index, data=np.NAN)

        positions: List[pd.Series] = [initial_positions]
        # initialize positions
        previous_positions: pd.Series = initial_positions

        if self.show_progress:
            name = self.name if self.name is not None else str(self.estimator)
            progress = tqdm(
                range(n_samples - 1),
                desc=f"Backtesting {name}",
                leave=False,
            )
        else:
            progress = range(n_samples - 1)
        for idx in progress:
            start_positions = previous_positions
            next_idx = idx + self.warmup_period + 1
            start_portfolio_value = start_positions.sum()
            cash_return = self.risk_free_rate
            margin_return = self.cash_borrow_rate
            if 0 <= start_positions["CASH"]:
                # Zero or positive cash
                row_returns = 1 + pd.concat(
                    (pd.Series({"CASH": cash_return}), asset_returns.iloc[idx, :]),
                    axis=0,
                )
            else:
                # Negative cash
                row_returns = 1 + pd.concat(
                    (
                        pd.Series({"CASH": margin_return}),
                        asset_returns.iloc[idx, :],
                    ),
                    axis=0,
                )
            # Apply current day's returns and calculate the end-of-row positions
            end_date = X.index[next_idx]
            end_positions = start_positions * row_returns
            end_portfolio_value = end_positions.sum()
            end_asset_weights = end_positions.div(end_portfolio_value)

            needs_rebalance = (
                (next_idx % self.rebalance_frequency == 0)
                or (rebalance_signal[next_idx])
            ) and (idx > 0)
            if needs_rebalance:
                # check we have enough data
                is_valid_window = next_idx >= self._min_window_size
                if is_valid_window:
                    # call the estimator of this slice of data
                    start_window = next_idx - self._max_window_size + 1
                    window_rows = np.arange(max(0, start_window), next_idx + 1)
                    asset_data = X.iloc[window_rows, :][self._asset_names]
                    end_asset_weights_new: pd.Series = self.estimator.fit(
                        X=asset_data,
                        y=y.iloc[window_rows] if y is not None else None,
                        **fit_params,
                    ).weights_.copy()
                    delta_weights: pd.Series = end_asset_weights_new - end_asset_weights
                    self.turnover_.loc[end_date] = delta_weights.abs().sum() * 0.5
                    buy_cost, sell_cost = self._transaction_cost_fcn(
                        delta_weights[self._asset_names] * end_portfolio_value
                    )
                    self.rebalance_dates_.append(end_date)
                    self.buy_sell_costs_.loc[end_date, :] = (buy_cost, sell_cost)
                    # pre_fees_portfolio_value = end_portfolio_value
                    end_portfolio_value -= buy_cost + sell_cost

                    # update end_position after transaction fees
                    end_asset_weights = end_asset_weights_new
            # needs to recompute the cash component
            end_asset_weights["CASH"] = 1 - end_asset_weights[self._asset_names].sum()
            end_positions = end_portfolio_value * end_asset_weights
            # this operation is faster than .loc
            positions.append(end_positions)
            self.returns_.loc[end_date] = (
                end_portfolio_value / start_portfolio_value - 1
            )
            previous_positions = end_positions

        # Finally converts the positions in a dataframe
        self.positions_ = pd.DataFrame(positions, index=X.index[self.warmup_period :])
        self.returns_.dropna(inplace=True)  # returns have one row less than original

        self.turnover_ = self.turnover_.reindex(
            self.returns_.index.drop_duplicates()
        ).fillna(value=0.0)
        # and computes the equity curve
        self.equity_curve_ = self.positions_.sum(axis=1)
        self.buy_sell_costs_ = self.buy_sell_costs_.reindex(
            self.equity_curve_.index
        ).fillna(value=0.0)
        last_position = positions[-1].drop("CASH")
        self.weights_ = last_position.div(last_position.sum())
        return self

    def fit_predict(self, X, y=None, **fit_kwargs):
        return self.fit(X, y, **fit_kwargs).equity_curve_

    def score(self, X, y=None, sample_weight=None):
        """
        Calculates the strategy score (like Sharpe ratio)
        Parameters
        ----------
        X: prices
        y: possibly other
        sample_weight

        Returns
        -------
        """
        return self.score_fcn(self.returns_)

    def summary(self):
        """
        Calculates the strategy metrics

        Returns
        -------
        pd.Series with all metrics
        """
        return summary(
            r=self.returns_,
            frequency=self.rates_frequency,
            risk_free_rate=self.risk_free_rate,
        ).rename(self.name)

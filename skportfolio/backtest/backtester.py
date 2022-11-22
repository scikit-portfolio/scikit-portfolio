# Here we should implement a backtester that takes one or more portfolio estimator objects,
# possibly a rebalance policy and transaction costs

from functools import partial
from typing import Union, Tuple, List, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.base import MetaEstimatorMixin, BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted  # type: ignore

from skportfolio._base import PortfolioEstimator
from skportfolio._constants import APPROX_DAYS_PER_YEAR
from skportfolio.backtest.fees import TransactionCostsFcn, basic_percentage_fee


class Backtester(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
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
        cash_borrow_rate: float = 0.0,
        rates_frequency: int = APPROX_DAYS_PER_YEAR,
    ):
        """
        Principal vectorized backtester for use with any PortfolioEstimator object.
        It runs the strategy intended by the PortfolioEstimator and periodically calculates a
        rebalance collecting all the positions over the complete interval.

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

        # private internals variables
        self._asset_names: Optional[List[str]] = None
        self._transaction_cost_fcn: Optional[TransactionCostsFcn] = None
        self._min_window_size: Optional[int] = 0
        self._max_window_size: Optional[int] = None

        # attributes to be read after fit
        self.positions_: Optional[pd.DataFrame] = None
        self.equity_curve_: Optional[pd.Series] = None
        self.returns_: Optional[pd.Series] = None
        self.rebalance_dates_: Optional[List] = []
        self.buy_sell_costs_: Optional[pd.DataFrame] = None
        self.turnover_: Optional[pd.Series] = None

    def fit(self, X, y=None, **fit_params) -> "Backtester":
        asset_returns = X.pct_change().dropna()
        self._asset_names = X.columns.tolist()
        n_samples, n_assets = X.shape

        # private variables conversion logic
        if isinstance(self.window_size, (list, tuple)):
            self._min_window_size = self.window_size[0]
            self._max_window_size = self.window_size[1]
        elif isinstance(self.window_size, int):
            # in case 0 is specified, add one for having at least one row of data, hence expanding window
            self._min_window_size, self._max_window_size = (
                self.window_size,
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
            initial_positions = pd.Series(
                {"CASH": self.initial_portfolio_value}
                | dict(zip(self._asset_names, (0,) * n_assets))
            )
        else:
            initial_positions = self.initial_portfolio_value * pd.Series(
                {
                    "CASH": 1 - self.initial_weights.sum(),
                    **self.initial_weights.to_dict(),
                }
            )

        self.buy_sell_costs_ = pd.DataFrame(
            columns=["buy", "sell"], index=self.rebalance_dates_, data=0
        )
        self.turnover_ = pd.Series(dtype=float, index=self.rebalance_dates_, data=0)
        self.returns_ = pd.Series(dtype=float, index=X.index, data=np.NAN)

        positions: List[pd.Series] = [initial_positions]
        # initialize
        previous_positions = initial_positions
        for idx in range(n_samples - 1):
            start_positions = previous_positions
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
            # start_date = X.index[idx]
            end_date = X.index[idx + 1]
            end_positions = start_positions * row_returns
            end_portfolio_value = end_positions.sum()
            end_asset_weights = end_positions.div(end_portfolio_value)

            needs_rebalance = ((idx + 1) % self.rebalance_frequency == 0) and (idx > 0)
            if needs_rebalance:
                # check we have enough data
                valid_window = idx - self._min_window_size >= 0
                if valid_window:
                    # call the estimator of this slice of data
                    end_asset_weights_new: pd.Series = self.estimator.fit(
                        X.iloc[:idx, :], y, **fit_params
                    ).weights_
                    delta_weights: pd.Series = end_asset_weights_new - end_asset_weights
                    self.turnover_.loc[end_date] = delta_weights.abs().sum() * 0.5
                    buy_cost, sell_cost = self._transaction_cost_fcn(
                        delta_weights[self._asset_names] * end_portfolio_value
                    )
                    self.buy_sell_costs_.loc[end_date, :] = (buy_cost, sell_cost)
                    # pre_fees_portfolio_value = end_portfolio_value
                    end_portfolio_value -= buy_cost + sell_cost

                    # update end_position after transaction fees
                    end_asset_weights = end_asset_weights_new

            # needs to recompute the cash component
            end_asset_weights["CASH"] = 1 - end_asset_weights.sum()
            end_positions = end_portfolio_value * end_asset_weights
            # this operation is faster than .loc
            positions.append(end_positions)
            self.returns_.loc[end_date] = (
                end_portfolio_value / start_portfolio_value - 1
            )
            previous_positions = end_positions

        # Finally converts the positions in a dataframe
        self.positions_ = pd.DataFrame(positions, index=X.index[self.warmup_period :])
        # and computes the equity curve
        self.equity_curve_ = self.positions_.sum(1)
        return self

    def fit_predict(self, X, y, **fit_kwargs):
        return self.fit(X, y, **fit_kwargs).positions_

from typing import Union, Tuple, List, Optional, Callable
import pandas as pd
from skportfolio import PortfolioEstimator


class Strategy:
    def __init__(
        self,
        estimator: PortfolioEstimator,
        initial_weights: pd.Series,
        initial_portfolio_value: float,
        rebalance_frequency: Union[int, str],
        lookback_periods: Union[
            int,
            Tuple[int, int],
            pd.offsets.BaseOffset,
            Tuple[pd.offsets.BaseOffset, pd.offsets.BaseOffset],
        ],
        transaction_costs_fcn: Callable[
            [
                pd.Series,
            ],
            Tuple[float, float],
        ],
        management_fee_schedule: Optional[Tuple[pd.DatetimeIndex, float]] = None,
    ) -> None:
        """

        Parameters
        ----------
        initial_weights
        estimator: PortfolioEstimator
            Portfolio estimator to evaluate the strategy. It works as a rebalance function that takes current weights in
            and returns new weights out.
        initial_portfolio_value: float
            Initial value
        rebalance_frequency: Union[int,str]
            If integer, number of rows in the input data to trigger a rebalance
        lookback_periods:
            Minimum and maximum lookback periods. If a scalar is specified the same value is set for both
            fee_scheme
        transaction_costs_fcn:
            Function to compute the fees
        management_fee_schedule:
            The portfolio management fee schedule with the percentage of the cost
            from total portfolio value to subtract from total value
        """
        self.estimator = estimator
        self.initial_weights = initial_weights
        self.initial_portfolio_value = initial_portfolio_value
        self.rebalance_frequency = rebalance_frequency
        self.lookback_periods = lookback_periods
        self.transaction_costs_fcn = transaction_costs_fcn
        self.management_fee_schedule = management_fee_schedule

        # Populated by calls to fit method on **all** initial dataset.
        self._last_position: Optional[pd.Series] = None
        self.all_positions_: pd.DataFrame = pd.DataFrame(columns=initial_weights.index)
        self.all_weights_: List[pd.Series] = []
        # Populated by calls to partial_fit method
        self.transaction_costs_: List[Tuple[float, float]] = [
            (0, 0)
        ]  # first value is zero as we enter the portfolio
        self.positions_: Optional[pd.DataFrame] = None
        self.positions_at_rebalance_: Optional[pd.DataFrame] = pd.DataFrame(
            columns=initial_weights.index
        )
        self.rebalance_dates_ = []
        self._current_value = None
        self.initial_value = None
        self.turnover_: List[float] = []

    def step(self, X, y=None, **kwargs):
        warmup_step = kwargs.get("warmup_step", False)
        trigger_rebalance = kwargs.get("trigger_rebalance", False)
        if warmup_step:
            weights = self.initial_weights
            # prepare the next weights
            self.estimator.fit(X, y, **kwargs)
            self.all_weights_.append(weights)
            self.initial_value = X.iloc[-1, :].sum()
            self._last_position = (
                weights * self.initial_portfolio_value / self.initial_value
            ) * self.initial_value
            self.positions_at_rebalance_.loc[X.index[-1]] = self._last_position
        else:
            if not trigger_rebalance:
                weights = self.estimator.weights_
            else:  # otherwise use the old weights
                old_weights = self.estimator.weights_
                weights = self.estimator.fit(X, y, **kwargs).weights_
                delta_pos = (
                    (weights - old_weights)
                    * self.initial_portfolio_value
                    / self.initial_value
                ) * self._current_value
                self.rebalance_dates_.append(X.index[-1])
                self.turnover_.append((weights - old_weights).abs().sum() * 0.5)
                self.transaction_costs_.append(self.transaction_costs_fcn(delta_pos))

        self._current_value = X.iloc[-1, :].sum()
        self.all_weights_.append(weights)
        # calculates the position
        self._last_position = (
            weights * self.initial_portfolio_value / self.initial_value
        ) * self._current_value - sum(self.transaction_costs_[-1])
        # TODO implement the handling of management fees, with approximation to closest date
        # in case a rebalancing just happened, take into consideration the transaction costs
        self.all_positions_.loc[
            X.index[-1], :
        ] = self._last_position - trigger_rebalance * sum(
            self.transaction_costs_[-1]
        )  # sums both the buy and sell costs
        return self

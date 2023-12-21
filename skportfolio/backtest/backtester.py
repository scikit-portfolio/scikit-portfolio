"""
Implementation of a vectorial backtester taking a single portfolio estimator
and other parameters as the rebalancing frequency and transaction costs, to produce
an equity curve as result
"""
from typing import Union, Tuple, List, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.base import MetaEstimatorMixin, RegressorMixin
from tqdm.auto import tqdm

from skportfolio.backtest.rebalance import BacktestWindow, prepare_window
from skportfolio._base import PortfolioEstimator
from skportfolio._base import assert_is_prices
from skportfolio._constants import APPROX_DAYS_PER_YEAR
from skportfolio._simple import EquallyWeighted
from skportfolio.backtest.positions import (
    prepare_initial_positions,
    compute_row_returns,
)
from skportfolio.backtest.rebalance import (
    prepare_rebalance_signal,
    BacktestRebalancingFrequencyOrSignal,
)
from skportfolio.backtest.score import BacktestScorer, prepare_score_fcn
from skportfolio.backtest.transaction_costs import (
    TransactionCostsFcn,
    BacktestTransactionCosts,
    prepare_buy_sell_costs,
    prepare_turnover,
    prepare_transaction_costs_function,
)
from skportfolio.logger import get_logger
from skportfolio.metrics import (
    annualize_rets,
    annualize_vol,
    sortino_ratio,
    maxdrawdown,
    profit_factor,
)


logger = get_logger()


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
        rebalance_frequency_or_signal: BacktestRebalancingFrequencyOrSignal = 1,
        initial_weights: Optional[pd.Series] = None,
        initial_portfolio_value: float = 10_000,
        warmup_period: int = 0,
        window_size: BacktestWindow = None,
        transaction_costs: BacktestTransactionCosts = 0.0,
        transactions_budget: float = 0.1,
        risk_free_rate: float = 0.0,
        cash_borrow_rate: float = 0.0,
        rates_frequency: int = APPROX_DAYS_PER_YEAR,
        scorer: BacktestScorer = None,
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
        rebalance_frequency_or_signal: BacktestRebalancingFrequencyOrSignal, default=1
            If an integer is passed it is the number of rows after when to trigger a
            portfolio rebalance.
            If a string is passed it can be one of:
                - "day"
                - "week"
                -"month"
                - "quarter"
                - "semester"
                - "yearly"
                - "Bday"
                - "bmonth"
                - "Bbmonth"
                - "Bemonth"
                - "bquarter"
            If a Pandas.offset is passed is the frequency offset to which trigger the rebalance.
            If a pandas Series is passed, everything else is ignored and that is used to trigger
            the rebalance events

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
        scorer: Callable
            The score function to evaluate the strategy performance.
            When None, default is set to Sharpe Ratio scorer
        show_progress: bool, default False
            Whether to show the actual progress
        """

        self.estimator = estimator
        self.name = name
        self.rebalance_frequency_or_signal = rebalance_frequency_or_signal
        self.initial_weights = initial_weights
        self.initial_portfolio_value = initial_portfolio_value
        self.warmup_period = warmup_period
        self.window_size = window_size
        self.transaction_costs = transaction_costs
        self.transactions_budget = transactions_budget
        self.risk_free_rate = risk_free_rate
        self.cash_borrow_rate = cash_borrow_rate
        self.rates_frequency = rates_frequency
        self.scorer = scorer
        self.show_progress = show_progress

        # private internals variables
        self._asset_names: Optional[List[str]] = None
        self._transaction_cost_fcn: Optional[TransactionCostsFcn] = None

        self.__reset__()

    def __reset__(self):
        # attributes to be read after fit
        self.positions_: Optional[pd.DataFrame] = None
        self.equity_curve_: Optional[pd.Series] = None
        self.returns_: Optional[pd.Series] = None
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

        Returns
        -------
        self
        """
        if self.estimator is None:
            self.estimator = EquallyWeighted()
        assert_is_prices(X)
        n_samples, n_assets = X.shape
        asset_returns = X.pct_change().fillna(0.0)

        self._asset_names = X.columns.tolist()
        if "CASH" in self._asset_names:
            raise ValueError(
                "CASH is reserved for liquidity hence cannot be an asset name. Please "
                "rename your dataframe column."
            )

        # private variables conversion logic
        # 1. define the rolling-expanding window size
        min_window_size, max_window_size = prepare_window(
            self.window_size, n_samples=n_samples
        )
        # 2. define the transaction cost functions starting from the user-provided
        # transaction cost argument
        transaction_costs_fcn = prepare_transaction_costs_function(
            transaction_costs=self.transaction_costs
        )
        # 3. define the initial positions, either using the user-provided initial
        # weights or by complete initialization
        initial_positions = prepare_initial_positions(
            initial_weights=self.initial_weights,
            n_assets=n_assets,
            initial_portfolio_value=self.initial_portfolio_value,
            asset_names=self._asset_names,
        )
        # 4. computes the rebalance signal and the list of rebalancing events
        self.rebalance_signal_, self.rebalance_events_ = prepare_rebalance_signal(
            rebalance_frequency_or_signal=self.rebalance_frequency_or_signal,
            index=X.index,
        )

        # 5. define the scoring function starting from the user-provided scorer
        self.score_fcn = prepare_score_fcn(score_fcn=self.scorer)

        # 6. define the logics of buy and sell as well as the turnover
        self.buy_sell_costs_ = prepare_buy_sell_costs()
        self.turnover_ = prepare_turnover()
        self.returns_: pd.Series = pd.Series(
            dtype=float, index=X.index, data=(np.NAN,) * len(X.index)
        )

        # 7. initialize positions
        positions: List[pd.Series] = [initial_positions]
        previous_positions: pd.Series = initial_positions

        # 9. Give a name to this run
        backtester_name = self.name if self.name is not None else str(self.estimator)

        # 8. Start backtesting loop
        with tqdm(
            iterable=range(n_samples - 1),
            desc=f"Backtesting {backtester_name}...",
            disable=not self.show_progress,
            mininterval=1,
        ) as progress:
            for idx in progress:
                # This forms the validation set, as we are using the available weights
                # up to the point where we refit the model on new data and let the process
                # start again.
                next_idx = idx + self.warmup_period + 1
                start_positions = previous_positions
                start_portfolio_value = start_positions.sum()

                # Apply current day's returns and calculate the end-of-row positions
                row_returns = compute_row_returns(
                    idx=idx,
                    asset_returns=asset_returns,
                    start_positions=start_positions,
                    cash_return=self.risk_free_rate,
                    margin_return=self.cash_borrow_rate,
                )
                end_positions = start_positions * row_returns
                end_portfolio_value = end_positions.sum()
                end_asset_weights = end_positions.div(end_portfolio_value)
                end_date = X.index[next_idx]
                # Triggered by a rebalance event, we refit the portfolio estimator
                # to get the new weights based on the last slice of input data
                needs_rebalance = self.rebalance_signal_.iloc[next_idx]
                is_valid_window = (
                    next_idx >= min_window_size
                )  # check we have enough data
                if needs_rebalance and is_valid_window:
                    # call the estimator of this slice of data
                    start_window = next_idx - max_window_size + 1
                    # having window_rows here boils down to selecting training data
                    window_rows = np.arange(max(0, start_window), next_idx + 1)
                    # print(
                    #     f"Fitting estimator on "
                    #     f"{X.index[window_rows[0]].strftime('%Y-%m-%d')}:"
                    #     f"{X.index[window_rows[-1]].strftime('%Y-%m-%d')} - "
                    #     f"{window_rows[0]} - {window_rows[-1]} - "
                    #     f"# {X.index[window_rows].shape[0]} rows"
                    # )
                    end_asset_weights_new: pd.Series = self.estimator.fit(
                        X=X.iloc[window_rows, :][self._asset_names],
                        y=y.iloc[window_rows] if y is not None else None,
                        **fit_params,
                    ).weights_.copy()
                    delta_weights: pd.Series = end_asset_weights_new - end_asset_weights
                    self.turnover_.loc[end_date] = delta_weights.abs().sum() * 0.5
                    # Compute the buy and sell transaction costs using the provided function
                    buy_cost, sell_cost = transaction_costs_fcn(
                        delta_weights[self._asset_names] * end_portfolio_value
                    )
                    self.buy_sell_costs_.loc[end_date, :] = (buy_cost, sell_cost)
                    end_portfolio_value -= abs(buy_cost) + abs(sell_cost)
                    # update end_position after transaction fees
                    end_asset_weights = end_asset_weights_new
                # needs to recompute the cash component
                end_asset_weights["CASH"] = (
                    1.0 - end_asset_weights[self._asset_names].sum()
                )
                end_positions = end_portfolio_value * end_asset_weights
                # this operation is faster than .loc
                positions.append(end_positions)
                self.returns_.loc[end_date] = (
                    end_portfolio_value / start_portfolio_value - 1
                )
                previous_positions = end_positions

        # Finally converts the positions in a dataframe
        self.positions_ = pd.DataFrame(
            data=positions, index=X.index[self.warmup_period :]
        )
        self.returns_.dropna(inplace=True)  # returns have one row less than original
        self.turnover_ = self.turnover_.reindex(
            self.returns_.index.drop_duplicates()
        ).fillna(value=0.0)
        # and finally computes the equity curve based on the sum of the positions over columns
        self.equity_curve_ = self.positions_.sum(axis=1)
        self.buy_sell_costs_ = self.buy_sell_costs_.reindex(
            self.equity_curve_.index
        ).fillna(value=0.0)
        last_position = positions[-1].drop("CASH")
        self.weights_ = last_position.div(last_position.sum())
        return self

    def fit_predict(self, X, y=None, **fit_params):
        """
        Run the backtester on data and returns the equity curve
        Parameters
        ----------
        X
        y
        fit_params

        Returns
        -------

        """
        return self.fit(X=X, y=y, **fit_params).equity_curve_

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
        Calculates the strategy metrics over the entire backtesting history

        Returns
        -------
        pd.Series with all portfolio metrics
        """
        start_date = self.equity_curve_.index.min()
        end_date = self.equity_curve_.index.max()
        duration = end_date - start_date
        equity_final = self.equity_curve_.iloc[-1]
        equity_peak = self.equity_curve_.max()
        return_pct = equity_final / self.initial_portfolio_value

        ann_ret = annualize_rets(self.returns_, frequency=252)
        ann_vol = annualize_vol(self.returns_, frequency=252)
        ann_sharpe = ann_ret / ann_vol
        ann_sortino = sortino_ratio(self.returns_, frequency=252, period="daily")

        all_metrics = {
            "Annualized Sharpe ratio": ann_sharpe,
            "Start date": start_date,
            "End date": end_date,
            "Duration": duration,
            "Equity final": equity_final,
            "Equity peak": equity_peak,
            "Return [%]": return_pct * 100,
            "Annualized return [%]": ann_ret * 100,
            "Annualized volatility [%]": ann_vol * 100,
            "Annualized Sortino [%]": ann_sortino * 100,
            "Avg drawdown [%]": (self.returns_ - self.returns_.cummax()).mean() * 100,
            "Max drawdown [%]": maxdrawdown(self.returns_) * 100,
            "Profit factor": profit_factor(self.returns_) * 100,
            "Average turnover [%]": self.turnover_.mean() * 100,
            "Total transaction costs $": self.buy_sell_costs_.sum().sum(),
        }
        return pd.Series(all_metrics)

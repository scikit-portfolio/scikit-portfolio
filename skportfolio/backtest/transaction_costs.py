"""
Some functions for the modeling of transaction costs in backtesting
"""
from functools import partial
from typing import Tuple, Union, Sequence, Dict, Callable

import numpy as np
import pandas as pd

FeePctCost = Union[float, Tuple[float, float]]
FeeRelativeScheme = Sequence[FeePctCost]
FeeBuySellRelativeScheme = Dict[str, FeeRelativeScheme]

TransactionCostsFcn = Callable[
    [
        pd.Series,
    ],
    float,
]

BacktestTransactionCosts = Union[float, Tuple[float, float], TransactionCostsFcn]


def basic_percentage_fee(delta_pos, transaction_costs) -> Tuple[float, float]:
    """
    Computes basic percentage transaction costs based on the change in positions.

    Parameters
    ----------
    delta_pos: pd.Series
        Difference in positions.
    transaction_costs: Union[float, Tuple[float, float]]
        Transaction cost as a percentage for buying and selling.

    Returns
    -------
    Tuple[float, float]
        Buy and sell transaction costs.
    """
    if isinstance(transaction_costs, (float, int)):
        transaction_costs = [transaction_costs, transaction_costs]
    return (
        abs(sum(delta_pos * (delta_pos > 0))) * transaction_costs[0],
        abs(sum(delta_pos * (delta_pos < 0))) * transaction_costs[1],
    )


def variable_transaction_costs(
    delta_pos: pd.Series, fee_scheme: Union[FeeRelativeScheme, FeeBuySellRelativeScheme]
) -> Tuple[float, float]:
    """
    Compute scaled transaction costs based on the change in market value of
    each asset after a rebalance.

    For example, if `fee_scheme` is specified as:

    {
        "buy": [(10_000, 0.005), (None, 0.035)],
        "sell": [(1_000, 0.0075), (None, 0.035)]
    }

    Costs are computed at the following rates:

    Buys:
      $0-$10,000 : 0.5%
      $10,000+   : 0.35%

    Sells:
       $0-$1,000  : 0.75%
       $1,000+    : 0.5%

    Parameters
    ----------
    delta_pos: pd.Series
        Difference in positions measured in currency value.
    fee_scheme: Union[FeeRelativeScheme, FeeBuySellRelativeScheme]
        Relative transaction costs specifying buy and sell limits and associated fees.

    Returns
    -------
    Tuple[float, float]
        A tuple representing buy and sell transaction costs.
    """
    # replace None with Inf
    buy = sell = pd.Series(data=0, index=delta_pos.index)

    for limit, fee_pct in fee_scheme["buy"]:
        if limit is None:
            limit = np.inf
        idx_buy = (0 < delta_pos) & (delta_pos < limit)
        buy[idx_buy] = fee_pct * delta_pos[idx_buy]

    for limit, fee_pct in fee_scheme["sell"]:
        if limit is None:
            limit = -np.inf
        idx_sell = (-limit < delta_pos) & (delta_pos < 0)
        sell[idx_sell] = fee_pct * -delta_pos[idx_sell]

    buy_cost = buy.sum()
    sell_cost = sell.sum()
    return buy_cost, sell_cost


def prepare_transaction_costs_function(
    transaction_costs: BacktestTransactionCosts,
) -> TransactionCostsFcn:
    """
    Creates and returns a function that computes transaction costs based on the given input.

    Parameters
    ----------
    transaction_costs: BacktestTransactionCosts
        A specification of transaction costs, which can be a numeric value, a tuple representing
        buy and sell fees, or a custom callable function.

    Returns
    -------
    TransactionCostsFcn
        A function that calculates transaction costs based on the specified input.

    Raises
    ------
    ValueError
        If the provided 'transaction_costs' is not a supported type.
    """
    if isinstance(transaction_costs, (float, int, tuple, list)):
        return partial(basic_percentage_fee, transaction_costs=transaction_costs)
    if callable(transaction_costs):
        return transaction_costs
    raise ValueError("'transaction_costs' is not a supported type")


def prepare_buy_sell_costs() -> pd.DataFrame:
    """
    Utility for convenience. Creates an empty dataframe with buy and sell costs.
    Returns
    -------

    """
    return pd.DataFrame(columns=["buy", "sell"], index=[], data=0.0)


def prepare_turnover() -> pd.Series:
    """
    Utility for convenience. Creates an empty series holding the turnover of assets.
    Returns
    -------

    """
    return pd.Series(index=[], data=0.0, name="turnover")

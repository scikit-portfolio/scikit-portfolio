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
    For example if cost_scheme is
    {
        "buy": [(10_000, 0.005), (None, 0.035)]
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
    delta_pos: Pd.Series
        Difference in positions as measured in currency value
    fee_scheme:
        Relative transaction costs

    Returns
    -------
    Buy and sell transaction costs
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
    Creates a function to be called that computes the transaction costs

    Parameters
    ----------
    transaction_costs: TransactionCosts

    Returns
    -------
    """
    if isinstance(transaction_costs, (float, int, tuple, list)):
        return partial(basic_percentage_fee, transaction_costs=transaction_costs)
    elif callable(transaction_costs):
        return transaction_costs
    else:
        raise ValueError("'transaction_costs' is not a supported type")


def prepare_buy_sell_costs() -> pd.DataFrame:
    return pd.DataFrame(columns=["buy", "sell"], index=[], data=0.0)


def prepare_turnover() -> pd.Series:
    return pd.Series(index=[], data=0.0, name="turnover")

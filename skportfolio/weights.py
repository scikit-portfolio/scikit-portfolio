"""
All functions related to weights manipulation and post-processing
"""

import numpy as np
import pandas as pd
import cvxpy as cp


def redistribute_non_allocatable(
    weights: pd.Series,
    total_portfolio_value: float,
    min_amount: float,
    rank_gamma: float = 0.5,
    min_weight_tolerance: float = 1e-6,
):
    """
    Redistributes the weights of non-allocatable assets (those who once multiplied by the total
    available investment amount are under a `min_amount` threshold, hence tend to remain in the
    non-allocable liquidity)

    Parameters
    ----------
    weights: pd.Series
        Portfolio weights. Long only, sum to 1
    total_portfolio_value: float
        The total available amount to allocate
    min_amount: float
        The minimum amount that can be allocated
    rank_gamma: float
        A regularization term. The higher the more similar the weights to the original weights.
        If rank_gamma=0, the resulting reallocation tends to push all left allocatabble to the asset
        with largest weight. If rank_gamma>>1 then a large portion of the leftovers still remain
        non allocable, given the min_amount constraint.
        Default: 0.5 as from many experiments it looks like with this value, the number of violations
        is smallest with small cosine distance from the original portfolio.
    min_weight_tolerance: float
        The minimum weight to consider as zero

    Returns
    -------
    The weights, redistributed in a way such that the most assets which are closest to the
    allocatable amount tend to increment at the expenses of the smallest weights.
    """
    amounts = weights * total_portfolio_value
    mask = (amounts < min_amount) & (amounts > min_weight_tolerance)
    amount_min = amounts[mask].sort_values(ascending=False)  # descending order
    remaining_weights: pd.Series = weights[mask]

    if remaining_weights.empty:
        return weights

    # we should reallocate some weight to reach minimum amount
    amount_min_rank = amount_min.rank(ascending=False, method="min")
    V = amount_min_rank.values
    # number of asset left to allocate
    n_left = amount_min.shape[0]
    x = cp.Variable(shape=(n_left,), pos=True)
    # here we setup an optimization problem. Weights with low rank are preferred to be
    # incremented, being V based on the ranking
    obj = cp.sum(cp.multiply(V, x))
    if rank_gamma > 0:
        obj += n_left * rank_gamma * cp.sum_squares(remaining_weights.values - x)

    constraints = [x >= 0, x <= 1, cp.sum(x) == remaining_weights.sum()]
    problem = cp.Problem(objective=cp.Minimize(obj), constraints=constraints)
    problem.solve()
    # Now put back the redistributed weights
    relloacated_weights = weights.copy()
    # Put back the redistributed weights
    relloacated_weights.loc[amount_min_rank.index] = x.value
    return relloacated_weights


def discrete_allocation(
    weights: pd.Series,
    latest_prices,
    total_portfolio_value_cents: float,
    multiplier: int = 100,
):
    """
    Allocates the weights into specific amounts of stake currency,
    respecting the fact that final amounts must be integers.
    Parameters
    ----------
    :param weights: pd.Series
    :param latest_prices: pd.DataFrame
    :param total_portfolio_value_cents: float
    :param multiplier: int

    Returns
    -------
    Allocation of assets in terms of integers in stake currency
    """
    from pypfopt import DiscreteAllocation

    lda = DiscreteAllocation(
        weights.to_dict(),
        latest_prices,
        total_portfolio_value=total_portfolio_value_cents * multiplier,
    )
    lda_weights = lda.greedy_portfolio()[0]

    return (
        pd.Series(lda_weights).sort_values(ascending=False).fillna(0) / multiplier
    ) * latest_prices


def clean_weights(weights: pd.Series, cutoff=1e-4, rounding=5):
    """
    Helper method to clean the raw weights, setting any weights whose absolute
    values are below the cutoff to zero, and rounding the rest.

    :param weights: pd.Series
        Portfolio weights
    :param cutoff: the lower bound, defaults to 1e-4
    :type cutoff: float, optional
    :param rounding: number of decimal places to round the weights, defaults to 5.
                     Set to None if rounding is not desired.
    :type rounding: int, optional
    :return: asset weights
    :rtype: OrderedDict
    """
    if weights is None:
        raise AttributeError("Weights not yet computed")
    cleaned_weights = weights.copy()
    cleaned_weights[np.abs(cleaned_weights) < cutoff] = 0
    if rounding is not None:
        if not isinstance(rounding, int) or rounding < 1:
            raise ValueError("rounding must be a positive integer")
        cleaned_weights = cleaned_weights.round(rounding)

    return cleaned_weights

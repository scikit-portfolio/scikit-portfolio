"""
Collection of cardinality based constraints
"""

from typing import List

import cvxpy as cp
from cvxpy.constraints.constraint import Constraint


def asset_cardinality_long(
    x: cp.Variable, eps: float = 1e-6, buy_in_threshold: float = 0.01
) -> List[Constraint]:
    """
    Implements a max number of assets constraint
    Parameters
    ----------
    x
    eps

    Returns
    -------

    """
    y = cp.Variable(n=x.shape[0], boolean=True)
    return [-1 + eps <= (x - y), x - y <= 0, x - y >= buy_in_threshold - 1]


def number_effective_assets(
    x: cp.Variable, effective_number: float
) -> List[Constraint]:
    """
    Constraints the portfolio to have a minimum number of assets.
    https://en.wikipedia.org/wiki/Herfindahl%E2%80%93Hirschman_index
    Parameters
    ----------
    x
    effective_number

    Returns
    -------
    """
    return [cp.sum_squares(x) <= (effective_number ** (-1.0))]

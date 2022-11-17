import cvxpy as cp


def asset_cardinality_long(x, eps=1e-6, buy_in_threshold=0.01):
    """
    Implements a max number of assets constraint
    Parameters
    ----------
    x

    Returns
    -------

    """
    y = cp.Variable(n=x.shape[0], boolean=True)
    return [-1 + eps <= x - y, x - y <= 0, x - y >= buy_in_threshold - 1]

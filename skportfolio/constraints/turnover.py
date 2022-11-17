# here we implement some turnover constraints to be added to portfolio estimators
import cvxpy as cp


def basic_turnover_constraint(w, w0, tau):
    return 0.5 * cp.sum(cp.abs(w - w0)) <= tau

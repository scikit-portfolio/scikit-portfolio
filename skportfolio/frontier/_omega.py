"""
A module for portfolio optimization based on Omega Ratio
"""
import copy
from typing import Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
from pypfopt import objective_functions
from pypfopt.exceptions import OptimizationError

from ._base_frontier import BaseConvexFrontier


class EfficientOmegaRatio(BaseConvexFrontier):
    """
    The EfficientOmegaRatio class allows for optimization along the omega ratio frontier, using the
    formulation from "Optimizing the Omega Ratio using Linear Programming" of Kapsos et al. (2011)
    https://cs.uwaterloo.ca/~yuying/Courses/CS870_2012/Omega_paper_Short_Cm.pdf

    Instance variables:

    - Inputs:

        - ``n_assets`` - int
        - ``tickers`` - str list
        - ``bounds`` - float tuple OR (float tuple) list
        - ``returns`` - pd.DataFrame
        - ``expected_returns`` - np.ndarray
        - ``solver`` - str
        - ``solver_options`` - {str: str} dict


    - Output: ``weights`` - np.ndarray

    Public methods:

    - ``max_omega_ratio()`` maximizes the Omega ratio
    - ``efficient_risk()`` maximises return for a given omega denominator (risk)
    - ``efficient_return()`` minimises denominator (risk) for a given Omega numerator (return)
    - ``min_omega_risk()`` minimizes the omega ratio denominator.
    - ``add_objective()`` adds a (convex) objective to the optimization problem
    - ``add_constraint()`` adds a constraint to the optimization problem
    - ``portfolio_performance()`` calculates the expected return and Omega of the portfolio
    - ``set_weights()`` creates self.weights (np.ndarray) from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(
        self,
        expected_returns: pd.Series,
        returns: pd.DataFrame,
        minimum_acceptable_return: float = 0.0,
        fraction: float = 1.0,
        weight_bounds: Tuple[float, float] = (0, 1),
        solver=None,
        verbose=False,
        solver_options=None,
    ):
        """
        :param expected_returns: expected returns for each asset. Can be None if

                                optimising for semideviation only.
        :type expected_returns: pd.Series, list, np.ndarray
        :param returns: (historic) returns for all your assets (no NaNs).
                                 See ``expected_returns.returns_from_prices``.
        :type returns: pd.DataFrame or np.array
        :param weight_bounds: minimum and maximum weight of each asset OR single min/max pair
                              if all identical, defaults to (0, 1). Must be changed to (-1, 1)
                              for portfolios with shorting.
        :type weight_bounds: tuple OR tuple list, optional
        :param solver: name of solver. list available solvers with: `cvxpy.installed_solvers()`
        :type solver: str
        :param verbose: whether performance and debugging info should be printed, defaults to False
        :type verbose: bool, optional
        :param solver_options: parameters for the given solver
        :type solver_options: dict, optional
        :raises TypeError: if ``expected_returns`` is not a series, list or array
        """
        self.expected_returns = expected_returns
        self.returns = self._validate_returns(returns)
        self.minimum_acceptable_return = minimum_acceptable_return
        self.weight_bounds = weight_bounds
        self.fraction = fraction  # subsample data

        self._max_return_value = None
        # Labels
        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        else:  # use integer labels
            tickers = list(range(len(expected_returns)))

        super().__init__(
            len(tickers),
            tickers,
            weight_bounds,
            solver=solver,
            verbose=verbose,
            solver_options=solver_options,
        )

    def max_omega_ratio(self):
        """
        Maximizes portfolio omega ratio (see docs for further explanation).
        :return: asset weights for the maximum omega ratio portfolio
        :rtype: OrderedDict
        """

        n_samples, n_assets = self.returns.shape

        # in case a smaller subset of historical returns is required we simply take the average
        if self.fraction == 1.0:
            r = self.returns
        else:  # otherwise we rely on the user specific returns subsample
            r = self.returns.sample(frac=self.fraction)

        r_bar = self.expected_returns
        s: cp.Variable = cp.Variable(shape=(n_assets,), name="s")
        q: cp.Variable = cp.Variable(shape=(n_samples,), name="q")
        z: cp.Variable = cp.Variable(name="z")

        w_min, w_max = self.weight_bounds
        self._objective = s.T @ r_bar - z * cp.Constant(
            value=self.minimum_acceptable_return
        )

        for obj in self._additional_objectives:
            self._objective += obj

        self._constraints = (
            [
                q[i] >= self.minimum_acceptable_return * z - s.T @ r.iloc[i, :]
                for i in range(n_samples)
            ]
            + [q[i] >= 0 for i in range(n_samples)]
            + [cp.sum(q) == 1]
            + [cp.sum(s) == z]
            + [(z * w_min) <= s]
            + [s <= (z * w_max)]
            + [s.T @ r_bar >= self.minimum_acceptable_return * z]
            + [z >= 0.0]
        )
        self._opt = cp.Problem(
            objective=cp.Maximize(self._objective), constraints=self._constraints
        )
        try:
            self._opt.solve()
        except (TypeError, cp.DCPError) as e:
            raise OptimizationError from e

        if self._opt.status not in {"optimal", "optimal_inaccurate"}:
            raise OptimizationError("Solver status: {}".format(self._opt.status))

        self.weights = s.value / s.value.sum()
        return self._make_output_weights()

    def efficient_return(self, target_return, market_neutral=False):
        """
        Maximise Omega risk for a target return. The resulting portfolio will have a return
        larger than the target (but not guaranteed to be equal).

        Parameters
        ----------
        target_return: float
             The desired return of the resulting portfolio.
        market_neutral: bool
            Whether the portfolio should be market neutral (weights sum to zero),
            defaults to False. Requires negative lower weight bound.

        Raises
        ------
        ValueError if ``target_return`` is not a positive float
        ValueError: if no portfolio can be found with volatility equal to ``target_volatility``
        ValueError: if ``risk_free_rate`` is non-numeric

        Returns
        -------
        (OrderedDict) asset weights for the efficient risk portfolio
        """
        if (
            not isinstance(target_return, float)
            or (target_return - self.minimum_acceptable_return) < 0
        ):
            raise ValueError(
                "target_return should be a float, larger than 'minimum_acceptable_return'"
            )
        if not self._max_return_value:
            self._max_return_value = copy.deepcopy(self)._max_return()
        if target_return > self._max_return_value:
            raise ValueError(
                "target_return must be lower than the maximum possible return"
            )

        update_existing_parameter = self.is_parameter_defined("target_return")
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value("target_return", target_return)
        else:
            n_samples = self.returns.shape[0]
            ret = (
                objective_functions.portfolio_return(
                    self._w, self.expected_returns, negative=False
                )
                - self.minimum_acceptable_return
            )

            # omega denominator as the problem objective
            self._objective = (
                cp.sum(
                    cp.pos(self.minimum_acceptable_return - self._w.T @ self.returns.T)
                )
                / n_samples
            )

            for obj in self._additional_objectives:
                self._objective += obj

            target_return_par = cp.Parameter(name="target_return", value=target_return)
            self.add_constraint(lambda _: ret >= target_return_par)
        self._make_weight_sum_constraint(is_market_neutral=market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_risk(self, target_risk, market_neutral=False):
        """
        Maximise return for a target risk. The resulting portfolio will have a volatility
        less than the target (but not guaranteed to be equal).

        Parameters
        ----------
        target_risk: float
             The desired maximum risk of the resulting portfolio.
        market_neutral: bool
            Whether the portfolio should be market neutral (weights sum to zero),
            defaults to False. Requires negative lower weight bound.

        Raises
        ------
        ValueError if ``target_risk`` is not a positive float
        ValueError: if no portfolio can be found with Omega risk equal to ``target_risk``
        ValueError: if ``risk_free_rate`` is non-numeric

        Returns
        -------
        (OrderedDict) asset weights for the efficient risk portfolio
        """
        if not isinstance(target_risk, (float, int)) or target_risk < 0:
            raise ValueError("target_risk should be a positive float")

        update_existing_parameter = self.is_parameter_defined("target_risk")
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value("target_risk", target_risk)
        else:
            self._objective = -(
                objective_functions.portfolio_return(self._w, self.expected_returns)
                - self.minimum_acceptable_return
            )
            omega_denominator = (
                cp.sum(
                    cp.pos(self.minimum_acceptable_return - self._w.T @ self.returns.T)
                )
                / self.returns.shape[0]
            )

            for obj in self._additional_objectives:
                self._objective += obj

            target_risk = cp.Parameter(
                name="target_risk", value=target_risk, nonneg=True
            )
            self.add_constraint(lambda _: omega_denominator <= target_risk)
            self._make_weight_sum_constraint(is_market_neutral=market_neutral)
        return self._solve_cvxpy_opt_problem()

    def min_omega_risk(self, market_neutral=False):
        """
        Minimizes the Omega risk.

        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),
                               defaults to False. Requires negative lower weight bound.
        :param market_neutral: bool, optional
        :raises ValueError: if ``target_volatility`` is not a positive float
        :raises ValueError: if no portfolio can be found with volatility equal to ``target_volatility``
        :raises ValueError: if ``risk_free_rate`` is non-numeric
        :return: asset weights for the efficient risk portfolio
        :rtype: OrderedDict
        """

        update_existing_parameter = self.is_parameter_defined("target_risk")
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
        else:
            omega_denominator = (
                cp.sum(
                    cp.pos(self.minimum_acceptable_return - self._w.T @ self.returns.T)
                )
                / self.returns.shape[0]
            )
            self._objective = omega_denominator

            for obj in self._additional_objectives:
                self._objective += obj

            self._make_weight_sum_constraint(is_market_neutral=market_neutral)
        return self._solve_cvxpy_opt_problem()


def omega_reward(
    w: pd.Series, expected_returns: pd.Series, minimum_acceptable_return: float
):
    """
    Calculates the reward of the portfolio in the Omega efficient frontier
    Parameters
    ----------
    w
    expected_returns
    minimum_acceptable_return

    Returns
    -------

    """
    return w.T @ expected_returns - minimum_acceptable_return


def omega_risk(w: pd.Series, returns: pd.Series, minimum_acceptable_return: float):
    """
    Calculates the risk of the portfolio in the Omega efficient frontier
    Parameters
    ----------
    w
    returns
    minimum_acceptable_return

    Returns
    -------

    """
    return np.mean(np.maximum(minimum_acceptable_return - returns.dot(w), 0))

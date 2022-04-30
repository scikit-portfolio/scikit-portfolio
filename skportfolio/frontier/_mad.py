"""
A module for portfolio optimization based on Mean Absolute Deviation
"""
import copy
from typing import Dict
from typing import Optional
from typing import Tuple

import cvxpy as cp
import pandas as pd
from pypfopt import objective_functions

from ._base_frontier import BaseConvexFrontier


class EfficientMeanAbsoluteDeviation(BaseConvexFrontier):
    """
    The EfficientMeanAbsoluteDeviation class allows for optimization along the Mean Absolute Deviation frontier.

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

    - ``min_mad()`` minimizes the Mean Absolute Deviation (MAD).
    - ``tangency()`` maximizes the portfolio return to MAD ratio, given a risk_free_rate
    - ``efficient_risk()`` maximises return for a given MAD denominator (risk)
    - ``efficient_return()`` minimises MAD denominator (risk) for a given portfolio return
    - ``add_objective()`` adds a (convex) objective to the optimization problem
    - ``add_constraint()`` adds a constraint to the optimization problem
    - ``set_weights()`` creates self.weights (np.ndarray) from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(
        self,
        expected_returns: pd.Series,
        returns: pd.DataFrame,
        weight_bounds: Tuple[float, float] = (0, 1),
        solver: Optional[str] = None,
        verbose: bool = False,
        solver_options: Optional[Dict[str, str]] = None,
    ):
        """
        Initialization of the MAD portfolio as a convex optimization problem.

        Parameters
        ----------
        expected_returns: pd.Series, list, np.ndarray
            Expected returns for each asset.
        returns: pd.DataFrame or np.ndarray
            Historic returns for all your assets (no NaNs).
        weight_bounds: tuple or tuple list, optional
             Minimum and maximum weight of each asset OR single min/max pair if all identical, defaults to (0, 1).
             Must be changed to (-1, 1) for portfolios with shorting.
        solver: str
            Name of the convex optimization solver. list available solvers with: `cvxpy.installed_solvers()`
        verbose: bool
            Whether performance and debugging info should be printed, defaults to False
        solver_options: dict, optional
            Prameters for the given solver

        Raises
        ------
        TypeError if expected_returns is not a pd.Series, list or array
        """
        super().__init__(
            n_assets=len(expected_returns),
            tickers=expected_returns.index.tolist(),
            weight_bounds=(0, 1),
            solver=solver,
            verbose=verbose,
            solver_options=solver_options,
        )
        self.expected_returns = self._validate_expected_returns(expected_returns)
        self.returns = self._validate_returns(returns)
        self.weight_bounds = weight_bounds
        # needed in the creation of the efficient frontier
        self._max_return_value = None

    def min_mad(self):
        """
        Minimizes portfolio Mean Absolute Deviation. See the documentation page for further support and formulae.

        Returns
        -------
            OrderedDict: Asset weights for the minimum mean absolute deviation portfolio
        """
        n_samples, n_assets = self.returns.shape
        self._objective = (
            cp.sum(cp.abs(self._w @ (self.returns - self.expected_returns).T))
            / n_samples
        )
        self._make_weight_sum_constraint(is_market_neutral=False)
        return self._solve_cvxpy_opt_problem()

    def efficient_return(self, target_return: float, market_neutral: bool = False):
        """
        Calculate the efficient portfolio, minimising MAD-risk for a given target return.

        Parameters
        ----------
        target_return: float
            The desired return of the resulting portfolio.

        market_neutral: bool
            whether the portfolio should be market neutral (weights sum to zero),
            defaults to False. Requires negative lower weight bound.

        Raises
        ------
        ValueError: if ``target_return`` is not a positive float
        ValueError: if no portfolio can be found with return equal to ``target_return``

        Returns
        -------
        asset weights for the MAD portfolio
        """
        if not isinstance(target_return, float) or target_return < 0:
            raise ValueError("target_return should be a positive float")
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
            ret = objective_functions.portfolio_return(
                self._w, self.expected_returns, negative=False
            )

        n_samples, n_assets = self.returns.shape
        # minimize L1-risk
        self._objective = (
            cp.sum(cp.abs(self._w @ (self.returns - self.expected_returns).T))
            / n_samples
        )

        # add other objectives
        for obj in self._additional_objectives:
            self._objective += obj

        target_return_par = cp.Parameter(name="target_return", value=target_return)
        # given bound on target return
        self.add_constraint(lambda _: ret >= target_return_par)
        self._make_weight_sum_constraint(is_market_neutral=market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_risk(self, target_risk: float, market_neutral: bool = False):
        """
        Maximise return for a target MAD-risk. The resulting portfolio will have a volatility
        less than the target (but not guaranteed to be equal).

        :param target_risk: the desired maximum volatility of the resulting portfolio.
        :type target_risk: float
        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),
                               defaults to False. Requires negative lower weight bound.
        :param market_neutral: bool, optional
        :raises ValueError: if ``target_volatility`` is not a positive float
        :raises ValueError: if no portfolio can be found with volatility equal to ``target_volatility``
        :raises ValueError: if ``risk_free_rate`` is non-numeric
        :return: asset weights for the efficient risk portfolio
        :rtype: OrderedDict
        """
        if not isinstance(target_risk, (float, int)) or target_risk < 0:
            raise ValueError("target_risk should be a positive float")

        update_existing_parameter = self.is_parameter_defined("target_risk")
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value("target_risk", target_risk)
        else:
            # maximize return
            self._objective = objective_functions.portfolio_return(
                self._w, self.expected_returns, negative=True
            )

            for obj in self._additional_objectives:
                self._objective += obj

            T = self.returns.shape[0]
            mad_risk = (
                cp.sum(cp.abs(self._w @ (self.returns - self.expected_returns).T)) / T
            )
            target_risk = cp.Parameter(
                name="target_risk", value=target_risk, nonneg=True
            )
            self.add_constraint(lambda _: mad_risk <= target_risk)
            self._make_weight_sum_constraint(is_market_neutral=market_neutral)
        return self._solve_cvxpy_opt_problem()

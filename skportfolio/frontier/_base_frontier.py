import warnings
from abc import ABCMeta
from collections import OrderedDict

import cvxpy as cp
import numpy as np
import pandas as pd
from pypfopt import objective_functions
from pypfopt.base_optimizer import BaseConvexOptimizer
from pypfopt.exceptions import InstantiationError


class BaseConvexFrontier(BaseConvexOptimizer, metaclass=ABCMeta):
    def _make_weight_sum_constraint(self, is_market_neutral):
        """
        Helper method to make the weight sum constraint. If market neutral,
        validate the weights proided in the constructor.
        """
        if is_market_neutral:
            # Check and fix bounds
            portfolio_possible = np.any(self._lower_bounds < 0)
            if not portfolio_possible:
                warnings.warn(
                    "Market neutrality requires shorting - bounds have been amended",
                    RuntimeWarning,
                )
                self._map_bounds_to_constraints((-1, 1))
                # Delete original constraints
                del self._constraints[0]
                del self._constraints[0]

            self.add_constraint(lambda w: cp.sum(w) == 0)
        else:
            self.add_constraint(lambda w: cp.sum(w) == 1)
        self._market_neutral = is_market_neutral

    @staticmethod
    def _validate_expected_returns(expected_returns):
        if expected_returns is None:
            return None
        elif isinstance(expected_returns, pd.Series):
            return expected_returns.values
        elif isinstance(expected_returns, list):
            return np.array(expected_returns)
        elif isinstance(expected_returns, np.ndarray):
            return expected_returns.ravel()
        else:
            raise TypeError("expected_returns is not a series, list or array")

    @staticmethod
    def _validate_risk_matrix(risk_matrix):
        if risk_matrix is None:
            raise ValueError("cov_matrix must be provided")
        elif isinstance(risk_matrix, pd.DataFrame):
            return risk_matrix.values
        elif isinstance(risk_matrix, np.ndarray):
            return risk_matrix
        else:
            raise TypeError("risk_matrix is not a dataframe or array")

    def _validate_returns(self, returns):
        """
        Helper method to validate daily returns (needed for some efficient frontiers)
        """
        if not isinstance(returns, (pd.DataFrame, np.ndarray)):
            raise TypeError("returns should be a pd.Dataframe or np.ndarray")

        returns_df = pd.DataFrame(returns)
        if returns_df.isnull().values.any():
            warnings.warn(
                "Removing NaNs from returns",
                UserWarning,
            )
            returns_df = returns_df.dropna(axis=0, how="any")

        if self.expected_returns is not None:
            if returns_df.shape[1] != len(self.expected_returns):
                raise ValueError(
                    "returns columns do not match expected_returns. Please check your tickers."
                )

        return returns_df

    def _max_return(self, return_value=True):
        """
        Helper method to maximise return. This should not be used to optimize a portfolio.

        :return: asset weights for the return-minimising portfolio
        :rtype: OrderedDict
        """
        if self.expected_returns is None:
            raise ValueError("no expected returns provided")

        self._objective = objective_functions.portfolio_return(
            self._w, self.expected_returns
        )

        self.add_constraint(lambda w: cp.sum(w) == 1)

        res = self._solve_cvxpy_opt_problem()

        if return_value:
            return -self._opt.value
        else:
            return res

    def _validate_market_neutral(self, market_neutral: bool) -> None:
        if self._market_neutral != market_neutral:
            raise InstantiationError(
                "A new instance must be created when changing market_neutral."
            )

    def populate_constraints(self, constraints):
        for cnstr in constraints:
            self.add_constraint(cnstr)
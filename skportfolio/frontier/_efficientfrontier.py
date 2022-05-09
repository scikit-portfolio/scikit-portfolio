import warnings
from abc import ABCMeta
from abc import abstractmethod
from copy import deepcopy
from itertools import product
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from cvxpy.error import SolverError
from joblib import Parallel
from joblib import delayed
from pypfopt import objective_functions
from pypfopt.base_optimizer import BaseConvexOptimizer
from pypfopt.exceptions import OptimizationError
from pypfopt.expected_returns import returns_from_prices
from pypfopt.objective_functions import portfolio_return
from pypfopt.objective_functions import portfolio_variance

from skportfolio._base import PortfolioEstimator
from skportfolio._constants import (
    APPROX_BDAYS_PER_YEAR,
    BASE_RISK_FREE_RATE,
    BASE_TARGET_RETURN,
    BASE_TARGET_RISK,
    BASE_MIN_ACCEPTABLE_RETURN,
)
from skportfolio.frontier._mixins import _EfficientCDarMixin
from skportfolio.frontier._mixins import _EfficientCVarMixin
from skportfolio.frontier._mixins import _EfficientMADMixin
from skportfolio.frontier._mixins import _EfficientMeanVarianceMixin
from skportfolio.frontier._mixins import _EfficientOmegaRatioMixin
from skportfolio.frontier._mixins import _EfficientSemivarianceMixin
from skportfolio.frontier._mixins import _TargetReturnMixin
from skportfolio.frontier._mixins import _TargetRiskMixin
from skportfolio.riskreturn import BaseReturnsEstimator
from skportfolio.riskreturn import BaseRiskEstimator
from skportfolio.riskreturn import MeanHistoricalLinearReturns
from skportfolio.riskreturn import SampleCovariance


class _BaseEfficientFrontierPortfolioEstimator(PortfolioEstimator, metaclass=ABCMeta):
    _min_risk_method_name = ""

    @abstractmethod
    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
    ):
        self.returns_data = returns_data
        self.frequency = frequency
        self.weight_bounds = weight_bounds
        self.l2_gamma = l2_gamma

    @abstractmethod
    def _optimizer(self, X: pd.DataFrame) -> BaseConvexOptimizer:
        raise NotImplementedError(
            "Each derived subclass must implement `_optimizer` method"
        )

    @abstractmethod
    def risk_reward(
        self, model: Optional[BaseConvexOptimizer] = None
    ) -> Tuple[float, float]:
        raise NotImplementedError(
            "Each derived subclass must implement specific `risk_reward`"
        )

    def grid_parameters(self) -> Dict[str, Sequence[Any]]:
        xmin = np.linspace(0, 0.3, 11)
        xmax = np.linspace(0.7, 1, 11)
        all_weight_bounds = list(product(xmin, xmax))
        return {
            "weight_bounds": all_weight_bounds,
            "l2_gamma": np.logspace(-4, 1, 10),
        }


    def _fit_method(self, X, method: str, **kwargs) -> pd.Series:
        """
        Helper method to simplify the call of various frontier methods on _BaseEfficientFrontierPortfolioEstimator
        derived classes. It takes the method name as a string, and runs it on data.

        Parameters
        ----------
        X: pd.DataFrame
            Asset prices or returns
        method: str
            Name of the method to call
        kwargs:
            Additional parameters to the method caller

        Returns
        -------
        Portfolio weights as a pd.Series with indices representing the assets and values representing the percentage
        allocation.

        Warnings
        --------
        Prints a warning if portfolio optimization did not work, hence returning NaN weights for all assets.
        """
        self.model = self._optimizer(X)
        try:
            # calls the method `method` on the self.model PyPortfolioOpt model instance
            getattr(self.model, method)(**kwargs)
            # if everything was ok from the call, returns the cleaned weights as a pd.Series
            return pd.Series(
                self.model.clean_weights(),
                index=X.columns,
                name=str(self),
            )
        except (SolverError, OptimizationError, ValueError) as ex:
            warnings.warn(str(ex))
            return pd.Series(
                data=[np.nan] * X.shape[1],
                index=X.columns,
                name=str(self),
            )

    def estimate_frontier(
        self, X, num_portfolios: int = 20, n_jobs: int = 1, random_seed: int = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimates the efficient frontier given either returns or prices (depending on `self.returns_data` attribute),
        and number of points along the frontier. It always starts from the least risky portfolio and increasing the
        portfolio returns, it looks for the efficient risk portfolio until the riskiest portfolio (and the one with
        the highest return) is met.

        Parameters
        ----------
        X: pd.DataFrame
            The prices or returns to fit multiple times the efficient risk/return portfolio
        num_portfolios: int
            Number of portfolios along the frontier.
        n_jobs: int
            Number of parallel jobs to distribute the calculations. Uses joblib parallel capabilities.
        random_seed:
            Only when the portfolio returns estimator is PerturbedReturns, this is needed to lock the same stochastic
            sample of the expected returns across the entire frontier.
        kwargs:
            - prefer: str
                Default "threads" instead of joblib loky, in case you are using multiple threads
        Returns
        -------
        tuple
            The first two elements represent, the risk and return coordinates of the portfolios along the
            efficient frontier. The last element contains the portfolio weights.
            Portfolios are indexed from the least risky to the maximum return along the frontier.

        Examples
        --------
        If you need to compute the efficient frontier for the Markowitz Mean-Variance portfolio for 128 points using
        16 parallel CPU cores you can simply do:

        >>> from skportfolio import MinimumVolatility
        >>> from skportfolio.datasets import load_tech_stock_prices
        >>> MinimumVolatility().estimate_frontier(load_tech_stock_prices(), num_portfolios=128, n_jobs=16)
        """
        import warnings

        warnings.filterwarnings("error")
        # First estimate the frontier limits in terms of risk and returns.
        n_assets = X.shape[1]
        self.rets_estimator.reseed(random_seed)
        # We must find 2 points with two coordinates of (risk,return), namely
        # (MIN_RISK, RETURN_MIN_RISK) and (RISK_MAX_RETURN, MAX_RETURN)
        # 1. We first find return of minimum risk portfolio
        min_risk_model: BaseConvexOptimizer = self._optimizer(X)
        # fill the internal values of min_risk_model by calling it
        # TODO refactoring this ugly call requires some rework of pyportfolio opt object oriented approach.
        (getattr(min_risk_model, self._min_risk_method_name)())
        # reset it again
        self.rets_estimator.reseed(random_seed)
        min_return_model: BaseConvexOptimizer = self._optimizer(X)
        # this workaround is necessary to get the return of the minimum risk portfolio
        # however there is a catch here, that if the objective function contains regularization its
        # objective value will be distorted
        try:
            min_return_model.efficient_risk(
                    self.risk_function(min_risk_model._opt.value)
                )
        except SolverError:
            return (np.nan * np.zeros(num_portfolios),) * 3
        self.rets_estimator.reseed(random_seed)
        # if fitting was possible finds the return of the minimum risk portfolio
        min_return_value: float = self.risk_reward(min_return_model)[1]
        self.rets_estimator.reseed(random_seed)
        # 2. Then we find risk of maximum return portfolio
        max_return_value = self._optimizer(X)._max_return()
        risks = np.zeros(num_portfolios)
        # the returns y-axis
        returns = np.linspace(
            start=min_return_value, stop=max_return_value, num=num_portfolios
        )
        # 3. for every return value, find the portfolio with best risk along the frontier
        frontier_weights = np.zeros((num_portfolios, len(X.columns))) * np.nan

        def get_risk_weights(r: float):
            self.rets_estimator.reseed(random_seed)
            _model_risk = self._optimizer(X)
            try:
                _model_risk.efficient_return(r)
            except (SolverError, ValueError, OptimizationError):
                return [np.nan, np.nan * np.zeros(n_assets)]
            return self.risk_reward(_model_risk)[0], _model_risk.weights

        if n_jobs is not None and n_jobs != 1:
            # perform embarassingly parallel operations
            out = Parallel(n_jobs=n_jobs, prefer=kwargs.get("prefer", None))(delayed(get_risk_weights)(r) for r in returns)
            risks = np.array([v[0] for v in out])
            frontier_weights = np.array([v[1] for v in out])
        else:
            # do not even use joblib and sequentially creates all the points on the frontier
            for i, ret in enumerate(returns):
                risks[i], frontier_weights[i, :] = get_risk_weights(ret)
        # portfolio risks (x-coordinate), portfolio returns (y-coordinate) and associated portfolio weights
        return risks, returns, frontier_weights

    def add_constraints(self, cnstr: Union[Callable, List[Callable]]):
        """
        Add a list of constraints to the convex optimization problem
        Parameters
        ----------
        cnstr: list
            List of constraints, to be added to the optimization problem

        Returns
        -------
        object
            The portfolio estimator object
        """
        if isinstance(cnstr, list):
            self.constraints = cnstr
        else:
            raise TypeError("Only list of constraints supported")
        return self


class _BaseMeanVariancePortfolio(
    _BaseEfficientFrontierPortfolioEstimator,
    _EfficientMeanVarianceMixin,
    metaclass=ABCMeta,
):
    _min_risk_method_name = "min_volatility"

    @abstractmethod
    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        risk_estimator: BaseRiskEstimator = SampleCovariance(),
    ):
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            weight_bounds=weight_bounds,
            l2_gamma=l2_gamma,
        )
        self.rets_estimator = rets_estimator
        self.risk_estimator = risk_estimator

    def _optimizer(self, X) -> BaseConvexOptimizer:
        """
        Creates the base optimizer as from pypfopt. Can be one of
            EfficientFrontier
            EfficientCVaR,
            EfficientSemivariance,
            EfficientCDaR
        Parameters
        ----------
        X: pd.DataFrame
            The price data to be fed
        Returns
        -------
            The optimizer model
        """

        expected_returns = (
            self.rets_estimator.set_returns_data(self.returns_data)
            .set_frequency(self.frequency)
            .fit(X)
            .expected_returns_
        )
        cov_matrix = (
            self.risk_estimator.set_returns_data(self.returns_data)
            .set_frequency(self.frequency)
            .fit(X)
            .risk_matrix_
        )
        eff_front_model = self._get_model(
            expected_returns=expected_returns,
            risk_matrix=cov_matrix,
            weight_bounds=self.weight_bounds,
        )
        if hasattr(self, "constraints"):
            for cnstr in self.constraints:
                eff_front_model.add_constraint(cnstr)

        if self.l2_gamma > 0:
            eff_front_model.add_objective(
                objective_functions.L2_reg, gamma=self.l2_gamma
            )
        return deepcopy(eff_front_model)

    def risk_reward(self, model: BaseConvexOptimizer = None):
        """
        Computes the risk-return for the current portfolio
        Returns
        -------
        """
        if model is None:
            model = self.model
        if model.weights is None or not hasattr(model, "weights"):
            raise AttributeError("Must fit model successfully to define weights")
        try:
            ret = portfolio_return(
                model.weights, model.expected_returns, negative=False
            )
            volatility = np.sqrt(portfolio_variance(model.weights, model.cov_matrix))
            return volatility, ret
        except Exception as opt_err:
            return np.nan, np.nan

    def grid_parameters(self) -> Dict[str, Sequence[Any]]:
        params = super().grid_parameters()
        # params["rets_estimator"] = all_returns_estimators
        # params["risk_estimator"] = all_risk_estimators
        return params


class MinimumVolatility(_BaseMeanVariancePortfolio):
    """
    Minimum volatility (minimum variance) portfolio. It does not need estimates of the expected returns, but only
    estimation of the covariance matrix.
    The following optimization problem is being solved.
    """

    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        risk_estimator: BaseRiskEstimator = SampleCovariance(),
    ):
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            weight_bounds=weight_bounds,
            l2_gamma=l2_gamma,
            risk_estimator=risk_estimator,
        )

    def fit(self, X, y=None) -> PortfolioEstimator:
        self.weights_ = self._fit_method(X, method="min_volatility")
        return self


class MaxSharpe(_BaseMeanVariancePortfolio):
    """
    Maximum Sharpe ratio portfolio. It maximizes the ratio between the risk-corrected portfolio expected return and
    returns volatility.
    By default here we use risk-free asset value of 0.
    """

    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        risk_estimator: BaseRiskEstimator = SampleCovariance(),
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        risk_free_rate=BASE_RISK_FREE_RATE,
    ):
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            weight_bounds=weight_bounds,
            l2_gamma=l2_gamma,
            rets_estimator=rets_estimator,
            risk_estimator=risk_estimator,
        )
        self.risk_free_rate = risk_free_rate

    def fit(self, X, y=None) -> PortfolioEstimator:
        self.weights_ = self._fit_method(
            X, "max_sharpe", risk_free_rate=self.risk_free_rate
        )
        return self


class MeanVarianceEfficientRisk(_TargetRiskMixin, _BaseMeanVariancePortfolio):
    """
    Efficient risk in the Markowitz's Mean Variance framework.
    """

    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        risk_estimator: BaseRiskEstimator = SampleCovariance(),
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        target_risk: float = BASE_TARGET_RISK,
    ):
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            weight_bounds=weight_bounds,
            l2_gamma=l2_gamma,
            rets_estimator=rets_estimator,
            risk_estimator=risk_estimator,
        )
        self.target_risk = target_risk

    def fit(self, X, y=None, **kwargs) -> PortfolioEstimator:
        self.weights_ = self._fit_method(
            X, "efficient_risk", target_volatility=self.target_risk
        )
        return self


class MeanVarianceEfficientReturn(_TargetReturnMixin, _BaseMeanVariancePortfolio):
    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        risk_estimator: BaseRiskEstimator = SampleCovariance(),
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        target_return: float = BASE_TARGET_RETURN,
    ):
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            weight_bounds=weight_bounds,
            l2_gamma=l2_gamma,
            rets_estimator=rets_estimator,
            risk_estimator=risk_estimator,
        )
        self.target_return = target_return

    def fit(self, X, y=None, **kwargs) -> PortfolioEstimator:
        self.weights_ = self._fit_method(
            X, "efficient_return", target_return=self.target_return
        )
        return self


class _BaseMeanSemiVariancePortfolio(
    _EfficientSemivarianceMixin,
    _BaseEfficientFrontierPortfolioEstimator,
    metaclass=ABCMeta,
):
    _min_risk_method_name = "min_semivariance"

    @abstractmethod
    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
    ):
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            weight_bounds=weight_bounds,
            l2_gamma=l2_gamma,
        )
        self.frequency = frequency
        self.rets_estimator = rets_estimator

    def _optimizer(self, X) -> BaseConvexOptimizer:
        # compute expected returns using the expected returns estimator
        expected_returns = (
            self.rets_estimator.set_returns_data(self.returns_data)
            .set_frequency(self.frequency)
            .fit(X)
            .expected_returns_
        )
        # create the model
        eff_front_model = self._get_model(
            expected_returns=expected_returns,
            returns=X if self.returns_data else returns_from_prices(X),
            weight_bounds=self.weight_bounds,
            frequency=self.frequency,
        )
        if hasattr(self, "constraints"):
            for cnstr in self.constraints:
                eff_front_model.add_constraint(cnstr)
        if self.l2_gamma > 0:
            eff_front_model.add_objective(
                objective_functions.L2_reg, gamma=self.l2_gamma
            )
        return deepcopy(eff_front_model)

    def risk_reward(self, model: BaseConvexOptimizer = None):
        if model is None:
            model = self.model
        if model.weights is None or not hasattr(model, "weights"):
            raise AttributeError("Must fit model successfully to define weights")
        ret = portfolio_return(model.weights, model.expected_returns, negative=False)
        portfolio_returns = model.returns @ model.weights
        drops = np.fmin(portfolio_returns - model.benchmark, 0)
        semivariance = np.sum(np.square(drops)) / model._T * model.frequency
        semi_deviation = np.sqrt(semivariance)
        return semi_deviation, ret


class MinimumSemiVolatility(
    _BaseMeanSemiVariancePortfolio,
):
    """
    Minimum semi-volatility portfolio.
    """

    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
    ):
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            weight_bounds=weight_bounds,
            l2_gamma=l2_gamma,
        )

    def fit(self, X, y=None) -> PortfolioEstimator:
        self.weights_ = self._fit_method(X, "min_semivariance")
        return self


class MeanSemiVarianceEfficientRisk(_TargetRiskMixin, _BaseMeanSemiVariancePortfolio):
    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        target_risk: float = BASE_TARGET_RISK,
    ):
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            weight_bounds=weight_bounds,
            l2_gamma=l2_gamma,
        )
        self.target_risk = target_risk

    def fit(self, X, y=None) -> PortfolioEstimator:
        self.weights_ = self._fit_method(
            X, "efficient_risk", target_semideviation=self.target_risk
        )
        return self


class MeanSemiVarianceEfficientReturn(
    _TargetReturnMixin, _BaseMeanSemiVariancePortfolio
):
    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        target_return: float = BASE_TARGET_RETURN,
    ):
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            weight_bounds=weight_bounds,
            l2_gamma=l2_gamma,
        )
        self.target_return = target_return

    def fit(self, X, y=None) -> PortfolioEstimator:
        self.weights_ = self._fit_method(
            X, "efficient_return", target_return=self.target_return
        )
        return self


class _BaseCVarCDarEstimator(
    _BaseEfficientFrontierPortfolioEstimator,
    metaclass=ABCMeta,
):
    """
    Efficient Conditional Value At risk/ Conditional Drawdown at risk portfolio base class
    It implements both the two variants, the mechanism of specification of PyPortfolioOPT model is left
    to the mixin classes _EfficientCVarMixin and _EfficientCDarMixin
    """

    @abstractmethod
    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        beta: float = 0.95,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        market_neutral: bool = False,
    ):
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            weight_bounds=weight_bounds,
            l2_gamma=l2_gamma,
        )
        self.beta = beta
        self.rets_estimator = rets_estimator
        self.market_neutral = market_neutral

    def _optimizer(self, X) -> BaseConvexOptimizer:
        expected_returns = (
            self.rets_estimator.set_returns_data(self.returns_data)
            .set_frequency(self.frequency)
            .fit(X)
            .expected_returns_
        )

        eff_front_model = self._get_model(
            expected_returns=expected_returns,
            returns=X if self.returns_data else returns_from_prices(X),
            weight_bounds=self.weight_bounds,
            beta=self.beta,
        )
        if hasattr(self, "constraints"):
            for cnstr in self.constraints:
                eff_front_model.add_constraint(cnstr)
        if self.l2_gamma > 0:
            eff_front_model.add_objective(
                objective_functions.L2_reg, gamma=self.l2_gamma
            )
        return deepcopy(eff_front_model)


class _EfficientCVarEstimator(
    _EfficientCVarMixin, _BaseCVarCDarEstimator, metaclass=ABCMeta
):
    """
    Virtually the same implementation as of CVAR, changing inheritance we overwrite the self.model with the
    EfficientCDAR optimizer, so nothing else is required.
    The replacement of self.model is done by the order of inheritance, using _EfficientCVarMixin as the first
    inherited class
    """

    _min_risk_method_name = "min_cvar"

    def risk_reward(self, model: Optional[BaseConvexOptimizer] = None):
        if model is None:
            model = self.model
        if model.weights is None or not hasattr(model, "weights"):
            raise AttributeError("Must fit model successfully to define weights")
        ret = portfolio_return(model.weights, model.expected_returns, negative=False)
        cvar = model._alpha.value + 1.0 / (
            len(model.returns) * (1 - model._beta)
        ) * np.sum(model._u.value)
        return cvar, ret


class MinimumCVar(_EfficientCVarEstimator):
    """
    Minimum Conditional Var portfolio
    """

    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        beta: float = 0.95,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        market_neutral: bool = False,
    ):
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            l2_gamma=l2_gamma,
            weight_bounds=weight_bounds,
            beta=beta,
            rets_estimator=rets_estimator,
            market_neutral=market_neutral,
        )

    def fit(self, X, y=None) -> PortfolioEstimator:
        self.weights_ = self._fit_method(
            X, "min_cvar", market_neutral=self.market_neutral
        )
        return self


class CVarEfficientRisk(_TargetRiskMixin, _EfficientCVarEstimator):
    """
    Efficient CVar at given target return
    """

    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        beta: float = 0.95,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        market_neutral: bool = False,
        target_risk: float = BASE_TARGET_RISK,
    ):
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            l2_gamma=l2_gamma,
            weight_bounds=weight_bounds,
            beta=beta,
            rets_estimator=rets_estimator,
            market_neutral=market_neutral,
        )
        self.target_risk = target_risk

    def fit(self, X, y=None) -> PortfolioEstimator:
        self.weights_ = self._fit_method(
            X,
            "efficient_risk",
            target_cvar=self.target_risk,
            market_neutral=self.market_neutral,
        )
        return self


class CVarEfficientReturn(_TargetReturnMixin, _EfficientCVarEstimator):
    """
    Efficient return at given target CVar
    """

    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        beta: float = 0.95,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        market_neutral: bool = False,
        target_return: float = BASE_TARGET_RETURN,
    ):
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            l2_gamma=l2_gamma,
            weight_bounds=weight_bounds,
            beta=beta,
            rets_estimator=rets_estimator,
            market_neutral=market_neutral,
        )
        self.target_return = target_return

    def fit(self, X, y=None) -> PortfolioEstimator:
        self.weights_ = self._fit_method(
            X,
            "efficient_return",
            target_return=self.target_return,
            market_neutral=self.market_neutral,
        )
        return self


class _EfficientCDarEstimator(
    _EfficientCDarMixin, _BaseCVarCDarEstimator, metaclass=ABCMeta
):
    """
    Virtually the same implementation as of CVAR, changing inheritance we overwrite the self.model with the
    EfficientCDAR optimizer, so nothing else is required.
    The replacement of self.model is done by the order of inheritance, using _EfficientCDarMixin as the first
    inherited class
    """

    _min_risk_method_name = "min_cdar"

    def risk_reward(self, model: Optional[BaseConvexOptimizer] = None):
        if model is None:
            model = self.model
        if model.weights is None or not hasattr(model, "weights"):
            raise AttributeError("Must fit model successfully to define weights")
        ret = portfolio_return(model.weights, model.expected_returns, negative=False)
        cdar = model._alpha.value + 1.0 / (
            len(model.returns) * (1 - model._beta)
        ) * np.sum(model._z.value)
        return cdar, ret


class MinimumCDar(_EfficientCDarEstimator):
    """
    Minimum Conditional DaR portfolio
    """

    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        beta: float = 0.95,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        market_neutral: bool = False,
    ):
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            l2_gamma=l2_gamma,
            weight_bounds=weight_bounds,
            beta=beta,
            rets_estimator=rets_estimator,
            market_neutral=market_neutral,
        )

    def fit(self, X, y=None) -> PortfolioEstimator:
        self.weights_ = self._fit_method(
            X, "min_cdar", market_neutral=self.market_neutral
        )
        return self


class CDarEfficientRisk(_TargetRiskMixin, _EfficientCDarEstimator):
    """
    Efficient CVar at given target return
    """

    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        beta: float = 0.95,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        market_neutral: bool = False,
        target_risk: float = BASE_TARGET_RISK,
    ):
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            l2_gamma=l2_gamma,
            weight_bounds=weight_bounds,
            beta=beta,
            rets_estimator=rets_estimator,
            market_neutral=market_neutral,
        )
        self.target_risk = target_risk

    def fit(self, X, y=None) -> PortfolioEstimator:
        self.weights_ = self._fit_method(
            X,
            "efficient_risk",
            target_cdar=self.target_risk,
            market_neutral=self.market_neutral,
        )
        return self


class CDarEfficientReturn(_TargetReturnMixin, _EfficientCDarEstimator):
    """
    Efficient return at given target CVar
    """

    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        beta: float = 0.95,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        market_neutral: bool = False,
        target_return: float = BASE_TARGET_RETURN,
    ):
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            l2_gamma=l2_gamma,
            weight_bounds=weight_bounds,
            beta=beta,
            rets_estimator=rets_estimator,
            market_neutral=market_neutral,
        )
        self.target_return = target_return

    def fit(self, X, y=None) -> PortfolioEstimator:
        self.weights_ = self._fit_method(
            X,
            "efficient_return",
            target_return=self.target_return,
            market_neutral=self.market_neutral,
        )
        return self


"""
Omega Ratio portfolios
"""


class _BaseOmegaPortfolio(
    _EfficientOmegaRatioMixin,
    _BaseEfficientFrontierPortfolioEstimator,
    metaclass=ABCMeta,
):
    _min_risk_method_name = "min_omega_risk"

    @abstractmethod
    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency=APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        minimum_acceptable_return: float = BASE_MIN_ACCEPTABLE_RETURN,
    ):
        """
        Base estimator of the omega ratio portfolio.
        Please consider that in using a non-zero minimum acceptable return one should look for returns specified in the
        same frequency units as those of the returns estimator.
        """
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            weight_bounds=weight_bounds,
            l2_gamma=l2_gamma,
        )
        self.rets_estimator = rets_estimator
        self.minimum_acceptable_return = minimum_acceptable_return

    def _optimizer(self, X) -> BaseConvexOptimizer:
        eff_front_model = self._get_model(
            expected_returns=self.rets_estimator.set_returns_data(
                self.returns_data
            ).fit_transform(X),
            returns=X if self.returns_data else returns_from_prices(X),
            frequency=self.frequency,
            minimum_acceptable_return=self.minimum_acceptable_return,
            weight_bounds=self.weight_bounds,
        )
        if hasattr(self, "constraints"):
            for cnstr in self.constraints:
                eff_front_model.add_constraint(cnstr)
        if self.l2_gamma > 0:
            eff_front_model.add_objective(
                objective_functions.L2_reg, gamma=self.l2_gamma
            )
        return deepcopy(eff_front_model)

    def risk_reward(self, model: Optional[BaseConvexOptimizer] = None):
        if model is None:
            model = self.model
        if model.weights is None or not hasattr(model, "weights"):
            raise AttributeError("Must fit model successfully to define weights")
        omega_numerator = (
            portfolio_return(model.weights, model.expected_returns, negative=False)
            - model.minimum_acceptable_return
        )
        omega_denominator = np.mean(
            np.maximum(
                model.minimum_acceptable_return - (model.weights @ model.returns.T),
                0,
            )
        )
        return omega_denominator, omega_numerator


class MaxOmegaRatio(_BaseOmegaPortfolio):
    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency=APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        minimum_acceptable_return: float = BASE_MIN_ACCEPTABLE_RETURN,
    ):
        """
        Estimator of the maximum omega ratio portfolio.
        Please consider that in using a non-zero minimum acceptable return one should look for returns specified in the
        same frequency units as those of the returns estimator.
        """
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            weight_bounds=weight_bounds,
            l2_gamma=l2_gamma,
            rets_estimator=rets_estimator,
            minimum_acceptable_return=minimum_acceptable_return,
        )

    def fit(self, X, y=None) -> PortfolioEstimator:
        self.weights_ = self._fit_method(X, "max_omega_ratio")
        return self


class OmegaEfficientRisk(_TargetRiskMixin, _BaseOmegaPortfolio):
    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency=APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        minimum_acceptable_return: float = BASE_MIN_ACCEPTABLE_RETURN,
        target_risk: float = BASE_TARGET_RISK,
    ):
        """
        Estimator of the maximum omega ratio portfolio.
        Please consider that in using a non-zero minimum acceptable return one should look for returns specified in the
        same frequency units as those of the returns estimator.
        """
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            weight_bounds=weight_bounds,
            l2_gamma=l2_gamma,
            rets_estimator=rets_estimator,
            minimum_acceptable_return=minimum_acceptable_return,
        )
        self.target_risk = target_risk

    def fit(self, X, y=None) -> PortfolioEstimator:
        self.weights_ = self._fit_method(
            X, "efficient_risk", target_risk=self.target_risk
        )
        return self


class OmegaEfficientReturn(_TargetReturnMixin, _BaseOmegaPortfolio):
    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency=APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        minimum_acceptable_return: float = BASE_MIN_ACCEPTABLE_RETURN,
        target_return: float = BASE_TARGET_RETURN,
    ):
        """
        Estimator of the maximum omega ratio portfolio.
        Please consider that in using a non-zero minimum acceptable return one should look for returns specified in the
        same frequency units as those of the returns estimator.
        """
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            weight_bounds=weight_bounds,
            l2_gamma=l2_gamma,
            rets_estimator=rets_estimator,
            minimum_acceptable_return=minimum_acceptable_return,
        )
        self.target_return = target_return

    def fit(self, X, y=None) -> PortfolioEstimator:
        self.weights_ = self._fit_method(
            X, "efficient_return", target_return=self.target_return
        )
        return self


class MinimumOmegaRisk(_BaseOmegaPortfolio):
    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency=APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        minimum_acceptable_return: float = BASE_MIN_ACCEPTABLE_RETURN,
    ):
        """
        Estimator of the maximum omega ratio portfolio.
        Please consider that in using a non-zero minimum acceptable return one should look for returns specified in the
        same frequency units as those of the returns estimator.
        """
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            weight_bounds=weight_bounds,
            l2_gamma=l2_gamma,
            rets_estimator=rets_estimator,
            minimum_acceptable_return=minimum_acceptable_return,
        )

    def fit(self, X, y=None) -> PortfolioEstimator:
        self.weights_ = self._fit_method(X, "min_omega_risk")
        return self


class _BaseMADPortfolio(
    _EfficientMADMixin,
    _BaseEfficientFrontierPortfolioEstimator,
    metaclass=ABCMeta,
):
    _min_risk_method_name = "min_mad"
    """
    Provides the base class for the implementation of portfolios along the
    Mean-Absolute-Deviation efficient frontier.
    It is based on the EfficientMeanAbsoluteDeviation
    """

    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency=APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
    ):
        """
        Estimator of the Mean-Absolute-Deviation portfolio.
        Please consider that in using a non-zero target return one should look for returns specified in the same
        frequency as those of the returns estimator.

        Parameters
        ----------
        returns_data: bool
            Whether input data are historical asset returns or historical asset prices
        l2_gamma: float
            Weights regularization constant for the L2 loss of weights.
        weight_bounds: Tuple[float,float]
            Minimum and maximum allowed weights
        frequency: int
            Number of periods to consider. As typical input data are daily close prices, and annualized
            returns or volatilies are needed, the default value for frequency=252 business days.
        rets_estimator: BaseReturnsEstimator
            Expected returns estimator, default sample mean of linear returns.
        """
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            weight_bounds=weight_bounds,
            l2_gamma=l2_gamma,
        )
        self.rets_estimator = rets_estimator

    def _optimizer(self, X) -> BaseConvexOptimizer:
        eff_front_model = self._get_model(
            expected_returns=self.rets_estimator.set_returns_data(self.returns_data)
            .set_frequency(
                1
            )  # here frequency must be 1 because we simply need the returns average
            .fit_transform(X),
            returns=X if self.returns_data else returns_from_prices(X),
            weight_bounds=self.weight_bounds,
        )
        if hasattr(self, "constraints"):
            for cnstr in self.constraints:
                eff_front_model.add_constraint(cnstr)
        if self.l2_gamma > 0:
            eff_front_model.add_objective(
                objective_functions.L2_reg, gamma=self.l2_gamma
            )
        return deepcopy(eff_front_model)

    def risk_reward(self, model: Optional[BaseConvexOptimizer] = None):
        if model is None:
            model = self.model
        if model.weights is None or not hasattr(model, "weights"):
            raise AttributeError("Must fit model successfully to define weights")
        ret = portfolio_return(model.weights, model.expected_returns, negative=False)
        mad = np.mean(
            np.abs((model._w.value @ (model.returns - model.expected_returns).T))
        )
        return mad, ret


class MinimumMAD(_BaseMADPortfolio):
    """
    Estimator of the portfoloi with the Minimum mean-absolute deviation
    """

    def fit(self, X, y=None) -> _BaseMADPortfolio:
        self.weights_ = self._fit_method(X, "min_mad")
        return self


class MADEfficientReturn(_TargetReturnMixin, _BaseMADPortfolio):
    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency=APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        target_return: float = BASE_TARGET_RETURN,
    ):
        """
        Estimator of the maximum Mean-Absolute-Deviation portfolio, given target return.
        Please consider that in using a non-zero target return one should look for returns specified in the same
        frequency as those of the returns estimator.

        Parameters
        ----------
        returns_data: bool
            Whether input data are historical asset returns or historical asset prices
        l2_gamma: float
            Weights regularization constant for the L2 loss of weights.
        weight_bounds: Tuple[float,float]
            Minimum and maximum allowed weights
            Maximum weight
        frequency: int
            Number of periods to consider. As typical input data are daily close prices, and annualized
            returns or volatilies are needed, the default value for frequency=252 business days.
        rets_estimator: BaseReturnsEstimator
            Expected returns estimator, default sample mean of linear returns.
        target_return: float
            The target value of portfolio return (x-axis in the frontier plot)
        """
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            weight_bounds=weight_bounds,
            l2_gamma=l2_gamma,
            rets_estimator=rets_estimator,
        )
        self.target_return = target_return

    def fit(self, X, y=None) -> _BaseMADPortfolio:
        self.weights_ = self._fit_method(
            X, "efficient_return", target_return=self.target_return
        )
        return self


class MADEfficientRisk(_TargetRiskMixin, _BaseMADPortfolio):
    def __init__(
        self,
        returns_data: bool = False,
        *,
        frequency=APPROX_BDAYS_PER_YEAR,
        weight_bounds: Tuple[float, float] = (0, 1),
        l2_gamma: float = 0.0,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        target_risk: float = BASE_TARGET_RISK,
    ):
        """
        Estimator of the maximum Mean-Absolute-Deviation portfolio, given target risk.
        Please consider that in using a non-zero target return one should look for returns specified in the same
        frequency as those of the returns estimator.

        Parameters
        ----------
        returns_data: bool
            Whether input data are historical asset returns or historical asset prices
        l2_gamma: float
            Weights regularization constant for the L2 loss of weights.
        weight_bounds: Tuple[float,float]
            Minimum and maximum allowed weights
            Maximum weight
        frequency: int
            Number of periods to consider. As typical input data are daily close prices, and annualized
            returns or volatilies are needed, the default value for frequency=252 business days.
        rets_estimator: BaseReturnsEstimator
            Expected returns estimator, default sample mean of linear returns.
        target_risk: float
            The target value of portfolio risk (x-axis in the frontier plot)
        """
        super().__init__(
            returns_data=returns_data,
            frequency=frequency,
            weight_bounds=weight_bounds,
            l2_gamma=l2_gamma,
            rets_estimator=rets_estimator,
        )
        self.target_risk = target_risk

    def fit(self, X, y=None) -> _BaseMADPortfolio:
        self.weights_ = self._fit_method(
            X, "efficient_risk", target_risk=self.target_risk
        )
        return self

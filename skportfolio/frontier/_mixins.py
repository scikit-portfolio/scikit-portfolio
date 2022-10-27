from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd

from pypfopt.efficient_frontier import (
    EfficientFrontier,
    EfficientCVaR,
    EfficientSemivariance,
    EfficientCDaR,
)
from skportfolio.frontier._omega import EfficientOmegaRatio
from skportfolio.frontier._mad import EfficientMeanAbsoluteDeviation
from skportfolio._constants import (
    BASE_TARGET_RISK,
    BASE_TARGET_RETURN,
    APPROX_BDAYS_PER_YEAR,
)
from pypfopt.base_optimizer import BaseConvexOptimizer


class _TargetReturnMixin:
    # Mixin to provide base efficient frontier objects with the set_target_return and target_return attributes
    target_return: float = BASE_TARGET_RETURN

    def set_target_return(self, target_return: float):
        self.target_return = target_return
        return self


class _TargetRiskMixin:
    # Mixin to provide base efficient frontier objects with the set_target_risk and target_risk attributes
    target_risk: float = BASE_TARGET_RISK

    def set_target_risk(self, target_risk: float):
        self.target_risk = target_risk
        return self


class _BaseFrontierMixin(ABC):
    @staticmethod
    @abstractmethod
    def _get_model(
        expected_returns=None,
        returns=None,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        risk_matrix: Union[pd.DataFrame, np.array] = None,
        weight_bounds=(0, 1),
        solver=None,
        verbose=None,
        solver_options=None,
        **kwargs
    ) -> BaseConvexOptimizer:
        raise NotImplementedError(
            "Must implement the _get_model method based on the efficient frontier object"
        )

    def risk_function(self, value):
        return value


class _EfficientMeanVarianceMixin(_BaseFrontierMixin):
    # Mixin to add the markowitz efficient frontier model
    @staticmethod
    def _get_model(
        expected_returns=None,
        returns=None,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        risk_matrix: Union[pd.DataFrame, np.array] = None,
        weight_bounds=(0, 1),
        solver=None,
        verbose=None,
        solver_options=None,
        **kwargs
    ) -> EfficientFrontier:
        if expected_returns is None:
            raise AttributeError("Must specify parameter 'expected_returns'")
        if risk_matrix is None:
            raise AttributeError("Must specify parameter 'risk_matrix'")
        return EfficientFrontier(
            expected_returns=expected_returns,
            cov_matrix=risk_matrix,
            weight_bounds=weight_bounds,
        )

    # override the risk function because the optimization target is the variance
    # but the model.efficient_risk requires the target volatility

    def risk_function(self, value):
        return np.sqrt(value)


class _EfficientSemivarianceMixin(_BaseFrontierMixin):
    # Mixin to add the efficient mean-semivariance frontier model
    @staticmethod
    def _get_model(
        expected_returns=None,
        returns=None,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        risk_matrix: Union[pd.DataFrame, np.array] = None,
        weight_bounds=(0, 1),
        solver=None,
        verbose=None,
        solver_options=None,
        **kwargs
    ) -> EfficientSemivariance:
        if expected_returns is None:
            raise AttributeError("Must specify parameter 'expected_returns'")
        if returns is None:
            raise AttributeError("Must specify parameter 'returns'")

        return EfficientSemivariance(
            expected_returns=expected_returns,
            returns=returns,
            frequency=frequency,
            benchmark=kwargs.get("benchmark", 0),
            weight_bounds=weight_bounds,
        )


class _EfficientCDarMixin(_BaseFrontierMixin):
    # Mixin to add the return-cdar efficient frontier model
    @staticmethod
    def _get_model(
        expected_returns=None,
        returns=None,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        risk_matrix: Union[pd.DataFrame, np.array] = None,
        weight_bounds=(0, 1),
        solver=None,
        verbose=None,
        solver_options=None,
        **kwargs
    ) -> EfficientCDaR:
        if expected_returns is None:
            raise AttributeError("Must specify parameter 'expected_returns'")
        if returns is None:
            raise AttributeError("Must specify parameter 'returns'")

        return EfficientCDaR(
            expected_returns=expected_returns,
            returns=returns,
            beta=kwargs.get("beta", 0.95),
            weight_bounds=weight_bounds,
        )


class _EfficientCVarMixin(_BaseFrontierMixin):
    # Mixin to add the return-cdar efficient frontier model
    @staticmethod
    def _get_model(
        expected_returns=None,
        returns=None,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        risk_matrix: Union[pd.DataFrame, np.array] = None,
        weight_bounds=(0, 1),
        solver=None,
        verbose=None,
        solver_options=None,
        **kwargs
    ) -> EfficientCVaR:
        if expected_returns is None:
            raise AttributeError("Must specify parameter 'expected_returns'")
        if returns is None:
            raise AttributeError("Must specify parameter 'returns'")

        return EfficientCVaR(
            expected_returns=expected_returns,
            returns=returns,
            beta=kwargs.get("beta", 0.95),
            weight_bounds=weight_bounds,
        )


class _EfficientOmegaRatioMixin(_BaseFrontierMixin):
    # Mixin to add the omega ratio frontier model, no need to add min_risk, since it's coded within skportfolio
    @staticmethod
    def _get_model(
        expected_returns=None,
        returns=None,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        risk_matrix: Union[pd.DataFrame, np.array] = None,
        weight_bounds=(0, 1),
        solver=None,
        verbose=None,
        solver_options=None,
        **kwargs
    ) -> EfficientOmegaRatio:
        if expected_returns is None:
            raise AttributeError("Must specify parameter 'expected_returns'")
        if returns is None:
            raise AttributeError("Must specify parameter 'returns'")

        return EfficientOmegaRatio(
            expected_returns=expected_returns,
            returns=returns,
            minimum_acceptable_return=kwargs.get("minimum_acceptable_return", 0.0),
            fraction=kwargs.get("fraction", 1.0),
            weight_bounds=weight_bounds,
        )


class _EfficientMADMixin(_BaseFrontierMixin):
    # Mixin to add the mean absolute deviation frontier model
    @staticmethod
    def _get_model(
        expected_returns=None,
        returns=None,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        risk_matrix: Union[pd.DataFrame, np.array] = None,
        weight_bounds=(0, 1),
        solver=None,
        verbose=None,
        solver_options=None,
        **kwargs
    ) -> EfficientMeanAbsoluteDeviation:
        if expected_returns is None:
            raise AttributeError("Must specify parameter 'expected_returns'")
        if returns is None:
            raise AttributeError("Must specify parameter 'returns'")

        return EfficientMeanAbsoluteDeviation(
            expected_returns=expected_returns,
            returns=returns,
            weight_bounds=weight_bounds,
        )


class _EfficientEDaRMixin(_BaseFrontierMixin):
    @staticmethod
    def _get_model(
        expected_returns=None,
        returns=None,
        frequency: int = APPROX_BDAYS_PER_YEAR,
        risk_matrix: Union[pd.DataFrame, np.array] = None,
        weight_bounds=(0, 1),
        solver=None,
        verbose=None,
        solver_options=None,
        **kwargs
    ) -> BaseConvexOptimizer:
        if expected_returns is None:
            raise AttributeError("Must specify parameter 'expected_returns'")
        if returns is None:
            raise AttributeError("Must specify parameter 'returns'")

        return EfficientEDaR(
            expected_returns=expected_returns,
            returns=returns,
            weight_bounds=weight_bounds,
        )

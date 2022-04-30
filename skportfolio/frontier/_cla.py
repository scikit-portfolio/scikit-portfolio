import abc
from typing import Any
from typing import Dict
from typing import Sequence

import pandas as pd
from pypfopt.cla import CLA

from skportfolio._base import PortfolioEstimator
from skportfolio.riskreturn import BaseReturnsEstimator
from skportfolio.riskreturn import BaseRiskEstimator
from skportfolio.riskreturn import MeanHistoricalLinearReturns
from skportfolio.riskreturn import SampleCovariance
from skportfolio.riskreturn import SemiCovariance


class _MeanVarianceCLA(PortfolioEstimator, metaclass=abc.ABCMeta):
    """
    Base class for Markowitz Critical Line Algorithm optimization
    on efficient frontier methods.
    All the other special points are obtained from this class
    CLAMaxSharpe
    CLAMinimumVolatility
    CLAMinimumSemiVolatility
    """

    def __init__(
        self,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        risk_estimator: BaseRiskEstimator = SampleCovariance(),
    ):
        super(_MeanVarianceCLA, self).__init__()
        self.rets_estimator = rets_estimator
        self.risk_estimator = risk_estimator

    def _get_model(self, X, y=None) -> CLA:
        expected_returns = self.rets_estimator.fit(X).expected_returns_
        risk_matrix = self.risk_estimator.fit(X).risk_matrix_
        return CLA(
            expected_returns=expected_returns,
            cov_matrix=risk_matrix,
            weight_bounds=self.weight_bounds,
        )

    def fit(self, X, y=None) -> PortfolioEstimator:
        pass


class CLAMaxSharpe(_MeanVarianceCLA):
    def fit(self, X, y=None) -> PortfolioEstimator:
        model = self._get_model(X)
        self.weights_ = pd.Series(model.max_sharpe())
        return self


class CLAMinimumVolatility(_MeanVarianceCLA):
    def fit(self, X, y=None) -> PortfolioEstimator:
        model = CLA(
            expected_returns=self.rets_estimator.fit(X).expected_returns_,
            cov_matrix=self.risk_estimator.fit(X).risk_matrix_,
            weight_bounds=self.weight_bounds,
        )
        self.weights_ = pd.Series(model.min_volatility())
        return self


class CLAMinimumSemiVolatility(_MeanVarianceCLA):
    """
    Uses the Markowitz Critical Line Algorithm to get the efficient frontier for the minimum semivolatility portfolio.
    It is based on the excellent PyPortfolioOpt CLA implementation as well as the EfficientSemiVar.
    """

    def fit(self, X, y=None) -> PortfolioEstimator:
        model = CLA(
            expected_returns=self.rets_estimator.fit(X).expected_returns_,
            cov_matrix=SemiCovariance().fit(X).risk_matrix_,
            weight_bounds=self.weight_bounds,
        )
        self.weights_ = pd.Series(model.min_volatility())
        return self

    def grid_parameters(self) -> Dict[str, Sequence[Any]]:
        return {}

    def optuna_parameters(self) -> Dict[str, Any]:
        return {}


class CLAMaxSemiSharpe(CLAMinimumSemiVolatility):
    def fit(self, X, y=None) -> PortfolioEstimator:
        model = CLA(
            expected_returns=self.rets_estimator.fit(X).expected_returns_,
            cov_matrix=SemiCovariance().fit(X).risk_matrix_,
            weight_bounds=self.weight_bounds,
        )
        self.weights_ = pd.Series(model.max_sharpe())
        return self

    def grid_parameters(self) -> Dict[str, Sequence[Any]]:
        return {}

    def optuna_parameters(self) -> Dict[str, Any]:
        return {}

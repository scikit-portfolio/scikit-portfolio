from abc import ABC
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import Sequence

import pandas as pd
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from pypfopt.expected_returns import returns_from_prices
from pypfopt.hierarchical_portfolio import HRPOpt

from skportfolio._base import BaseRiskEstimator
from skportfolio._base import PortfolioEstimator
from skportfolio.riskreturn import SampleCovariance
from skportfolio.riskreturn import all_risk_estimators
from skportfolio.riskreturn import all_risk_models


class _HierarchicalRiskPortfolioEstimator(PortfolioEstimator, metaclass=ABCMeta):
    def __init__(
        self,
        returns_data: bool = False,
        linkage: str = "single",
        risk_estimator: BaseRiskEstimator = SampleCovariance(),
    ):
        super().__init__()
        self.returns_data = returns_data
        self.linkage: str = linkage
        self.risk_estimator = risk_estimator

    def fit(self, X, y=None) -> PortfolioEstimator:
        if self.returns_data:
            returns = X
        else:
            returns = returns_from_prices(X)

        hropt = HRPOpt(
            returns=returns,
            cov_matrix=self.risk_estimator.set_returns_data(self.returns_data)
            .fit(returns)
            .risk_matrix_,
        )
        hropt.optimize(self.linkage)
        self.weights_ = pd.Series(hropt.clean_weights(), name=self.__class__.__name__)
        return self

    def grid_parameters(self) -> Dict[str, Sequence[Any]]:
        return {
            "linkage": ["average", "ward", "single", "complete"],
            "risk_estimator": all_risk_estimators,
        }

    def optuna_parameters(self) -> Dict[str, BaseDistribution]:
        return {
            "linkage": CategoricalDistribution(
                ["average", "ward", "single", "complete"]
            ),
            "cov_estimator": CategoricalDistribution(all_risk_models),
        }


class HierarchicalRisk(_HierarchicalRiskPortfolioEstimator, ABC):
    """
    Hierarchical Risk estimator based on Lopez-De Prado book
    """

    pass

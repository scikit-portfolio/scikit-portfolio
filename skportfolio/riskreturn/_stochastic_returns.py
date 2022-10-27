"""
Contains the definition of the meta-class Perturbed Returns, which is a
way to produce randomized versions of the expected returns by sampling
expected returns from a multivariate normal distribution with mean
as from the rets_estimator and covariance as from the risk_estimator
"""
from typing import Optional

import numpy as np

from skportfolio.riskreturn._returns_estimators import BaseReturnsEstimator
from skportfolio.riskreturn._returns_estimators import MeanHistoricalLinearReturns
from skportfolio.riskreturn._risk_estimators import BaseRiskEstimator
from skportfolio.riskreturn._risk_estimators import SampleCovariance


class PerturbedReturns(BaseReturnsEstimator, BaseRiskEstimator):
    """
    Estimator of expected returns based on random normal perturbation around the expected returns based on the
    sample covariance.
    """

    _required_parameters = ["rets_estimator", "risk_estimator"]

    def __init__(
        self,
        returns_data: bool = False,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        risk_estimator: BaseRiskEstimator = SampleCovariance(),
        random_state: Optional[np.random.RandomState] = None,
    ):
        super().__init__()
        self.returns_data = returns_data
        self.rets_estimator = rets_estimator
        self.risk_estimator = risk_estimator
        self.random_state = random_state
        self._estimator_expected_returns = None
        self._estimator_expected_risk = None

    def _set_risk(self, X):
        self.risk_matrix_ = self.risk_estimator.set_returns_data(
            self.returns_data
        ).fit_transform(X)

    def _set_expected_returns(self, X, y=None):
        self._estimator_expected_returns = self.rets_estimator.set_returns_data(
            self.returns_data
        ).expected_returns_

    def fit(self, X, y=None):
        if self.random_state is None:
            self.random_state = np.random.default_rng()
        elif isinstance(self.random_state, int):
            self.random_state = np.random.default_rng(self.random_state)

        self._set_expected_returns(X)
        self._set_risk(X)

        # Overwrite self.expected_returns_ from the base returns estimator with the ones as observed from
        # the perturbed sample using
        self.expected_returns_ = self.random_state.multivariate_normal(
            self._estimator_expected_returns,
            self.risk_matrix_,
        )
        return self

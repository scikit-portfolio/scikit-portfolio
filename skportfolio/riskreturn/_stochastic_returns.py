"""
Contains the definition of the meta-class PerturbedReturns, which is a
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
    Estimator of expected returns based on random normal perturbation around
    the expected returns based on the sample covariance.
    """

    _required_parameters = ["rets_estimator", "risk_estimator"]

    def __init__(
        self,
        returns_data: bool = False,
        rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
        risk_estimator: BaseRiskEstimator = SampleCovariance(),
        random_state: Optional[np.random.Generator] = None,
    ):
        super().__init__()
        self.returns_data = returns_data
        self.rets_estimator = rets_estimator
        self.risk_estimator = risk_estimator
        self.random_state = random_state
        self._estimator_expected_returns = None
        self._estimator_expected_risk = None

    def _set_risk(self, X, y=None, **fit_params):
        self.risk_matrix_ = self.risk_estimator.set_returns_data(
            self.returns_data
        ).fit_transform(X)

    def _set_expected_returns(self, X, y=None, **fit_params):
        self._estimator_expected_returns = (
            self.rets_estimator.set_returns_data(self.returns_data)
            .fit(X, y, **fit_params)
            .expected_returns_
        )

    def fit(self, X, y=None, **fit_params):
        if self.random_state is None:
            generator = np.random.default_rng()
        else:
            generator = np.random.default_rng(seed=self.random_state)

        self._set_expected_returns(
            X,
            y,
        )
        self._set_risk(X, y, **fit_params)

        # Overwrite self.expected_returns_ from the base returns estimator
        # with the ones as observed from the perturbed sample using
        self.expected_returns_ = generator.multivariate_normal(
            mean=self._estimator_expected_returns,
            cov=self.risk_matrix_,
        )
        return self

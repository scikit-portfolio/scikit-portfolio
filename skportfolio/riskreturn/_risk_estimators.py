from abc import abstractmethod

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from skportfolio._constants import APPROX_BDAYS_PER_YEAR
from skportfolio.riskreturn.expected_risk import covariance_exp
from skportfolio.riskreturn.expected_risk import covariance_glasso
from skportfolio.riskreturn.expected_risk import covariance_hierarchical_filter_average
from skportfolio.riskreturn.expected_risk import covariance_hierarchical_filter_complete
from skportfolio.riskreturn.expected_risk import covariance_hierarchical_filter_single
from skportfolio.riskreturn.expected_risk import covariance_ledoit_wolf
from skportfolio.riskreturn.expected_risk import covariance_oracle_approx
from skportfolio.riskreturn.expected_risk import covariance_rmt
from skportfolio.riskreturn.expected_risk import sample_covariance
from skportfolio.riskreturn.expected_risk import semicovariance


class BaseRiskEstimator(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        returns_data=False,
    ):
        self.returns_data = returns_data
        self.risk_matrix_ = None

    def set_returns_data(self, returns_data):
        self.returns_data = returns_data
        return self

    @abstractmethod
    def _set_risk(self, X, y=None, **fit_params):
        pass

    def fit(self, X, y=None, **fit_params):
        self._set_risk(X, y=y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.risk_matrix_


class SampleCovariance(BaseRiskEstimator):
    """
    The standard sample covariance estimator, based on historical data.
    """

    def _set_risk(self, X, y=None, **fit_params):
        self.risk_matrix_ = sample_covariance(X, self.returns_data)


class SemiCovariance(BaseRiskEstimator):
    """
    The semicovariance, a.k.a. the covariance matrix estimated from only the positive returns.
    """

    def _set_risk(self, X, y=None, **fit_params):
        self.risk_matrix_ = semicovariance(X, self.returns_data)


class CovarianceRMT(BaseRiskEstimator):
    """
    Estimator of covariance based on Random Matrix Theory
    """

    def _set_risk(self, X, y=None, **fit_params):
        self.risk_matrix_ = covariance_rmt(X, returns_data=self.returns_data)


class CovarianceGlasso(BaseRiskEstimator):
    """
    Estimator of covariance based on GLASSO algorithm
    """

    def _set_risk(self, X, y=None, **fit_params):
        self.risk_matrix_ = covariance_glasso(X, returns_data=self.returns_data)


class CovarianceOAS(BaseRiskEstimator):
    """
    Estimator of covariance based on the oracle shrinkage approximation
    """

    def _set_risk(self, X, y=None, **fit_params):
        self.risk_matrix_ = covariance_oracle_approx(X, returns_data=self.returns_data)


class CovarianceExp(BaseRiskEstimator):
    """
    Estimator of covariance based on exponential weighted average of returns
    """

    def __init__(
        self,
        returns_data: bool = False,
        span: int = 60,
    ):
        super().__init__(returns_data=returns_data)
        self.span = span

    def _set_risk(self, X, y=None, **fit_params):
        self.risk_matrix_ = covariance_exp(
            X, returns_data=self.returns_data, span=self.span
        )


class CovarianceLedoitWolf(BaseRiskEstimator):
    """
    Estimator of covariance based on the Ledoit-Wolf shrinkage
    """

    def __init__(
        self,
        returns_data: bool = False,
        shrinkage_target: str = "constant_variance",
    ):
        super().__init__(returns_data=returns_data)
        self.shrinkage_target = shrinkage_target

    def _set_risk(self, X, y=None, **fit_params):
        self.risk_matrix_ = covariance_ledoit_wolf(
            X,
            returns_data=self.returns_data,
            shrinkage_target=self.shrinkage_target,
        )


class CovarianceHierarchicalFilterAverage(BaseRiskEstimator):
    """
    Estimator of covariance based on hierarchical filtering approach.
    """

    def _set_risk(self, X, y=None, **fit_params):
        self.risk_matrix_ = covariance_hierarchical_filter_average(X, self.returns_data)


class CovarianceHierarchicalFilterSingle(BaseRiskEstimator):
    """
    Estimator of covariance based on hierarchical filtering approach.
    """

    def _set_risk(self, X, y=None, **fit_params):
        self.risk_matrix_ = covariance_hierarchical_filter_single(X, self.returns_data)


class CovarianceHierarchicalFilterComplete(BaseRiskEstimator):
    """
    Estimator of covariance based on hierarchical filtering approach.
    """

    def _set_risk(self, X, y=None, **fit_params):
        self.risk_matrix_ = covariance_hierarchical_filter_complete(
            X, self.returns_data
        )


all_risk_estimators = [
    SampleCovariance(),
    SemiCovariance(),
    CovarianceRMT(),
    CovarianceGlasso(),
    CovarianceOAS(),
    CovarianceExp(),
    CovarianceLedoitWolf(),
    CovarianceHierarchicalFilterAverage(),
    CovarianceHierarchicalFilterSingle(),
    CovarianceHierarchicalFilterComplete(),
]

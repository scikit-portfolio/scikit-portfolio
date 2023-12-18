"""
Implementation of the basic Black-Litterman estimator that can
blend a-priori results with investors' views in a Bayesian manner.

An AnalystView is a tuple composed of 3 distinct elements
- Either a single symbol to provide absolute view of
- or a pair of symbols tuples to specify relative views
- the view value (second element)
- the view confidence (third element, as a rate in [0,1]
Relative views are grouped so that one can ask a group of tickers
outperforms another group of tickers
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pypfopt.black_litterman import (
    BlackLittermanModel,
)
from sklearn.base import TransformerMixin, BaseEstimator

from skportfolio.riskreturn._returns_estimators import (
    BaseReturnsEstimator,
    MarketImpliedReturns,
)
from skportfolio.riskreturn._risk_estimators import BaseRiskEstimator, SampleCovariance

AnalystView = Tuple[
    Union[Tuple[str], Tuple[Sequence[str], Sequence[str]], float, float]
]


def flatten(iterable):
    for item in iterable:
        if isinstance(item, (list, tuple, set)):
            yield from flatten(item)
        else:
            yield item


def parse_views_to_q_p_omega(
    views: Sequence[AnalystView],
    tickers: Sequence[str],
):
    """
    Parse the human-readable absolute or relative views using the Idzorek method
    to the Q, P and Omega matrices
    Parameters
    ----------
    views: Sequence[AnalystView]
        Sequence of individual views
    tickers: Sequence[str]
        The tickers universe
    Returns
    -------
    """
    n_views = len(views)
    if n_views == 0:
        return None, None, None

    views_confidence = []
    q_matrix = []
    p_matrix = pd.DataFrame(columns=tickers)

    # validate views: they must contain the same tickers we have
    for view in views:
        for s in flatten(view[0]):
            if s not in tickers:
                raise ValueError(
                    f"Ticker {s} not present in the data. Check your views"
                )

    # builds the Q matrix
    for view in views:
        if len(view) != 3:
            raise ValueError(
                "A view is composed of three elements: [symbol, value, confidence]"
            )
        q_matrix.append(view[-2])
        if view[-1] > 1.0 or view[-1] < 0.0:
            raise ValueError("View confidence must be in [0,1] domain")
        views_confidence.append(view[-1])

    views_confidence = np.array(views_confidence).reshape(-1, 1)
    q_matrix = np.array(q_matrix).reshape(-1, 1)
    # builds the P matrix
    idx_absolute_views = []
    for i_view, view in enumerate(views):
        symbol_or_symbols_sets, view_value = view[0], view[1]
        if (
            isinstance(symbol_or_symbols_sets, (list, tuple))
            and len(symbol_or_symbols_sets) == 1
        ):  # it's an absolute view
            p_matrix.loc[i_view, symbol_or_symbols_sets] = 1
            idx_absolute_views.append(i_view)
        elif (
            isinstance(symbol_or_symbols_sets, (tuple, list))
            and len(symbol_or_symbols_sets) == 2
        ):  # it's a relative view
            if isinstance(symbol_or_symbols_sets[0], (tuple, list)) and isinstance(
                symbol_or_symbols_sets[1], (tuple, list)
            ):
                n_left = len(symbol_or_symbols_sets[0])
                n_right = len(symbol_or_symbols_sets[1])
                for l_sym in symbol_or_symbols_sets[0]:
                    p_matrix.loc[i_view, l_sym] = 1 / n_left
                for r_sym in symbol_or_symbols_sets[1]:
                    p_matrix.loc[i_view, r_sym] = -1 / n_right
            else:
                raise ValueError(
                    "When specifying relative views, specify sets of tickers"
                )
        else:
            ValueError("Malformed view, please check input view format")

    p_matrix.fillna(0.0, inplace=True)
    return q_matrix, p_matrix, views_confidence


class _BlackLittermanBaseEstimator(TransformerMixin, BaseEstimator):
    """
    The base class for the Black-Litterman estimator that defines
    the parent for both the returns and risk estimator
    """

    def __init__(
        self,
        views: Optional[Sequence[AnalystView]] = None,
        rets_estimator: Optional[BaseReturnsEstimator] = None,
        covariance_estimator: Optional[BaseRiskEstimator] = None,
        returns_data=False,
        omega: Optional[pd.DataFrame] = None,
        tau: Optional[float] = None,
        risk_aversion: float = 1,
        risk_free_rate: Optional[float] = 0.02,
    ):
        """
        https://www.portfoliovisualizer.com/black-litterman-model
        Here we use the Idzorek method for computing the Omega matrix based on the investor view confidence

        Parameters
        ----------
        rets_estimator: BaseReturnsEstimator
            Returns estimator
        covariance_estimator: BaseRiskEstimator
            Covariance matrix estimator
        views: Sequence[AnalystViews]
            Analyst views, either absolute or relative
        returns_data: bool
            Whether the passed data are prices or returns
        omega: pd.DataFrame, np.ndarray
            Squared matrix with asset views uncertainties
        risk_aversion: float
            Investor risk aversion parameter
        risk_free_rate: float

        Returns
        -------

        """
        self.rets_estimator = rets_estimator
        self.covariance_estimator = covariance_estimator
        self.views = views
        self.returns_data = returns_data
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.risk_free_rate = risk_free_rate
        self.omega = omega

    def _set_pi(self, X, y):
        """
        In the absence of any views, the equilibrium returns are likely equal to
        the implied returns from the equilibrium portfolio holding.
        In practice, the applicable equilibrium portfolio holding can be any optimal portfolio
        that the investment analyst would use in the absence of additional views on the market,
        such as the portfolio benchmark, an index, or even the current portfolio [2].
        One can use linear regression to find a market portfolio that tracks the returns of the benchmark.
        Then, use the market portfolio as the equilibrium portfolio and the equilibrium returns
        are implied from the market portfolio.
        The MarketImpliedReturns class implements the equilibrium returns.
        The estimator takes historical asset returns and benchmark returns as inputs and outputs the market
        portfolio and the corresponding implied returns.

        Parameters
        ----------
        X
        y

        Returns
        -------

        """
        if self.rets_estimator is None:
            # MarketImpliedReturns estimator
            self.rets_estimator = MarketImpliedReturns()
        self.pi_ = self.rets_estimator.fit_transform(X=X, y=y)

    def _set_p_q(self, X):
        """
        Calculates the P and Q matrices from views.
        Please bear in mind to provide the views levels at the same frequency as the data,
        For example, if you have annuals views but provide daily data, divide the views by 252 business days.
        This class does not make any frequency conversion.
        Parameters
        ----------
        X

        Returns
        -------

        """
        self.Q_, self.P_, self.views_confidence_ = parse_views_to_q_p_omega(
            views=self.views, tickers=X.columns
        )

    def _set_tau(self, T):
        """
        The Black-Litterman model makes the assumption that the structure of C is proportional to the covariance Σ.
        Therefore, C=τΣ, where τ is a small constant.
        A smaller τ indicates a higher confidence in the prior belief of μ.
        The work of He and Litterman uses a value of 0.025, other use tau=0.05.
        Some authors suggest using 1/T where T is the number of data points used to generate the covariance matrix [3].
        If set to None we use 1/N

        Parameters
        ----------
        X
        Returns
        -------
        None
        """
        if self.tau is None:
            self.tau = 1 / T  # as in Matlab

    def _set_sigma(self, X, y):
        """
        Calculates the prior covariance matrix.
        Σ  is the covariance of the historical asset returns.
        We use the sample covariance as the default estimator, but one can specify any other covariance estimator

        Parameters
        ----------
        X
        y
        Returns
        -------
        """
        if self.covariance_estimator is None:
            self.covariance_estimator = SampleCovariance()
        self.sigma_ = self.covariance_estimator.fit_transform(X, y)

    def fit(self, X, y=None):
        """
        Fit the Black-Litterman estimator of posterior returns and risk matrix
        Parameters
        ----------
        X:
        y: Optionally the benchmark portfolio to calculate the implied returns
        provide market_caps

        Returns
        -------
        """
        # these arguments are just for notational simplicity

        self._set_pi(X=X, y=y)
        self._set_sigma(X=X, y=y)
        self._set_p_q(X=X)

        # Look this for the Idzorek method
        # https://people.duke.edu/~charvey/Teaching/BA453_2006/Idzorek_onBL.pdf
        # here we have computed a way to pass the fundamental matrices to the pypfopt optimization method
        self.bl_model_ = BlackLittermanModel(
            cov_matrix=self.sigma_,
            pi=self.pi_,
            Q=self.Q_,
            P=self.P_,
            omega="idzorek" if self.omega is None else self.omega,
            view_confidences=self.views_confidence_ if self.omega is None else None,
            risk_aversion=self.risk_aversion,
            risk_free_rate=self.risk_free_rate,
        )
        self.expected_returns_ = self.bl_model_.bl_returns()
        self.risk_matrix_ = self.bl_model_.bl_cov() + self.sigma_
        return self

    def get_p_matrix(self) -> np.ndarray:
        """
        Returns the picking matrix, a KxN matrix where K is the number of views (absolute or relative)
        and N is the number of assets in the universe.

        Returns
        -------
        The picking matrix
        """
        return self.P_

    def get_q_matrix(self) -> np.ndarray:
        """
        Returns the views values matrix
        Returns
        -------
        """
        return self.Q_


class BlackLittermanReturnsEstimator(
    _BlackLittermanBaseEstimator, BaseReturnsEstimator
):
    """
    Estimator of posterior expected returns given the prior expected returns and the views (absolute or relative)
    explicitly given as views or as $\boldsymbol Omega$ matrix.
    """

    def fit_transform(self, X, y=None, **fit_params):
        self.expected_returns_ = self.fit(X=X, y=y).bl_model_.bl_returns()
        return self.expected_returns_

    def _set_expected_returns(self, X, y=None, **fit_params):
        self.expected_returns_ = self.fit_transform(X, y, **fit_params)


class BlackLittermanRiskEstimator(_BlackLittermanBaseEstimator, BaseRiskEstimator):
    """
    Estimator of posterior covariance given the prior expected returns and the views (absolute or relative) explicitly
    given as views or $\boldsymbol Omega$ matrix.
    """

    def fit_transform(self, X, y=None, **fit_params):
        self.risk_matrix_ = self.fit(X=X, y=y).bl_model_.bl_cov() + self.sigma_
        return self.risk_matrix_

    def _set_risk(self, X, y=None, **fit_params):
        self.risk_matrix_ = self.fit_transform(X, y, **fit_params)
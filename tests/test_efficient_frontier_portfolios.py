#!/usr/bin/env python

"""Tests for `skportfolio` package."""
import pytest
import numpy as np
import pandas as pd  # type: ignore
from sklearn.base import clone  # type: ignore
from sklearn.utils.estimator_checks import check_estimator

from skportfolio import PortfolioEstimator

# efficient frontier portfolios
from skportfolio.frontier import CDarEfficientReturn
from skportfolio.frontier import CDarEfficientRisk
from skportfolio.frontier import CVarEfficientReturn
from skportfolio.frontier import CVarEfficientRisk
from skportfolio.frontier import MADEfficientReturn
from skportfolio.frontier import MADEfficientRisk
from skportfolio.frontier import MaxOmegaRatio
from skportfolio.frontier import MaxSharpe
from skportfolio.frontier import MeanSemiVarianceEfficientReturn
from skportfolio.frontier import MeanSemiVarianceEfficientRisk
from skportfolio.frontier import MeanVarianceEfficientReturn
from skportfolio.frontier import MeanVarianceEfficientRisk
from skportfolio.frontier import MinimumCDar
from skportfolio.frontier import MinimumCVar
from skportfolio.frontier import MinimumMAD
from skportfolio.frontier import MinimumSemiVolatility
from skportfolio.frontier import MinimumVolatility
from skportfolio.frontier import OmegaEfficientReturn
from skportfolio.frontier import OmegaEfficientRisk
from skportfolio.frontier._efficientfrontier import (
    _BaseEfficientFrontierPortfolioEstimator,
)

from .datasets_fixtures import prices, returns, log_returns
from logger import get_logger

logger = get_logger()


def check_model_fit(model: PortfolioEstimator, X: pd.DataFrame):
    try:
        fitted_model = model.fit(X)
        if not isinstance(fitted_model, PortfolioEstimator):
            raise TypeError("not a portfolio estimator returned")
    except Exception as ex:
        assert False, f"Cannot fit model {model} to data:\nReason:\n" + str(ex)
    return fitted_model


def check_model_predict(model: PortfolioEstimator, prices: pd.DataFrame):
    try:
        out = model.predict(prices)
        if not isinstance(out, pd.Series):
            raise TypeError("not a pd.Series returned")
    except Exception as ex:
        assert False, f"Cannot run portfolio {model} predict to data.\nReason:\n" + str(
            ex
        )


def check_model_score(model: PortfolioEstimator, prices: pd.DataFrame):
    try:
        score = model.score(prices)
        if not isinstance(score, float):
            raise TypeError("not a float returned")
    except Exception as ex:
        assert False, f"Cannot score model {model}.\nReason\n" + str(ex)
    return score


def check_model_weights(model: PortfolioEstimator):
    assert isinstance(model.weights_, pd.Series)


def check_risk_reward(model: PortfolioEstimator, X: pd.DataFrame):
    try:
        out = np.array(model.risk_reward())
    except AttributeError as has_no_weights:
        return
    assert len(out) == 2, "len should be 2"
    assert not np.any(np.isnan(out)), "risk or reward are Nan"


def check_estimate_frontier(
    model: _BaseEfficientFrontierPortfolioEstimator, X: pd.DataFrame
):
    risks_, returns_, weights_ = model.estimate_frontier(X, num_portfolios=10)
    assert len(risks_) == 10
    assert len(returns_) == 10


def check_clone_portfolio(model: PortfolioEstimator):
    """
    https://scikit-learn.org/stable/developers/develop.html?highlight=check_estimator#cloning
    """
    # Tests that clone creates a correct deep copy.
    # We create an estimator, make a copy of its original state
    # (which, in this case, is the current state of the estimator),
    # and check that the obtained copy is a correct deep copy.
    model_cloned = clone(model)
    assert model is not model_cloned
    # assert model.get_params() == model_cloned.get_params()


# def test_efficient_frontier_portfolios(prices, returns):
#     models = [
#         # Markowitz MVO
#         MinimumVolatility,
#         MeanVarianceEfficientRisk,
#         MeanVarianceEfficientReturn,
#         MaxSharpe,
#         # Semivariance frontier
#         MinimumSemiVolatility,
#         MeanSemiVarianceEfficientReturn,
#         MeanSemiVarianceEfficientRisk,
#         # CVar efficient frontier
#         MinimumCVar,
#         CVarEfficientReturn,
#         CVarEfficientRisk,
#         # CDar efficient frontier
#         MinimumCDar,
#         CDarEfficientReturn,
#         CDarEfficientRisk,
#         # # Omega frontier
#         MaxOmegaRatio,
#         OmegaEfficientReturn,
#         OmegaEfficientRisk,
#         # MAD Frontier
#         MinimumMAD,
#         MADEfficientReturn,
#         MADEfficientRisk,
#     ]
#
#     for m in models:
#         for prices_or_returns, returns_data in zip([returns, prices], [True, False]):
#             model = m(returns_data=returns_data)
#             check_clone_portfolio(model)
#             try:
#                 fitted_model = check_model_fit(model, X=prices_or_returns)
#                 check_risk_reward(fitted_model, X=prices_or_returns)
#                 check_model_predict(fitted_model, prices=prices)
#                 check_model_score(fitted_model, prices=prices)
#                 check_model_weights(fitted_model)
#                 check_estimate_frontier(fitted_model, X=prices_or_returns)
#
#                 check_clone_portfolio(model)
#                 try:
#                     check_estimator(model)
#                 except AttributeError as ex:  # because we use dataframes
#                     pass
#             except AssertionError as ex:
#                 logger.warning(ex)
#             assert isinstance(prices_or_returns, pd.DataFrame), "stop here"


def test_minimumvolatility_returns(prices, returns):
    model = MinimumVolatility(returns_data=True)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_minimumvolatility_prices(prices, returns):
    model = MinimumVolatility(returns_data=False)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_meanvarianceefficientrisk_returns(prices, returns):
    model = MeanVarianceEfficientRisk(returns_data=True)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_meanvarianceefficientrisk_prices(prices, returns):
    model = MeanVarianceEfficientRisk(returns_data=False)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_meanvarianceefficientreturn_returns(prices, returns):
    model = MeanVarianceEfficientReturn(returns_data=True)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_meanvarianceefficientreturn_prices(prices, returns):
    model = MeanVarianceEfficientReturn(returns_data=False)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_maxsharpe_returns(prices, returns):
    model = MaxSharpe(returns_data=True)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_maxsharpe_prices(prices, returns):
    model = MaxSharpe(returns_data=False)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_minimumsemivolatility_returns(prices, returns):
    model = MinimumSemiVolatility(returns_data=True)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_minimumsemivolatility_prices(prices, returns):
    model = MinimumSemiVolatility(returns_data=False)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_meansemivarianceefficientreturn_returns(prices, returns):
    model = MeanSemiVarianceEfficientReturn(returns_data=True)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_meansemivarianceefficientreturn_prices(prices, returns):
    model = MeanSemiVarianceEfficientReturn(returns_data=False)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_meansemivarianceefficientrisk_returns(prices, returns):
    model = MeanSemiVarianceEfficientRisk(returns_data=True)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_meansemivarianceefficientrisk_prices(prices, returns):
    model = MeanSemiVarianceEfficientRisk(returns_data=False)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_minimumcvar_returns(prices, returns):
    model = MinimumCVar(returns_data=True)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_minimumcvar_prices(prices, returns):
    model = MinimumCVar(returns_data=False)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_cvarefficientreturn_returns(prices, returns):
    model = CVarEfficientReturn(returns_data=True)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_cvarefficientreturn_prices(prices, returns):
    model = CVarEfficientReturn(returns_data=False)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_cvarefficientrisk_returns(prices, returns):
    model = CVarEfficientRisk(returns_data=True)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_cvarefficientrisk_prices(prices, returns):
    model = CVarEfficientRisk(returns_data=False)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_minimumcdar_returns(prices, returns):
    model = MinimumCDar(returns_data=True)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_minimumcdar_prices(prices, returns):
    model = MinimumCDar(returns_data=False)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_cdarefficientreturn_returns(prices, returns):
    model = CDarEfficientReturn(returns_data=True)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_cdarefficientreturn_prices(prices, returns):
    model = CDarEfficientReturn(returns_data=False)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_cdarefficientrisk_returns(prices, returns):
    model = CDarEfficientRisk(returns_data=True)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_cdarefficientrisk_prices(prices, returns):
    model = CDarEfficientRisk(returns_data=False)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_maxomegaratio_returns(prices, returns):
    model = MaxOmegaRatio(returns_data=True)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_maxomegaratio_prices(prices, returns):
    model = MaxOmegaRatio(returns_data=False)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_omegaefficientreturn_returns(prices, returns):
    model = OmegaEfficientReturn(returns_data=True)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_omegaefficientreturn_prices(prices, returns):
    model = OmegaEfficientReturn(returns_data=False)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_omegaefficientrisk_returns(prices, returns):
    model = OmegaEfficientRisk(returns_data=True)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_omegaefficientrisk_prices(prices, returns):
    model = OmegaEfficientRisk(returns_data=False)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_minimummad_returns(prices, returns):
    model = MinimumMAD(returns_data=True)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_minimummad_prices(prices, returns):
    model = MinimumMAD(returns_data=False)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_madefficientreturn_returns(prices, returns):
    model = MADEfficientReturn(returns_data=True, target_return=0.1)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_madefficientreturn_prices(prices, returns):
    model = MADEfficientReturn(returns_data=False, target_return=0.1)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_madefficientrisk_returns(prices, returns):
    model = MADEfficientRisk(returns_data=True)
    fitted_model = check_model_fit(model, X=returns)
    check_risk_reward(fitted_model, X=returns)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=returns)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


def test_madefficientrisk_prices(prices, returns):
    model = MADEfficientRisk(returns_data=False)
    fitted_model = check_model_fit(model, X=prices)
    check_risk_reward(fitted_model, X=prices)
    check_model_predict(fitted_model, prices=prices)
    check_model_score(fitted_model, prices=prices)
    check_model_weights(fitted_model)
    check_estimate_frontier(fitted_model, X=prices)
    check_clone_portfolio(model)
    try:
        check_estimator(fitted_model)
    except AttributeError as ex:
        logger.warning(ex)


if __name__ == "__main__":
    pytest.main(args=[__file__])

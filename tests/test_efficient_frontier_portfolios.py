#!/usr/bin/env python

"""Tests for `skportfolio` package."""
import numpy as np
import pandas as pd  # type: ignore
from sklearn.base import clone  # type: ignore
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


def check_model_fit(model: PortfolioEstimator, X: pd.DataFrame):
    try:
        fitted_model = model.fit(X)
        if not isinstance(fitted_model, PortfolioEstimator):
            raise TypeError("not a portfolio estimator returned")
    except Exception as ex:
        assert False, f"Cannot fit model {model} to data:\nReason:\n" + str(ex)
    return fitted_model


def check_model_predict(model: PortfolioEstimator, X: pd.DataFrame):
    try:
        out = model.predict(X)
        if not isinstance(out, pd.Series):
            raise TypeError("not a pd.Series returned")
    except Exception as ex:
        assert False, f"Cannot run portfolio {model} predict to data.\nReason:\n" + str(
            ex
        )
    return out


def check_model_score(model: PortfolioEstimator, X: pd.DataFrame):
    try:
        score = model.score(X)
        if not isinstance(score, float):
            raise TypeError("not a float returned")
    except Exception as ex:
        assert False, f"Cannot score model {model}.\nReason\n" + str(ex)
    return score


def check_model_weights(model: PortfolioEstimator):
    assert isinstance(model.weights_, pd.Series)
    return model.weights_


def check_grid_params(model: PortfolioEstimator):
    assert isinstance(
        model.grid_parameters(), dict
    ), "Not a dictionary for grid parameters"


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
    risks, returns, weights = model.estimate_frontier(X, num_portfolios=10)
    assert len(risks) == 10
    assert len(returns) == 10
    return risks, returns, weights


def check_clone_portfolio(model: PortfolioEstimator):
    # Tests that clone creates a correct deep copy.
    # We create an estimator, make a copy of its original state
    # (which, in this case, is the current state of the estimator),
    # and check that the obtained copy is a correct deep copy.
    model_cloned = clone(model)
    assert model is not model_cloned
    # assert model.get_params() == model_cloned.get_params()


def check_clone_portfolio2(model: PortfolioEstimator):
    model_clone = clone(model)
    model_clone.some_parameter = None
    assert not hasattr(model, "some_parameter")


def test_efficient_frontier_portfolios(prices, returns):
    models = [
        # Markowitz MVO
        MinimumVolatility(),
        MeanVarianceEfficientRisk(),
        MeanVarianceEfficientReturn(),
        MaxSharpe(),
        # Semivariance frontier
        MinimumSemiVolatility(),
        MeanSemiVarianceEfficientReturn(),
        MeanSemiVarianceEfficientRisk(),
        # CVar efficient frontier
        MinimumCVar(),
        CVarEfficientReturn(),
        CVarEfficientRisk(),
        # CDar efficient frontier
        MinimumCDar(),
        CDarEfficientReturn(),
        CDarEfficientRisk(),
        # # Omega frontier
        MaxOmegaRatio(),
        OmegaEfficientReturn(),
        OmegaEfficientRisk(),
        # MAD Frontier
        MinimumMAD(),
        MADEfficientReturn(),
        MADEfficientRisk(),
    ]
    for m in models:
        check_clone_portfolio(m)

    for returns_data in [True, False]:
        if returns_data:
            prices_or_returns = returns
        else:
            prices_or_returns = prices
        for model in models:
            model = model.set_returns_data(returns_data)
            fitted_model = check_model_fit(model, X=prices_or_returns)
            check_risk_reward(fitted_model, X=prices_or_returns)
            ptf_series = check_model_predict(fitted_model, X=prices)
            score = check_model_score(fitted_model, X=prices_or_returns)
            weights = check_model_weights(fitted_model)
            risks, returns, weights = check_estimate_frontier(
                fitted_model, X=prices_or_returns
            )
            check_grid_params(model)


def test_estimate_frontier_exact(prices):
    model = MinimumVolatility()
    # import warnings
    #
    # warnings.filterwarnings("error")
    model.estimate_frontier(prices, n_jobs=8)

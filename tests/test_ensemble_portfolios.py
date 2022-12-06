#!/usr/bin/env python

"""Tests for `skportfolio` package."""
import numpy as np

from skportfolio.ensemble import MichaudResampledFrontier
from skportfolio.ensemble import SubsetResampling
from skportfolio import InverseVariance
from skportfolio.frontier import MinimumVolatility
from itertools import product
from .datasets_fixtures import prices, returns

TOLERANCE = 1e-5
COMMON_SEED = 42


def test_portfolio_michaud_resampled_min_volatilility(prices, returns):

    resampled_model1 = MichaudResampledFrontier(
        estimator=MinimumVolatility(returns_data=False),
        n_iter=1,
        random_state=COMMON_SEED,
    ).fit(prices)

    resampled_model2 = MichaudResampledFrontier(
        estimator=MinimumVolatility(returns_data=True),
        n_iter=1,
        random_state=COMMON_SEED,
    ).fit(returns)

    assert (
        np.abs(resampled_model1.weights_ - resampled_model2.weights_).mean() < TOLERANCE
    )


def test_michaud_resampled_parameters(prices, returns):

    all_random_state = (None, 42, np.random.default_rng())
    all_agg_func = ("mean", "median")
    all_models = (MinimumVolatility(), InverseVariance())

    for random_state, agg_func, model in product(
        all_random_state, all_agg_func, all_models
    ):
        resampled_model = MichaudResampledFrontier(
            estimator=model,
            n_iter=1,
            random_state=random_state,
            agg_func=agg_func,
        ).fit(prices)


def test_portfolio_michaud_resampled_max_sharpe(prices, returns):

    model = MichaudResampledFrontier(
        estimator=InverseVariance(returns_data=False),
        n_iter=2,
        random_state=COMMON_SEED,
    ).fit(prices)


def test_subset_resampling(prices, returns):
    a = 1
    weights_1 = (
        SubsetResampling(
            estimator=MinimumVolatility(returns_data=False),
            random_state=COMMON_SEED,
        )
        .fit(prices)
        .weights_
    )
    weights_2 = (
        SubsetResampling(
            estimator=MinimumVolatility(returns_data=True),
            random_state=COMMON_SEED,
            n_iter=2,
        )
        .fit(returns)
        .weights_
    )
    assert np.abs(weights_1 - weights_2).mean() < TOLERANCE


def test_subset_resampling_sharpe(prices, returns):
    (
        SubsetResampling(
            estimator=InverseVariance(returns_data=False),
            random_state=COMMON_SEED,
            n_iter=2,
        ).fit(prices)
    )


# from skportfolio.ensemble import RobustBayesian
# def test_robust_bayesian_allocation(prices):
#     model = RobustBayesian(
#         window=10,
#         n_portfolios=2,
#         robustness_param_loc=0.1,
#         robustness_param_scatter=0.1,
#         rets_estimator=MeanHistoricalLinearReturns(),
#         risk_estimator=SampleCovariance(),
#         n_jobs=1,
#     ).fit(prices.iloc[0:50, :])
#     return

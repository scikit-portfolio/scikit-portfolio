#!/usr/bin/env python

"""Tests for `skportfolio` package."""
import numpy as np

from skportfolio.ensemble import MichaudResampledFrontier
from skportfolio.ensemble import RobustBayesian
from skportfolio.ensemble import SubsetResampling
from skportfolio.frontier import MaxSharpe
from skportfolio.frontier import MinimumVolatility
from skportfolio.riskreturn import MeanHistoricalLinearReturns
from skportfolio.riskreturn import SampleCovariance

from .datasets_fixtures import prices, returns

TOLERANCE = 1e-5
common_seed = 42


def test_portfolio_michaud_resampled_min_volatilility(prices, returns):

    resampled_model1 = MichaudResampledFrontier(
        ptf_estimator=MinimumVolatility(returns_data=False),
        n_iter=10,
        n_jobs=1,
        random_state=common_seed,
    ).fit(prices)

    resampled_model2 = MichaudResampledFrontier(
        ptf_estimator=MinimumVolatility(returns_data=True),
        n_iter=10,
        n_jobs=1,
        random_state=common_seed,
    ).fit(returns)

    assert (
        np.abs(resampled_model1.weights_ - resampled_model2.weights_).mean() < TOLERANCE
    )


def test_michaud_resampled_parameters(prices, returns):
    for n_jobs in (None, 1, 4):
        for seed in (None, 42):
            for X, returns_data in zip((returns, prices), (True, False)):
                for agg_func in ("mean", "median"):
                    for rs in (None, 42):
                        for model in (
                            MinimumVolatility(returns_data=returns_data),
                            MaxSharpe(returns_data=returns_data),
                        ):
                            resampled_model1 = MichaudResampledFrontier(
                                ptf_estimator=model,
                                n_iter=10,
                                n_jobs=n_jobs,
                                random_state=seed,
                                returns_estimator_random_state=rs,
                                agg_func=agg_func,
                            ).fit(X)
                            assert resampled_model1.weights_.isna().sum() == 0


def test_portfolio_michaud_resampled_max_sharpe(prices, returns):

    model = MichaudResampledFrontier(
        ptf_estimator=MaxSharpe(returns_data=False, risk_free_rate=0.02),
        n_iter=2,
        n_jobs=1,
        random_state=common_seed,
    ).fit(prices)


def test_subset_resampling(prices, returns):
    w1 = (
        SubsetResampling(
            ptf_estimator=MinimumVolatility(returns_data=False),
            random_state=common_seed,
        )
        .fit(prices)
        .weights_
    )
    w2 = (
        SubsetResampling(
            ptf_estimator=MinimumVolatility(returns_data=True),
            random_state=common_seed,
            n_iter=2,
            n_jobs=1,
        )
        .fit(returns)
        .weights_
    )
    assert np.abs(w1 - w2).mean() < TOLERANCE


def test_subset_resampling_sharpe(prices, returns):
    (
        SubsetResampling(
            ptf_estimator=MaxSharpe(returns_data=False),
            random_state=common_seed,
            n_iter=2,
            n_jobs=1,
        ).fit(prices)
    )


def test_robust_bayesian_allocation(prices):
    model = RobustBayesian(
        window=10,
        n_portfolios=2,
        robustness_param_loc=0.1,
        robustness_param_scatter=0.1,
        rets_estimator=MeanHistoricalLinearReturns(),
        risk_estimator=SampleCovariance(),
        n_jobs=1,
    ).fit(prices.iloc[0:50, :])
    return

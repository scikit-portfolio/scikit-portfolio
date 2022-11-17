#!/usr/bin/env python

"""Tests for `skportfolio` package."""

# from skportfolio.model_selection import *
#
# from .datasets_fixtures import prices
# from skportfolio.frontier import MinimumVolatility
# from skportfolio import OptunaSearchCV, GridSearchCV, all_scorers, sharpe_ratio_scorer

# def test_grid_searchcv(prices):
#     model = MinimumVolatility()
#     gridcv = GridSearchCV(
#         model,
#         scoring=all_scorers,
#         cv=5,
#         verbose=True,
#         # error_score="raise",
#         n_jobs=1,
#         refit="sharpe_ratio",
#     ).fit(prices.iloc[0:50, :])
#
#     gridcv.fit(prices.iloc[0:50, :]).predict(prices)
#     return
#
#
# def test_optuna_search_min_vol(prices):
#     model = MinimumVolatility()
#     optcv = OptunaSearchCV(
#         model,
#         param_distributions=model.optuna_parameters(),
#         cv=5,
#         scoring=sharpe_ratio_scorer,
#         error_score="raise",
#         n_jobs=1,
#         n_trials=10,
#     ).fit(prices.iloc[0:50, :])
#     optcv.fit(prices.iloc[0:50, :]).predict(prices)
#     return optcv
#
#
# def test_cross_validate(prices):
#     from skportfolio.model_selection import portfolio_cross_validate
#
#     portfolio_cross_validate(
#         estimator=MinimumVolatility(),
#         prices_or_returns=prices,
#         cv=2,
#         scoring=sharpe_ratio_scorer,
#         n_jobs=1,
#     )

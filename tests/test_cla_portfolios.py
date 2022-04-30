#!/usr/bin/env python
"""Tests for `skportfolio` package."""
from skportfolio.frontier._cla import (
    CLAMaxSharpe,
    CLAMinimumVolatility,
    CLAMinimumSemiVolatility,
    CLAMaxSemiSharpe,
)

from .datasets_fixtures import prices, returns, log_returns


# def test_cla_mean_variance(prices):
#     model = CLAMaxSharpe().fit(prices)
#     model = CLAMinimumVolatility().fit(prices)
#     return
#
#
# def test_cla_semivariance(prices):
#     model = CLAMinimumSemiVolatility().fit(prices)
#     model = CLAMaxSemiSharpe().fit(prices)
#     model.score(prices.pct_change().dropna())
#     return

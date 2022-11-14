#!/usr/bin/env python

"""Tests for `carrottrade` package."""
import numpy as np
from pytest import fixture
from skportfolio._constants import APPROX_BDAYS_PER_YEAR
from skportfolio.datasets import load_dow_prices
from skportfolio.riskreturn import SampleCovariance
from skportfolio.riskreturn.blacklitterman import (
    BlackLittermanReturnsEstimator,
    BlackLittermanRiskEstimator,
)
from skportfolio.riskreturn import MarketImpliedReturns

TOLERANCE = 1e-5


@fixture
def prices():
    yield load_dow_prices()[["AA", "AIG", "WMT", "MSFT", "BA", "GE", "IBM"]]


@fixture
def returns():
    yield load_dow_prices()[
        ["AA", "AIG", "WMT", "MSFT", "BA", "GE", "IBM"]
    ].pct_change().dropna()


@fixture
def benchmark():
    yield load_dow_prices()["DJI"]


@fixture
def benchmark_returns():
    yield load_dow_prices()["DJI"].pct_change().dropna()


def test_implied_returns(returns, prices, benchmark, benchmark_returns):
    MarketImpliedReturns(returns_data=True).fit_transform(
        X=returns, y=benchmark_returns
    )


def test_blacklitterman_expected_returns(returns, benchmark_returns):
    omega = np.diag([1e-3, 1e-3, 1e-5]) / APPROX_BDAYS_PER_YEAR
    rets_estimator = MarketImpliedReturns(returns_data=True).fit(
        X=returns, y=benchmark_returns
    )
    blmodel = BlackLittermanReturnsEstimator(
        views=None,
        rets_estimator=rets_estimator,
        omega=omega,
        covariance_estimator=SampleCovariance(),
    )

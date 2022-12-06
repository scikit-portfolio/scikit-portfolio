#!/usr/bin/env python

"""Tests for `carrottrade` package."""
import numpy as np
import pandas as pd
from pytest import fixture

from skportfolio._constants import APPROX_BDAYS_PER_YEAR
from skportfolio.datasets import load_dow_prices
from skportfolio.riskreturn import (
    MarketImpliedReturns,
    MeanHistoricalLinearReturns,
)
from skportfolio.riskreturn import SampleCovariance
from skportfolio.riskreturn.blacklitterman import (
    BlackLittermanReturnsEstimator,
)

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


def test_blacklitterman_expected_returns(prices, returns, benchmark_returns):
    omega = np.diag([1e-3, 1e-3, 1e-5]) / APPROX_BDAYS_PER_YEAR
    rets_estimator = MarketImpliedReturns(returns_data=True).fit(
        X=returns, y=benchmark_returns
    )
    blmodel_with_omega = BlackLittermanReturnsEstimator(
        views=None,
        rets_estimator=rets_estimator,
        omega=omega,
        covariance_estimator=SampleCovariance(),
    )

    blmodel_with_views = BlackLittermanReturnsEstimator(
        views=None,
        rets_estimator=rets_estimator,
        omega=omega,
        covariance_estimator=SampleCovariance(),
    )


def test_blacklitterman_risk_matrix(prices, returns, benchmark_returns):
    prices = pd.read_parquet("/Users/carlo/Desktop/prices.parquet")
    views = [
        [["AMZN"], 0.05 / APPROX_BDAYS_PER_YEAR, 0.5],
        [["BAC"], -0.03 / APPROX_BDAYS_PER_YEAR, 0.25],
        [
            [["TSLA"], prices.columns.drop("TSLA").tolist()],
            0.03 / APPROX_BDAYS_PER_YEAR,
            0.9,
        ],
    ]

    blmodel = BlackLittermanReturnsEstimator(
        returns_data=False,
        views=views,
        rets_estimator=MeanHistoricalLinearReturns(),
        covariance_estimator=SampleCovariance(),
    ).fit(prices)
    blmodel.expected_returns_

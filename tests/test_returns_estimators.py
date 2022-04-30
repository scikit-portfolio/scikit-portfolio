#!/usr/bin/env python

"""Tests for `skportfolio` package."""
from pytest import fixture
import numpy as np
import pandas as pd

from skportfolio import (
    MeanHistoricalLinearReturns,
    MeanHistoricalLogReturns,
    MedianHistoricalLinearReturns,
    MedianHistoricalLogReturns,
    EMAHistoricalReturns,
    PerturbedReturns,
    CAPMReturns,
    RollingMedianReturns,
    CompoundedHistoricalLinearReturns,
)
import pkg_resources

TOLERANCE = 1e-5
LOGTOLERANCE = 1e-3

from .datasets_fixtures import prices, returns, log_returns


def test_mean_historical_returns(prices, returns):
    delta = (
        MeanHistoricalLinearReturns(returns_data=False).fit(prices).expected_returns_
        - MeanHistoricalLinearReturns(returns_data=True).fit(returns).expected_returns_
    )
    assert delta.mean() < TOLERANCE


def test_mean_historical_log_returns(prices, log_returns):
    delta = (
        MeanHistoricalLogReturns(returns_data=False).fit(prices).expected_returns_
        - MeanHistoricalLogReturns(returns_data=True).fit(log_returns).expected_returns_
    )
    assert delta.mean() < TOLERANCE


def test_median_historical_returns(prices, returns):
    delta = (
        MedianHistoricalLinearReturns(returns_data=False).fit(prices).expected_returns_
        - MedianHistoricalLinearReturns(returns_data=True)
        .fit(returns)
        .expected_returns_
    )
    assert delta.mean() < TOLERANCE


def test_median_historical_log_returns(prices, log_returns):
    delta = (
        MedianHistoricalLogReturns(returns_data=False).fit(prices).expected_returns_
        - MedianHistoricalLogReturns(returns_data=True)
        .fit(log_returns)
        .expected_returns_
    )
    assert delta.mean() < LOGTOLERANCE


def test_ema_historical_returns(prices, returns):
    delta = (
        EMAHistoricalReturns(returns_data=False).fit(prices).expected_returns_
        - EMAHistoricalReturns(returns_data=True).fit(returns).expected_returns_
    )
    assert delta.mean() < TOLERANCE


def test_perturbed_returns(prices, returns):
    delta = PerturbedReturns(returns_data=False).reseed(42).fit_transform(
        prices
    ) - PerturbedReturns(returns_data=True).reseed(42).fit_transform(returns)
    assert delta.mean() < TOLERANCE


def test_capm_returns(prices, returns):
    delta = (
        CAPMReturns(returns_data=False).fit(prices).expected_returns_
        - CAPMReturns(returns_data=True).fit(returns).expected_returns_
    )
    assert delta.mean() < TOLERANCE


def test_rolling_median_returns(prices, returns):
    delta = (
        RollingMedianReturns(returns_data=False).fit(prices).expected_returns_
        - RollingMedianReturns(returns_data=True).fit(returns).expected_returns_
    )
    assert delta.mean() < TOLERANCE

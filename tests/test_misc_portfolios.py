#!/usr/bin/env python

"""Tests for `skportfolio` package."""
from skportfolio.misc import HierarchicalRisk, REPO
from skportfolio import SingleAsset, InverseVolatility, InverseVariance
from .datasets_fixtures import prices, returns


def test_repo(prices, returns):
    REPO(returns_data=False).fit(prices)
    REPO(returns_data=True).fit(returns)


def test_hropt(prices, returns):
    HierarchicalRisk(returns_data=False).fit(prices)
    HierarchicalRisk(returns_data=True).fit(returns)


def test_inverse_vol(prices):
    InverseVolatility().fit(prices)


def test_inverse_var(prices):
    InverseVariance().fit(prices)


def test_single_asset(prices):
    SingleAsset(asset="MSTR").fit(prices)
    try:
        SingleAsset(asset="XXX").fit(prices)
    except ValueError:
        pass

#!/usr/bin/env python

"""Tests for `skportfolio` package."""
import pytest
import numpy as np
import pandas as pd  # type: ignore
from skportfolio.backtest.backtester import Strategy
from skportfolio import EquallyWeighted, InverseVariance, InverseVolatility
from sklearn.base import clone  # type: ignore
from skportfolio.datasets import load_dataset
from .datasets_fixtures import prices, returns, log_returns
from skportfolio.backtest.backtester import Backtester


def test_backtester_inverse_variance(prices):
    prices.index = pd.to_datetime(prices.index)
    prices = prices.resample("D").mean()

    strategy = Strategy(
        initial_weights=EquallyWeighted().fit(prices).weights_,
        initial_portfolio_value=1_000, # units of currency
        estimator=InverseVariance(),
        rebalance_frequency="M",
        lookback_periods=(126, 126),
        buy_sell_fees_pct=(1, 1),  # buy and sell costs
    )

    backtester = Backtester(strategy=strategy, warmup_period=10).fit(prices)
    return

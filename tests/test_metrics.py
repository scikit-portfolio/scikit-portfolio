#!/usr/bin/env python

"""Tests for `carrottrade` package."""
from pytest import fixture
import numpy as np
import pandas as pd
from skportfolio.metrics import (
    sharpe_ratio,
    omega_ratio,
    sortino_ratio,
    semistd,
    skewness,
    calmar_ratio,
    var_gaussian,
    portfolio_return,
    portfolio_vol,
    annualize_vol,
    annualize_rets,
    tail_ratio,
    drawdown,
    downside_risk,
    maxdrawdown,
    kurtosis,
    cdar,
    cvar,
)

from .datasets_fixtures import prices, returns, log_returns, weights, ptf_return


def test_sharpe_ratio(ptf_return):
    val = sharpe_ratio(ptf_return, 0.0, frequency=365)


def test_omega_ratio(ptf_return):
    val = omega_ratio(ptf_return, target_ret=0)


def test_sortino_ratio(ptf_return):
    val = sortino_ratio(ptf_return, riskfree_rate=0, frequency=365)


def test_semistd(ptf_return):
    val = semistd(ptf_return)


def test_skewness(ptf_return):
    val = skewness(ptf_return)


def test_calmar_ratio(ptf_return):
    val = calmar_ratio(ptf_return)


def test_var_gaussian(ptf_return):
    var_gaussian(ptf_return)


def test_portfolio_return(prices, weights):
    portfolio_return(prices, weights)


def test_portfolio_vol(returns, weights):
    portfolio_vol(returns, weights, frequency=252)


def test_annualize_vol(ptf_return):
    annualize_vol(ptf_return.dropna())


def test_annualize_rets(ptf_return):
    annualize_rets(ptf_return)


def test_cdar(ptf_return):
    cdar(ptf_return)


def test_cvar(ptf_return):
    cvar(ptf_return)


def test_tail_ratio(ptf_return):
    tail_ratio(ptf_return)


def test_drawdown(ptf_return):
    drawdown(ptf_return)


def test_downside_risk(ptf_return):
    downside_risk(ptf_return)


def test_maxdrawdown(ptf_return):
    maxdrawdown(ptf_return)


def test_kurtosis(ptf_return):
    kurtosis(ptf_return)

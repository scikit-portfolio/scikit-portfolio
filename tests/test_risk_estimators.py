#!/usr/bin/env python

"""Tests for `carrottrade` package."""
import numpy as np
from pytest import fixture

from skportfolio.datasets import load_tech_stock_prices
from skportfolio.riskreturn import *
from skportfolio.riskreturn import CovarianceExp
from skportfolio.riskreturn import CovarianceGlasso
from skportfolio.riskreturn import CovarianceHierarchicalFilterAverage
from skportfolio.riskreturn import CovarianceHierarchicalFilterComplete
from skportfolio.riskreturn import CovarianceHierarchicalFilterSingle
from skportfolio.riskreturn import CovarianceLedoitWolf
from skportfolio.riskreturn import CovarianceRMT
from skportfolio.riskreturn import MeanHistoricalLinearReturns
from skportfolio.riskreturn import SampleCovariance
from skportfolio.riskreturn import SemiCovariance

TOLERANCE = 1e-5


def assert_similar_matrix(est1, est2):
    assert (
        np.sqrt(((est1.risk_matrix_ - est2.risk_matrix_) ** 2).mean().mean())
        < TOLERANCE
    )


@fixture
def prices():
    yield load_tech_stock_prices()


@fixture
def returns():
    yield load_tech_stock_prices().pct_change().dropna(how="all")


def check_size(X):
    assert X.shape[0] == X.shape[1], "not square covariance"


def test_sample_cov(returns, prices):
    C_returns = sample_covariance(returns, returns_data=True)
    C_prices = sample_covariance(prices, returns_data=False)
    assert np.allclose((C_returns - C_prices).sum().sum(), 0, rtol=TOLERANCE)

    assert_similar_matrix(
        SampleCovariance(returns_data=False).fit(prices),
        SampleCovariance(returns_data=True).fit(returns),
    )


def test_semicovariance(returns, prices):
    C_returns = semicovariance(returns, returns_data=True)
    C_prices = semicovariance(prices, returns_data=False)
    assert np.allclose((C_returns - C_prices).sum().sum(), 0, rtol=TOLERANCE)

    assert_similar_matrix(
        SemiCovariance(returns_data=False).fit(prices),
        SemiCovariance(returns_data=True).fit(returns),
    )


def test_covariance_exp(returns, prices):
    C_returns = covariance_exp(returns, returns_data=True)
    C_prices = covariance_exp(prices, returns_data=False)
    assert np.allclose((C_returns - C_prices).sum().sum(), 0, rtol=TOLERANCE)

    assert_similar_matrix(
        CovarianceExp(returns_data=False).fit(prices),
        CovarianceExp(returns_data=True).fit(returns),
    )


def test_covariance_crr_denoise(returns, prices):
    C_returns = covariance_crr_denoise(returns, returns_data=True)
    C_prices = covariance_crr_denoise(prices, returns_data=False)
    assert np.allclose((C_returns - C_prices).sum().sum(), 0, rtol=TOLERANCE)

    # TODO implement covariancecrrdenoise estimator


def test_covariance_crr_denoise_detoned(returns, prices):
    C_returns = covariance_crr_denoise_detoned(returns, returns_data=True)
    C_prices = covariance_crr_denoise_detoned(prices, returns_data=False)
    assert np.allclose((C_returns - C_prices).sum().sum(), 0, rtol=TOLERANCE)

    # TODO implement covariancecrrdenoise detoned estimator


def test_covariance_spectral(returns, prices):
    C_returns = covariance_denoise_spectral(returns, returns_data=True)
    C_prices = covariance_denoise_spectral(prices, returns_data=False)
    assert np.allclose((C_returns - C_prices).sum().sum(), 0, rtol=TOLERANCE)

    # TODO implement covariancedenoisespectral estimator


def test_covariance_rmt(returns, prices):
    C_returns = correlation_rmt(returns, returns_data=True)
    C_prices = correlation_rmt(prices, returns_data=False)
    assert np.allclose((C_returns - C_prices).sum().sum(), 0, rtol=TOLERANCE)

    assert_similar_matrix(
        CovarianceRMT(returns_data=False).fit(prices),
        CovarianceRMT(returns_data=True).fit(returns),
    )


def test_covariance_glasso(returns, prices):
    C_prices = covariance_glasso(prices, returns_data=False)
    C_returns = covariance_glasso(returns, returns_data=True)
    assert np.allclose((C_returns - C_prices).sum().sum(), 0, rtol=TOLERANCE)
    assert_similar_matrix(
        CovarianceGlasso(returns_data=False).fit(prices),
        CovarianceGlasso(returns_data=True).fit(returns),
    )


def test_covariance_hierarchical_filter_complete(returns, prices):
    C_prices = covariance_hierarchical_filter_complete(prices, returns_data=False)
    C_returns = covariance_hierarchical_filter_complete(returns, returns_data=True)
    assert np.allclose((C_returns - C_prices).sum().sum(), 0, rtol=TOLERANCE)


def test_covariance_hierarchical_filter_complete(returns, prices):
    C_prices = covariance_hierarchical_filter_complete(prices, returns_data=False)
    C_returns = covariance_hierarchical_filter_complete(returns, returns_data=True)
    assert np.allclose((C_returns - C_prices).sum().sum(), 0, rtol=TOLERANCE)

    assert_similar_matrix(
        CovarianceHierarchicalFilterComplete(returns_data=False).fit(prices),
        CovarianceHierarchicalFilterComplete(returns_data=True).fit(returns),
    )


def test_covariance_hierarchical_filter_single(returns, prices):
    C_prices = covariance_hierarchical_filter_single(prices, returns_data=False)
    C_returns = covariance_hierarchical_filter_single(returns, returns_data=True)
    assert np.allclose((C_returns - C_prices).sum().sum(), 0, rtol=TOLERANCE)

    assert_similar_matrix(
        CovarianceHierarchicalFilterSingle(returns_data=False).fit(prices),
        CovarianceHierarchicalFilterSingle(returns_data=True).fit(returns),
    )


def test_covariancehierarchical_filter_average(returns, prices):
    C_prices = covariance_hierarchical_filter_average(prices, returns_data=False)
    C_returns = covariance_hierarchical_filter_average(returns, returns_data=True)
    assert np.allclose((C_returns - C_prices).sum().sum(), 0, rtol=TOLERANCE)

    assert_similar_matrix(
        CovarianceHierarchicalFilterAverage(returns_data=False).fit(prices),
        CovarianceHierarchicalFilterAverage(returns_data=True).fit(returns),
    )


def test_covariance_ledoit_wolf(returns, prices):
    C_prices = covariance_ledoit_wolf(prices, returns_data=False)
    C_returns = covariance_ledoit_wolf(returns, returns_data=True)
    assert np.allclose((C_returns - C_prices).sum().sum(), 0, rtol=TOLERANCE)

    assert_similar_matrix(
        CovarianceLedoitWolf(returns_data=False).fit(prices),
        CovarianceLedoitWolf(returns_data=True).fit(returns),
    )


def test_covariance_oracle_approx(returns, prices):
    C_prices = covariance_oracle_approx(prices, returns_data=False)
    C_returns = covariance_oracle_approx(returns, returns_data=True)
    assert np.allclose((C_returns - C_prices).sum().sum(), 0, rtol=TOLERANCE)

    # TODO implementare covariance OAS


def test_perturbed_returns(returns, prices):
    rets_estimator = PerturbedReturns(
        rets_estimator=MeanHistoricalLinearReturns(),
        risk_estimator=SampleCovariance(),
        random_state=None,
    )
    a = rets_estimator.reseed(42).fit_transform(prices)
    b = rets_estimator.reseed(42).fit_transform(prices)
    assert np.abs((a - b).mean()) < 1e-3

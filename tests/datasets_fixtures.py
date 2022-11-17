from pytest import fixture
import pandas as pd
import numpy as np
from pypfopt.expected_returns import returns_from_prices
from skportfolio import EquallyWeighted
from skportfolio.datasets import load_tech_stock_prices


@fixture
def prices():
    yield load_tech_stock_prices()


@fixture
def returns(prices):
    yield returns_from_prices(prices)


@fixture
def log_returns(prices):
    yield np.log(1 + prices.pct_change().dropna(how="all"))


@fixture
def weights(prices):
    yield EquallyWeighted().fit(prices).weights_


@fixture
def ptf_return(returns, weights):
    yield returns.dot(weights)


@fixture
def dow_prices():
    cols = ["BA", "CAT", "DIS", "GE", "IBM", "MCD", "MSFT"]
    prices = pd.read_parquet("/Users/carlo/Desktop/dowportfolio.parquet")[cols]
    yield

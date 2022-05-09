"""Tests for `skportfolio` package."""

# efficient frontier portfolios
import pandas as pd

from skportfolio.frontier import MinimumVolatility
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import clone
from skportfolio.datasets import load_dataset


def test_estimate_frontier():
    model = MinimumVolatility()
    prices = load_dataset("tech_stocks")
    model.estimate_frontier(prices)

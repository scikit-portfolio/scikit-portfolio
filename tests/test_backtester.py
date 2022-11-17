import numpy as np
import pandas as pd  # type: ignore
from skportfolio.backtest.backtester import Strategy
from skportfolio import EquallyWeighted, InverseVariance, InverseVolatility
from sklearn.base import clone  # type: ignore
from skportfolio.datasets import load_dataset
from skportfolio.backtest.backtester import Backtester
from skportfolio.backtest.fees import variable_transaction_costs


def test_backtester_example1():
    cols = [
        "AA",
        "CAT",
        "DIS",
        "GM",
        "HPQ",
        "JNJ",
        "MCD",
        "MMM",
        "MO",
        "MRK",
        "MSFT",
        "PFE",
        "PG",
        "T",
        "XOM",
    ]
    prices = load_dataset("dowportfolio")[cols]
    prices.index = pd.to_datetime(prices.index)

    backtester = Backtester(
        estimator=EquallyWeighted(),
        name="EquallyWeighted",
        initial_portfolio_value=1_000_000,
        rebalance_frequency=1,
        window_size=0,
        transaction_costs=0.0,
    )

    backtester.partial_fit(X=prices.iloc[0:5, :], y=None)
    backtester.partial_fit(X=prices.iloc[5:10, :], y=None)
    backtester.equity_curve_

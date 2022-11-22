import numpy as np
import pandas as pd  # type: ignore
from skportfolio import EquallyWeighted, InverseVariance, InverseVolatility
from sklearn.base import clone  # type: ignore
from skportfolio.datasets import load_dataset
from skportfolio.backtest.backtester import Backtester
from tabulate import tabulate


def test_backtester_example_partial_fit():
    """
    See this: https://it.mathworks.com/help/finance/backtestengine.equitycurve.html
    """
    cols = ["BA", "CAT", "DIS", "GE", "IBM", "MCD", "MSFT"]
    prices = pd.read_parquet("/Users/carlo/Desktop/dowportfolio.parquet")[cols]

    backtester = Backtester(
        estimator=EquallyWeighted(),
        name="EquallyWeighted",
        initial_weights=EquallyWeighted().fit(prices).weights_,
        initial_portfolio_value=10_000,
        rebalance_frequency=0,
        window_size=0,
        rates_frequency=252,
        risk_free_rate=0.0,
        transaction_costs=0.0,
        warm_start=False,
    )

    for i in range(1, prices.shape[0]):
        backtester.partial_fit(X=prices.iloc[:i, :], y=None, full_index=prices.index)

    print(tabulate(backtester.equity_curve_.add(-10000).to_frame(), headers="keys"))
    # print(tabulate(backtester.positions_.head(10), headers="keys"))
    # print(tabulate(backtester.returns_.head(10).to_frame(), headers="keys"))


def test_backtester_example_fit():
    cols = ["BA", "CAT", "DIS", "GE", "IBM", "MCD", "MSFT"]
    prices = pd.read_parquet("/Users/carlo/Desktop/dowportfolio.parquet")[cols]
    from skportfolio import MaxSharpe

    backtester = Backtester(
        estimator=EquallyWeighted(),
        name="EquallyWeighted",
        warmup_period=0,
        initial_weights=EquallyWeighted().fit(prices).weights_,
        initial_portfolio_value=10_000,
        rebalance_frequency=60,
        window_size=0,
        rates_frequency=252,
        risk_free_rate=0.00,
        transaction_costs=0.005,
    )

    backtester.fit(prices)
    # same values as in Matlab
    # https://it.mathworks.com/help/finance/backtestengine.runbacktest.html
    # average turnover
    average_turnover = backtester.turnover_.reindex_like(prices).fillna(0).mean()
    # total return
    total_return = (
        1 - backtester.equity_curve_.iloc[-1] / backtester.equity_curve_.iloc[0]
    )
    # average sell cost
    average_sell_cost = (
        backtester.buy_sell_costs_["sell"].reindex_like(prices).fillna(0).mean()
    )
    average_buy_cost = (
        backtester.buy_sell_costs_["buy"].reindex_like(prices).fillna(0).mean()
    )
    from skportfolio import maxdrawdown

    maxdrawdown = -maxdrawdown(backtester.equity_curve_.pct_change())

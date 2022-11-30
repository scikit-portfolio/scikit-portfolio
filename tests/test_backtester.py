from pytest import fixture
import numpy as np
import pandas as pd  # type: ignore
from skportfolio import EquallyWeighted
from sklearn.base import clone  # type: ignore
from skportfolio import InverseVariance, maxdrawdown, sharpe_ratio, MaxSharpe
from skportfolio.backtest.backtester import Backtester
from sklearn.model_selection import GridSearchCV  # type: ignore
from skportfolio.datasets import load_dow_prices
from sklearn.model_selection import cross_val_score


@fixture
def prices():
    cols = ["BA", "CAT", "DIS", "GE", "IBM", "MCD", "MSFT"]
    return load_dow_prices()[cols]


def test_backtester_equally_weighted(prices):
    backtester = Backtester(
        estimator=EquallyWeighted(),
        name="EquallyWeighted",
        warmup_period=0,
        initial_weights=EquallyWeighted().fit(prices).weights_,
        initial_portfolio_value=10_000,
        rebalance_frequency=20,
        window_size=(10, None),
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
    max_drawdown = -maxdrawdown(backtester.equity_curve_.pct_change())


def test_backtester_inverse_variance(prices):
    n_assets = prices.shape[1]
    backtester = Backtester(
        estimator=InverseVariance(),
        name="InverseVariance",
        warmup_period=0,
        initial_weights=EquallyWeighted().fit(prices).weights_,
        initial_portfolio_value=10_000,
        rebalance_frequency=25,
        window_size=(30, 60),
        rates_frequency=252,
        risk_free_rate=0.00,
        transaction_costs=(0.0025, 0.005),
    )
    backtester.fit(prices)
    # same values as in Matlab
    # https://it.mathworks.com/help/finance/backtestengine.runbacktest.html
    # average turnover
    average_turnover = backtester.turnover_.mean()
    max_turnover = backtester.turnover_.max()
    # total return
    total_return = ((1 + backtester.equity_curve_.pct_change()).prod()) - 1
    # average sell cost
    average_sell_cost = backtester.buy_sell_costs_["sell"].mean()
    average_buy_cost = backtester.buy_sell_costs_["buy"].mean()

    sharpe = sharpe_ratio(backtester.returns_, frequency=1)
    max_drawdown = -maxdrawdown(backtester.equity_curve_.pct_change())

    assert np.allclose(
        total_return, 0.22655096214384
    ), "Incorrect result 'total_return'"
    assert np.allclose(
        sharpe, 0.121656520897682, atol=1e-3
    ), "Incorrect result 'sharpe_ratio'"
    assert np.allclose(
        max_drawdown, 0.0945709265778057
    ), "Incorrect result 'sharpe_ratio'"
    assert np.allclose(
        average_turnover, 0.00359812855388656
    ), "Incorrect result 'average_turnover"
    assert np.allclose(max_turnover, 0.2170087594002), "Incorrect result max_turnover"
    backtester.summary(prices)


def test_backtester_grid_search(prices):
    backtester = Backtester(
        estimator=EquallyWeighted(),
        name="EquallyWeighted",
        warmup_period=0,
        initial_weights=EquallyWeighted().fit(prices).weights_,
        initial_portfolio_value=10_000,
        rebalance_frequency=25,
        window_size=0,
        rates_frequency=252,
        risk_free_rate=0.00,
        transaction_costs=(0.005, 0.005),
        score_fcn=None,  # default Sharpe ratio
    )

    grid_search_cv = GridSearchCV(
        estimator=backtester,
        return_train_score=True,
        param_grid=[
            {"estimator": [InverseVariance()]},
            # {"estimator": [EquallyWeighted()]},
            # {"rebalance_frequency": [10, 20, 30, 40, 50, 60]}
        ],
        n_jobs=1,
        cv=[(slice(None), slice(None))],
        verbose=True,
        refit=True,
    ).fit(prices)

from .datasets_fixtures import prices, returns
from skportfolio.frontier import (
    MinimumVolatility,
    MinimumMAD,
    MinimumCDar,
    MinimumCVar,
    MinimumSemiVolatility,
    MinimumOmegaRisk,
)
from skportfolio.plotting import plot_frontier


def test_plot_frontier(prices, returns):

    plot_frontier(
        MinimumVolatility().fit(prices), prices, num_portfolios=10, show_assets=False
    )

    plot_frontier(
        MinimumSemiVolatility().fit(prices),
        prices,
        num_portfolios=10,
        show_assets=False,
    )

    plot_frontier(
        MinimumMAD().fit(prices), prices, num_portfolios=10, show_assets=False
    )

    plot_frontier(
        MinimumOmegaRisk().fit(prices), prices, num_portfolios=10, show_assets=False
    )

    plot_frontier(
        MinimumCVar().fit(prices), prices, num_portfolios=10, show_assets=False
    )

    plot_frontier(
        MinimumCDar().fit(prices), prices, num_portfolios=10, show_assets=False
    )

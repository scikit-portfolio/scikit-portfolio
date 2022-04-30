"""
# Plotting mean-variance efficient frontier
Here we define all the necessary steps to find the maximum sharpe ratio portfolio for the tech stock dataset
"""
# mkdocs_gallery_thumbnail_path = '_static/sharpe_frontier.png'

import matplotlib.pyplot as plt
from pypfopt.expected_returns import returns_from_prices
from skportfolio.datasets import load_tech_stock_prices
from skportfolio import MeanHistoricalLinearReturns, SampleCovariance, EquallyWeighted
from skportfolio.frontier import MaxSharpe

import seaborn as sns

# %%
# We setup all the aesthetics for the plot
#
plt.style.use("ggplot")

# %%
# We then load the tech stock prices and compute the asset historical returns
#


def main():
    prices = load_tech_stock_prices()
    model = MaxSharpe().fit(prices)
    model.plot_frontier(prices, num_portfolios=10)
    plt.show()


if __name__ == "__main__":
    main()

# Here we should implement a backtester that takes one or more portfolio estimator objects,
# possibly a rebalancing policy, transaction costs
from skportfolio import PortfolioEstimator
from .rebalancing import RebalancePolicy


class Strategy:
    def __init__(self, estimator: PortfolioEstimator, rebalancer: RebalancePolicy):
        pass

    def fit(self):
        pass


class StrategySummary:
    def __init__(self):
        pass

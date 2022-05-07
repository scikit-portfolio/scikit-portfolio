"""Tests for `skportfolio` package."""
import numpy as np
import pandas as pd  # type: ignore
from sklearn.base import clone  # type: ignore
from skportfolio import PortfolioEstimator

# efficient frontier portfolios
from skportfolio.frontier import CDarEfficientReturn
from skportfolio.frontier import CDarEfficientRisk
from skportfolio.frontier import CVarEfficientReturn
from skportfolio.frontier import CVarEfficientRisk
from skportfolio.frontier import MADEfficientReturn
from skportfolio.frontier import MADEfficientRisk
from skportfolio.frontier import MaxOmegaRatio
from skportfolio.frontier import MaxSharpe
from skportfolio.frontier import MeanSemiVarianceEfficientReturn
from skportfolio.frontier import MeanSemiVarianceEfficientRisk
from skportfolio.frontier import MeanVarianceEfficientReturn
from skportfolio.frontier import MeanVarianceEfficientRisk
from skportfolio.frontier import MinimumCDar
from skportfolio.frontier import MinimumCVar
from skportfolio.frontier import MinimumMAD
from skportfolio.frontier import MinimumSemiVolatility
from skportfolio.frontier import MinimumVolatility
from skportfolio.frontier import OmegaEfficientReturn
from skportfolio.frontier import OmegaEfficientRisk


from sklearn.utils.estimator_checks import check_estimator


def test_efficient_frontier_portfolios():
    models = [
        # Markowitz MVO
        MinimumVolatility(),
        MeanVarianceEfficientRisk(),
        MeanVarianceEfficientReturn(),
        MaxSharpe(),
        # Semivariance frontier
        MinimumSemiVolatility(),
        MeanSemiVarianceEfficientReturn(),
        MeanSemiVarianceEfficientRisk(),
        # CVar efficient frontier
        MinimumCVar(),
        CVarEfficientReturn(),
        CVarEfficientRisk(),
        # CDar efficient frontier
        MinimumCDar(),
        CDarEfficientReturn(),
        CDarEfficientRisk(),
        # # Omega frontier
        MaxOmegaRatio(),
        OmegaEfficientReturn(),
        OmegaEfficientRisk(),
        # MAD Frontier
        MinimumMAD(),
        MADEfficientReturn(),
        MADEfficientRisk(),
    ]
    for model in models:
        check_estimator(model)

def test_clone_estimator():
    from sklearn.base import clone
    model = MinimumVolatility()
    cloned_model = clone(model)

"""
Portfolio optimization for Python
==================================

scikit-portfolio is a Python module integrating classical portfolio optimization methods in the tightly-knit world of
scientific Python packages (numpy, scipy, matplotlib).
It aims to provide simple and efficient solutions to portfolio optimization problems
that are accessible to everybody and reusable in various contexts.
"""

from skportfolio import misc
from skportfolio.misc._hrp import HierarchicalRisk
from skportfolio.misc._repo import REPO

# Efficient frontier portfolios
from skportfolio import frontier
from skportfolio.frontier import *

# Returns and risk estimators
from skportfolio import riskreturn
from skportfolio.riskreturn import *

# Metrics
from skportfolio import metrics
from skportfolio.metrics import *

# Model selection
from skportfolio import model_selection
from skportfolio.model_selection import *

# Plotting utilities
from skportfolio import plotting
from skportfolio.plotting import *

# backtesting utilities
from skportfolio.backtest import backtester
from skportfolio.backtest import *

# constraints utilities
from skportfolio import constraints
from skportfolio.constraints import *

# utilities and other portfolio estimators
# portfolio estimators
from ._base import PortfolioEstimator
from ._simple import (
    EquallyWeighted,
    InverseVariance,
    InverseVolatility,
    SingleAsset,
    CapWeighted,
    single_asset_weights,
)

# Utilities needed to post-process weights
from .weights import (
    redistribute_non_allocatable,
    discrete_allocation,
    clean_weights,
)

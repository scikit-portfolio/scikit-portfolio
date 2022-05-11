"""
Portfolio optimization for Python
==================================

scikit-portfolio is a Python module integrating classical portfolio optimization methods in the tightly-knit world of
scientific Python packages (numpy, scipy, matplotlib).
It aims to provide simple and efficient solutions to portfolio optimization problems
that are accessible to everybody and reusable in various contexts.
"""

# portfolio estimators
from ._base import (
    PortfolioEstimator,
    EquallyWeighted,
    InverseVariance,
    InverseVolatility,
    SingleAsset,
    CapWeighted,
)

# Utilities needed to post-process weights
from ._base import (
    redistribute_non_allocable,
    discrete_allocation,
    clean_weights,
    single_asset_weights,
)

import skportfolio.misc as misc
from skportfolio.misc._hrp import HierarchicalRisk
from skportfolio.misc._repo import REPO

# Efficient frontier portfolios
import skportfolio.frontier as frontier
from skportfolio.frontier import *

# Returns and risk estimators
import skportfolio.riskreturn as riskreturn
from skportfolio.riskreturn import *

# Metrics
import skportfolio.metrics as metrics
from skportfolio.metrics import *

# Model selection
import skportfolio.model_selection as model_selection
from skportfolio.model_selection import *

# Plotting utilities
import skportfolio.plotting as plotting
from skportfolio.plotting import *

# backtesting utilities
import skportfolio.backtest as backtest
from skportfolio.backtest import *

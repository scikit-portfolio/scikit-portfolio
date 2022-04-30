from ._scorer import PortfolioScorer
from ._scorer import make_scorer
from .metrics import aggregate_returns
from .metrics import annualize_rets
from .metrics import annualize_vol
from .metrics import calmar_ratio
from .metrics import cdar
from .metrics import cvar
from .metrics import corrected_sharpe_ratio
from .metrics import cumulative_returns
from .metrics import cvar_historic
from .metrics import downside_risk
from .metrics import drawdown
from .metrics import final_cum_returns
from .metrics import l1_risk_ratio
from .metrics import maxdrawdown
from .metrics import number_effective_assets
from .metrics import omega_ratio
from .metrics import portfolio_return
from .metrics import portfolio_vol
from .metrics import semistd
from .metrics import sharpe_ratio
from .metrics import kurtosis
from .metrics import sharpe_ratio_se
from .metrics import skewness
from .metrics import sortino_ratio
from .metrics import tail_ratio
from .metrics import value_at_risk
from .metrics import var_gaussian
from .metrics import var_historic

# Here is a list of sklearn compatible scorers
sharpe_ratio_scorer = make_scorer(
    sharpe_ratio, greater_is_better=True, riskfree_rate=0.0
)
omega_ratio_scorer = make_scorer(omega_ratio, greater_is_better=True, target_ret=0.0)
sortino_ratio_scorer = make_scorer(
    sortino_ratio, greater_is_better=True, riskfree_rate=0.0
)
maxdrawdown_scorer = make_scorer(maxdrawdown, greater_is_better=False)
downside_risk_scorer = make_scorer(downside_risk, greater_is_better=False)
semistd_scorer = make_scorer(semistd, greater_is_better=False)
calmar_ratio_scorer = make_scorer(calmar_ratio, greater_is_better=True)
var_gaussian_scorer = make_scorer(var_gaussian, greater_is_better=False)
cvar_scorer = make_scorer(cvar, greater_is_better=False)
var_scorer = make_scorer(value_at_risk, greater_is_better=False)
cdar_scorer = make_scorer(cdar, greater_is_better=False)

all_scorers = {
    "sharpe_ratio": sharpe_ratio_scorer,
    "omega_ratio": omega_ratio_scorer,
    "sortino_ratio": sortino_ratio_scorer,
    "maxdrawdown": maxdrawdown_scorer,
    "downside_risk": downside_risk_scorer,
    "semistd": semistd_scorer,
    "calmar_ratio": calmar_ratio_scorer,
    # "cvar_historic": cvar_historic_scorer,
    # "var_historic": var_historic_scorer,
    "var_gaussian": var_gaussian_scorer,
    "value_at_risk": var_scorer,
    "cvar": cvar_scorer,
    "cdar": cdar_scorer,
}

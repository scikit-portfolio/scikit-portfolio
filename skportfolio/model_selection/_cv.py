from typing import Iterable
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import cross_validate

from skportfolio._base import PortfolioEstimator
from skportfolio.metrics import PortfolioScorer
from skportfolio.metrics import sharpe_ratio_scorer
from skportfolio.model_selection import CrossValidator


def portfolio_cross_validate(
    estimator: Union[PortfolioEstimator, Iterable[PortfolioEstimator]],
    prices_or_returns: pd.DataFrame,
    cv: Union[BaseCrossValidator, CrossValidator, int],
    scoring: Union[PortfolioScorer, Iterable[PortfolioScorer]] = sharpe_ratio_scorer,
    n_jobs: int = -1,
    verbose: int = 0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    error_score=np.nan,
    return_train_score: bool = False,
    return_dataframe: bool = True,
):
    """

    Parameters
    ----------
    estimator: Union[PortfolioEstimator, Iterable[PortfolioEstimator]]
        The portfolio estimator or an iterable of portfolio estimators to run multiple estimators
    prices_or_returns:
        The prices or returns dataframe. When feeding returns, please make sure to set "returns_data=True" as attribute
        in each estimator
    cv: Union[int, BaseCrossValidator, CrossValidator]
        A cross validator object or simply the number of folds to divide train and test.
    scoring: UnionPortfolioScorer
        A portfolio scorer object, by default sharpe_ratio_scorer or, alternatively an iterable for performing multiple
        portfolio scoring.
    n_jobs: int
        Number of jobs. If -1 then all processors are utilized
    verbose: int
        Whether to print debugging information
    fit_params:
        Dictionary of key-value parameters to be fed to each individual estimator.fit(X,y,**kwargs) method
    pre_dispatch:
        Same as in sklearn cross_validate
    error_score:
        Default score to assing in case of errors arose from the portfolio `.fit` method. Default returns np.nan
    return_train_score:
        Also includes training set score or not. Default False
    return_dataframe:
        True if a dataframe is returned instead of a dictionary

    Returns
    -------
    Either a dictionary or a dataframe with all the selected portfolio metrics over all train-test folds defined
    from the cv object.
    """
    cv_out = None
    if isinstance(estimator, PortfolioEstimator):
        cv_out = cross_validate(
            estimator=estimator,
            X=prices_or_returns,
            y=None,
            groups=None,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            return_train_score=return_train_score,
            verbose=verbose,
            fit_params=fit_params,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
        )
        if return_dataframe:
            cv_out = pd.concat(
                [pd.DataFrame(cv_out)], keys=[estimator.__class__.__name__], axis=1
            )  # this prepends a level to columns with the estimator name
    elif all(isinstance(p, PortfolioEstimator) for p in estimator):
        cv_out = {
            str(p): cross_validate(
                estimator=p,
                X=prices_or_returns,
                y=None,
                groups=None,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                fit_params=fit_params,
                pre_dispatch=pre_dispatch,
                return_train_score=return_train_score,
                error_score=error_score,
            )
            for p in estimator
        }
        if return_dataframe:
            cv_out = pd.concat(
                (
                    pd.DataFrame(v).rename(index={"index": "scorer"})
                    for k, v in cv_out.items()
                ),
                axis=1,
                keys=list(cv_out.keys()),
                names=["estimator"],
            )
    cv_out.index.rename("fold", inplace=True)
    return cv_out

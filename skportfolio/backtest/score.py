from typing import Union, Callable, Optional
import pandas as pd

from skportfolio import sharpe_ratio, all_scorers

BacktestScorer = Optional[
    Union[
        str,
        Callable[
            [
                pd.Series,
            ],
            float,
        ],
    ]
]


def prepare_score_fcn(score_fcn: BacktestScorer) -> Callable[[pd.Series], float]:
    """
    Defines the backtester scoring function when the Backtester class
    is used inside hyperparameters search estimators like GridSearchCV

    Parameters
    ----------
    score_fcn: Union[str, Callable]

    Returns
    -------
    Callable
    """
    if score_fcn is None:
        return sharpe_ratio
    if isinstance(score_fcn, str):
        if score_fcn not in all_scorers.keys():
            raise ValueError(
                f"Must select from the available scorers {list(all_scorers.keys())}"
            )
        return all_scorers[score_fcn]._score_func
    elif callable(score_fcn):
        return score_fcn
    else:
        raise TypeError(f"Not a supported scoring method {score_fcn}")

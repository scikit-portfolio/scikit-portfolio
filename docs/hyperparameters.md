# Portfolio hyperparameters optimization
One of the strongest abilities on `scikit-portfolio` is the possibility to backtest portfolios on historical data and automatically select the best portfolio hyperparameters.

You can optimize a set of `PortfolioEstimator` hyperparameters, or better, optimize a backtesting strategy that embeds a number of portfolio estimators. 

## scikit-learn `GridSearchCV` and similar

For example you can make a grid search over a large cartesian product of estimators and parameters for a `Backtester` estimator like follows:

!!! warning
	Importantly, and differently from classical `sklearn` you have to specify no test data both to the `GridSearchCV`, `RandomizedSearchCV` and all the `...CV` methods.
	Specify `[(slice(None), slice(None))]` to the `cv` parameter of `GridSearchCV` to disable the division of train and test set when searching over the parameter space of a `Backtester`.

```python
import pandas as pd
from skportfolio.riskreturn import EMAHistoricalReturns, MedianHistoricalLinearReturns, MeanHistoricalLinearReturns,
    SampleCovariance, CovarianceGlasso, CovarianceExp, CovarianceRMT
from sklearn.model_selection import GridSearchCV
from skportfolio.backtest.backtester import Backtester
from skportfolio import InverseVariance, EquallyWeighted, MinimumMAD, MaxOmegaRatio, MinimumCDar, MinimumCVar,
    sharpe_ratio

returns_estimators = [EMAHistoricalReturns(), MedianHistoricalLinearReturns(), MeanHistoricalLinearReturns()]
risk_estimators = [SampleCovariance(), CovarianceExp(), CovarianceGlasso(), CovarianceRMT()]
from skportfolio.datasets import load_dow_prices

prices = load_dow_prices()
cols = prices.columns
grid_search_cv = GridSearchCV(
    estimator=Backtester(
        estimator=InverseVariance(),
        name="BestEstimatorStrategy",
        warmup_period=0,
        initial_weights=pd.Series([1.0 / 7.0] * len(cols), index=cols),  # initialize with equal weights
        initial_portfolio_value=10_000,
        rebalance_frequency_or_signal=25,
        window_size=0,  # to have an expanding window approach
        rates_frequency=252,
        risk_free_rate=0.00,
        transaction_costs=(0.005, 0.005),
        score_fcn=sharpe_ratio  # to sort the results based on the strategy Omega ratio
    ),
    param_grid=[
        {'estimator': [InverseVariance()]},
        {"estimator": [EquallyWeighted()]},
        {"rebalance_frequency": [10, 20, 30, 40, 50, 60]},
        {"estimator": [MinimumMAD()], "estimator__rets_estimator": returns_estimators},
        {"estimator": [MinimumCVar()], "estimator__rets_estimator": returns_estimators},
        {"estimator": [MinimumCDar()], "estimator__rets_estimator": returns_estimators},
        {"estimator": [MaxOmegaRatio()], "estimator__rets_estimator": returns_estimators,
         "estimator__l2_gamma": np.logspace(-2, 1, 20)},
    ],
    n_jobs=-1,
    cv=[(slice(None), slice(None))],  # because we are unsupervised and need to test set.
    verbose=True,
    refit=True,
).fit(prices)
```



### Optuna

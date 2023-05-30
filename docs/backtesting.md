# Backtesting

`scikit-portfolio` offers a method for backtesting a long-only investment strategy based on the usage of the portfolio 
estimators shipped with it.

In developing the backtesting of a portfolio strategy, you only need to define a couple of parameters that are fundamental
for the run of the backtest.


```python
import pandas as pd
from skportfolio.backtest.backtester import Backtester
from skportfolio import InverseVariance
from skportfolio.datasets import load_dow_prices
from skportfolio import sharpe_ratio

# load some price data
prices = load_dow_prices()
# define the backtest strategy
backtester=Backtester(
    estimator=InverseVariance(),
    name="BestEstimatorStrategy",
    warmup_period=0,
    initial_weights=None,
    initial_portfolio_value=10_000,
    rebalance_frequency=25,
    window_size=None, # to have an expanding window approach
    rates_frequency=252,
    risk_free_rate=0.00,
    transaction_costs=(0.005,0.005),
    score_fcn=sharpe_ratio # to sort the results based on the strategy Sharpe ratio
)

# run it on data
backtester.fit(prices)

# get the results
backtester.equity_curve_
```

!!! warning
    The `backtester` object is a sklearn estimator that has the only the `fit` and the `fit_predict` method, 
    as it is impossible to have a `predict` method. You can look at it more like a clustering algorithm rather than a 
    supervised method. Backtesting is **not inductive** and so cannot be directly applied to new data samples 
    without recomputing everything. After all, it's called **back**-tester. **However** any portfolio estimator is an
    inductive method, as they have the `.predict` method. For this reason, for example after a grid-search for the best
    parameters on some historical data, you can take the `best_estimator_.estimator` and call the `.predict` on new data.

The results of the backteter are encoded in three important variables:

```python
backtester.positions_
backtester.equity_curve_
backtester.returns_
```

The `positions_` contains all the asset positions over the course of the time the backtest has ran over.
The `equity_curve_` is a `pd.Series` starting at `initial_portfolio_value` representing the value of the strategy.
Finally, the `returns_` encodes nothing more than the linear returns of the strategy and it has one row less than the 
equity curve.

### Encoding the transaction costs 💸
The `transaction_costs` variable defines: 

- constant buy or sell relative percentange transaction costs (single scalar value)
- specific buy or sell relative percentage transaction costs (pair of scalars in the form of a 2-valued tuple)
- A more complex callable.

Examples of complex transaction costs policies are implemented in the [fee](fees.py) module.
For example you can encode variable transaction costs policies based on position sizes as follows

````python
def variable_transaction_costs(
    delta_pos: pd.Series, fee_scheme: Union[FeeRelativeScheme, FeeBuySellRelativeScheme]
) -> Tuple[float, float]:
    """
    Compute scaled transaction costs based on the change in market value of
    each asset after a rebalance.
    For example if cost_scheme is
    {
        "buy": [(10_000, 0.005), (None, 0.035)]
        "sell": [(1_000, 0.0075), (None, 0.035)]
    }
    Costs are computed at the following rates:

    Buys:
      $0-$10,000 : 0.5%
      $10,000+   : 0.35%
    Sells:
       $0-$1,000  : 0.75%
       $1,000+    : 0.5%

    Parameters
    ----------
    delta_pos: Pd.Series
        Difference in positions as measured in currency value
    fee_scheme:
        Relative transaction costs

    Returns
    -------
    Buy and sell transaction costs
    """
    # replace None with Inf
    buy = sell = pd.Series(data=0, index=delta_pos.index)

    for limit, fee_pct in fee_scheme["buy"]:
        if limit is None:
            limit = np.inf
        idx_buy = (0 < delta_pos) & (delta_pos < limit)
        buy[idx_buy] = fee_pct * delta_pos[idx_buy]

    for limit, fee_pct in fee_scheme["sell"]:
        if limit is None:
            limit = -np.inf
        idx_sell = (-limit < delta_pos) & (delta_pos < 0)
        sell[idx_sell] = fee_pct * -delta_pos[idx_sell]

    buy_cost = buy.sum()
    sell_cost = sell.sum()
    return buy_cost, sell_cost
````

## Hyperparameters optimization with a Backtester

Yes, you can perform exhaustive or randomized grid search for the best parameters also on a `Backtester` object, since
it is another `sklearn` estimator. Always remember though that the `Backtester` object is not an inductive estimator, so
in optimizing over historical data you have no `predict` method for new data. For this reason you need to specify to cross
validation options, by setting `cv=[(slice(None), slice(None))]` in both the `GridSearchCV`, `RandomizedSearchCV` or any
other derived hyperparameters optimization estimator, such as the one provided by 
[Optuna](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.OptunaSearchCV.html).

Here is an example implementation:

````python
risk_estimators = [SampleCovariance(), CovarianceExp(span=30), CovarianceGlasso(), CovarianceRMT()]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    grid_search_cv = GridSearchCV(
        estimator=Backtester(
            transaction_costs=(0.005,0.005),
            score_fcn=sharpe_ratio,
            show_progress=True
        ),
        param_grid=[
            {'estimator': [InverseVariance()], "rebalance_frequency": [60,120,252,], "window_size": [(60,120),(60,252),(60,504)]},
            {"estimator": [EquallyWeighted()], "rebalance_frequency": [60,120,252,], "window_size": [(60,120),(60,252),(60,504),]},
            {"estimator": [MinimumVolatility()], "estimator__risk_estimator": risk_estimators, "rebalance_frequency": [60,120,252,], "window_size": [(60,120),(60,252),(60,504)]},
            {"estimator": [MinimumCVar()], "rebalance_frequency": [60,120,252,], "window_size": [(60,120),(60,252),(60,504),]},
            {"estimator": [MinimumCDar()], "rebalance_frequency": [60,120,252,], "window_size": [(60,120),(60,252),(60,504),]},
        ],
        n_jobs=8,
        cv=[(slice(None), slice(None))],
        verbose=False,
        refit=True,
    ).fit(prices_train)
````

We are hyperoptimizing not only over the portfolio attributes, but also on meta-hyperparameters which are the 
`rebalance_frequency` as well as the `window_size`.
Clearly there is a strong historical selection bias with this approach, nonetheless it may help to discard very bad strategies.
Here we obviously consider the transaction costs **not** as a hyperparameters, as they are supposed to be fixed.


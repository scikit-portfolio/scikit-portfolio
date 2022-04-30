# Estimators in `scikit-portfolio` class

At the very core of `skportfolio` lies the `PortfolioEstimator` class which borrows many of the ideas of the well-known scikit-learn estimators.

Begin an `Estimator` object, the `PortfolioEstimator` follows all the best-practices suggested when developing new estimators. 
As from the official scikit-learn documentation:

> To have a uniform API, we try to have a common basic API for all the objects. In addition, to avoid the proliferation of framework code, we try to adopt simple conventions and limit to a minimum the number of methods an object must implement.

Here we have strictly followed this suggestion.

## Portfolio Estimator 
The API has one predominant object: the estimator.
An estimator is an object that fits a model based on some training data and is capable of inferring some properties on new data. All portfolio estimators implement the `.fit` method:

```
estimator.fit(X, y)
``` 

All built-in portfolio estimators also have a `.set_params` method, which sets data-independent parameters (overriding previous parameter values passed to `__init__`).

All portfolio estimators in the `skportfolio` codebase should inherit from `skportfolio._base.PortfolioEstimator`.

## Instantiation
This concerns the creation of an object. The object’s `__init__` method might accept constants as arguments that determine the estimator’s behavior. It should **not**, however, take the actual training data as an argument, as this is left to the `fit()` method:

The arguments accepted by `__init__` should all be keyword arguments with a default value.
In other words, a user should be able to instantiate an estimator without passing any arguments to it.

The arguments should all correspond to hyperparameters describing the portfolio model or the class tries to solve.
These initial arguments (or parameters) are always remembered by the estimator.
Also note that they should not be documented under the “Attributes” section, but rather under the “Parameters” section for that estimator.

In addition, every keyword argument accepted by `__init__` should correspond to an attribute on the instance.
`Scikit-learn` and `skportfolio` relies on this to find the relevant attributes to set on an estimator when doing model selection.

To summarize, an `__init__` should look like:

```python
def __init__(self, param1=1, param2=2):
    self.param1 = param1
    self.param2 = param2
```

**There should be no logic, not even input validation, and the parameters should not be changed.**
The corresponding logic should be put where the parameters are used, typically in `.fit`.
The following **is wrong**:

```python
def __init__(self, param1=1, param2=2, param3=3):
    # WRONG: parameters should not be modified
    if param1 > 1:
        param2 += 1
    self.param1 = param1
    # WRONG: the object's attributes should have exactly the name of
    # the argument in the constructor
    self.param3 = param2
```

The reason for postponing the validation is that the same validation would have to be performed in `set_params`, which is used in algorithms like GridSearchCV.

## Fitting
The next thing you will probably want to do is to estimate some parameters in the portfolio model.
This is implemented in the `fit()` method.

The `fit()` method takes the training data as arguments, which can be one array in the case of unsupervised learning, or two arrays in the case of supervised learning.

Note that the model is fitted using `X` and `y`, but the object **holds no reference to X and y**.

|Parameters|
|X| array-like of shape (n_timesteps, n_assets)|
|y| array-like of shape (n_samples,)|
|kwargs| optional data-dependent parameters|

`X.shape[0]` should be the same as `y.shape[0]` whenever `y` is necessary. Most portfolio methods don't need it.
If this requisite is not met, an exception of type ValueError should be raised.
`y` is almost always ignored. However, to make it possible to use the estimator as part of a pipeline.
For the same reason, `fit_predict`, `fit_transform`, `score` and `partial_fit` methods need to accept a y argument in the second place if they are implemented.

The method should return the object (`self`). This pattern is useful to be able to implement quick **one liners** in an IPython session such as:

```python
prices_test_predicted = MinimumVolatility().fit(prices_train).predict(prices_test)
```

Depending on the nature of the algorithm, fit can sometimes also accept additional keywords arguments. However, any parameter that can have a value assigned prior to having access to the data should be an `__init__` keyword argument.
`fit` parameters should be restricted to directly data dependent variables. A tolerance stopping criterion `tol` is not directly data dependent (although the optimal value according to some scoring function probably is).

When fit is called, any previous call to fit should be ignored. 
In general, calling `estimator.fit(X1)` and then `estimator.fit(X2)` should be the same as only calling `estimator.fit(X2)`.
However, this may not be true in practice when fit depends on some random process, see `random_state`.
Another exception to this rule is when the hyper-parameter warm_start is set to True for estimators that support it. warm_start=True means that the previous state of the trainable parameters of the estimator are reused instead of using the default initialization strategy.

## Estimated Attributes
Attributes that have been estimated from the data must always have a name ending with trailing underscore `_', for example the portfolio weights would be stored in a `weight_` attribute after `fit` has been called.

The estimated attributes are expected to be overridden when you call fit a second time.

## Optional Arguments
In iterative algorithms, the number of iterations should be specified by an integer called n_iter.

## get_params and set_params

All scikit-learn estimators have ``get_params`` and ``set_params`` functions.
The ``get_params`` function takes no arguments and returns a dict of the
``__init__`` parameters of the estimator, together with their values.

It must take one keyword argument, ``deep``, which receives a boolean value
that determines whether the method should return the parameters of
sub-estimators (for most estimators, this can be ignored). The default value
for ``deep`` should be `True`. For instance considering the following
estimator::

```python
>>> from sklearn.base import BaseEstimator
>>> from sklearn.linear_model import LogisticRegression
>>> class MyEstimator(BaseEstimator):
...     def __init__(self, subestimator=None, my_extra_param="random"):
...         self.subestimator = subestimator
...         self.my_extra_param = my_extra_param
```

The parameter `deep` will control whether or not the parameters of the
`subsestimator` should be reported. Thus when `deep=True`, the output will be::

```python
>>> my_estimator = MyEstimator(subestimator=LogisticRegression())
>>> for param, value in my_estimator.get_params(deep=True).items():
...     print(f"{param} -> {value}")
my_extra_param -> random
subestimator__C -> 1.0
subestimator__class_weight -> None
subestimator__dual -> False
subestimator__fit_intercept -> True
subestimator__intercept_scaling -> 1
subestimator__l1_ratio -> None
subestimator__max_iter -> 100
subestimator__multi_class -> auto
subestimator__n_jobs -> None
subestimator__penalty -> l2
subestimator__random_state -> None
subestimator__solver -> lbfgs
subestimator__tol -> 0.0001
subestimator__verbose -> 0
subestimator__warm_start -> False
subestimator -> LogisticRegression()
```

Often, the `subestimator` has a name (as e.g. named steps in a
`~sklearn.pipeline.Pipeline` object), in which case the key should
become `<name>__C`, `<name>__class_weight`, etc.

While when `deep=False`, the output will be::

    >>> for param, value in my_estimator.get_params(deep=False).items():
    ...     print(f"{param} -> {value}")
    my_extra_param -> random
    subestimator -> LogisticRegression()

The ``set_params`` on the other hand takes as input a dict of the form
``'parameter': value`` and sets the parameter of the estimator using this dict.
Return value must be estimator itself.

While the ``get_params`` mechanism is not essential (see :ref:`cloning` below),
the ``set_params`` function is necessary as it is used to set parameters during
grid searches.

The easiest way to implement these functions, and to get a sensible
``__repr__`` method, is to inherit from ``sklearn.base.BaseEstimator``. If you
do not want to make your code dependent on scikit-learn, the easiest way to
implement the interface is::

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"alpha": self.alpha, "recursive": self.recursive}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


## Parameters and init

As `model_selection.GridSearchCV` uses ``set_params``
to apply parameter setting to estimators,
it is essential that calling ``set_params`` has the same effect
as setting parameters using the ``__init__`` method.
The easiest and recommended way to accomplish this is to
**not do any parameter validation in** ``__init__``.
All logic behind estimator parameters,
like translating string arguments into functions, should be done in ``fit``.

Also it is expected that parameters with trailing ``_`` are **not to be set
inside the** ``__init__`` **method**. All and only the public attributes set by
fit have a trailing ``_``. As a result the existence of parameters with
trailing ``_`` is used to check if the estimator has been fitted.

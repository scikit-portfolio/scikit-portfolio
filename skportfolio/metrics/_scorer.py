from functools import partial


def _cached_call(cache, estimator, method, *args, **kwargs):
    """Call estimator with method and args and kwargs."""
    if cache is None:
        return getattr(estimator, method)(*args, **kwargs)

    try:
        return cache[method]
    except KeyError:
        result = getattr(estimator, method)(*args, **kwargs)
        cache[method] = result
        return result


class _BasePortfolioScorer:
    def __init__(self, score_func, sign, kwargs):
        self._kwargs = kwargs
        self._score_func = score_func
        self._sign = sign

    def __repr__(self):
        kwargs_string = "".join(
            [", %s=%s" % (str(k), str(v)) for k, v in self._kwargs.items()]
        )
        return "make_scorer(%s%s%s%s)" % (
            self._score_func.__name__,
            "" if self._sign > 0 else ", greater_is_better=False",
            self._factory_args(),
            kwargs_string,
        )

    def _factory_args(self) -> str:
        """Return non-default make_scorer arguments for repr."""
        return ""

    def __call__(self, estimator, X):
        """Evaluate predicted target values for X (prices or returns)

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        return self._score(
            partial(_cached_call, None),
            estimator,
            X,
        )


class PortfolioScorer(_BasePortfolioScorer):
    def _score(self, method_caller, estimator, X):
        """Evaluate predicted target values for X

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        estimator : object
            Trained estimator to use for scoring. Must have a predict
            method; the output of that is used to compute the score.

        X : {array-like, sparse matrix, pd.Series}
            Test data that will be fed to estimator.predict.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        y_pred = method_caller(estimator, "predict", X).pct_change()
        return self._sign * self._score_func(y_pred, **self._kwargs)


def make_scorer(
    score_func,
    greater_is_better=True,
    **kwargs,
):
    """Make a scorer from a performance metric or loss function.
    This is a modified version from sklearn make_scorer function for compatibility
    purpose. Please be careful not to mix it in the same code.

    This factory function wraps scoring functions for use in
    :class:`~sklearn.model_selection.GridSearchCV` and
    :func:`~sklearn.model_selection.cross_val_score`.
    It takes a metric function, such as :func:`~carrottrade.portfolio.metrics.sharpe_ratio`
    and returns a callable that scores an estimator's output.
    The signature of the call is `(estimator, X, y)` where `estimator`
    is the model to be evaluated, `X` is the data and `y` is the
    ground truth labeling (or `None` in the case of unsupervised models).

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    greater_is_better : bool, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.

    Examples
    --------
    """
    sign = 1 if greater_is_better else -1
    return PortfolioScorer(score_func, sign, kwargs)

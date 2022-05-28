import pandas as pd
import numpy as np


def parkinson_volatility(high_price: pd.Series, low_price: pd.Series):
    """
    Parkinson volatility is a volatility measure that uses the stock’s high and low price of the day.
    The main difference between regular volatility and Parkinson volatility is that the latter uses high and low prices
    for a day, rather than only the closing price. That is useful as close to close prices could show little
    difference while large price movements could have happened during the day. Thus Parkinson's volatility is
    considered to be more precise and requires less data for calculation than the close-close volatility.

    One drawback of this estimator is that it doesn't take into account price movements after market close.
    Hence it systematically undervalues volatility. That drawback is taken into account in the Garman-Klass's
     volatility estimator.
    """
    if not (high_price.ndim == low_price.ndim):
        raise ValueError("Inconsistent dimensions")

    ht = high_price
    lt = low_price
    T = ht.shape[0]
    return np.sqrt(1 / (4 * T * np.log(2)) * np.sum(np.log(ht / lt) ** 2))


def garman_klass_volatility(
    open_price: pd.Series,
    high_price: pd.Series,
    low_price: pd.Series,
    close_price: pd.Series,
) -> float:
    """
    Estimates the garman-klass volatility using OHLC prices
    Garman Klass is a volatility estimator that incorporates open, low, high, and close prices of a security.
    Garman-Klass volatility extends Parkinson's volatility by taking into account the opening and closing price.
    As markets are most active during the opening and closing of a trading session, it makes volatility estimation
    more accurate.

    Garman and Klass also assumed that the process of price change is a process of continuous diffusion
     (geometric Brownian motion). However, this assumption has several drawbacks. The method is not robust for
     opening jumps in price and trend movements.

    Despite its drawbacks, the Garman-Klass estimator is still more effective than the basic formula since it takes
    into account not only the price at the beginning and end of the time interval but also intraday price extremums.

    Researchers Rogers and Satchel have proposed a more efficient method for assessing historical volatility that takes
    into account price trends. See Rogers-Satchell Volatility for more detail.

    Parameters
    ----------
    open_price: pd.Series
    high_price: pd.Series
    low_price: pd.Series
    close_price: pd.Series

    Returns
    -------
    Garman-Klass volatility estimate

    See Also
    -------
    https://portfolioslab.com/tools/garman-klass

    References
    ---------
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2508648
    """
    if not (open_price.ndim == high_price.ndim == low_price.ndim == close_price.ndim):
        raise ValueError("Inconsistent dimensions")

    o = open_price
    h = high_price
    l = low_price
    c = close_price
    T = o.shape[0]
    const = 2 * (np.log(2) - 1) / T
    return np.sqrt(
        1 / (2 * T) * np.sum(np.log((h / l)) ** 2) - const * np.log((c / o) ** 2)
    )


def rogers_satchell_volatility(
    open_price: pd.Series,
    high_price: pd.Series,
    low_price: pd.Series,
    close_price: pd.Series,
) -> float:
    """
    Rogers-Satchell is an estimator for measuring the volatility of securities with an average return not equal to zero.
    Unlike Parkinson and Garman-Klass estimators, Rogers-Satchell incorporates drift term (mean return not equal to zero).
    As a result, it provides a better volatility estimation when the underlying is trending.

    The main disadvantage of this method is that it does not take into account price movements between trading sessions.
    It means an underestimation of volatility since price jumps periodically occur in the market precisely at the
    moments between sessions.


    A more comprehensive estimator that also considers the gaps between sessions was developed based on the
    Rogers-Satchel formula in the 2000s by Yang-Zhang. See Yang Zhang Volatility for more detail.
    Returns
    -------
    """

    if not (open_price.ndim == high_price.ndim == low_price.ndim == close_price.ndim):
        raise ValueError("Inconsistent dimensions")

    o = open_price
    h = high_price
    l = low_price
    c = close_price
    T = o.shape[0]
    return np.sqrt(
        np.sum(np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)) / T
    )


#
# def yang_zhang_volatility(
#     open_price: pd.Series,
#     high_price: pd.Series,
#     low_price: pd.Series,
#     close_price: pd.Series,
# ) -> float:
#     """
#     The Yang-Zhang estimator had the following properties:
#         (a) unbiased,
#         (b) independent of the drift,
#         (c) consistent in dealing with the opening jump
#         (d) smallest variance among all the estimators with similar properties (the typical
#         biweekly Yang-Zhang variance estimation was over 7 times more efficient than the classical variance estimator)
#     Parameters
#     ----------
#     open_price
#     high_price
#     low_price
#     close_price
#
#     Returns
#
#     References:
#     https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2508648
#     -------
#
#     """
#     if not (open_price.ndim == high_price.ndim == low_price.ndim == close_price.ndim):
#         raise ValueError("Inconsistent dimensions")
#
#     o = open_price
#     h = high_price
#     l = low_price
#     c = close_price
#     T = o.shape[0]
#
#     log_ho = np.log(h / o)
#     log_lo = np.log(l / o)
#     log_co = np.log(c / o)
#     log_oc = np.log(o / c.shift(1))
#     log_oc_sigmac = log_oc ** 2
#
#     log_cc = np.log(c / c.shift(1))
#     log_cc_sq = log_cc ** 2
#
#     rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
#
#     close_vol = log_cc_sq.rolling(window=window, center=False).sum() * (
#         1.0 / (window - 1.0)
#     )
#     open_vol = log_oc_sq.rolling(window=window, center=False).sum() * (
#         1.0 / (window - 1.0)
#     )
#     window_rs = rs.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
#
#     k = 0.34 / (1.34 + (window + 1) / (window - 1))
#     result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(
#         np.sqrt
#     ) * math.sqrt(trading_periods)

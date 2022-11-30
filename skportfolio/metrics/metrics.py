from typing import Union
import pandas as pd
import numpy as np
import warnings
from skportfolio.riskreturn import (
    BaseReturnsEstimator,
    BaseRiskEstimator,
    SampleCovariance,
    MeanHistoricalLinearReturns,
)
from skportfolio._constants import (
    APPROX_BDAYS_PER_YEAR,
    WEEKLY,
    MONTHLY,
    QUARTERLY,
    YEARLY,
    FREQUENCIES,
)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import empyrical as ep


def omega_ratio(
    r: pd.Series,
    target_ret: float = 0.0,
    risk_free_rate: float = 0.0,
    frequency: int = APPROX_BDAYS_PER_YEAR,
) -> float:
    """
    Returns the omega ratio of a strategy.
    Omega is a ratio of winning size weighted by probabilities to losing size weighted by probabilities.
    It considers size and odds of winning and losing trades, as well as all moments because the definition incorporates the whole distribution of returns.

    Important advantages are:

    - There is no parameter (estimation).
    - There is no need to estimate (higher) moments.
    - Works with all kinds of distributions.
    - Use a function (of Loss Threshold) to measure performance rather than a single number (as in Sharpe Ratio).
    - It is as smooth as the return distribution.
    - It is monotonic decreasing

    Look here for further details:

    - [Omega ratio paper](https://cs.uwaterloo.ca/~yuying/Courses/CS870_2012/Omega_paper_Short_Cm.pdf)
    - [Beyond Markowitz optimization](https://nmfin.tech/wp-content/uploads/2019/02/beyond-Markowitz-portfolio-optimization.pdf)

    Parameters
    ----------
    r: pd.Series
        Return of the strategy, typically obtained by the dot product of the asset returns dataframe and weights vector.
    target_ret:
        The desidered return per sample, aka the minimum acceptable return.
    risk_free_rate:
        The default risk free rate (default 0)
    frequency:
        The annualization frequency (default APPROX_BDAYS_PER_YEAR)
    Returns
    -------
    The omega ratio of the portfolio returns series

    See Also
    --------
    sharpe_ratio, sortino_ratio, calmar_ratio
    """
    return ep.omega_ratio(
        r,
        risk_free=risk_free_rate,
        required_return=target_ret,
        annualization=frequency,
    )
    # num = 1 + (ret.mean() - target_ret)
    # den = np.maximum(target_ret - ret, 0).mean()
    # if den == 0:
    #     return np.nan
    # else:
    #     return num / den


def annualize_rets(r: pd.Series, frequency: int = APPROX_BDAYS_PER_YEAR) -> float:
    """
    Compunded annual growth rate, also called cagr

    Parameters
    ----------
    r: pd.Series
    frequency: annualization frequency, as number of rows in a period.
    For example with daily returns (rows), frequency=252 calculates the annualized CAGR

    Returns
    -------
    CAGR
    """
    return ep.annual_return(r, "daily", annualization=frequency)


def annualize_vol(
    r: pd.Series, frequency: int = APPROX_BDAYS_PER_YEAR, levy_alpha: float = 2.0
) -> float:
    """
    Annualizes the volatility of a set of returns
    """
    return ep.annual_volatility(r, "daily", alpha=levy_alpha, annualization=frequency)


def sharpe_ratio(
    r: pd.Series,
    riskfree_rate: float = 0.0,
    period: str = "DAILY",
    frequency: int = APPROX_BDAYS_PER_YEAR,
) -> float:
    """
    Computes the annualized sharpe ratio of a set of returns.
    See the notes above about the annualization

    Returns
    -------
    The Sharpe ratio of the portfolio returns
    See Also
    -------
    """
    return ep.sharpe_ratio(
        r, risk_free=riskfree_rate, period=period, annualization=frequency
    )


def info_ratio(
    r: pd.Series,
    benchmark: pd.Series,
    period: str = "DAILY",
    frequency: int = APPROX_BDAYS_PER_YEAR,
) -> float:
    """
    Computes the annualized information ratio of a set of returns.
    Although originally called the “appraisal ratio” by Treynor and Black, the information ratio is the ratio of
    relative return to relative risk (known as “tracking error”). Whereas the Sharpe ratio looks at returns relative
    to a riskless asset, the information   ratio is based on returns relative to a risky benchmark which is known
    colloquially as a “bogey.” Given an asset or portfolio of assets with random returns designated by Asset and a
    benchmark with random returns designated by Benchmark, the information ratio has the form:

    ```
    Mean(Asset − Benchmark) / Sigma (Asset − Benchmark)
    ```

    Here `Mean(Asset − Benchmark)` is the mean of Asset minus Benchmark returns, and `Sigma(Asset - Benchmark)` is the
    standard deviation of Asset minus Benchmark returns. A higher information ratio is considered better than a
    lower information ratio.

    See the notes above about the annualization

    Parameters
    ----------
    r: pd.Series
        Returns of the portfolio
    benchmark:
        Returns of the benchmark
    period: str
        periodicy of the data
    frequency:
        Annualization constant
    Returns
    -------
    The Information Ratio (IR) of the portfolio

    See Also
    -------
    sharpe_ration, omega_ratio
    """
    # assert pd.testing.assert_index_equal(r.index, benchmark.index)
    return np.mean(r - benchmark, axis=0) / np.std(r - benchmark, axis=0)


def l1_risk_ratio(
    r: pd.Series,
    riskfree_rate: float = 0.0,
    frequency: int = APPROX_BDAYS_PER_YEAR,
) -> float:
    """
    Computes the annualized sharpe ratio of a set of returns.

    See the notes above about the annualization
    See Also
    """
    return np.mean(np.abs(r * frequency) - riskfree_rate)


def sharpe_ratio_se(
    r: pd.Series,
    riskfree_rate: float = 0.0,
    period: str = "DAILY",
    frequency: int = APPROX_BDAYS_PER_YEAR,
) -> float:
    """
    Computes the annualized sharpe ratio standard error from a set of returns
    Equation 9 from "The Statistic of Sharpe Ratio" by Andrew Lo
    """
    sr_hat = sharpe_ratio(r, riskfree_rate, period, frequency)
    T = r.shape[0]
    std_err_sr_hat = np.sqrt((1 + 0.5 * sr_hat**2) / T)
    return std_err_sr_hat


def corrected_sharpe_ratio(
    r: pd.Series,
    riskfree_rate: float = 0.0,
    period: str = "DAILY",
    frequency: int = APPROX_BDAYS_PER_YEAR,
) -> float:
    """
    Computes the annualizatin correction of the sharpe ratio of a set of returns.
    Sharpe Ratio:
    Estimation, Confidence Intervals, and Hypothesis Testing
    https://www.twosigma.com/wp-content/uploads/sharpe-tr-1.pdf Equation 10
    """
    # 1. infer the ratio between desidered frequency and actual frequency
    t = pd.infer_freq(r.index)

    if t is None:
        t = 1  # inferred as daily frequency
    else:
        t = FREQUENCIES.get(t, 1)

    q = int(np.floor(frequency / t))
    sr = sharpe_ratio_se(r, riskfree_rate, period, frequency=1)
    var = r.var()

    return (
        sr
        * (frequency / t)
        / np.sqrt(
            q + 2 * np.sum([(q - k) * r.cov(r.shift(k)) / var for k in range(1, q - 1)])
        )
    )


excess_sharpe = ep.excess_sharpe


def semistd(r: pd.Series) -> float:
    """
    Returns the semideviation aka negative semideviation of a strategy r
    Parameters
    ----------
    r: Strategy returns

    Returns
    -------
    The semi-standard deviation
    """
    return r[r < 0].std(ddof=0)


def sortino_ratio(
    r: pd.Series, riskfree_rate: float = 0.0, frequency: int = APPROX_BDAYS_PER_YEAR
) -> float:
    """
    The Sortino ratio is an improvement of the Sharpe ratio.
    What sets the Sortino ratio apart is that it acknowledges the difference between upside and downward risks.
    More specifically, it provides an accurate rate of return, given the likelihood of downside risk, while the
    Sharpe ratio treats both upside and downside risks equally.
    As a rule of thumb, a Sortino ratio of 2 and above is considered ideal.
    Parameters
    ----------
    r: pd.Series
        portfolio returns obtained from (prices @ weights).pct_change()
    riskfree_rate: float
    frequency

    Returns
    -------

    """
    return ep.sortino_ratio(r, riskfree_rate, "daily", annualization=frequency)


def calmar_ratio(r: pd.Series, frequency: int = APPROX_BDAYS_PER_YEAR) -> float:
    """
    Determines the Calmar ratio, a measure of risk-adjusted returns for investment funds based on
    the maximum drawdown. The Calmar ratio is a modified version of the Sterling ratio.
    Parameters
    ----------
    r
    frequency

    Returns
    -------

    """
    return ep.calmar_ratio(r, "daily", annualization=frequency)


def var_historic(r: Union[pd.DataFrame, pd.Series], level: int = 5) -> float:
    """
    VaR Historic (Value At Risk)
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return np.percentile(r, level)
    else:
        raise TypeError("Expected r to be pd.Series or pd.DataFrame")


def skewness(r: pd.Series) -> float:
    """
    Alternative to scipy.stats.skewness
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp / (sigma_r**3)


def kurtosis(r: pd.Series) -> float:
    """
    Alternative to scipy.stats.kurtosis
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp / (sigma_r**4)


def var_gaussian(
    r: Union[pd.Series, pd.DataFrame], level: int = 5, modified: bool = False
) -> float:
    """
    Returns the parametric gaussian VaR of a Series or DataFrame

    http://www.nematrian.com/ValueAtRiskCoherentForGaussian
    Parameters
    ----------
    r: pd.Series
        Strategy returns
    level: float
    modified: bool, default False
        True if Cornish-Fisher
    Returns
    -------
    The gaussian value-at-risk
    """
    from scipy.stats import norm

    z = norm.ppf(level / 100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (
            z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * (k - 3) / 24
            - (2 * z**3 - 5 * z) * (s**2) / 36
        )
    return -(r.mean() + z * r.std(ddof=0))


def cvar(r: pd.Series, level: float = 0.05) -> float:
    """
    Calculates the conditional value at risk, with cutoff level of 5%.

    Parameters
    ----------
    r: pd.Series
        Strategy returns
    level: float
        percentile cutoff tail value
    Returns
    -------
    """
    return ep.stats.conditional_value_at_risk(returns=r.dropna(), cutoff=level)


def cdar(r: pd.Series, level: float = 0.05) -> float:
    """
    Calculate the conditional drawdown of risk (CDaR) of a portfolio/asset.

    Parameters
    ----------
    r: pd.Series
        Historical returns for an asset / portfolio
    level: float
        Confidence level (alpha)

    Returns
    -------
    Conditional drawdown risk
    """
    dd = r.expanding().max() - r  # drawdown
    max_drawdown = dd.expanding().max()
    max_drawdown_at_level = max_drawdown.quantile(1 - level, interpolation="higher")
    return np.nanmean(max_drawdown[max_drawdown >= max_drawdown_at_level])


def value_at_risk(r: pd.Series, level: float = 0.05):
    """
    Computes the value at risk with cutoff level of default 5%
    Parameters
    ----------
    r
    level

    Returns
    -------

    """
    return ep.stats.value_at_risk(returns=r.dropna(), cutoff=level)


def cvar_historic(r: Union[pd.Series, pd.DataFrame], level: int = 5) -> float:
    """
    Computes the conditional VaR of Series or DataFrame
    Parameters
    ----------
    r
    level:

    Returns
    -------
    The conditional value at risk
    """
    if isinstance(r, pd.Series):
        # it is based on all the returns that are less than the historic var
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def cumulative_returns(
    r: pd.Series,
    starting_value: float = 1000,
) -> pd.Series:
    """
    Computes (1+r).prod() * starting_value
    Parameters
    ----------
    r
    starting_value

    Returns
    -------

    """
    return ep.stats.cum_returns(returns=r, starting_value=starting_value)


def final_cum_returns(r: pd.Series, starting_value: float = 0):
    """
    Compute total returns from simple returns.

    Parameters
    ----------
    r : pd.DataFrame, pd.Series, or np.ndarray
       Noncumulative simple returns of one or more timeseries.
    starting_value : float, optional
       The starting returns.

    Returns
    -------
    total_returns : pd.Series, np.ndarray, or float
        If input is 1-dimensional (a Series or 1D numpy array), the result is a
        scalar.

        If input is 2-dimensional (a DataFrame or 2D numpy array), the result
        is a 1D array containing cumulative returns for each column of input.
    """
    if len(r) == 0:
        return np.nan

    if isinstance(r, pd.DataFrame):
        result = (r + 1).prod()
    else:
        result = np.nanprod(r + 1, axis=0)

    if starting_value == 0:
        result -= 1
    else:
        result *= starting_value
    return result


def aggregate_returns(r: pd.Series, convert_to: str) -> pd.Series:
    """
    Aggregates returns by week, month, or year.

    Parameters
    ----------
    r : pd.Series
       Daily returns of the strategy, noncumulative.
    convert_to : str
        Can be 'weekly', 'monthly', or 'yearly'.
    Returns
    -------
    aggregated_returns : pd.Series
    """

    def cumulate_returns(x):
        return cumulative_returns(x).iloc[-1]

    if convert_to == WEEKLY:
        grouping = [lambda x: x.year, lambda x: x.isocalendar()[1]]
    elif convert_to == MONTHLY:
        grouping = [lambda x: x.year, lambda x: x.month]
    elif convert_to == QUARTERLY:
        grouping = [lambda x: x.year, lambda x: int(np.ceil(x.month / 3.0))]
    elif convert_to == YEARLY:
        grouping = [lambda x: x.year]
    else:
        raise ValueError(f"convert_to must be {WEEKLY}, {MONTHLY} or {YEARLY}")

    return r.groupby(grouping).apply(cumulate_returns)


def portfolio_return(
    prices: pd.DataFrame,
    weights: pd.Series,
    rets_estimator: BaseReturnsEstimator = MeanHistoricalLinearReturns(),
) -> float:
    """
    Calculates the portfolio return as $\\mathbb{E}\\lbrack \\mathbf{r}^T \\mathbf{w} \\rbrack$.

    Parameters
    ----------
    prices: pd.Dataframe with the prices
    weights: pd.Series with the weights (must sum to 1)
    rets_estimator: a callable

    Returns
    -------
    The portfolio return
    """
    return (
        rets_estimator.set_returns_data(False)
        .fit(prices)
        .expected_returns_.dot(weights)
    )


def portfolio_vol(
    r: Union[pd.DataFrame, pd.Series],
    weights: pd.Series,
    frequency: int = APPROX_BDAYS_PER_YEAR,
    risk_estimator: BaseRiskEstimator = SampleCovariance(returns_data=True),
):
    """
    Computes the portfolio volatility using the risk_estimator, default set to sample covariance

    Parameters
    ----------
    r: pd.Dataframe with the prices
    weights: pd.Series with the weights (must sum to 1)
    risk_estimator: a skportfolio risk estimator instance, with returns_data set to True

    Returns
    -------
    Portfolio volatility
    """
    risk_estimator.frequency = frequency
    # Force the risk estimator to read from returns data rather than from price data
    cov = risk_estimator.set_returns_data(returns_data=True).fit(r).risk_matrix_
    return np.sqrt(weights.dot(cov).dot(weights))


def tail_ratio(
    r: pd.Series, upper_tail: float = 95.0, lower_tail: float = 5.0
) -> float:
    """
    Determines the ratio between the right (95%) and left tail (5%).
    For example, a ratio of 0.25 means that losses are four times
    as bad as profits.

    Parameters
    ----------
    r : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
         - See full explanation in :func:`~empyrical.stats.cum_returns`.
    upper_tail: float
        Upper percentile
    lower_tail: float
        Lower percentile
    Returns
    -------
    tail_ratio : float
    """

    if len(r) < 1:
        return np.nan

    r = np.asanyarray(r)
    # Be tolerant of nan's
    r = r[~np.isnan(r)]
    if len(r) < 1:
        return np.nan

    return np.abs(np.percentile(r, upper_tail)) / np.abs(np.percentile(r, lower_tail))


def downside_risk(r: pd.Series, target_return: float = 0.0) -> float:
    """
    Calculates downside risk
    Parameters
    ----------
    r: pd.Series
        Strategy retrurns
    target_return: float

    frequency: int, default APPROX_BDAYS_PER_YEAR
         Annualization frequency
    Returns
    -------
    The downside risk metric
    """
    return ep.downside_risk(r, required_return=target_return)


def drawdown(r: pd.Series):
    """
    Takes a time series of asset returns.
    returns a DataFrame with columns for the wealth index,
    the previous peaks and the percentage drawdown

    Parameters
    ----------
    r: Stategy returns

    Returns
    -------
    The drawdowns time series as a seris of the same lenght as strategy returns
    """
    wealth_index = r.add(1).cumprod(axis=0)
    previous_peaks = wealth_index.cummax(axis=0)
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return drawdowns


def maxdrawdown(r: pd.Series):
    """
    Returns the maxdrawdown measure of returns, with negative sign.

    Parameters
    ----------
    r: Stategy returns

    Returns
    -------
    The maximum drawdown
    """
    return ep.stats.max_drawdown(r)


def number_effective_assets(weights: pd.Series):
    """
    Returns a measure of portfolio diversification, known as number of effective assets.
    Its maximum value
    Parameters
    ----------
    weights: pd.Series
        Portfolio weights

    Returns
    -------
    The quantity `sum_i (w_i)^-2`
    """
    return 1.0 / (weights**2).sum()


def summary(
    r: pd.Series,
    frequency: int = APPROX_BDAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
):
    """
    Takes a dataframe with N columns (pairs) and T rows (time) containing the daily prices
    Computes return over rows, volatility over rows, sharpe ratio over rows.
    Most of the metrics are annualized with the standard approach of multiplying by 252 or sqrt(252) depending
    on the metrics. However more complex calculations can be done, as explained in
    [this resource](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3054517).
    """
    return pd.Series(
        {
            "annualized_return": annualize_rets(r=r, frequency=frequency),
            "annualized_volatility": annualize_vol(r=r, frequency=frequency),
            "final_cumulative_return": final_cum_returns(r=r),
            "volatility": r.std(),
            "semi_volatility": semistd(r=r),
            "sharpe_ratio": sharpe_ratio(
                r=r,
                riskfree_rate=risk_free_rate,
                frequency=frequency,
            ),
            "var_historic": var_historic(r=r),
            "cvar_historic": cvar_historic(r=r),
            "calmar_ratio": calmar_ratio(r=r, frequency=frequency),
            "sortino_ratio": sortino_ratio(
                r=r,
                riskfree_rate=risk_free_rate,
                frequency=frequency,
            ),
            "omega_ratio_0": omega_ratio(r=r, target_ret=0, frequency=frequency),
            "omega_ratio_target": omega_ratio(
                r=r, target_ret=target_return, frequency=frequency
            ),
            "max_drawdown": maxdrawdown(r=r),
            "skew": skewness(r=r),
            "var_gaussian_95": var_gaussian(r=r),
            "downside_risk": downside_risk(r=r, target_return=target_return),
            "kurtosis": kurtosis(r=r),
            "tail_ratio_95_5": tail_ratio(r=r, upper_tail=95.0, lower_tail=5.0),
        },
        name=r.name,
    )


def equity_curve(df: Union[pd.Series, pd.DataFrame], initial_value: float = 1):
    """
    An equity curve is a visual representation of the trend and variation in the value of an
    investment or trading account shown on a chart over a specified period of time.
    Normalizes the values, setting all assets to relative value of 1.
    While this function could in theory work on any dataframe, it's only
    useful to compare the performances of equities as result of many Portfolio.predit
    methods series

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe of asset prices
    initial_value:
        Capital at the first instant

    Returns
    -------
    The equity curve. First value is set to initial_value
    """
    return df.div(df.iloc[0]).mul(initial_value)

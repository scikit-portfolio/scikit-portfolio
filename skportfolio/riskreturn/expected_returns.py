"""
Collector of models for expected returns calculation
"""
import numpy as np
import pandas as pd
from pypfopt import expected_returns as expret

from skportfolio._constants import APPROX_BDAYS_PER_YEAR


def mean_historical_return(
    prices_or_returns: pd.DataFrame,
    returns_data=False,
    frequency=APPROX_BDAYS_PER_YEAR,
) -> pd.Series:
    """
    Returns the annualized (based on 252 working days) expected returns from
    average historical data.
    Compounding is set to false by default.
    If frequency=1 then you get the daily historical returns.
    Parameters
    ----------
    prices_or_returns: pd.DataFrame
        Provide either the prices or the returns, must be coherent with what
        is passed as returns_data.
    returns_data: bool
        If passing prices, you need to set returns_data=False, otherwise
        returns_data=True
    frequency: int
        How to annualize the returns, if frequency=1 you get the daily average
         returns, if frequency=252 you assume returns on business days.
         If frequency=30.5 you assume monthly returns.
    Returns
    -------
    The annualized returns.
    """
    return expret.mean_historical_return(
        prices_or_returns,
        returns_data=returns_data,
        compounding=False,
        frequency=frequency,
    )


def mean_historical_log_return(
    prices_or_returns: pd.DataFrame,
    returns_data=False,
    frequency=APPROX_BDAYS_PER_YEAR,
) -> pd.Series:
    """
    Returns the annualized (based on 252 working days) expected **log**-returns
    from average historical data.
    Compounding is set to false by default.
    If frequency=1 then you get the daily historical returns.
    Parameters
    ----------
    prices_or_returns: pd.DataFrame
        Provide either the prices or the returns, must be coherent with what is
         passed as returns_data.
    returns_data: bool
        If passing prices, you need to set returns_data=False, otherwise
        returns_data=True
    frequency: int
        How to annualize the returns, if frequency=1 you get the daily average
        returns, if frequency=252 you assume returns on business days.
        If frequency=30.5 you assume monthly returns.
    Returns
    -------
    The annualized returns.
    """
    return expret.mean_historical_return(
        prices_or_returns,
        returns_data=returns_data,
        compounding=False,
        frequency=frequency,
        log_returns=True,
    )


def median_historical_return(
    prices_or_returns: pd.DataFrame,
    returns_data=False,
    frequency=APPROX_BDAYS_PER_YEAR,
) -> pd.Series:
    """
    Returns the annualized (based on 252 working days) expected returns from average historical data using median.
    Compounding is set to false by default.
    If frequency=1 then you get the daily historical returns.
    Parameters
    ----------
    prices_or_returns: pd.DataFrame
        Provide either the prices or the returns, must be coherent with what is passed as returns_data.
    returns_data: bool
        If passing prices, you need to set returns_data=False, otherwise returns_data=True
    frequency: int
        How to annualize the returns, if frequency=1 you get the daily average returns, if frequency=252 you assume
        returns on business days. If frequency=30.5 you assume monthly returns.
    Returns
    -------
    The annualized returns.
    """
    if returns_data:
        returns = prices_or_returns
    else:
        returns = prices_or_returns.pct_change().dropna(how="all")
    return returns.median() * frequency


def median_historical_log_return(
    prices_or_returns: pd.DataFrame,
    returns_data=False,
    frequency=APPROX_BDAYS_PER_YEAR,
) -> pd.Series:
    """
    Returns the annualized (based on 252 working days) expected returns from average historical data using median.
    Compounding is set to false by default.
    If frequency=1 then you get the daily historical returns.
    Parameters
    ----------
    prices_or_returns: pd.DataFrame
        Provide either the prices or the returns, must be coherent with what is passed as returns_data.
    returns_data: bool
        If passing prices, you need to set returns_data=False, otherwise returns_data=True
    frequency: int
        How to annualize the returns, if frequency=1 you get the daily average returns, if frequency=252 you assume
        returns on business days. If frequency=30.5 you assume monthly returns.
    Returns
    -------
    The annualized returns.
    """
    if returns_data:
        returns = prices_or_returns
    else:
        returns = prices_or_returns.pct_change().dropna(how="all")
    return np.median(np.log(1 + returns)) * frequency


def capm_return(
    prices_or_returns: pd.DataFrame,
    returns_data=False,
    frequency=APPROX_BDAYS_PER_YEAR,
    risk_free_rate=0.0,
    benchmark=None,
) -> pd.Series:
    """
    Returns the annualized (based on 252 working days) expected returns from average historical data using the
    capital asset pricing model.
    Compounding is set to false by default.
    If frequency=1 then you get the daily historical returns.
    Parameters
    ----------
    prices_or_returns: pd.DataFrame
        Provide either the prices or the returns, must be coherent with what is passed as returns_data.
    returns_data: bool
        If passing prices, you need to set returns_data=False, otherwise returns_data=True
    frequency: int
        How to annualize the returns, if frequency=1 you get the daily average returns, if frequency=252 you assume
        returns on business days. If frequency=30.5 you assume monthly returns.
    risk_free_rate: float
        Risk free rate
    benchmark: pd.DataFrame
        Benchmark prices, typically an index.
    Returns
    -------
    The annualized returns.
    """
    return expret.capm_return(
        prices_or_returns,
        returns_data=returns_data,
        frequency=frequency,
        risk_free_rate=risk_free_rate,
        market_prices=benchmark,
    )


def ema_historical_return(
    prices_or_returns: pd.DataFrame,
    returns_data=False,
    frequency: int = APPROX_BDAYS_PER_YEAR,
    span: int = 180,
) -> pd.Series:
    """
    Returns the annualized (based on 252 working days) expected returns from exponential weighted moving average
    Compounding is set to false by default.
    If frequency=1 then you get the daily historical returns.
    Parameters
    ----------
    prices_or_returns: pd.DataFrame
        Provide either the prices or the returns, must be coherent with what is passed as returns_data.
    returns_data: bool
        If passing prices, you need to set returns_data=False, otherwise returns_data=True
    frequency: int
        How to annualize the returns, if frequency=1 you get the daily average returns, if frequency=252 you assume
        returns on business days. If frequency=30.5 you assume monthly returns.
    span: int
    Returns
    -------
    The annualized returns.
    """
    return expret.ema_historical_return(
        prices_or_returns, returns_data=returns_data, frequency=frequency, span=span
    )


def rolling_median_returns(
    prices_or_returns: pd.DataFrame,
    returns_data=False,
    frequency: int = APPROX_BDAYS_PER_YEAR,
    window=20,
):
    """
    Returns the expected returns using a composition of rolling window median average score

    Parameters
    ----------
    prices_or_returns: pd.DataFrame
        Provide either the prices or the returns, must be coherent with what is passed as returns_data.
    returns_data: bool
        If passing prices, you need to set returns_data=False, otherwise returns_data=True
    frequency: int
        How to annualize the returns, if frequency=1 you get the daily average returns, if frequency=252 you assume
        returns on business days. If frequency=30.5 you assume monthly returns.
    window

    Returns
    -------
    The rolling median average return
    """
    if returns_data:
        returns = prices_or_returns
    else:
        returns = prices_or_returns.pct_change().dropna(how="all")

    return returns.rolling(window).median().mean() * frequency


all_return_models = [
    mean_historical_return,
    mean_historical_log_return,
    median_historical_return,
    rolling_median_returns,
    median_historical_log_return,
    ema_historical_return,
    capm_return,
]

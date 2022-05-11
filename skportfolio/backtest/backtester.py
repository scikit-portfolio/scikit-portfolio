# Here we should implement a backtester that takes one or more portfolio estimator objects, possibly a rebalancing policy, transaction costs
import pandas as pd

def equity_curve(df: pd.DataFrame):
    """
    Normalizes the values, setting all assets to relative value of 1.
    While this function could in theory work on any dataframe, it's only
    useful to compare the performances of equities as result of many Portfolio.predit
    methods series

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    The equity curve. First value is set to 1.
    """
    return df.div(df.iloc[0])
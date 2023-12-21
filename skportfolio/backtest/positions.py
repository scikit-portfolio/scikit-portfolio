from typing import List, Union

import numpy as np
import pandas as pd


def prepare_initial_positions(
    initial_weights: pd.Series,
    n_assets: int,
    initial_portfolio_value: float,
    asset_names: List[str],
) -> pd.Series:
    """
    Computes the initial positions in a portfolio based on given parameters.

    Parameters
    ----------
    initial_weights : pd.Series
        The initial weights assigned to each asset in the portfolio.
    n_assets : int
        The total number of assets in the portfolio.
    initial_portfolio_value : float
        The initial total value of the portfolio.
    asset_names : List[str]
        A list of asset names.

    Returns
    -------
    pd.Series
        A pandas Series representing the initial positions in the portfolio,
        including cash and asset positions.

    Raises
    ------
    ValueError
        If the length of initial_weights is not equal to the number of assets.
    """
    if (
        isinstance(initial_weights, (np.ndarray, pd.Series))
        and len(initial_weights) != n_assets
    ):
        raise ValueError(
            "Invalid number of initial weights, provide all weights of each asset"
        )
    return create_positions_series(
        cash_weight=1.0 if initial_weights is None else 1 - initial_weights.sum(),
        asset_weights=[0.0] * n_assets
        if initial_weights is None
        else initial_weights.values,
        asset_names=asset_names,
        cash_name="CASH",
        portfolio_value=initial_portfolio_value,
    )


def create_positions_series(
    cash_weight: float,
    asset_weights: Union[np.ndarray, List[float]],
    asset_names: List[str],
    cash_name: str,
    portfolio_value: float,
) -> pd.Series:
    """
    Utility function to define a position as a pandas Series.

    Parameters
    ----------
    cash_weight : float
        The weight assigned to the cash position in the portfolio.
    asset_weights : Union[np.ndarray, List[float]]
        The weights assigned to each asset in the portfolio.
    asset_names : List[str]
        A list of asset names.
    cash_name : str
        The name assigned to the cash position.
    portfolio_value : float
        The total value of the portfolio.

    Returns
    -------
    pd.Series
        A pandas Series representing the positions in the portfolio, including
        cash and asset positions.
    """
    all_names = (cash_name, *asset_names)
    all_weights = (cash_weight, *asset_weights)
    return pd.Series(dict(zip(all_names, all_weights)), dtype=float).mul(
        portfolio_value
    )


def compute_row_returns(
    idx: int,
    asset_returns: pd.DataFrame,
    start_positions: pd.Series,
    cash_return: float,
    margin_return: float,
):
    """
    Computes the returns for each asset and cash position in the portfolio for a given time index.

    Parameters
    ----------
    idx : int
        The time index for which returns are computed.
    asset_returns : pd.DataFrame
        DataFrame containing historical returns for each asset.
    start_positions : pd.Series
        The initial positions in the portfolio at the beginning of the period.
    cash_return : float
        The return on the cash position.
    margin_return : float
        The return used for margin adjustments.

    Returns
    -------
    pd.Series
        A pandas Series representing the returns for each asset and cash position
        in the portfolio for the specified time index.
    """

    cash_key = "CASH"
    cash_adjustment = margin_return if start_positions[cash_key] < 0 else cash_return
    return pd.concat(
        (pd.Series({cash_key: cash_adjustment}), asset_returns.iloc[idx, :]), axis=0
    ).add(1.0)

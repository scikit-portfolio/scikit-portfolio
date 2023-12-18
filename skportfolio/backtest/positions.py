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
    Computes the initial positions

    Parameters
    ----------
    initial_weights
    n_assets
    initial_portfolio_value
    asset_names

    Returns
    -------
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
    Utility function to define a position as a pd.Series

    Parameters
    ----------
    cash_weight: float
    asset_weights: float
    asset_names: List[str]
    cash_name: str
    portfolio_value: float

    Returns
    -------
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

    Parameters
    ----------
    idx
    asset_returns
    start_positions
    cash_return
    margin_return

    Returns
    -------

    """
    cash_key = "CASH"
    cash_adjustment = margin_return if start_positions[cash_key] < 0 else cash_return
    return pd.concat(
        (pd.Series({cash_key: cash_adjustment}), asset_returns.iloc[idx, :]), axis=0
    ).add(1.0)


from numba import jit


@jit(nopython=True)
def calculate_positions(
    backtester_name,
    asset_returns,
    risk_free_rate,
    cash_borrow_rate,
    n_samples,
    warmup_period,
    min_window_size,
    max_window_size,
    rebalance_signal,
    asset_names,
    transaction_costs_fcn,
    show_progress,
    estimator,
    X,
    y,
    fit_params,
):
    # Pre-allocate arrays for performance
    turnover = np.zeros(n_samples)
    buy_sell_costs = np.zeros((n_samples, 2))
    returns = np.zeros(n_samples)
    positions = np.zeros_like(asset_returns)

    previous_positions = np.zeros_like(
        asset_returns[0]
    )  # Assuming this is initialized correctly outside the loop

    with tqdm(
            iterable=range(n_samples - 1),
            desc=f"Backtesting {backtester_name}...",
            disable=not show_progress,
        ) as progress:
    for idx in progress:
        next_idx = idx + warmup_period + 1
        start_positions = previous_positions
        start_portfolio_value = start_positions.sum()

        # Vectorized computation of row returns
        row_returns = compute_row_returns(
            idx, asset_returns, start_positions, risk_free_rate, cash_borrow_rate
        )
        end_positions = start_positions * row_returns
        end_portfolio_value = end_positions.sum()
        end_asset_weights = end_positions / end_portfolio_value

        needs_rebalance = rebalance_signal[next_idx]
        is_valid_window = next_idx >= min_window_size

        if needs_rebalance and is_valid_window:
            start_window = next_idx - max_window_size + 1
            window_rows = np.arange(max(0, start_window), next_idx + 1)

            # Assuming estimator.fit() is a costly operation and is already optimized
            end_asset_weights_new = estimator.fit(
                X.iloc[window_rows, :][asset_names],
                y.iloc[window_rows] if y is not None else None,
                **fit_params
            ).weights_.copy()
            delta_weights = end_asset_weights_new - end_asset_weights

            turnover[next_idx] = np.abs(delta_weights).sum() * 0.5
            buy_cost, sell_cost = transaction_costs_fcn(
                delta_weights[asset_names] * end_portfolio_value
            )
            buy_sell_costs[next_idx, :] = (buy_cost, sell_cost)
            end_portfolio_value -= buy_cost + sell_cost
            end_asset_weights = end_asset_weights_new

        end_asset_weights["CASH"] = 1.0 - end_asset_weights[asset_names].sum()
        end_positions = end_portfolio_value * end_asset_weights
        positions.append(end_positions)
        returns[next_idx] = end_portfolio_value / start_portfolio_value - 1
        previous_positions = end_positions

    return positions, turnover, buy_sell_costs, returns

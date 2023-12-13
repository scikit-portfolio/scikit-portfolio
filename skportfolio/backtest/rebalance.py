from typing import Union, Literal, Tuple, Optional
import numpy as np
import pandas as pd

_FREQUENCIES = (
    "day",
    "week",
    "month",
    "quarter",
    "semester",
    "year",
    "Bday",
    "bmonth",
    "Bbmonth",
    "Bemonth",
    "bquarter",
)

_FREQUENCY_TO_OFFSET = {
    "day": pd.offsets.Day(),
    "week": pd.offsets.Week(),
    "month": pd.offsets.MonthEnd(),
    "quarter": pd.offsets.QuarterEnd(1),
    "semester": pd.offsets.QuarterEnd(2),
    "year": pd.offsets.YearEnd(),
    "Bday": pd.offsets.BusinessDay(),
    "bmonth": pd.offsets.MonthBegin(),
    "Bbmonth": pd.offsets.BusinessMonthBegin(),
    "Bemonth": pd.offsets.BusinessMonthEnd(),
    "bquarter": pd.offsets.QuarterBegin(),
    "bsemester": pd.offsets.QuarterBegin(2),
}

BacktestRebalancingFrequencyOrSignal = Union[
    Literal["day", "week", "month", "quarter", "semester", "year"],
    pd.offsets.BaseOffset,
    int,
    pd.Series,
]


def make_rebalance_signal(
    start,
    end,
    rebalance_freq: Union[str, pd.offsets.BaseOffset],
    base_freq: Optional[Union[str, pd.offsets.BaseOffset]] = None,
):
    """
    Utility function to create a rebalance signal in the given date range.

    Parameters
    ----------
    start : str or datetime-like
        The start date of the range.
    end : str or datetime-like
        The end date of the range.
    rebalance_freq : str or pd.offsets.BaseOffset
        The frequency at which rebalancing should occur.
    base_freq : str or pd.offsets.BaseOffset, optional
        The frequency of the resulting rebalance signal.
        If not provided, it defaults to rebalance_freq.

    Other Parameters
    ----------------
    Parameters to be passed to pd.date_range.

    Returns
    -------
    pd.Series
        A rebalance signal as a boolean Series, indicating when rebalancing should occur.
    """

    # Create a base index using pd.date_range with the specified rebalance frequency
    base_index = pd.date_range(
        start=start, end=end, freq=rebalance_freq, inclusive="left"
    )

    # Create a Series with 1s at the rebalance dates and 0s elsewhere
    rebalance_signal = pd.Series(index=base_index, data=(1,) * len(base_index))

    # Resample the Series to the desired frequency and get the first value in each interval
    rebalance_signal = rebalance_signal.resample(
        base_freq, closed="left", convention="start"
    ).first()

    # Fill NaN values with 0 and downcast to boolean
    rebalance_signal = rebalance_signal.fillna(0, downcast="bool")
    return rebalance_signal


# def prepare_rebalance_signal2(
#     rebalance_frequency_or_signal: BacktestRebalancingFrequencyOrSignal,
#     index: Union[pd.Index, pd.DatetimeIndex, pd.Series],
#     **resampling_kwargs,
# ) -> pd.Series:
#     """
#     Computes the transaction event series containing False and True
#     for when to trigger rebalance.
#     Parameters
#     ----------
#     rebalance_frequency_or_signal
#     index
#     resampling_kwargs
#     Returns
#     -------
#     The rebalance signal, a series with the same index as the data, filled with False.
#     At rebalance event, a True is specified at the given date.
#     """
#     validate_rebalance_frequency(rebalance_frequency_or_signal)
#     index = convert_index_to_series(index)
#
#     # Initialize rebalance signal with False
#     rebalance_signal = pd.Series(index=index.index, data=False)
#
#     # Calculate rebalance events based on rebalance_frequency
#     rebalance_events = calculate_rebalance_events(
#         rebalance_frequency_or_signal, index, resampling_kwargs
#     )
#
#     # Set True at the rebalance frequency events
#     rebalance_signal.loc[rebalance_events] = True
#
#     return rebalance_signal, rebalance_events.index


def prepare_rebalance_signal(
    rebalance_frequency_or_signal: BacktestRebalancingFrequencyOrSignal,
    index: Union[pd.Index, pd.DatetimeIndex, pd.Series],
    **resampling_kwargs,
) -> Tuple[pd.Series, pd.Index]:
    """
    Computes the transaction event series containing False and True
    for when to trigger rebalance.

    Parameters
    ----------
    rebalance_frequency_or_signal
    index
    resampling_kwargs

    Returns
    -------
    The rebalance signal, a series with the same index as the data, filled with False.
    At rebalance event, a True is specified at the given date.
    """
    if not isinstance(
        rebalance_frequency_or_signal, (pd.Series, int, pd.offsets.BaseOffset, str)
    ):
        raise ValueError("Not a supported type for rebalance_frequency parameters")
    # in this case the user already provides a rebalance signal, we return it as it is and
    # together with the rebalance events
    if isinstance(rebalance_frequency_or_signal, pd.Series):
        if rebalance_frequency_or_signal.dtypes == bool:
            if not rebalance_frequency_or_signal.index.equals(index):
                raise ValueError(
                    "Must provide a rebalance signal with the same index as data"
                )
            rebalance_events = rebalance_frequency_or_signal[
                rebalance_frequency_or_signal
            ].index
            return (
                rebalance_frequency_or_signal,
                rebalance_events,
            )
        else:
            raise TypeError("Not a supported rebalance signal. Must be boolean")
    if isinstance(rebalance_frequency_or_signal, str):
        if rebalance_frequency_or_signal not in _FREQUENCIES:
            raise ValueError(
                f"Supported strings for rebalance frequencies are {_FREQUENCIES}"
            )
    elif isinstance(rebalance_frequency_or_signal, int):
        if rebalance_frequency_or_signal < 0:
            raise ValueError("Rebalancing frequency must be greater or equal than 1")

    if isinstance(index, (pd.Index, pd.DatetimeIndex)):
        index = index.to_series()

    rebalance_signal = pd.Series(index=index.index, data=False)
    # fill with True at the rebalance frequency events
    if isinstance(rebalance_frequency_or_signal, int):
        rebalance_events = index[
            np.mod(np.arange(len(index)), rebalance_frequency_or_signal) == 0
        ]
        rebalance_signal.loc[rebalance_events] = True
    elif isinstance(rebalance_frequency_or_signal, str):
        rebalance_frequency_or_signal = _FREQUENCY_TO_OFFSET[
            rebalance_frequency_or_signal
        ]

    if isinstance(rebalance_frequency_or_signal, pd.offsets.BaseOffset):
        rebalance_events = index.resample(
            rebalance_frequency_or_signal, **resampling_kwargs
        ).first()
        rebalance_signal.loc[rebalance_events] = True

    # in any case do not allow rebalancing at step 0, it makes no sense in any case!
    if rebalance_signal.iloc[0]:
        rebalance_events.drop(rebalance_signal.index[0], errors="ignore")
        rebalance_signal.iloc[0] = False
    return rebalance_signal, rebalance_events.index

from typing import Union, Tuple, Any, Optional

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

BacktestRebalancingFrequencyOrSignal = Union[
    None, bool, int, str, list, pd.offsets.BaseOffset, pd.Series, np.ndarray
]


def validate_index(
    index: Union[pd.Index, pd.DatetimeIndex, pd.Series, pd.DataFrame, np.ndarray]
) -> pd.DatetimeIndex:
    """
    Validates and converts the input index to a pandas DatetimeIndex if necessary.

    Parameters
    ----------
    index : Union[pd.Index, pd.DatetimeIndex, pd.Series, pd.DataFrame, np.ndarray]
        The index to validate and, if needed, convert.

    Returns
    -------
    pd.DatetimeIndex
        A validated pandas DatetimeIndex.

    Raises
    ------
    TypeError
        If the input index is a NumPy array and its dtype is not datetime64.

    Notes
    -----
    This function accepts various types of input indices, including Pandas Index,
    DatetimeIndex, Series, DataFrame, and NumPy array. It ensures that the output is
    always a valid DatetimeIndex. If the input is a NumPy array, it checks whether
    the dtype is datetime64; otherwise, it raises a TypeError. If the input is a
    Pandas DataFrame or Series, it extracts the index.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np

    >>> index1 = pd.date_range('2022-01-01', '2022-01-10')
    >>> validate_index(index1)
    DatetimeIndex(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04',
                   '2022-01-05', '2022-01-06', '2022-01-07', '2022-01-08',
                   '2022-01-09', '2022-01-10'],
                  dtype='datetime64[ns]', freq='D')

    >>> index2 = np.array(['2022-01-01', '2022-01-02'], dtype='object')
    >>> validate_index(index2)
    TypeError: Input must be datetime64 type
    """
    if isinstance(index, list):
        index = np.array(index)
        if isinstance(index, np.ndarray) and not np.issubdtype(
            index.dtype, np.datetime64
        ):
            raise TypeError("Input must be datetime64 type")
    if isinstance(index, (pd.DataFrame, pd.Series)):
        index = index.index
    # Use pd.to_datetime for consistent DatetimeIndex
    return pd.to_datetime(index)


def validate_rebalance_frequency_or_signal(
    rebalance_frequency_or_signal: BacktestRebalancingFrequencyOrSignal,
) -> BacktestRebalancingFrequencyOrSignal:
    """
    Validates the input for rebalance frequency or signal.

    Parameters
    ----------
    rebalance_frequency_or_signal : BacktestRebalancingFrequencyOrSignal
        The rebalance frequency or signal to be validated.

    Returns
    -------
    BacktestRebalancingFrequencyOrSignal
        The validated rebalance frequency or signal.

    Raises
    ------
    ValueError
        If the input is not a supported type or value.
    TypeError
        If the input is a Series but not boolean or if it's an integer less than 0.

    Notes
    -----
    This function ensures that the input for rebalance frequency or signal is of a
    supported type or value. It accepts Pandas Series, integers, Pandas offsets, and
    strings. If the input is a Pandas Series, it checks that the dtype is boolean. If
    the input is an integer, it checks that it is greater than or equal to 0. If the
    input is not of a supported type or value, a ValueError is raised. If the input
    is a Pandas Series and its dtype is not boolean or if it's an integer less than 0,
    a TypeError is raised.

    Examples
    --------
    >>> import pandas as pd

    >>> signal = pd.Series([True, False, True])
    >>> validate_rebalance_frequency_or_signal(signal)
    0     True
    1    False
    2     True
    dtype: bool

    >>> frequency = 5
    >>> validate_rebalance_frequency_or_signal(frequency)
    5

    >>> unsupported_input = 'daily'
    >>> validate_rebalance_frequency_or_signal(unsupported_input)
    ValueError: Not a supported value for rebalance_frequency parameters

    >>> invalid_series = pd.Series([1, 0, 1])
    >>> validate_rebalance_frequency_or_signal(invalid_series)
    TypeError: Not a supported rebalance signal. Must be boolean
    """
    if rebalance_frequency_or_signal is None:
        rebalance_frequency_or_signal = False
    if not isinstance(
        rebalance_frequency_or_signal, (pd.Series, int, pd.offsets.BaseOffset, str)
    ):
        raise ValueError(
            f"Not a supported type {type(rebalance_frequency_or_signal)} for "
            "rebalance_frequency parameters"
        )

    if isinstance(rebalance_frequency_or_signal, pd.Series):
        if rebalance_frequency_or_signal.dtypes != bool:
            raise TypeError("Not a supported rebalance signal. Must be all boolean")

    if isinstance(rebalance_frequency_or_signal, int):
        if rebalance_frequency_or_signal < 0:
            raise ValueError("Rebalancing frequency must be greater or equal than 1")

    return rebalance_frequency_or_signal


def generate_rebalance_signal(
    rebalance_frequency_or_signal: BacktestRebalancingFrequencyOrSignal,
    index: pd.DatetimeIndex,
    **resampling_kwargs,
) -> pd.Series:
    """
    Generates a rebalance signal based on the provided frequency or signal.

    Parameters
    ----------
    rebalance_frequency_or_signal : BacktestRebalancingFrequencyOrSignal
        The rebalance frequency or signal. If it's an integer, it represents the
        rebalance frequency in periods. If it's a string, it should be a valid string
        representation of a pandas offset (e.g., 'D' for day, 'W' for week). If it's
        a pandas offset object, it will be used directly for resampling.
    index : pd.DatetimeIndex
        The index for which to generate the rebalance signal.
    resampling_kwargs : dict
        Additional keyword arguments for resampling, applicable when using a pandas
        offset for rebalance frequency.

    Returns
    -------
    pd.Series
        The generated rebalance signal series. It has True at rebalance events and
        False otherwise.

    Notes
    -----
    This function generates a rebalance signal series based on the provided
    rebalance_frequency_or_signal. If the input is an integer, it sets True at
    rebalance events, defined as every rebalance_frequency_or_signal periods. If the
    input is a string, it converts it to a pandas offset object and uses it for
    resampling. Additional resampling_kwargs can be provided for customization. If
    the input is a pandas offset object, it directly resamples the index. The resulting
    signal series has True at rebalance events and False otherwise.
    If the input is None or False, set the rebalance signal to False everywhere.
    The first element of the signal series is always False, indicating no rebalancing at step 0.


    Examples
    --------
    >>> import pandas as pd

    >>> frequency = 5
    >>> index = pd.date_range('2022-01-01', '2022-01-10')
    >>> generate_rebalance_signal(frequency, index)
    2022-01-01    False
    2022-01-02    False
    2022-01-03    False
    2022-01-04    False
    2022-01-05     True
    2022-01-06    False
    2022-01-07    False
    2022-01-08    False
    2022-01-09    False
    2022-01-10     True
    Freq: D, dtype: bool

    >>> offset = 'W'
    >>> generate_rebalance_signal(offset, index)
    2022-01-01    False
    2022-01-02    False
    2022-01-03     True
    2022-01-04    False
    2022-01-05    False
    2022-01-06    False
    2022-01-07    False
    2022-01-08    False
    2022-01-09    False
    2022-01-10     True
    Freq: D, dtype: bool
    """

    rebalance_signal = pd.Series(index=index, data=False)
    if isinstance(rebalance_frequency_or_signal, bool):
        rebalance_signal.loc[:] = rebalance_frequency_or_signal
    elif rebalance_frequency_or_signal is None:
        rebalance_signal.loc[:] = False
    elif isinstance(rebalance_frequency_or_signal, int):
        # fill with True at the rebalance frequency events
        rebalance_events_mask = (
            np.mod(np.arange(len(index)), rebalance_frequency_or_signal) == 0
        )
        rebalance_signal.iloc[rebalance_events_mask] = True
    elif isinstance(rebalance_frequency_or_signal, str):
        # Convert string frequency to a pandas offset object
        rebalance_frequency_or_signal = to_offset(rebalance_frequency_or_signal)
    if isinstance(rebalance_frequency_or_signal, pd.offsets.BaseOffset):
        # Resample using the provided BaseOffset object
        rebalance_events = (
            (
                ~rebalance_signal.resample(
                    rebalance_frequency_or_signal, **resampling_kwargs
                ).first()
            )
            .reindex(index=index)
            .fillna(False)
        )
        rebalance_signal.loc[rebalance_events[rebalance_events].index] = True

    # Ensure that the first element is always False (no rebalancing at step 0)
    rebalance_signal.iloc[0] = False
    return rebalance_signal


def prepare_rebalance_signal(
    rebalance_frequency_or_signal: BacktestRebalancingFrequencyOrSignal,
    index: Union[pd.Index, pd.DatetimeIndex, pd.Series, np.ndarray],
    **resampling_kwargs: Any,
) -> Tuple[pd.Series, pd.Index]:
    """
    Computes the transaction event series containing False and True
    for when to trigger a rebalance.

    Parameters
    ----------
    rebalance_frequency_or_signal : BacktestRebalancingFrequencyOrSignal
        The rebalance frequency or signal. If it's an integer, it represents the
        rebalance frequency in periods. If it's a string, it should be a valid string
        representation of a pandas offset (e.g., 'D' for day, 'W' for week). If it's
        a pandas offset object, it will be used directly for resampling. If it's a
        pandas Series, it serves as a custom rebalance signal.
    index : Union[pd.Index, pd.DatetimeIndex, pd.Series, np.ndarray]
        The index for which to generate the rebalance signal.
    resampling_kwargs : dict
        Additional keyword arguments for resampling, applicable when using a pandas
        offset for rebalance frequency.

    Returns
    -------
    Tuple[pd.Series, pd.Index]
        The rebalance signal, a series with the same index as the data, filled with
        False. At rebalance events, a True is specified at the given date.

    Notes
    -----
    This function prepares a rebalance signal series based on the provided
    rebalance_frequency_or_signal. If the input is a Pandas Series, it is assumed to
    be a custom rebalance signal, and it is returned as is, with True at rebalance
    events. If the input is an integer, string, or pandas offset object, a rebalance
    signal series is generated using the generate_rebalance_signal function. The
    resulting signal series has True at rebalance events and False otherwise. The
    function returns a tuple containing the rebalance signal series and the index
    corresponding to True values in the signal series.

    Examples
    --------
    >>> import pandas as pd

    >>> frequency = 'W'
    >>> index = pd.date_range('2022-01-01', '2022-01-10')
    >>> prepare_rebalance_signal(frequency, index)
    (2022-01-01    False
    2022-01-02    False
    2022-01-03     True
    2022-01-04    False
    2022-01-05    False
    2022-01-06    False
    2022-01-07    False
    2022-01-08    False
    2022-01-09    False
    2022-01-10     True
    Freq: D, dtype: bool, DatetimeIndex(['2022-01-03', '2022-01-10'], dtype='datetime64[ns]', freq=None))

    >>> custom_signal = pd.Series([False, True, False], index=index[0:3])
    >>> prepare_rebalance_signal(custom_signal, index[0:3])
    (2022-01-01    False
    2022-01-02     True
    2022-01-03    False
    2022-01-04    False
    2022-01-05    False
    2022-01-06    False
    2022-01-07    False
    2022-01-08    False
    2022-01-09    False
    2022-01-10    False
    Freq: D, dtype: bool, DatetimeIndex(['2022-01-02'], dtype='datetime64[ns]', freq=None))
    """
    index = validate_index(index)
    rebalance_frequency_or_signal = validate_rebalance_frequency_or_signal(
        rebalance_frequency_or_signal
    )

    # If the user provides a rebalance signal, we return it as it is.
    if isinstance(rebalance_frequency_or_signal, pd.Series):
        if not rebalance_frequency_or_signal.index.equals(index):
            raise ValueError(
                "Must provide a rebalance signal with the same index as data"
            )
        rebalance_events = rebalance_frequency_or_signal[
            rebalance_frequency_or_signal
        ].index
        return rebalance_frequency_or_signal, rebalance_events

    # Otherwise, generate the rebalance signal.
    rebalance_signal = generate_rebalance_signal(
        rebalance_frequency_or_signal, index, **resampling_kwargs
    )
    return rebalance_signal, rebalance_signal[rebalance_signal].index


BacktestWindow = Union[
    int,
    Tuple[int, Optional[int]],
]


def prepare_window(window_size: BacktestWindow, n_samples: int):
    """
    Converts the window_size parameter into a minimum and maximum window size
    to be used to create the data windows

    Parameters
    ----------
    window_size: Window
        The minimum and maximum window size. Default None means expanding window.
        Each time the backtesting engine calls a strategy rebalance function,
        a window of asset price data (and possibly signal data) is passed to the
        rebalance function. The rebalance function can then make trading and
        allocation decisions based on a rolling window of market data. The window_size
        property sets the size of these rolling windows.
        Set the window in terms of time steps. The window determines the number of
        rows of data from the asset price timetable that are passed to the
        rebalance function.
        The window_size property can be set in two ways. For a fixed-sized
        rolling window of data (for example, "50 days of price history"), the
        window_size property is set to a single scalar value
        (N = 50). The software then calls the rebalance function with a price
        timetable containing exactly N rows of rolling price data.
        Alternatively, you can define the window_size property by using a
        1-by-2 vector [min max]
        that specifies the minimum and maximum size for an expanding window of data.
        In this way, you can set flexible window sizes. For example:

        [10, None] — At least 10 rows of data
        [0, 50] — No more than 50 rows of data
        [0, None] — All available data (that is, no minimum, no maximum): default
        [20, 20] — Exactly 20 rows of data; equivalent to setting window_size to 20

        The software does not call the rebalance function if the data is insufficient to
        create a valid rolling window, regardless of the value of the RebalanceFrequency
        property.
        If the strategy does not require any price or signal data history, then you can
        indicate that the rebalance function requires no data by setting the window_size
         property to 0.
    n_samples:
        Number of samples in the dataframe

    Returns
    -------

    """
    if isinstance(window_size, (list, tuple)):
        if len(window_size) != 2:
            raise ValueError(
                "When specifying minimum and maximum window size, only two values "
                "are allowed"
            )
        min_window_size = window_size[0]
        if not isinstance(min_window_size, int):
            raise TypeError("Must specify integer minimum window")
        max_window_size = window_size[1] if window_size[1] is not None else n_samples
        if max_window_size < min_window_size:
            raise ValueError("Maximum window is shorter than minimum window")
    elif isinstance(window_size, int):
        # in case 0 is specified, add one for having at least one row of data,
        # hence expanding window
        min_window_size, max_window_size = (window_size, n_samples)
    elif window_size is None:
        min_window_size, max_window_size = (
            0,
            n_samples,
        )
    else:
        raise ValueError("Not a supported window size specification")
    return min_window_size, max_window_size

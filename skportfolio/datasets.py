import os
import re
import numpy as np
import pandas as pd
import pkg_resources
from urllib.request import urlopen, urlretrieve
from skportfolio._base import EquallyWeighted
from pypfopt.expected_returns import returns_from_prices


def get_dataset_names():
    """Report available example datasets, useful for reporting issues.
    Requires an internet connection.
    """
    url = "https://github.com/CarloNicolini/skportfolio-data"
    with urlopen(url) as resp:
        html = resp.read()

    pat = r"/CarloNicolini/skportfolio-data/blob/main/(\w*).csv"
    print(pat)
    datasets = re.findall(pat, html.decode())
    return datasets


def get_data_home(data_home=None):
    """Return a path to the cache directory for example datasets.
    This directory is then used by :func:`load_dataset`.
    If the ``data_home`` argument is not specified, it tries to read from the
    ``SKPORTFOLIO_DATA`` environment variable and defaults to ``~/skportfolio-data``.
    """
    if data_home is None:
        data_home = os.environ.get(
            "SKPORTFOLIO_DATA", os.path.join("~", "skportfolio-data")
        )
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def load_dataset(name, cache=True, data_home=None, **kws):
    """Load an example dataset from the online repository (requires internet).
    This function provides quick access to a small number of example datasets
    that are useful for documenting seaborn or generating reproducible examples
    for bug reports. It is not necessary for normal usage.
    Note that some of the datasets have a small amount of preprocessing applied
    to define a proper ordering for categorical variables.
    Use :func:`get_dataset_names` to see a list of available datasets.
    Parameters
    ----------
    name : str
        Name of the dataset (``{name}.csv`` on
        https://github.com/CarloNicolini/skportfolio-data).
    cache : boolean, optional
        If True, try to load from the local cache first, and save to the cache
        if a download is required.
    data_home : string, optional
        The directory in which to cache data; see :func:`get_data_home`.
    kws : keys and values, optional
        Additional keyword arguments are passed to passed through to
        :func:`pandas.read_csv`.
    Returns
    -------
    df : :class:`pandas.DataFrame`
        Tabular data, possibly with some preprocessing applied.
    """
    # A common beginner mistake is to assume that one's personal data needs
    # to be passed through this function to be usable with seaborn.
    # Let's provide a more helpful error than you would otherwise get.
    if isinstance(name, pd.DataFrame):
        err = (
            "This function accepts only strings (the name of an example dataset). "
            "You passed a pandas DataFrame. If you have your own dataset, "
            "it is not necessary to use this function before plotting."
        )
        raise TypeError(err)

    url = f"https://raw.githubusercontent.com/CarloNicolini/skportfolio-data/main/{name}.csv"

    if cache:
        cache_path = os.path.join(get_data_home(data_home), os.path.basename(url))
        if not os.path.exists(cache_path):
            if name not in get_dataset_names():
                raise ValueError(f"'{name}' is not one of the example datasets.")
            urlretrieve(url, cache_path)
        full_path = cache_path
    else:
        full_path = url

    df = pd.read_csv(full_path, **kws)

    if df.iloc[-1].isnull().all():
        df = df.iloc[:-1]
    if "date" in df.columns:
        return df.set_index("date")
    else:
        return df


def load_tech_stock_prices():
    """
    Loads adjusted close prices of five technological stocks ['AAPL', 'MSTR', 'TSLA', 'MSFT', 'AMZN'], in the interval
    from 2016-08-22 to 2021-08-20.
    Returns
    -------
    A dataframe with the
    """
    df = pd.read_csv(
        pkg_resources.resource_filename(
            "skportfolio", "data/stock_prices_2016_2021.csv"
        ),
        index_col="date",
        infer_datetime_format=True,
    )
    df.index = pd.to_datetime(df.index)
    return df


def load_sp500_adj_close():
    """
    Loads adjusted close prices of the SP500 index (^GSPC), in the interval from 2016-08-22 to 2021-08-20.
    Returns
    -------

    """
    df = pd.read_csv(
        pkg_resources.resource_filename("skportfolio", "data/sp500_2016_2021.csv"),
        index_col="date",
        infer_datetime_format=True,
    )
    df.index = pd.to_datetime(df.index)
    return df


def load_nasdaq_adj_close():
    """
    Loads adjusted close prices of the Nasdaq index (^IXIC), in the interval from 2016-08-22 to 2021-08-20.
    Returns
    -------
    A pandas dataframe of adjusted close prices of the Nasdaq index (^IXIC), from 2016-08-22 to 2021-08-20.
    """
    df = pd.read_csv(
        pkg_resources.resource_filename("skportfolio", "data/nasdaq_100.csv"),
        index_col="date",
        infer_datetime_format=True,
    )
    df.index = pd.to_datetime(df.index)
    return df


def load_matlab_random_returns():
    """
    Loads random returns for 5 assets generated with matlab with rng(42) seed.

    Returns
    -------
    Returns random returns for 5 assets generated with matlab
    """
    data = np.loadtxt(
        pkg_resources.resource_filename(
            "skportfolio", "data/random_returns_matlab.csv"
        ),
        delimiter=",",
    )
    return pd.DataFrame(
        data=data, index=pd.date_range(start="2020-01-01", periods=data.shape[0])
    )


def load_crypto_prices():
    """
    Loads prics of 74 cryptocurrencies from 2020-01-01 to 2022-03-02 at 4h timeframe.

    Returns
    -------
    Returns a pandas dataframe of prics of 74 cryptocurrencies from 2020-01-01 to 2022-03-02 at 4h timeframe.
    """
    df = pd.read_csv(
        pkg_resources.resource_filename("skportfolio", "data/crypto_large.csv")
    ).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df

"""
Algorithm to separate a given training set into input X and outputs y
"""

import warnings
from abc import ABC, ABCMeta, abstractmethod
from itertools import combinations
from logging import getLogger
from typing import Any, List, Optional, Sequence, Tuple, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from tabulate import tabulate
from skportfolio.model_selection._model_selection import (
    split_blocking_time_series,
    ml_get_train_times,
    _get_number_of_backtest_paths,
)
from skportfolio.backtest.rebalance import (
    BacktestRebalancingFrequencyOrSignal,
    prepare_rebalance_signal,
)
from skportfolio.backtest.rebalance import prepare_window, BacktestWindow

logger = getLogger(__name__)


def make_split_df(fold_generator, titles, columns=None, silent=True):
    """
    Creates a visualizatin of our train-test split
    Parameters
    ----------
    fold_generator
    titles
    columns

    Returns
    -------

    """
    X = pd.DataFrame()
    mapping = {"train": "🟢", "test": "🔴", "cv_train": "🟩", "cv_test": "🟥"}
    for ith_fold, fold_values in enumerate(fold_generator):
        for title, vals in zip(titles, fold_values):
            X.loc[ith_fold, vals] = mapping[title]
    X = X.fillna("").sort_index(axis=1)
    X.columns.name = "indices"
    X.index.name = "Folds"
    if columns is not None:
        X = X.rename(
            columns=dict(zip(range(X.shape[1]), columns)),
        )
    for title, val in mapping.items():
        X = X.assign(**{title: (X == val).sum(1)})
        if X[title].nunique() > 1 and not silent:
            msg = f"Length uniformity not respected in '{title}' set"
            warnings.warn(msg)

    return X, {v: k for k, v in mapping.items()}


def print_split(fold_generator, titles, columns=None):
    """

    Parameters
    ----------
    fold_generator
    titles
    columns

    Returns
    -------

    """
    X, mapping = make_split_df(fold_generator, titles, columns=columns)
    print(tabulate([(v, k) for k, v in mapping.items()]))
    print(tabulate(X, headers=X.columns))


class CrossValidator(ABC):
    """
    The base class for all cross validation splitters.
    Taken from sklearn with some small modifications
    """

    @abstractmethod
    def __init__(self, n_splits, *, shuffle, random_state):
        if not isinstance(n_splits, int):
            raise ValueError(
                f"The number of folds must of integer type."
                f"{n_splits} of type {type(n_splits)} was passed."
            )
        n_splits = int(n_splits)

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False; got {0}".format(shuffle))

        if not shuffle and random_state is not None:  # None is the default
            raise ValueError(
                "Setting a random_state has no effect since shuffle is "
                "False. You should leave "
                "random_state to its default (None), or set shuffle=True.",
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    @abstractmethod
    def split(
        self,
        X: Sequence[Any],
        y: Optional[Sequence[Any]] = None,
        groups: Optional[Sequence[Any]] = None,
    ):
        pass

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(list(self.split(X, y, groups))[0])

    def visualize(
        self,
        X=None,
        y=None,
        **kwargs,
    ):
        viz, legend = make_split_df(
            self.split(X, y, groups=kwargs.get("groups", None)),
            titles=("train", "test"),
        )
        if "columns" in kwargs:
            viz = viz.rename(
                columns={i: kwargs["columns"][i] for i in range(len(viz.columns[:-4]))}
            )
        title = f"Cross validation table for {str(self)} train 🟢 test 🔴"
        return viz.style.set_caption(title)


class NoSplit(CrossValidator):
    """
    A cross-validation strategy that does not do any cross-validation, keeps all the data.
    Returns all the indices for both training and testing
    Only present for compatibility purpose.
    """

    def __init__(self):
        super().__init__(n_splits=1, shuffle=False, random_state=None)

    def _iter_test_indices(self, X=None, y=None, groups=None):
        pass

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        yield indices, indices


class WalkForward(CrossValidator, metaclass=ABCMeta):
    """
    A class to generate indices for training TimeSeries models splitting the training set
    It can be used both for generating train-test examples following a sliding windows approach
    or to generate the train X,y samples to train a model on data X with input y
    with the specific behaviour that the model uses labels y[i+1,i+2,...,i+num_y]
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        warmup: int = 0,
        gap: int = 0,
        anchored: bool = False,
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.warmup = warmup
        self.anchored = anchored
        self.gap = gap

    def split(self, X: Sequence, y: Optional[Sequence] = None, groups: Optional = None):
        if y is not None and len(y) != len(X):
            raise ValueError("Check y and X have same length")
        indices = np.arange(len(X))
        for i in range(
            self.train_size + self.warmup, len(X) - self.test_size, self.test_size
        ):
            # we exceeded the data limit, we stop otherwise the test set has different length
            if i + self.test_size + self.gap > len(X):
                break

            if not self.anchored:
                train_slice, test_slice = slice(
                    i - self.train_size - self.warmup, i, 1
                ), slice(self.gap + i - self.warmup, i + self.test_size + self.gap, 1)
            else:
                train_slice, test_slice = slice(0 + self.warmup, i, 1), slice(
                    self.gap + i + self.warmup, i + self.test_size + self.gap, 1
                )
            yield indices[train_slice], indices[test_slice]


def group_slices_old(
    change_indices: np.ndarray,
):
    """
    Generator yielding slices of train and test indices with a blocking time series approach
    There are no intersections
    Parameters
    ----------
    change_indices: np.ndarray

    Returns
    -------

    """
    # Initialize start index of the first group
    start_idx = 0

    # Iterate over the change indices to create pairs of slices
    for i, train_end in enumerate(change_indices):

        # If it's not the last element, set the next start to the next change index
        if i < len(change_indices) - 1:
            next_train_start = change_indices[i + 1]
        else:
            # For the last group, set the next start to the end of the array
            next_train_start = None

        # Yield the pair of slices
        yield slice(start_idx, train_end), slice(train_end, next_train_start)

        # Update the start index for the next group
        start_idx = train_end


def group_slices(
    change_indices: np.ndarray,
    min_window_size: Optional[int] = None,
    max_window_size: Optional[int] = None,
):
    """
    Generator yielding slices of train and test indices with a blocking time series approach.
    There are no intersections.

    Parameters
    ----------
    change_indices : np.ndarray
        Array of indices where the rebalance event occurs.
    min_window_size : Optional[int]
        Minimum size of the training or testing window.
    max_window_size : Optional[int]
        Maximum size of the training or testing window.

    Returns
    -------
    Generator of (train_slice, test_slice) tuples.
    """
    # Iterate over the change indices to create pairs of slices
    already_yielded = set()
    for i, train_end in enumerate(change_indices):
        # Ensure minimum window size is respected for the train set
        train_start = max(
            0,
            train_end
            - (max_window_size if max_window_size is not None else len(change_indices)),
        )
        train_end = max(
            train_start + (min_window_size if min_window_size is not None else 0),
            train_end,
        )

        # If it's not the last element, determine the end of the test set based on the next change index
        if i < len(change_indices) - 1:
            test_end = change_indices[i + 1]
            # Ensure maximum window size is respected for the test set
            if max_window_size is not None:
                test_end = min(test_end, train_end + max_window_size)
        else:
            # For the last group, set the end of the test set to the end of the array
            test_end = len(change_indices)

        # Ensure minimum window size is respected for the test set
        test_end = max(
            train_end + (min_window_size if min_window_size is not None else 1),
            test_end,
        )

        # Do not yield a new tuple if we've reached the end of the sequence
        if train_end == len(change_indices) or test_end == len(change_indices):
            break

        # Yield the pair of slices
        to_yield = (train_start, train_end), (train_end, test_end)
        if to_yield not in already_yielded:
            yield slice(*to_yield[0]), slice(*to_yield[1])
        already_yielded.add(to_yield)


# class RebalanceSplitter(CrossValidator, metaclass=ABCMeta):
#     """
#     A class to create train/test splits to be used in a backtesting estimator.
#     It follow the BlockingTimeSeriesSplit logic but it accepts a rebalance_signal array.
#     It is supposed to be used by a BacktesterCV class, where the portfolio weights
#     are computed at train and the portfolio evolution is applied by the .predict method.
#     """
#
#     def __init__(
#         self,
#         rebalance_frequency_or_signal: BacktestRebalancingFrequencyOrSignal,
#         window_size: BacktestWindow,
#     ):
#         self.rebalance_frequency_or_signal = rebalance_frequency_or_signal
#         self.window_size = window_size
#
#     def split(
#         self,
#         X: Sequence[Any],
#         y: Optional[Sequence[Any]] = None,
#         groups: Optional[Sequence[Any]] = None,
#     ):
#         if not isinstance(X, (pd.Series, pd.DataFrame)):
#             raise TypeError("Only dataframe or series with an index is accepted")
#
#         # Prepare the rebalance signal based on the provided frequency or signal
#         rebalance_signal, _ = prepare_rebalance_signal(
#             self.rebalance_frequency_or_signal, index=X.index
#         )
#
#         indices = np.arange(len(X))
#         change_indices = np.where(rebalance_signal)[0]
#
#         # Define default window sizes if none provided
#
#         min_window_size, max_window_size = prepare_window(
#             window_size=self.window_size, n_samples=X.shape[0]
#         )
#         print(min_window_size, max_window_size)
#
#         for train_fold, test_fold in group_slices(
#             change_indices=change_indices,
#             min_window_size=min_window_size,
#             max_window_size=max_window_size,
#         ):
#             yield indices[train_fold], indices[test_fold]
class RebalanceSplitter(CrossValidator, metaclass=ABCMeta):
    def __init__(self, rebalance_frequency_or_signal, window_size):
        self.rebalance_frequency_or_signal = rebalance_frequency_or_signal
        self.window_size = window_size

    def split(
        self,
        X: Sequence[Any],
        y: Optional[Sequence[Any]] = None,
        groups: Optional[Sequence[Any]] = None,
    ):
        n_samples = X.shape[0]
        min_window_size, max_window_size = prepare_window(
            self.window_size, n_samples=n_samples
        )

        indices = np.arange(len(X))
        rebalance_signal, rebalance_events = prepare_rebalance_signal(
            self.rebalance_frequency_or_signal, index=X.index.to_series()
        )
        for idx in range(n_samples - 1):
            next_idx = idx + 1
            needs_rebalance = rebalance_signal.iloc[next_idx]
            is_valid_window = next_idx >= min_window_size
            if is_valid_window and needs_rebalance:
                start_window = next_idx - max_window_size + 1
                train_indices = slice(max(0, start_window), next_idx + 1)
                test_indices = slice(next_idx + 1, next_idx + 2)
                yield indices[train_indices], indices[
                    slice(max(0, start_window) + 1 - next_idx, next_idx + 2)
                ]


class BlockingTimeSeriesSplit(CrossValidator):
    """
    Implements the Blocking Time Series Split cross validation method.
    It is the same as the RebalanceSplitter method, but it accepts different parameters
    which may not be easy to apply in the context of a specific rebalance_signal.
    Here for convenience.
    """

    def __init__(self, n_splits: int, train_test_ratio: float = 0.7):
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.train_test_ratio = train_test_ratio

    def split(self, X, y=None, groups=None):
        return split_blocking_time_series(X, self.n_splits, self.train_test_ratio)


class PurgedKFold(CrossValidator):
    """
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle = False), w/o training samples in between
    """

    def __init__(self, n_splits: int = 5, pct_embargo: float = 0.0):
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        indices = np.arange(len(X))
        embargo = int(len(X) * self.pct_embargo)
        test_starts = [
            (i[0], i[-1] + 1) for i in np.array_split(np.arange(len(X)), self.n_splits)
        ]
        for i, j in test_starts:
            t_0 = i  # start of test set
            test_indices = indices[i:j]
            max_t1_idx = indices.searchsorted(indices[test_indices].max())
            train_indices = indices.searchsorted(indices[indices <= t_0])
            if max_t1_idx < len(X):  # right train (with embargo)
                train_indices = np.concatenate(
                    (train_indices, indices[max_t1_idx + embargo :])
                )
            yield train_indices, test_indices


class WindowedTestTimeSeriesSplit(TimeSeriesSplit, ABC):
    """
    parameters
    ----------
    n_test_folds: int
        number of folds to be used as testing at each iteration.
        by default, 1.
    """

    def __init__(self, n_splits=5, max_train_size=None, n_test_folds=1):
        super().__init__(n_splits, max_train_size=max_train_size)
        self.n_test_folds = n_test_folds

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + self.n_test_folds
        if n_folds > n_samples:
            raise ValueError(
                (
                    "Cannot have number of folds ={0} greater"
                    " than the number of samples: {1}."
                ).format(n_folds, n_samples)
            )
        indices = np.arange(n_samples)
        fold_size = n_samples // n_folds
        test_size = fold_size * self.n_test_folds  # test window
        test_starts = range(
            fold_size + n_samples % n_folds, n_samples - test_size + 1, fold_size
        )  # splits based on fold_size instead of test_size
        for test_start in test_starts:
            if self.max_train_size and self.max_train_size < test_start:
                yield (
                    indices[test_start - self.max_train_size : test_start],
                    indices[test_start : test_start + test_size],
                )
            else:
                yield (
                    indices[:test_start],
                    indices[test_start : test_start + test_size],
                )


class CombinatorialPurgedKFold(KFold, CrossValidator):
    """
    Advances in Financial Machine Learning, Chapter 12.

    Implements Combinatial Purged Cross Validation (CPCV)

    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between

    :param n_splits: (int) The number of splits. Default to 3
    :param pct_embargo: (float) Percent that determines the embargo size.
    """

    def __init__(
        self,
        n_splits: int = 3,
        n_test_splits: int = 2,
        pct_embargo: float = 0.0,
        samples_info_sets=None,
    ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo
        self.n_test_splits = n_test_splits
        self.num_backtest_paths = _get_number_of_backtest_paths(
            self.n_splits, self.n_test_splits
        )
        self.backtest_paths = []  # Array of backtest paths

    def _generate_combinatorial_test_ranges(self, splits_indices: dict) -> List:
        """
        Using start and end indices of test splits from KFolds and number of test_splits (self.n_test_splits),
        generates combinatorial test ranges splits

        :param splits_indices: (dict) Test fold integer index: [start test index, end test index]
        :return: (list) Combinatorial test splits ([start index, end index])
        """

        # Possible test splits for each fold
        combinatorial_splits = list(
            combinations(list(splits_indices.keys()), self.n_test_splits)
        )
        combinatorial_test_ranges = (
            []
        )  # List of test indices formed from combinatorial splits
        for combination in combinatorial_splits:
            temp_test_indices = (
                []
            )  # Array of test indices for current split combination
            for int_index in combination:
                temp_test_indices.append(splits_indices[int_index])
            combinatorial_test_ranges.append(temp_test_indices)
        return combinatorial_test_ranges

    def _fill_backtest_paths(self, train_indices: list, test_splits: list):
        """
        Using start and end indices of test splits and purged/embargoed train indices from CPCV, find backtest path and
        place in the path where these indices should be used.

        :param test_splits: (list) of lists with first element corresponding to test start index and second - test end
        """
        # Fill backtest paths using train/test splits from CPCV
        for split in test_splits:
            found = False  # Flag indicating that split was found and filled in one of backtest paths
            for path in self.backtest_paths:
                for path_el in path:
                    if (
                        path_el["train"] is None
                        and split == path_el["test"]
                        and found is False
                    ):
                        path_el["train"] = np.array(train_indices)
                        path_el["test"] = list(range(split[0], split[-1]))
                        found = True

    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None):
        """
        The main method to call for the PurgedKFold class

        :param X: (pd.DataFrame) Samples dataset that is to be split
        :param y: (pd.Series) Sample labels series
        :param groups: (array-like), with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        :return: (tuple) [train list of sample indices, and test list of sample indices]
        """
        # self.samples_info_sets = pd.Series(
        #     data=np.arange(len(X)), index=np.arange(len(X))
        # )
        if X.shape[0] != self.samples_info_sets.shape[0]:
            raise ValueError(
                "X and the 'samples_info_sets' series param must be the same length"
            )

        test_ranges: [(int, int)] = [
            (ix[0], ix[-1] + 1)
            for ix in np.array_split(np.arange(X.shape[0]), self.n_splits)
        ]
        splits_indices = {}
        for index, [start_ix, end_ix] in enumerate(test_ranges):
            splits_indices[index] = [start_ix, end_ix]

        combinatorial_test_ranges = self._generate_combinatorial_test_ranges(
            splits_indices
        )
        # Prepare backtest paths
        for _ in range(self.num_backtest_paths):
            path = []
            for split_idx in splits_indices.values():
                path.append({"train": None, "test": split_idx})
            self.backtest_paths.append(path)

        embargo: int = int(X.shape[0] * self.pct_embargo)
        for test_splits in combinatorial_test_ranges:

            # Embargo
            test_times = pd.Series(
                index=[self.samples_info_sets[ix[0]] for ix in test_splits],
                data=[
                    self.samples_info_sets[ix[1] - 1]
                    if ix[1] - 1 + embargo >= X.shape[0]
                    else self.samples_info_sets[ix[1] - 1 + embargo]
                    for ix in test_splits
                ],
            )

            test_indices = []
            for [start_ix, end_ix] in test_splits:
                test_indices.extend(list(range(start_ix, end_ix)))

            # Purge
            train_times = ml_get_train_times(self.samples_info_sets, test_times)

            # Get indices
            train_indices = []
            for train_ix in train_times.index:
                train_indices.append(self.samples_info_sets.index.get_loc(train_ix))

            self._fill_backtest_paths(train_indices, test_splits)
            yield np.array(train_indices), np.array(test_indices)


class RebalanceSplitter2(CrossValidator):
    def __init__(
        self,
        rebalance_frequency_or_signal: BacktestRebalancingFrequencyOrSignal,
        window_size: BacktestWindow,
    ):
        self.rebalance_frequency_or_signal = rebalance_frequency_or_signal
        self.window_size = window_size

    def split(
        self,
        X: Sequence[Any],
        y: Optional[Sequence[Any]] = None,
        groups: Optional[Sequence[Any]] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        min_window_size, max_window_size = prepare_window(
            self.window_size, n_samples=n_samples
        )
        indices = np.arange(len(X))
        rebalance_signal, rebalance_events = prepare_rebalance_signal(
            self.rebalance_frequency_or_signal, index=X.index.to_series()
        )
        for idx in range(n_samples - 1):
            next_idx = idx + 1
            needs_rebalance = rebalance_signal.iloc[next_idx]
            is_valid_window = next_idx >= min_window_size
            if is_valid_window and needs_rebalance and idx > 0:
                start_window = next_idx - max_window_size - 1
                train_indices = slice(max(0, start_window), idx + 1)
                test_indices = slice(idx, idx + 1)
                # when there are both training and test, the user should concatenate them
                yield indices[train_indices], indices[test_indices]
            else:
                # there are no new training data, we only let the user evolve the positions on the test
                yield indices[slice(0, 0)], indices[slice(0, next_idx)]

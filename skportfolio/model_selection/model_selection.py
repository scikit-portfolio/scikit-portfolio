"""
Algorithm to separate a given training set into input X and outputs y
"""

import warnings
from abc import ABC, ABCMeta, abstractmethod
from itertools import combinations
from logging import getLogger
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from tabulate import tabulate

from skportfolio.model_selection._model_selection import (
    split_train_val_forward_chaining,
    split_blocking_time_series,
    ml_get_train_times,
    _get_number_of_backtest_paths,
    split_train_val_k_fold,
    split_train_val_test_forward_chaining,
    split_train_val_group_k_fold,
    split_train_val_test_k_fold,
    split_train_val_test_group_k_fold,
)

logger = getLogger(__name__)


def make_split_df(fold_generator, titles, columns=None):
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
        for tit, vals in zip(titles, fold_values):
            X.loc[f"Fold({ith_fold+1})", vals] = mapping[tit]
    X = X.fillna("").sort_index(axis=1)
    X.columns.name = "indices"
    X.index.name = "Folds"
    if columns is not None:
        X = X.rename(
            columns=dict(zip(range(X.shape[1]), columns)),
        )
    for tit, val in mapping.items():
        X = X.assign(**{tit: (X == val).sum(1)})
        if X[tit].nunique() > 1:
            warnings.warn(f"Length uniformity not respected in {tit} set")

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

    def visualize(self, X=None, y=None, titles: Tuple[str] = ("train", "test")):
        viz, legend = make_split_df(self.split(X), titles=titles)
        print({k: v for k, v in legend.items() if v in titles})
        return viz


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


class WalkForwardRolling(CrossValidator, metaclass=ABCMeta):
    """
    A class to generate indices for training TimeSeries models splitting the training set
    It can be used both for generating train-test examples following a sliding windows approach
    or to generate the train X,y samples to train a model on data X with input y
    with the specific behaviour that the model uses labels y[i+1,i+2,...,i+num_y]
    """

    def __init__(self, train_size: int, test_size: int, warmup: int = 0,
                 gap:int =0, anchored:bool=False):
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
            if not self.anchored:
                train_slice, test_slice = slice(
                    i - self.train_size - self.warmup, i, 1
                ), slice(self.gap + i - self.warmup, i + self.test_size+self.gap, 1)
            else:
                train_slice, test_slice = slice(
                    0+ self.warmup, i, 1
                ), slice(self.gap + i + self.warmup, i + self.test_size+self.gap, 1)
            yield indices[train_slice], indices[test_slice]


class SplitTrainValForwardChaining(CrossValidator):
    """
    Splits data using forward chaining technique
    """

    def __init__(self, num_X: int, num_y: int, num_jumps: int):
        self.num_X = num_X
        self.num_y = num_y
        self.num_jumps = num_jumps

    def split(self, X, y=None, groups=None):
        out_X, out_y, out_x_cv, out_y_cv = split_train_val_forward_chaining(
            X, self.num_X, self.num_y, self.num_jumps
        )
        for x, y, xcv, ycv in zip(out_X, out_y, out_x_cv, out_y_cv):
            yield x, y, xcv, ycv


class SplitTrainValFCKFold(CrossValidator):
    """
    Splits data using forward chaining and K-Fold cross-validation
    """

    def __init__(self, num_X: int, num_y: int, num_jumps: int):
        self.num_X = num_X
        self.num_y = num_y
        self.num_jumps = num_jumps

    def split(self, X, y=None, groups=None):
        out_X, out_y, out_x_cv, out_y_cv = split_train_val_k_fold(
            X, self.num_X, self.num_y, self.num_jumps
        )
        for x, y, xcv, ycv in zip(out_X, out_y, out_x_cv, out_y_cv):
            yield x, y, xcv, ycv


class SplitTrainValTestFcKFold(CrossValidator):
    def __init__(self, num_X: int, num_y: int, num_jumps: int):
        self.num_X = num_X
        self.num_y = num_y
        self.num_jumps = num_jumps

    def split(self, X, y=None, groups=None):
        out_X, out_y, out_x_cv, out_y_cv = sp(X, self.num_X, self.num_y, self.num_jumps)
        for x, y, xcv, ycv in zip(out_X, out_y, out_x_cv, out_y_cv):
            yield x, y, xcv, ycv


class BlockingTimeSeriesSplit(CrossValidator):
    """
    Implements the Blocking Time Series Split cross validation method
    """

    def __init__(self, n_splits: int, train_test_ratio: float = 0.7):
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.train_test_ratio = train_test_ratio

    def split(self, X, y=None, groups=None):
        return split_blocking_time_series(X, self.n_splits, self.train_test_ratio)


# class PurgedKFold(CrossValidator):
#     """
#     Extend KFold class to work with labels that span intervals
#     The train is purged of observations overlapping test-label intervals
#     Test set is assumed contiguous (shuffle = False), w/o training samples in between
#     """
#
#     def __init__(self, pct_embargo=0.0):
#         self.pct_embargo = pct_embargo
#
#     def split(self, X, y=None, groups=None):
#         indices = np.arange(len(X))
#         mbrg = int(len(X) * self.pct_embargo)
#         test_starts = [
#             (i[0], i[-1] + 1) for i in np.array_split(np.arange(len(X)), self.n_splits)
#         ]
#
#         for i, j in test_starts:
#             t0 = i  # start of test set
#             test_indices = indices[i:j]
#             maxT1Idx = indices.searchsorted(indices[test_indices].max())
#             train_indices = indices.searchsorted(indices[indices <= t0])
#             if maxT1Idx < len(X):  # right train (with embargo)
#                 train_indices = np.concatenate(
#                     (train_indices, indices[maxT1Idx + mbrg :])
#                 )
#             yield train_indices, test_indices
#


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

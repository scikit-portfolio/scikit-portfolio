"""
Algorithm to separate a given training set into input X and outputs y
"""

import warnings
from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from itertools import combinations
from logging import getLogger
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection._split import TimeSeriesSplit
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from tabulate import tabulate

from skportfolio.model_selection._model_selection import _get_number_of_backtest_paths
from skportfolio.model_selection._model_selection import ml_get_train_times
from skportfolio.model_selection._model_selection import split_blocking_time_series
from skportfolio.model_selection._model_selection import (
    split_train_val_forward_chaining,
)
from skportfolio.model_selection._model_selection import split_train_val_k_fold

logger = getLogger(__name__)


"""
# Problems with cross validation for time series data

Random train-test splits can lead to data leakage, and if traditional k-fold and leave-one-out CV are the default
procedures being followed, data leakage will happen.
Leakage is the major reason why traditional CV is not appropriate for time series. Using these cross-validators
when you shouldn't will inflate performance metrics since they allow the training model to cheat because observations
"from the future" (posterior samples) leak into the training set.

Since time series data is time ordered, we want to keep intact the fact that we must use past observations to predict
future observations. The randomization in the standard cross-validation algorithm does not preserve the time ordering,
and we end up making predictions for some samples using a model trained on posterior samples. While this is not
immediately a huge problem for some applications, it becomes a critical one if the time series data is often strongly
correlated along the time axis. The randomization of traditional CV will make it likely that for each sample in the
validation set, numerous strongly correlated samples exist in the train set. This defeats the very purpose of having
a validation set: the model essentially “knows” about the validation set already, leading to inflated performance
metrics on the validation set in case of overfitting.

One solution is to use Walk-forward cross-validation (closest package implementation being Time Series Split
in sklearn), which restricts the full sample set differently for each split, but this suffers from the problem that,
near the split point, we may have training samples whose evaluation time is posterior to the prediction time of
validation samples.

Such overlapping samples are unlikely to be independent, leading to information leaking from the  train set into the
validation set. To deal with this, purging can be used.

## Purging
Purging involves dropping from the train set any sample whose evaluation time is posterior to the earliest prediction
time in the validation set. This ensures that predictions on the validation set are free of look-ahead bias.
But since walk-forward CV has other grievances, like its lack of focus on the most recent data during training fold
construction, the following is becoming more widely adopted for time series data:

## Combinatorial cross-validation
Consider we abandon the requirement that all the samples in the train set precede the samples in the validation set.
This is not as problematic as it may sound. The crucial point is to ensure that the samples in the validation set are
reasonably independent from the samples in the training set. If this condition is verified, the validation set
performance will still be a good proxy for the performance on new data.

Combinatorial K-fold cross-validation is similar to K-fold cross-validation, except that we take the validation set to
consists in j<K blocks of samples. We then have K choose j possible different splits. This allows us to create
easily a large number of splits, just by taking j=2 or 3, addressing two other problems not mentioned that purging
did not address.

It is however clear that we cannot use combinatorial K-fold cross-validation as it stands. We have to make sure that
the samples in the train set and in the validation set are independent. We already saw that purging helps reduce
their dependence. However, when there are train samples occurring after validation samples, this is not sufficient.

## Embargoing
We obviously also need to prevent the overlap of train and validation samples at the right end(s) of the validation set.
But simply dropping any train sample whose prediction time occurs before the latest evaluation time in the preceding
block of validation samples may not be sufficient. There may be correlations between the samples over longer
periods of time. In order to deal with such long range correlation, we can define an embargo period after each right
end of the validation set. If a train sample prediction time falls into the embargo period, we simply drop the sample
from the train set. The required embargo period has to be estimated from the problem and dataset at hand.
A nice feature of combinatorial cross-validation is also that as each block of samples appears the same number of
times in the validation set, we can group them (arbitrarily) into validation predictions over the full dataset (keeping
in mind that these predictions have been made by models trained on different train sets). This is very useful to
extract performance statistics over the whole dataset.
"""


def make_split_df(fold_generator, titles, columns=None):
    X = pd.DataFrame()  # empty then we fill it Linello, [2/2/22 2:59 PM]

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
    X, mapping = make_split_df(fold_generator, titles, columns=columns)
    print(tabulate([(v, k) for k, v in mapping.items()]))
    print(tabulate(X, headers=X.columns))


class CrossValidator(ABC):
    @abstractmethod
    def __init__(self, n_splits, *, shuffle, random_state):
        if not isinstance(n_splits, int):
            raise ValueError(
                "The number of folds must of integer type. "
                "%s of type %s was passed." % (n_splits, type(n_splits))
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


class WalkForwardFixed(TimeSeriesSplit, CrossValidator, metaclass=ABCMeta):
    def split(self, X, y=None, groups=None):
        if self.max_train_size is None:
            raise ValueError(
                "Fixed walk forward method requires `max_train_size` to be specified in initialization"
            )
        return super().split(X, y, groups)


class SplitTrainFixed(CrossValidator):
    """
    A class to generate indices for training TimeSeries models splitting the training set
    It can be used both for generating train-test examples following a sliding windows approach
    or to generate the train X,y samples to train a model on data X with input y
    with the specific behaviour that the model uses labels y[i+1,i+2,...,i+num_y]
    """

    def __init__(
        self,
        num_X: int,
        num_y: int,
        num_jumps: int,
    ):
        """
        Parameters
        ----------
        num_X (int)   : Number of inputs X used at each training
        num_y (int)  : Number of outputs y used at each training
        num_jumps (int)    : Number of sequence samples to be ignored between (X,y) sets
        """
        self.num_X = num_X
        self.num_y = num_y
        self.num_jumps = num_jumps

    def split(
        self,
        X: Sequence[Any],
        y: Optional[Sequence[Any]] = None,
        groups: Optional[Sequence[Any]] = None,
    ):
        """
        Returns sets to train a model or sets to be used as train-test pairs

            i.e. X[0] = X[0], ..., X[numInputs]
                 y[0] = X[numInputs+1], ..., X[numInputs+numOutputs]
                 ...
                 X[k] = X[k*numJumps], ..., X[k*numJumps+numInputs]
                 y[k] = X[k*numJumps+numInputs+1], ..., X[k*numJumps+numInputs+numOutputs]

        Parameters
        ----------
        X: Sequence
        y: Optional[Sequence]
        groups: Sequence

        Yields
        -------
            X (2D array): Array of numInputs arrays.
                          len(X[k]) = numInputs
            y (2D array): Array of numOutputs arrays
                          len(y[k]) = numOutputs
        """
        sequence = np.arange(len(X)).astype(int)

        if y is not None and len(y) != len(X):
            raise ValueError("Check y and X have same length")

        if self.num_X + self.num_y > len(X):
            warnings.warn(
                "To have at least one X,y arrays, the sequence size needs to be bigger than numInputs+numOutputs"
            )
            yield sequence, sequence

        for i in range(len(X)):
            i = self.num_jumps * i
            end_ix = i + self.num_X
            # Once train data crosses time series length return
            if end_ix + self.num_y > len(X):
                break
            yield sequence[i:end_ix], sequence[end_ix : end_ix + self.num_y]

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(list(self.split(X, y, groups))[0])


class SplitTrainMinTrain(CrossValidator):
    """
    Returns sets to train a model with variable input length
        i.e. X[0] = sequence[0], ..., sequence[minSamplesTrain]
            y[0] = sequence[0], ..., sequence[minSamplesTrain+numOutputs]
            ...
            X[k] = sequence[0], ..., sequence[k*numJumps+minSamplesTrain]
            y[k] = sequence[0], ..., sequence[k*numJumps+minSamplesTrain+numOutputs]
    """

    def __init__(
        self,
        min_num_x: int,
        num_y: int,
        num_jumps: int,
    ):
        self.min_num_x = min_num_x
        self.num_y = num_y
        self.num_jumps = num_jumps

    def split(self, X, y=None, groups=None):
        i = 0
        sequence = np.arange(len(X)).astype(int)
        if self.min_num_x + self.num_y > len(sequence):
            warnings.warn(
                "To have at least one X,y arrays, the sequence size needs to be bigger than minSamplesTrain+numOutputs"
            )
            yield np.array([]), np.array([])

        # Iterate through all validation splits
        while 1:
            end_ix = self.min_num_x + self.num_jumps * i
            i += 1
            # Once val data crosses time series length return
            if (self.min_num_x + self.num_jumps * i + self.num_y) > len(sequence):
                break
            yield sequence[0:end_ix], sequence[end_ix : end_ix + self.num_y]


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
        super(CombinatorialPurgedKFold, self).__init__(
            n_splits, shuffle=False, random_state=None
        )
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

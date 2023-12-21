"""
Model selection tools for portfolio optimization methods
"""

from typing import Sequence, Any
from itertools import combinations
import numpy as np
import pandas as pd


def split_blocking_time_series(
    seq: Sequence[Any], n_splits: int, train_test_ratio: float
):
    """
    Implements the blocking time series split logic.
    The train and test sets over all folds have zero intersection.
    Very conservative method but it produces small train/test folds.

    Parameters
    ----------
    seq
    n_splits
    train_test_ratio

    Returns
    -------

    """
    n_samples = len(seq)
    k_fold_size = n_samples // n_splits
    indices = np.arange(n_samples)
    margin = 0
    for i in range(n_splits):
        start = i * k_fold_size
        stop = start + k_fold_size
        mid = int(train_test_ratio * (stop - start)) + start
        yield indices[start:mid], indices[(mid + margin) : stop]


def _generate_combinatorial_test_ranges(n_test_splits, splits_indices: dict) -> list:
    """
    Using start and end indices of test splits from KFolds and number of
    test_splits (self.n_test_splits), generates combinatorial test
    ranges splits

    Parameters
    ----------
    splits_indices: (dict)
        Test fold integer index  [start test index, end test index]

    Returns
    -------
    List of combinatorial test splits ([start index, end index])
    """

    # Possible test splits for each fold
    combinatorial_splits = list(
        combinations(list(splits_indices.keys()), n_test_splits)
    )
    combinatorial_test_ranges = (
        []
    )  # List of test indices formed from combinatorial splits
    for combination in combinatorial_splits:
        temp_test_indices = []  # Array of test indices for current split combination
        for int_index in combination:
            temp_test_indices.append(splits_indices[int_index])
        combinatorial_test_ranges.append(temp_test_indices)
    return combinatorial_test_ranges


def ml_get_train_times(
    samples_info_sets: pd.Series, test_times: pd.Series
) -> pd.Series:
    # pylint: disable=invalid-name
    """
    Advances in Financial Machine Learning, Snippet 7.1, page 106.

    Purging observations in the training set

    This function find the training set indexes given the information on which each record is based
    and the range for the test set.
    Given test_times, find the times of the training observations.

    :param samples_info_sets: (pd.Series) The information range on which each record is constructed from
        *samples_info_sets.index*: Time when the information extraction started.
        *samples_info_sets.value*: Time when the information extraction ended.
    :param test_times: (pd.Series) Times for the test dataset.
    :return: (pd.Series) Training set
    """
    train = samples_info_sets.copy(deep=True)
    for start_ix, end_ix in test_times.items():
        df0 = train[
            (start_ix <= train.index) & (train.index <= end_ix)
        ].index  # Train starts within test
        df1 = train[
            (start_ix <= train) & (train <= end_ix)
        ].index  # Train ends within test
        df2 = train[
            (train.index <= start_ix) & (end_ix <= train)
        ].index  # Train envelops test
        train = train.drop(df0.union(df1).union(df2))
    return train


def _get_number_of_backtest_paths(n_train_splits: int, n_test_splits: int) -> int:
    """
    Number of combinatorial paths for CPCV(N,K)
    :param n_train_splits: (int) number of train splits
    :param n_test_splits: (int) number of test splits
    :return: (int) number of backtest paths for CPCV(N,k)
    """
    from scipy.special import comb

    return int(
        comb(n_train_splits, n_train_splits - n_test_splits)
        * n_test_splits
        / n_train_splits
    )


def cpcv_generator(t_span, n, k):
    """
    Generator for the combinatorial purged cross validation method by Marcos Lopez DePrado.
    Parameters
    ----------
    t_span
    n
    k

    Returns
    -------

    """
    # split data into N groups, with N << T
    # this will assign each index position to a group position
    group_num = np.arange(t_span) // (t_span // n)
    group_num[group_num == n] = n - 1

    # generate the combinations
    test_groups = np.array(list(combinations(np.arange(n), k))).reshape(-1, k)
    C_nk = len(test_groups)
    n_paths = C_nk * k // n

    # is_test is a T x C(n, k) array where each column is a logical array
    # indicating which observation in in the test set
    is_test_group = np.full((n, C_nk), fill_value=False)
    is_test = np.full((t_span, C_nk), fill_value=False)

    # assign test folds for each of the C(n, k) simulations
    for k, pair in enumerate(test_groups):
        i, j = pair
        is_test_group[[i, j], k] = True

        # assigning the test folds
        mask = (group_num == i) | (group_num == j)
        is_test[mask, k] = True

    # for each path, connect the folds from different simulations to form a backtest path
    # the fold coordinates are: the fold number, and the simulation index e.g. simulation 0, fold 0 etc
    path_folds = np.full((n, n_paths), fill_value=np.nan)

    for i in range(n_paths):
        for j in range(n):
            s_idx = is_test_group[j, :].argmax().astype(int)
            path_folds[j, i] = s_idx
            is_test_group[j, s_idx] = False

    # finally, for each path we indicate which simulation we're building the path from and the time indices
    paths = np.full((t_span, n_paths), fill_value=np.nan)

    for p in range(n_paths):
        for i in range(n):
            mask = group_num == i
            paths[mask, p] = int(path_folds[i, p])

    return is_test, paths, path_folds

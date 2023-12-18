"""
Model selection tools for portfolio optimization methods
"""

import warnings
from typing import Sequence, Any
from itertools import combinations
import numpy as np
import pandas as pd


def split_blocking_time_series(
    seq: Sequence[Any], n_splits: int, train_test_ratio: float
):
    n_samples = len(seq)
    k_fold_size = n_samples // n_splits
    indices = np.arange(n_samples)
    margin = 0
    for i in range(n_splits):
        start = i * k_fold_size
        stop = start + k_fold_size
        mid = int(train_test_ratio * (stop - start)) + start
        yield indices[start:mid], indices[(mid + margin) : stop]


def split_train_val_forward_chaining(
    sequence: Sequence[Any], num_inputs: int, num_outputs: int, num_jumps: int
):
    """
    Returns sets to train and cross-validate a model using forward chaining technique

    Parameters:
        sequence (array)  : Full training dataset
        num_inputs (int)   : Number of inputs X and Xcv used at each training and validation
        num_outputs (int)  : Number of outputs y and ycv used at each training and validation
        num_jumps (int)    : Number of sequence samples to be ignored between (X,y) sets
    Returns:
        X (2D array)    : Array of numInputs arrays used for training
        y (2D array)    : Array of numOutputs arrays used for training
        Xcv (2D array)  : Array of numInputs arrays used for cross-validation
        ycv (2D array)  : Array of numOutputs arrays used for cross-validation
    """

    X = y = Xcv = ycv = {}
    j = 2
    # Tracks index of CV set at each train/val split
    # Iterate through all train/val splits
    while True:
        end_ix = 0
        X_it = y_it = Xcv_it = ycv_it = []
        i = 0
        # Index of individual training set at each train/val split
        # Iterate until index of individual training set is smaller than index of cv set
        while i < j:
            start_ix = num_jumps * i
            end_ix = start_ix + num_inputs
            seq_x = sequence[start_ix:end_ix]
            X_it.append(seq_x)
            seq_y = sequence[end_ix : end_ix + num_outputs]
            y_it.append(seq_y)
            i += 1

        # Once val data crosses time series length return
        if ((end_ix + num_inputs) + num_outputs) > len(sequence):
            break

        # CROSS-VALIDATION DATA
        startCv_ix = end_ix
        endCv_ix = end_ix + num_inputs

        seq_xcv = sequence[startCv_ix:endCv_ix]
        Xcv_it.append(seq_xcv)
        seq_ycv = sequence[endCv_ix : endCv_ix + num_outputs]
        ycv_it.append(seq_ycv)

        # Add another train/val split
        X[j - 2] = np.array(X_it)
        y[j - 2] = np.array(y_it)
        Xcv[j - 2] = np.array(Xcv_it)
        ycv[j - 2] = np.array(ycv_it)
        j += 1

    if len(X) == 0 or len(Xcv) == 0:
        warnings.warn(
            "The sequence provided does not has size enough to populate the return arrays"
        )

    return X, y, Xcv, ycv


def split_train_val_group_k_fold(
    seq: Sequence[Any], num_inputs: int, num_outputs: int, num_jumps: int
):
    """Returns sets to train and cross-validate a model using group K-Fold technique

    Parameters:
        seq (array)  : Full training dataset
        num_inputs (int)   : Number of inputs X and Xcv used at each training
        num_outputs (int)  : Number of outputs y and ycv used at each training
        num_jumps (int)    : Number of sequence samples to be ignored between (X,y) sets
    Returns:
        X (2D array)    : Array of numInputs arrays used for training
        y (2D array)    : Array of numOutputs arrays used for training
        Xcv (2D array)  : Array of numInputs arrays used for cross-validation
        ycv (2D array)  : Array of numOutputs arrays used for cross-validation

    """

    X = y = Xcv = ycv = {}

    # Iterate through 5 train/val splits
    for j in np.arange(5):
        start_ix = 0
        end_ix = 0
        startCv_ix = 0
        endCv_ix = 0
        X_it, y_it, Xcv_it, ycv_it = [], [], [], []
        i = 0
        # Index of individual training set at each train/val split
        n = 0
        # Number of numJumps

        while 1:
            if ((i + 1 + j) % 5) != 0:
                # TRAINING DATA
                start_ix = endCv_ix + num_jumps * n
                end_ix = start_ix + num_inputs
                n += 1

                # Leave train/val split loop once training data crosses time series length
                if end_ix + num_outputs > len(seq) - 1:
                    break

                seq_x = seq[start_ix:end_ix]
                X_it.append(seq_x)
                seq_y = seq[end_ix : end_ix + num_outputs]
                y_it.append(seq_y)
            else:
                # CROSS-VALIDATION DATA
                startCv_ix = end_ix
                endCv_ix = end_ix + num_inputs
                n = 0

                # Once val data crosses time series length return
                if (endCv_ix + num_outputs) > len(seq):
                    break

                seq_xcv = seq[startCv_ix:endCv_ix]
                Xcv_it.append(seq_xcv)
                seq_ycv = seq[endCv_ix : endCv_ix + num_outputs]
                ycv_it.append(seq_ycv)

            i += 1

        # Add another train/val split
        X[j] = np.array(X_it)
        y[j] = np.array(y_it)
        Xcv[j] = np.array(Xcv_it)
        ycv[j] = np.array(ycv_it)

    if len(X) == 0 or len(Xcv) == 0:
        print(
            "The sequence provided does not has size enough to populate the return arrays"
        )

    return X, y, Xcv, ycv


def split_train_val_test_forward_chaining(
    seq: Sequence[Any], num_inputs: int, num_outputs: int, num_jumps: int
):
    """
    Returns sets to train, cross-validate and test a model using forward chaining technique

    Parameters:
        seq (array)  : Full training dataset
        num_inputs (int)   : Number of inputs X and Xcv used at each training
        num_outputs (int)  : Number of outputs y and ycv used at each training
        num_jumps (int)    : Number of sequence samples to be ignored between (X,y) sets
    Returns:
        X (2D array)      : Array of numInputs arrays used for training
        y (2D array)      : Array of numOutputs arrays used for training
        Xcv (2D array)    : Array of numInputs arrays used for cross-validation
        ycv (2D array)    : Array of numOutputs arrays used for cross-validation
        Xtest (2D array)  : Array of numInputs arrays used for testing
        ytest (2D array)  : Array of numOutputs arrays used for testing
    """

    X = y = Xcv = ycv = Xtest = ytest = {}
    j = 2
    # Tracks index of CV set at each train/val/test split

    # Iterate through all train/val/test splits
    while 1:
        start_ix = 0
        end_ix = 0
        startCv_ix = 0
        endCv_ix = 0
        startTest_ix = 0
        endTest_ix = 0
        X_it = y_it = Xcv_it = ycv_it = Xtest_it = ytest_it = []
        i = 0
        # Index of individual training set at each train/val/test split

        # Iterate until index of individual training set is smaller than index of cv set
        while i < j:
            # TRAINING DATA
            start_ix = num_jumps * i
            end_ix = start_ix + num_inputs

            seq_x = seq[start_ix:end_ix]
            X_it.append(seq_x)
            seq_y = seq[end_ix : end_ix + num_outputs]
            y_it.append(seq_y)

            i += 1

        # Once test data crosses time series length return
        if (((end_ix + num_inputs) + num_inputs) + num_outputs) > (len(seq)):
            break

        # CROSS-VALIDATION DATA
        startCv_ix = end_ix
        endCv_ix = end_ix + num_inputs

        seq_xcv = seq[startCv_ix:endCv_ix]
        Xcv_it.append(seq_xcv)
        seq_ycv = seq[endCv_ix : endCv_ix + num_outputs]
        ycv_it.append(seq_ycv)

        # TEST DATA
        startTest_ix = endCv_ix
        endTest_ix = endCv_ix + num_inputs

        seq_xtest = seq[startTest_ix:endTest_ix]
        Xtest_it.append(seq_xtest)
        seq_ytest = seq[endTest_ix : endTest_ix + num_outputs]
        ytest_it.append(seq_ytest)

        # Add another train/val/test split
        X[j - 2] = np.array(X_it)
        y[j - 2] = np.array(y_it)
        Xcv[j - 2] = np.array(Xcv_it)
        ycv[j - 2] = np.array(ycv_it)
        Xtest[j - 2] = np.array(Xtest_it)
        ytest[j - 2] = np.array(ytest_it)
        j += 1

    if len(X) == 0 or len(Xcv) == 0 or len(Xtest) == 0:
        raise RuntimeWarning(
            "The sequence provided does not has size enough to populate the return arrays"
        )

    return X, y, Xcv, ycv, Xtest, ytest


def split_train_val_test_k_fold(
    seq: Sequence[Any], num_inputs: int, num_outputs: int, num_jumps: int
):
    """
    Returns sets to train, cross-validate and test a model using K-Fold technique

    Parameters:
        seq (array)  : Full training dataset
        num_inputs (int)   : Number of inputs X and Xcv used at each training
        num_outputs (int)  : Number of outputs y and ycv used at each training
        num_jumps (int)    : Number of sequence samples to be ignored between (X,y) sets
    Returns:
        X (2D array)      : Array of numInputs arrays used for training
        y (2D array)      : Array of numOutputs arrays used for training
        Xcv (2D array)    : Array of numInputs arrays used for cross-validation
        ycv (2D array)    : Array of numOutputs arrays used for cross-validation
        Xtest (2D array)  : Array of numInputs arrays used for testing
        ytest (2D array)  : Array of numOutputs arrays used for testing

    """

    X = y = Xcv = ycv = Xtest = ytest = {}
    j = 2
    # Tracks index of CV set at each train/val/test split
    theEnd = 0
    # Flag to terminate function

    # Iterate until test set falls outside time series length
    while 1:
        start_ix = 0
        end_ix = 0
        startCv_ix = 0
        endCv_ix = 0
        startTest_ix = 0
        endTest_ix = 0
        X_it = y_it = Xcv_it = ycv_it = Xtest_it = ytest_it = []
        i = 0
        # Index of individual training set at each train/val/test split
        n = 0
        # Number of numJumps

        # Iterate through all train/val/test splits
        while 1:
            if i != j:
                # TRAINING DATA
                start_ix = endTest_ix + num_jumps * n
                end_ix = start_ix + num_inputs
                n += 1

                # Leave train/val/test split loop once training data crosses time series length
                if end_ix + num_outputs > len(seq):
                    break

                seq_x = seq[start_ix:end_ix]
                X_it.append(seq_x)
                seq_y = seq[end_ix : end_ix + num_outputs]
                y_it.append(seq_y)
            else:

                # Once test data crosses time series length return
                if (((end_ix + num_inputs) + num_inputs) + num_outputs) > (len(seq)):
                    theEnd = 1
                    break

                n = 0
                i += 1

                # CROSS-VALIDATION DATA
                startCv_ix = end_ix
                endCv_ix = end_ix + num_inputs

                seq_xcv = seq[startCv_ix:endCv_ix]
                Xcv_it.append(seq_xcv)
                seq_ycv = seq[endCv_ix : endCv_ix + num_outputs]
                ycv_it.append(seq_ycv)

                # TEST DATA
                startTest_ix = endCv_ix
                endTest_ix = endCv_ix + num_inputs

                seq_xtest = seq[startTest_ix:endTest_ix]
                Xtest_it.append(seq_xtest)
                seq_ytest = seq[endTest_ix : endTest_ix + num_outputs]
                ytest_it.append(seq_ytest)

            i += 1

        # Only add a train/val/test split if the time series length has not been crossed
        if theEnd == 1:
            break

        # Add another train/val/test split
        X[j - 2] = np.array(X_it)
        y[j - 2] = np.array(y_it)
        Xcv[j - 2] = np.array(Xcv_it)
        ycv[j - 2] = np.array(ycv_it)
        Xtest[j - 2] = np.array(Xtest_it)
        ytest[j - 2] = np.array(ytest_it)

        j += 1

    if len(X) == 0 or len(Xcv) == 0 or len(Xtest) == 0:
        raise RuntimeWarning(
            "The sequence provided does not has size enough to populate the return arrays"
        )

    return X, y, Xcv, ycv, Xtest, ytest


def split_train_val_test_group_k_fold(
    seq: Sequence[Any], num_inputs: int, num_outputs: int, num_jumps: int
):
    """
    Returns sets to train, cross-validate and test a model using group K-Fold technique

    Parameters:
        seq (array)  : Full training dataset
        num_inputs (int)   : Number of inputs X and Xcv used at each training
        num_outputs (int)  : Number of outputs y and ycv used at each training
        num_jumps (int)    : Number of sequence samples to be ignored between (X,y) sets
    Returns:
        X (2D array)      : Array of numInputs arrays used for training
        y (2D array)      : Array of numOutputs arrays used for training
        Xcv (2D array)    : Array of numInputs arrays used for cross-validation
        ycv (2D array)    : Array of numOutputs arrays used for cross-validation
        Xtest (2D array)  : Array of numInputs arrays used for testing
        ytest (2D array)  : Array of numOutputs arrays used for testing

    """

    X = y = Xcv = ycv = Xtest = ytest = {}

    # Iterate through 5 train/val/test splits
    for j in np.arange(5):
        start_ix = 0
        end_ix = 0
        startCv_ix = 0
        endCv_ix = 0
        startTest_ix = 0
        endTest_ix = 0
        X_it = y_it = Xcv_it = ycv_it = Xtest_it = ytest_it = []
        i = 0
        # Index of individual training set at each train/val/test split
        n = 0
        # Number of numJumps

        while 1:
            if ((i + 1 + j) % num_jumps) != 0:
                # TRAINING DATA
                start_ix = endTest_ix + num_jumps * n
                end_ix = start_ix + num_inputs
                n += 1

                # Leave train/val/test split loop if train data crosses time series length
                if end_ix + num_outputs > len(seq):
                    break

                seq_x = seq[start_ix:end_ix]
                X_it.append(seq_x)
                seq_y = seq[end_ix : end_ix + num_outputs]
                y_it.append(seq_y)
            else:
                # CROSS-VALIDATION DATA
                startCv_ix = end_ix
                endCv_ix = end_ix + num_inputs

                # Leave train/val/test split loop if val data crosses time series length
                if (endCv_ix + num_outputs) > len(seq):
                    break

                seq_xcv = seq[startCv_ix:endCv_ix]
                Xcv_it.append(seq_xcv)
                seq_ycv = seq[endCv_ix : endCv_ix + num_outputs]
                ycv_it.append(seq_ycv)

                # TEST DATA
                startTest_ix = endCv_ix
                endTest_ix = endCv_ix + num_inputs

                # Leave train/val/test split loop if test data crosses time series length
                if (endTest_ix + num_outputs) > len(seq):
                    break

                seq_xtest = seq[startTest_ix:endTest_ix]
                Xtest_it.append(seq_xtest)
                seq_ytest = seq[endTest_ix : endTest_ix + num_outputs]
                ytest_it.append(seq_ytest)

                n = 0
                i += 1

            i += 1

        # Add another train/val split
        X[j] = np.array(X_it)
        y[j] = np.array(y_it)
        Xcv[j] = np.array(Xcv_it)
        ycv[j] = np.array(ycv_it)
        Xtest[j] = np.array(Xtest_it)
        ytest[j] = np.array(ytest_it)

    if len(X) == 0 or len(Xcv) == 0 or len(Xtest) == 0:
        raise RuntimeWarning(
            "The sequence provided does not has size enough to populate the return arrays"
        )

    return X, y, Xcv, ycv, Xtest, ytest


def split_train_val_k_fold(
    seq: Sequence[Any], num_inputs: int, num_outputs: int, num_jumps: int
):
    """Returns sets to train and cross-validate a model using K-Fold technique

    Parameters:
        seq (array)  : Full training dataset
        num_inputs (int)   : Number of inputs X and Xcv used at each training
        num_outputs (int)  : Number of outputs y and ycv used at each training
        num_jumps (int)    : Number of sequence samples to be ignored between (X,y) sets
    Returns:
        X (2D array)    : Array of numInputs arrays used for training
        y (2D array)    : Array of numOutputs arrays used for training
        Xcv (2D array)  : Array of numInputs arrays used for cross-validation
        ycv (2D array)  : Array of numOutputs arrays used for cross-validation

    """

    X, y, Xcv, ycv = {}, {}, {}, {}
    j = 2
    # Tracks index of CV set at each train/val split
    theEnd = 0  # Flag to terminate function

    # Iterate until val set falls outside time series length
    while 1:
        start_ix = 0
        end_ix = 0
        startCv_ix = 0
        endCv_ix = 0
        X_it = y_it = Xcv_it = ycv_it = []
        i = 0
        # Index of individual training set at each train/val split
        n = 0
        # Number of numJumps
        # Iterate through all train/val splits
        while 1:
            if i != j:
                # TRAINING DATA
                start_ix = endCv_ix + num_jumps * n
                end_ix = start_ix + num_inputs
                n += 1
                # Leave train/val split loop once training data crosses time series length
                if end_ix + num_outputs > len(seq):
                    break

                seq_x = seq[start_ix:end_ix]
                X_it.append(seq_x)
                seq_y = seq[end_ix : end_ix + num_outputs]
                y_it.append(seq_y)
            else:
                # CROSS-VALIDATION DATA
                startCv_ix = end_ix
                endCv_ix = end_ix + num_inputs
                n = 0

                # Once val data crosses time series length exit tran/val split loop and return
                if endCv_ix + num_outputs > len(seq):
                    theEnd = 1
                    break

                seq_xcv = seq[startCv_ix:endCv_ix]
                Xcv_it.append(seq_xcv)
                seq_ycv = seq[endCv_ix : endCv_ix + num_outputs]
                ycv_it.append(seq_ycv)
            i += 1

        # Only add a train/val split if the time series length has not been crossed
        if theEnd == 1:
            break

        # Add another train/val split
        X[j - 2] = np.array(X_it)
        y[j - 2] = np.array(y_it)
        Xcv[j - 2] = np.array(Xcv_it)
        ycv[j - 2] = np.array(ycv_it)
        j += 1

    if len(X) == 0 or len(Xcv) == 0:
        raise RuntimeWarning(
            "The sequence provided does not has size enough to populate the return arrays"
        )
    return X, y, Xcv, ycv


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
    for start_ix, end_ix in test_times.iteritems():
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

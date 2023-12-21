#!/usr/bin/env python

"""Tests for `skportfolio` package."""

import numpy as np

# import pandas as pd
from pytest import fixture

# from skportfolio.model_selection.deprecated import BlockingTimeSeriesSplit
# from skportfolio.model_selection.deprecated import KFold
# from skportfolio.model_selection.deprecated import NoSplit
# from skportfolio.model_selection.deprecated import SplitTrainFixed
# from skportfolio.model_selection.deprecated import SplitTrainMinTrain
# from skportfolio.model_selection.deprecated import SplitTrainValForwardChaining
# from skportfolio.model_selection.deprecated import WindowedTestTimeSeriesSplit
from skportfolio.model_selection import make_split_df


def flatten(S):
    if not S:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


def get_flatten_fold(fold):
    return sorted(set(flatten(fold)))


@fixture
def data():
    yield np.arange(20)


# def test_no_split(data):
#     cv = NoSplit()
#     make_split_df(cv.split(data), titles=("train", "test", "validation"))
#
#
# def test_k_fold(data):
#     cv = KFold()
#     make_split_df(cv.split(data), titles=("train", "test", "validation"))
#
#
# def test_train_fixed(data):
#     cv = SplitTrainFixed(num_X=5, num_y=3, num_jumps=2)
#     make_split_df(
#         cv.split(data),
#         titles=("train", "test", "validation"),
#         columns=pd.date_range(start="2020-01-01", periods=20),
#     )


# def test_split_train_min_fixed(data):
#     cv = SplitTrainMinTrain(min_num_x=5, num_y=3, num_jumps=2)
#     make_split_df(cv.split(data), titles=("train", "test", "validation"))


# def test_split_train_val_forward_chaining(data):
#     cv = SplitTrainValForwardChaining(num_X=5, num_y=3, num_jumps=3)
#     make_split_df(cv.split(data), titles=("train", "test", "cv_train", "cv_test"))


# def test_blocking_time_series_split(data):
#     cv = BlockingTimeSeriesSplit(n_splits=5, train_test_ratio=0.7)
#     make_split_df(cv.split(data), titles=("train", "test", "cv_train", "cv_test"))


#
# def test_purged_k_fold(data):
#     cv = PurgedKFold(pct_embargo=0.0)
#     print_split(cv.split(data), titles=("train", "test", "cv_train", "cv_test"))


# def test_windowed_test_timeseries_split(data):
#     cv = WindowedTestTimeSeriesSplit(n_splits=5, max_train_size=5, n_test_folds=2)
#     make_split_df(cv.split(data), titles=("train", "test", "cv_train", "cv_test"))


# def test_cpcv(data):
#     cv = CombinatorialPurgedKFold(n_splits=10, n_test_splits=5)
#     make_split_df(cv.split(data), titles=("train", "test", "cv_train", "cv_test"))

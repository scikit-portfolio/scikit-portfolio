from abc import ABC

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection._split import BaseCrossValidator
from torch.utils.data import DataLoader
from more_itertools import batched


def pandas_to_tensor(dataframe: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(dataframe.astype(np.float32).values)


class SlidingWindow(BaseCrossValidator, ABC):
    def __init__(self, window_size):
        self.window_size = window_size

    def split(self, X, y=None, groups=None):
        n, i = len(X), 0
        idx = tuple(range(n))
        while i + 2 * self.window_size < n - 1:
            yield idx[i : (i + self.window_size)], idx[
                (i + self.window_size) : (i + 2 * self.window_size)
            ]
            i = i + 1

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(tuple(_ for _ in self.split(X)))


class TimeSeriesCVDataLoader(DataLoader):
    def __init__(self, dataset, cv_splitter: BaseCrossValidator, batch_size):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=1,
            persistent_workers=True,
        )
        self.cv_splitter = cv_splitter

    def make_iter(self):
        for train_index, test_index in self.cv_splitter.split(self.dataset):
            yield self.dataset[train_index, :], self.dataset[test_index, :]

    def __iter__(self):
        train_getter = (x[0] for x in self.make_iter())
        test_getter = (x[1] for x in self.make_iter())
        for i, (batch_train_data, batch_test_data) in enumerate(
            zip(
                batched(train_getter, self.batch_size),
                batched(test_getter, self.batch_size),
            )
        ):
            train_data = torch.stack(batch_train_data)
            test_data = torch.stack(batch_test_data)
            yield train_data, test_data


class AssetDataModule(pl.LightningDataModule):
    def __init__(self, dataset, test_horizon: int, split_dataloader, batch_size: int):
        super().__init__()
        self.dataset = dataset[:-test_horizon]
        self.test_dataset = dataset[-test_horizon:]
        self.split_dataloader = split_dataloader
        self._split_data_loader = TimeSeriesCVDataLoader(
            self.dataset, cv_splitter=self.split_dataloader, batch_size=batch_size
        )

    def train_val_getter(self, loader, what):
        if what == "train":
            for train_data, _ in loader:
                yield train_data
        elif what == "val":
            for _, val_data in loader:
                yield val_data
        elif what in ("test", "predict"):
            for _, test_data in loader:
                yield test_data
        else:
            raise ValueError("Not supported fold")

    def train_dataloader(self):
        yield from self.train_val_getter(self._split_data_loader, what="train")

    # def val_dataloader(self):
    #     yield from self.train_val_getter(self._split_data_loader, what="val")

    def test_dataloader(self):
        yield from self.train_val_getter(self._split_data_loader, what="test")


if __name__ == "__main__":
    np.random.seed(0)
    T = 100
    num_assets = 5
    df = pd.DataFrame(data=np.random.randn(T, 2 * num_assets))
    splitter = SlidingWindow(
        window_size=20,
    )

    # print_split(splitter.split(df), titles=["train", "test"])

    datamodule = AssetDataModule(
        dataset=pandas_to_tensor(dataframe=df),
        test_horizon=5,
        split_dataloader=splitter,
        batch_size=8,
    )

    for train in datamodule.val_dataloader():
        print(train.shape)

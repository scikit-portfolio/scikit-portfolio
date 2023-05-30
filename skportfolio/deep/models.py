import matplotlib.pyplot as plt
from typing import Optional, List
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from skportfolio.deep.dataloaders import (
    AssetDataModule,
    pandas_to_tensor,
    SlidingWindow,
)
from skportfolio.model_selection import BlockingTimeSeriesSplit
from skportfolio.logger import get_logger

# logger = get_logger()


class Zhang(pl.LightningModule):
    def __init__(
        self,
        num_assets,
        hidden_dim,
        seq_length,
        asset_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.num_assets = num_assets
        self.input_dim = 2 * num_assets
        self.hidden_dim = hidden_dim
        self.output_dim = num_assets
        self.seq_length = seq_length
        self.asset_names = asset_names
        # Defines the network architecture
        self.neural_layer = nn.LSTM(
            self.input_dim, hidden_dim, num_layers=16, batch_first=True
        )
        self.dropout_layer = nn.Dropout(p=0.25)
        self.output_layer = nn.Linear(
            self.seq_length * self.hidden_dim, self.output_dim
        )
        self.flattener = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        x0, _ = self.neural_layer(x)
        x1 = self.flattener(x0)
        # x1 = self.dropout_layer(x1)
        x2 = self.output_layer(x1)
        x3 = F.softmax(x2, dim=1)
        return x3

    def sharpe_ratio(self, returns, weights):
        portfolio_return = torch.einsum("bti,bi->t", returns, weights)
        expected_ptf_return = portfolio_return.mean()
        expected_ptf_vol = portfolio_return.std()
        return expected_ptf_return / expected_ptf_vol

    def training_step(self, batch, batch_idx):
        weights_pred = self.forward(batch)
        returns = batch[:, :, self.input_dim // 2 :]
        loss = self.sharpe_ratio(returns, weights_pred)
        self.log("sharpe", loss, prog_bar=True)
        for i in range(self.output_dim):
            self.log(f"{self.asset_names[i]}", weights_pred.mean(0)[i])
        return loss

    # def validation_step(self, batch, batch_idx):
    #     weights_pred = self.forward(batch)
    #     returns = batch[:, :, self.input_dim // 2 :]
    #     loss = self.sharpe_ratio_loss(returns, weights_pred)
    #     self.log("sharpe", loss, prog_bar=True)
    #     for i in range(self.output_dim):
    #         self.log(f"{self.asset_names[i]}", weights_pred.mean(0)[i])

    def test_step(self, batch, batch_idx):
        weights_pred = self.forward(batch)
        returns = batch[:, :, self.input_dim // 2 :]
        loss = self.sharpe_ratio(returns, weights_pred)
        self.log("sharpe", loss, prog_bar=True)
        for i in range(self.output_dim):
            self.log(f"{self.asset_names[i]}", weights_pred.mean(0)[i])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, maximize=True)
        return optimizer


def make_data():
    from skportfolio.datasets import load_tech_stock_prices, load_crypto_prices

    prices = load_tech_stock_prices()
    rets = prices.pct_change()
    return (
        pandas_to_tensor(pd.concat((prices, rets), axis=1).dropna()),
        prices.columns.tolist(),
    )


def get_n_samples_n_assets(dataset):
    return dataset.shape[0], dataset.shape[1] // 2


def main():
    # define dataset
    dataset, asset_names = make_data()
    num_samples, num_assets = get_n_samples_n_assets(dataset)

    window_size = 50
    splitter = SlidingWindow(window_size=window_size)

    asset_datamodule = AssetDataModule(
        dataset=dataset, test_horizon=252, split_dataloader=splitter, batch_size=64
    )

    model = Zhang(
        num_assets=num_assets,
        hidden_dim=64,
        seq_length=window_size,
        asset_names=asset_names,
    )
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

    TensorBoardLogger(save_dir="/Users/carlo/models/scikit-portfolio/zhang", version=0)
    logger = CSVLogger(
        save_dir="/Users/carlo/models/scikit-portfolio/zhang",
        version=0,
        name="scikit-portfolio-logs",
    )
    trainer = pl.Trainer(
        accelerator="mps",
        logger=logger,
        log_every_n_steps=50,
        max_epochs=100,
        num_sanity_val_steps=None,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        reload_dataloaders_every_n_epochs=1,
    )

    # Train the model
    trainer.fit(model, datamodule=asset_datamodule)
    trainer.test(model=model, datamodule=asset_datamodule)

    output = pd.read_csv(
        "/Users/carlo/models/scikit-portfolio/zhang/scikit-portfolio-logs/version_0/metrics.csv"
    ).set_index("step")
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    output.drop(["sharpe", "epoch"], axis=1).plot(ax=ax[0], title="Asset weights")
    output["sharpe"].plot(ax=ax[1], title="Sharpe ratio")
    plt.show()


if __name__ == "__main__":
    main()

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl

import numpy as np

# config
from hydra import utils


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        test_batch_size,
        data_type,
        num_workers,
        pin_memory,
        **kwargs,
    ):
        super().__init__()

        # Save parameters to self
        self.data_dir = utils.get_original_cwd() + data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.permuted = kwargs["permuted"]

        # Determine sizes of dataset
        self.input_channels = 1
        self.output_channels = 10

        # Determine data_type
        if data_type == "default":
            self.data_type = "image"
            self.data_dim = 2
        elif data_type == "sequence":
            self.data_type = data_type
            self.data_dim = 1
        else:
            raise ValueError(f"data_type {data_type} not supported.")

        # Create transform
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        # Define permutation if needed
        if self.data_type == "sequence" and self.permuted:
            self.permutation = torch.tensor(
                np.random.permutation(784).astype(np.float64)
            ).long()

    def prepare_data(self):
        # download data, train then test
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # we set up only relevant datamodules when stage is specified
        if stage == "fit" or stage is None:
            mnist = datasets.MNIST(
                self.data_dir,
                train=True,
                transform=self.transform,
            )
            self.train_dataset, self.val_dataset = random_split(mnist, [55000, 5000])
        if stage == "test" or stage is None:
            self.test_dataset = datasets.MNIST(
                self.data_dir,
                train=False,
                transform=self.transform,
            )

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return test_dataloader

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if self.data_type == "sequence":
            # If sequential, flatten the input [B, C, Y, X] -> [B, C, -1]
            x, y = batch
            x_shape = x.shape
            x = x.view(x_shape[0], x_shape[1], -1)

            if self.permuted:
                # If permuted, apply self.permutation
                x = x[:, :, self.permutation]
            batch = x, y
        return batch
